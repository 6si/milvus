# Under-fill Compaction Pass

## Background

`dataCoord`'s L1 mix-compaction planner (`compaction_trigger.generatePlans`) classifies
every flushed candidate segment into one of three groups:

* **prioritized** — force / single compaction candidates (`ShouldDoSingleCompaction`),
* **small** — `getSegmentSize() < smallProportion * maxSize` (default `0.5 * maxSize`),
* **non-planned** — everything else.

Only *small* segments (plus prioritized ones) are ever merged. Any segment whose size is
between `smallProportion * maxSize` and `maxSize` is neither small nor full, so it lands in
`nonPlannedSegments` and is **never re-compacted** — a permanent dead-band.

### Observed symptoms (single collection, `dataCoord.segment.maxSize = 32MB`, 5M rows)

* ~600 output segments whose sizes spread from ~16MB to ~35MB (median ~25MB); roughly half sit
  at or below ~20MB — i.e. permanently half-full.
* Because the score-based query balancer equalizes **rows/memory** per querynode, unequal
  segment *sizes* translate directly into unequal segment *counts* after a scale-out
  (measured: 61 vs 89 segments per node at equal rows).

The dead-band is therefore the upstream cause of both "many small segments hang around" and
"segment counts are uneven across nodes".

Config tuning cannot close it:

| `smallProportion` | segments | notes |
|---|---|---|
| 0.25 | 1014 | floor drops to 8MB, more/smaller segments |
| 0.5 (default) | ~600 | fewest — sweet spot |
| 0.8 | ~6900 | merge never reconverges — ~10x regression |

Raising `compactableProportion` (0.85 → 0.98) has no effect: it changes how full an *accepted
bucket* must be, not which segments are *eligible*.

### Frequent-ingestion / upsert case

Upserts and deletes accumulate deltalogs. A segment with moderate deletes (below the
`SingleCompactionRatioThreshold`, default 0.2) is not picked up by single/L0 compaction, yet
its live data (residual size = `size * (1 - deletedRows/rows)`) may be well under target. Such
segments are also frozen in the dead-band, so their reclaimable space is never recovered.

## Design

Add a **low-priority under-fill pass** to `generatePlans`, gated behind a new config flag
(default **off**), that repacks dead-band segments into fewer, fuller segments.

New config (`dataCoord.compaction.underfill.*`):

* `enable` (bool, default `false`) — master switch.
* `groupFactor` (int, default `8`, min `2`) — under-filled segments are packed into buckets
  whose total residual size targets `groupFactor * maxSize`; the datanode `MultiSegmentWriter`
  then rotates each bucket into full `maxSize` output segments.

Classification adds a fourth bucket, evaluated only when `enable`:

```
isUnderFullSegment(seg) :=
    !isSmallSegment(seg) &&
    GetResidualSegmentSize(seg) < compactableProportion * maxSize
```

Using **residual** (delete-adjusted) size — not raw size — is what makes the pass reclaim
upsert/delete space: a raw-full segment with enough deletes becomes under-full and eligible.

The pass runs after the existing small-merge/squeeze phases and only when the compaction
pipeline has spare capacity:

```
if enable && !signal.isForce && !inspector.isFull():
    candidates = nonFullCandidates + leftover small segments
    pack candidates (greedy, residual-descending) into groups of ~groupFactor*maxSize
    for each group of >= minSegment segments:
        emit a MixCompaction task ONLY IF it strictly reduces segment count:
            ceil(sum(residual) / maxSize) < len(group)
```

### Safety properties

* **Default off** — zero change to existing behavior unless explicitly enabled.
* **Strict count reduction** — `underFillReducesCount` guarantees every emitted group produces
  fewer output segments than it consumes. This both avoids pointless churn (two near-full
  segments would just yield two full outputs) and guarantees **termination**: each round
  strictly reduces the count, and a lone leftover cannot form a group (needs `>= minSegment`),
  so the system converges instead of compacting forever.
* **Low priority** — skipped whenever `inspector.isFull()`, so it never starves L0/clustering/
  single compaction.
* **Reuses the existing compactor** — emits ordinary `MixCompaction` tasks; the datanode writer
  already splits output at `plan.MaxSize`, so no compactor change is required.

## Testing

`internal/datacoord/compaction_trigger_underfill_test.go`:

* dead-band medium segments produce **no** plan by default, and a single count-reducing group
  when enabled;
* tiny-segment merging and already-full segments are unaffected (no churn);
* a group that would not reduce the count is skipped;
* moderately-deleted (upsert) segments — frozen by default — are repacked when enabled;
* `isUnderFullSegment` / `underFillReducesCount` boundary tests.

## Rollout

Ship disabled. Enable per-cluster to tighten the segment-size distribution; measure segment
count and per-node segment distribution before/after. The separate score-based-balancer
segment-count skew is tracked independently.
