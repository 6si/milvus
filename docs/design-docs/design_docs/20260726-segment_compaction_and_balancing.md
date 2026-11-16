# Segment compaction under-fill + segment-count-aware balancing

Status: Draft
Date: 2026-07-26
Related PRs: 6si/milvus#16 (compaction), 6si/milvus#17 (balancer)
Supersedes/combines: `20260724-underfill_compaction.md`, `20260724-segment_count_balancing.md`

## 1. Motivation

Two operator-reported problems on Milvus 2.x, single collection, 32MB target
segments, after large ingestion:

1. **Small / half-full segments hang around.** After compaction settles many
   sealed L1 segments sit far below the `maxSize` target and are never merged
   again.
2. **The balancer does not distribute segments close to even.** After a
   scale-out, QueryNodes end up with near-equal *rows/memory* but very unequal
   *segment counts*.

Both were reproduced locally on a real **v2.6.18** cluster (custom image built
from the fork with both fixes compiled in), 32MB segments, single collection.

## 2. Reproduction summary (baseline, stock 2.6.18)

Single collection, 32MB segments:

- 5M rows → ~600 sealed L1 segments spread **16MB → 35MB** (≈2.2× size ratio);
  only ~35 hit the full 32MB target, ~162 sat at ≤20MB (half-full).
- Row/size distribution histogram showed a wide plateau, a hard floor at
  `smallProportion*maxSize` (=16MB at defaults), and a ceiling near
  `expansionRate*maxSize` (=40MB).
- Scale-out to 8 QueryNodes: rows equalized to ±1% per node, but segment counts
  were skewed ~2.18× (e.g. 34–41 vs 69–74 per node).

Config tuning cannot fix it (measured, each value the only change from default):

| `smallProportion` | segments | floor | verdict |
|---|---|---|---|
| 0.25 | ~1014 | 8MB | more, smaller (mild regression) |
| **0.5 (default)** | **~600** | 16MB | reference (best of the three) |
| 0.8 | ~6100–6900 | 2MB | ~10× regression, merge never reconverges |

`compactableProportion` 0.85→0.98: no material change (607 vs 600) — it only
controls how full a merge *bucket* must be, not *which* segments are eligible.

## 3. Root cause

### 3.1 Compaction dead-band (issue 1)

`internal/datacoord/compaction_trigger.go::generatePlans` classifies each
sealed segment as:

- prioritized (force / `ShouldDoSingleCompaction`), or
- small (`isSmallSegment`: raw size `< smallProportion*maxSize`), or
- otherwise **non-planned** (left untouched).

So any segment in `[smallProportion*maxSize, maxSize)` — the **dead-band** — is
neither small enough to merge nor full enough to be complete, and is **frozen
forever** even at half capacity. Eligibility uses *raw* size, so segments whose
live data shrank via deletes/upserts (but whose raw size is still near target,
below the `hasTooManyDeletions` single-compaction threshold) also never reclaim
space.

### 3.2 Balancer objective mismatch (issue 2)

The default 2.6 balancer is `ScoreBasedBalancer`. Its node score
(`internal/querycoordv2/assign/assign_policy_score.go`) is:

```
score = collectionRowCount + globalRowCount * globalRowCountFactor
```

There is **no segment-count term** in this path. (The existing
`queryCoord.segmentCountFactor` / `globalSegmentCountFactor` are consumed by
`MultiTargetBalancer`, not this default score path.) The balancer therefore
moves segments only until rows/memory equalize; with non-uniform segment sizes,
equal rows ⇒ unequal counts, and the skew is stable indefinitely.

### 3.3 Why sizes vary in the first place

`generatePlans` feeds a greedy first-fit-decreasing knapsack
(`knapsack.go`), bounded by `compactableProportion*maxSize` (accept floor) and
`expansionRate*maxSize` (overshoot ceiling). The leftover/"squeeze" bucket of
each round has no lower-remainder limit, so it lands anywhere down to the
dead-band floor — producing the plateau.

## 4. Design

### 4.1 Under-fill compaction pass (PR #16)

A 4th candidate class in `generatePlans`, gated behind a default-off flag:

- `isUnderFullSegment(seg)`: not small, and **delete-adjusted residual size**
  (`GetResidualSegmentSize()`) `< compactableProportion*maxSize`. Using residual
  (not raw) size is what also reclaims update/upsert/delete space.
- These segments (plus small leftovers) are packed into large buckets of
  `groupFactor*maxSize` via the existing packer and sent through the existing
  `MixCompaction` task + DataNode `MultiSegmentWriter` (which splits output at
  `plan.MaxSize`). No new compactor path.

Safety guards:

- Default off: `dataCoord.compaction.underfill.enable=false` → zero behavior
  change until opted in.
- **Strict count-reduction gate**: emit a group only if
  `ceil(Σ residual / maxSize) < len(pack)`. Never churns without shrinking
  count; also guarantees termination (candidates are consumed each pack).
- Low priority: skipped when the compaction inspector is full, so it never
  starves L0/clustering/single compaction.

Config: `dataCoord.compaction.underfill.enable` (bool, default false),
`dataCoord.compaction.underfill.groupFactor` (int, default 8).

### 4.2 Segment-count-aware balancer factor (PR #17)

Add an opt-in per-segment weight to the score-based path:

```
score  = collectionRowCount + globalRowCount * globalRowCountFactor
       + segmentCountFactor * (collectionSegmentCount
                               + globalSegmentCount * globalRowCountFactor)
```

and, consistently, in `CalculateSegmentScore`:

```
segScore = rows * (1 + globalRowCountFactor)
         + segmentCountFactor * (1 + globalRowCountFactor)
```

Each segment contributes `segmentCountFactor` "row-equivalents", using the same
collection-local + global structure and the same `globalRowCountFactor` weight
as rows, so it plugs into the existing `CurrentScore - AssignedScore` move-cost
mechanism.

Config: `queryCoord.scoreBasedBalanceSegmentCountFactor` (float, default `0` =
existing row-only behavior; refreshable at runtime).

## 5. Combined validation (both fixes, one v2.6.18 binary)

Built one custom v2.6.18 image with **both** changes; merged branch compiles and
both unit-test suites pass in a single build. Live cluster, single collection,
2M rows, 32MB segments.

**Compaction (both enabled):** 167 segments, median 29.4MB, mean ~12k rows/seg;
only 2 segments below 24MB (dead-band floor eliminated). Datacoord logs show the
under-fill pass firing (`nonFullCandidates` > 0, "under-fill repacking N
segments"). A tail of oversized outputs (36–64MB) remains — see §6.

**Balancer, 8 QueryNodes, A/B on identical data (factor is runtime-refreshable):**

| config | rows skew | segment-count skew |
|---|---|---|
| under-fill only (`factor=0`) | ~1.03× (very even) | **1.92×** (12 vs 21–23 / node) |
| both (`factor=10000`) | ~1.31× | **1.47×** (15 vs 21–22 / node) |

The factor reduces count skew (1.92× → 1.47×), directly addressing the
equal-rows/unequal-segments symptom.

Earlier 2.5M-row A/B on the compaction fix alone: 440 → 193 total segments,
149 tiny (<2k rows) → 0; after deleting 40% of rows, residual re-classification
repacked 193 → 96 segments (the delete/upsert residual path end-to-end).

## 6. Interaction / known limitation

Running both together exposes a real interaction. The single node that stays
count-skewed is the one holding under-fill's **oversized** outputs
(~60MB / ~20k rows each). To give that node more segments the balancer must move
~10k-row segments onto it, which overshoots its row count — so count-balancing
and row-balancing **conflict, but only because segment sizes are non-uniform.**

Cause of the oversized outputs: `MultiSegmentWriter.rotateWriterOrGrowCurrent`
checks rotation only *between whole `Write(record)` calls*, and
`GetWrittenUncompressed()` includes buffered bytes; under-fill feeds larger
grouped inputs, so the per-record overshoot is more visible (~2× vs ~1.15×).
This is a **pre-existing shared-writer property**, not under-fill logic.

Implications:
- Set `scoreBasedBalanceSegmentCountFactor` moderately (not as high as the
  10000 used above) while the caveat stands.
- A follow-up **shared-writer output cap** (split a record/batch at the
  `maxSize` boundary) would make outputs uniform and remove the tension so both
  objectives align. Scoped separately because it touches *all* compaction paths
  using `MultiSegmentWriter`.

## 7. Rollout

Both features default-off and independent; can merge in either order.
1. Ship under-fill (default off); operators enable per collection/cluster and
   watch segment count + inspector saturation.
2. Ship balancer factor (default 0); operators raise gradually and watch both
   row skew and segment-count skew.
3. Optionally add the shared-writer output cap to remove the §6 tension.

## 8. Test coverage

- `compaction_trigger_underfill_test.go`: dead-band frozen by default →
  repacked when enabled; tiny/full segments untouched; non-count-reducing packs
  skipped; delete/upsert residual reclamation; eligibility & count-reduction
  boundaries; adversarial multi-pack test (terminates, no duplicate/lost IDs,
  trailing sub-`minSegment` group deferred).
- `assign_policy_score_segmentcount_test.go`: equal rows / unequal counts →
  equal score when disabled, higher score for the high-count node when enabled;
  `CalculateSegmentScore` disabled/enabled; a new segment assigned to the
  fewer-segment node.
- Combined: both suites pass in one build; live v2.6.18 A/B above.

## 9. Related upstream GitHub issues (milvus-io/milvus)

Found while researching; linked for context.

**Compaction (issue 1):**
- **#49991 [Feature]: Target-based compaction** — asks for durable,
  target-size-driven compaction incl. "target-size optimize" rewrite campaigns.
  Closest conceptual match; the under-fill pass is a concrete step toward it.
  https://github.com/milvus-io/milvus/issues/49991
- **#49199 [Bug]: collection segment can not compact** — compaction "did not
  work" even after tuning `smallProportion`/`sealProportion` (2.5.12). Matches
  the symptom (low quality / `needs-information`).
  https://github.com/milvus-io/milvus/issues/49199
- **#50908 [Bug]: Manual force-merge collapses a large scope into one oversized
  compaction task** — force-merge planner grouping bug; adjacent to the planner
  changes here (and to the oversized-output caveat in §6).
  https://github.com/milvus-io/milvus/issues/50908

**Balancer / segment-count skew (issue 2):**
- **#49707 [Bug]: QueryNode read concurrency saturates on skewed segment
  distribution** (`triage/accepted`, area/performance) — the per-node
  segment-count skew symptom PR #17 targets.
  https://github.com/milvus-io/milvus/issues/49707
