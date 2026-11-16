# Segment-count-aware score-based balancing

## Background

QueryCoord's default balancer (`ChannelLevelScoreBalancer`, which reuses the
score-based segment assignment policy) equalizes each QueryNode's **row count /
memory**. A node's balance score is:

```
score(node) = collectionRowCount(node) + globalRowCount(node) * globalRowCountFactor
```

Segment **count** is never part of the score. This is fine when every segment is
the same size, but Milvus L1 segments vary widely in size (see the companion
under-fill compaction work: sealed/compacted segments range roughly
`0.5*maxSize`–`1.25*maxSize`). When segment sizes differ, "equal rows per node"
no longer implies "equal segments per node".

### Observed symptom

Single collection, 32MB target segments, scaled out from 4 to 8 QueryNodes:

| metric        | result                                              |
|---------------|-----------------------------------------------------|
| rows per node | near-perfectly even (score converges within ~0.1%)  |
| segments/node | ~34–41 on original nodes vs ~69–74 on new nodes (~2.18x skew), stable indefinitely |

The balancer moves segments only until row/memory score equalizes, then stops —
so the original nodes keep the larger segments and the scaled-out nodes keep the
smaller ones. Equal rows, unequal segment counts. This inflates per-node query
fan-out / read concurrency on the high-count nodes (see milvus-io/milvus #49707).

## Goal

Let operators make the score-based balancer *also* equalize segment count,
without changing default behavior.

## Design

Add a configurable per-segment weight, in row-count-equivalent units:

```
queryCoord.scoreBasedBalanceSegmentCountFactor   (default 0)
```

When `> 0`, each segment contributes `segmentCountFactor` "rows" to the node
score, weighted collection-locally and globally exactly like the row-count term:

```
score(node) = collectionRowCount + globalRowCount * globalRowCountFactor
            + segmentCountFactor * ( collectionSegmentCount
                                     + globalSegmentCount * globalRowCountFactor )
```

and the per-segment move cost (`CalculateSegmentScore`) gains the matching term:

```
segmentScore = rows * (1 + globalRowCountFactor)
             + segmentCountFactor * (1 + globalRowCountFactor)
```

The two are consistent: summing the per-segment cost over a node's segments
reproduces the node-level term, so the existing benefit-evaluation and priority
math (`CurrentScore - AssignedScore`) stay coherent.

### Behavior

- `segmentCountFactor = 0` (default): the extra terms are skipped entirely — byte
  -for-byte the current row-only behavior.
- `segmentCountFactor > 0`: a node holding more segments scores higher even at
  equal rows, so the balancer moves segments off high-count nodes until both row
  count and segment count are balanced. Setting it near the typical rows/segment
  makes segment count roughly as important as row count; smaller values make it a
  gentle tie-breaker.

### Why reuse the row-count units

Expressing the weight in row-equivalent units keeps a single, comparable score
scale (rows) rather than introducing a separately-normalized cost model. The
existing `MultiTargetBalancer` already has a normalized segment-count cost model;
this change brings a lightweight equivalent to the *default* score-based path
without forcing operators to switch balancer implementations.

## Compatibility & safety

- Default `0` ⇒ no behavior change; safe to ship dark.
- `refreshable:"true"` ⇒ tunable at runtime.
- Segment counts are gathered in the same `getWorkloadStatus` pass that already
  scans the segment distribution, so there is no extra scan cost.
- In-flight task deltas remain row-based (segment-count delta from scheduled-but
  -not-applied tasks is not modeled); this only slightly under-damps during churn
  and self-corrects on the next balance round.

## Testing

Unit tests (`assign_policy_score_segmentcount_test.go`) simulate the exact
failure state — two nodes, equal rows, 4 vs 1 segments:

- factor `0`: node scores are equal (no signal to rebalance count);
- factor `> 0`: the 4-segment node scores strictly higher by the expected gap;
- `CalculateSegmentScore` gains exactly `factor*(1+globalRowCountFactor)`;
- `AssignSegment` sends a new segment to the fewer-segment node when enabled.
