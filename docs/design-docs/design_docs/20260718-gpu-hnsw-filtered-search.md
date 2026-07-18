# MEP: GPU_HNSW Filtered Search — CPU-parity delete / TTL / partition filtering on GPU

- **Created:** 2026-07-18
- **Author(s):** @6si
- **Status:** Draft
- **Component:** Index | QueryNode
- **Related Issues:** milvus-io/milvus#50653, zilliztech/knowhere#1686
- **Related design docs:** [20260619-gpu-hnsw.md](20260619-gpu-hnsw.md),
  [20260711-gpu-hnsw-ocq-removal.md](20260711-gpu-hnsw-ocq-removal.md)
- **Released:** N/A

## 1. Summary

Today `GPU_HNSW` / `GPU_HNSW_SQ` **rejects** any search that carries a non-empty
`BitsetView` (deletes, TTL expiry, partition/visibility) with
`Status::invalid_args, "GPU_HNSW does not support filtered search"`. That limits
GPU_HNSW to append-mostly / immutable collections.

This MEP specifies **full CPU-HNSW-parity filtered search on the GPU**: the GPU
kernel will consume the same delete bitset Milvus already produces, exclude
filtered rows from results, keep traversing filtered rows as graph waypoints (so
recall is preserved), and fall back to a brute-force scan at high filter ratios —
matching the exact semantics of the CPU HNSW path in Knowhere.

**Scope of change:** faiss CUDA kernel (primary) + knowhere plumbing (minor) +
Milvus docs/tests (no functional Milvus code). Milvus already builds and passes
the bitset to `Search()`; nothing in the delegator / segcore delete pipeline
changes.

## 2. Background — how deletes reach the index

Milvus deletes are soft and MVCC-based (see the delete pipeline: L0 segments +
delta logs → per-segment `BitsetView` at query time, gated by `TSafe` /
guarantee timestamp). By the time a search reaches Knowhere, the delete / TTL /
partition state for a segment has been collapsed into a single `BitsetView` over
the segment's row offsets: **bit set ⇒ row is filtered out**. CPU indexes honor
this bitset; GPU_HNSW currently rejects it.

Current rejection (`knowhere/src/index/hnsw/faiss_hnsw.cc`,
`GpuHnswIndexNode::Search()`):

```cpp
if (!bitset.empty() && bitset.data() != nullptr) {
    return expected<DataSetPtr>::Err(Status::invalid_args,
                                     "GPU_HNSW does not support filtered search");
}
```

## 3. Target behavior — CPU HNSW parity (authoritative reference)

The GPU path must reproduce the CPU HNSW filtered-search semantics. Those live in
Knowhere:

### 3.1 Two-tier traversal: valid results vs. invalid frontier
`faiss/cppcontrib/knowhere/impl/Neighbor.h` — `NeighborSetDoublePopList` keeps
**two** lists:
- `valid_ns_` — the result beam; only **non-filtered** nodes. Produces top-k.
- `invalid_ns_` — filtered nodes retained **only for expansion** (navigation),
  and only while their distance beats the valid beam's back distance.

The search frontier pops the globally-closest of the two (`pop_based_on_distance`),
so filtered nodes are still expanded in distance order to preserve graph
connectivity, but never enter the result set (`Neighbor::kValid` /
`Neighbor::kInvalid` status, checked at `insert`).

### 3.2 `accumulated_alpha` / `kAlpha` admission gate
`faiss/cppcontrib/knowhere/impl/HnswSearcher.h` — `evaluate_single_node`:

```cpp
if (!filter.is_member(v1)) {          // v1 is filtered out (deleted)
    status = knowhere::Neighbor::kInvalid;
    accumulated_alpha += kAlpha;       // kAlpha = bitset.filter_ratio() * 0.7
    if (accumulated_alpha < 1.0f) {
        continue;                      // skip: don't even expand this filtered node
    }
    accumulated_alpha -= 1.0f;         // admit it as an (invalid) frontier node
}
```

Effect: at low filter ratios filtered nodes are rarely admitted as waypoints
(cheap); as the filter ratio rises `kAlpha → ~0.65` so more filtered nodes are
kept for navigation, preserving recall. `accumulated_alpha` starts at `1.0`, or
`FLT_MAX` (always-admit) once the filter ratio crosses the BF threshold.

### 3.3 Brute-force fallback (two triggers)
Thresholds in `thirdparty/hnswlib/hnswlib/hnswalg.h`:
```cpp
constexpr float kHnswSearchKnnBFFilterThreshold = 0.93f; // filtered fraction
constexpr float kHnswSearchBFTopkThreshold      = 0.5f;  // k vs live count
```
1. **Up-front:** if `filtered_out >= 0.93 * ntotal` **or** `k >= 0.5 * live`,
   skip graph search and scan all live rows directly.
2. **Per-query:** after graph search, if the number of valid results `< k` (and
   `>= k` live rows exist), redo that query as brute force
   (`bf_search_needed()` in `faiss_hnsw.cc`; guarded by
   `disable_fallback_brute_force`).

**These three mechanisms together are "CPU parity."** A GPU implementation that
only drops deleted ids at copy-out is NOT parity — it collapses recall exactly in
the cases (2) and (3) exist for. This MEP therefore specifies all three.

## 4. Requirements & non-goals

**Requirements**
- R1: Non-empty bitset no longer rejected; filtered rows never appear in results.
- R2: Recall on the surviving set matches CPU HNSW within tolerance across the
  full delete-ratio range (0–99%), for all 5 dtypes (fp32/fp16/bf16/int8-generic/
  int8-DP4A) and L2 / IP / COSINE.
- R3: Guarantee `k` valid results whenever `>= k` live rows exist (via the BF
  fallback), matching CPU.
- R4: No new memcheck / initcheck / racecheck findings; concurrent filtered
  searches + reload remain safe.
- R5: No functional Milvus code change; the existing delete/L0/bitset pipeline is
  untouched.

**Non-goals**
- Partition-key multi-index routing (`getIndexToSearchByScalarInfo`) — GPU_HNSW is
  single-index per segment; out of scope.
- Iterator / range-search filtered paths on GPU (knn only in phase 1).
- Changing the delete pipeline, L0 compaction, or TSafe logic.

## 5. Design

### 5.1 ID-space mapping (must be proven first)
The kernel's `result_ids` are FAISS internal ids; knowhere returns them straight
to `GenResultDataSet` with **no remap**. For a single `IndexHNSWFlat` /
`HNSW_SQ` segment the storage id == add order == segment row offset == the
`BitsetView` index. So `bitset.test(node_id)` is a direct 1:1 lookup.

**Action (blocking):** add a test asserting GPU raw id == Milvus row offset ==
bitset index on a segment with known deletes, before relying on it. If any
reorder exists this is a silent correctness bug. (The CPU multi-index
`label_to_internal_offset` / `internal_offset_to_most_external_id` path is not
used for GPU_HNSW.)

### 5.2 Bitset upload to device
- The bitset is identical for all `nq` queries in a `Search()` call → upload
  **once per call**, not per query. Size = `ceil(N/8)` bytes.
- Add to `GpuHnswSearchScratch` (`GpuHnswTypes.h`): `uint8_t* d_bitset` +
  `size_t bitset_bytes`, allocated in `ensure()` next to the visited bitmap. One
  buffer per scratch-pool slot ⇒ concurrent searches stay isolated (R4).
- Knowhere `Search()` `cudaMemcpyAsync`s the host bitset words onto the slot's
  stream before launch, and passes `d_bitset` (nullable ⇒ no filter), `N`, and
  the precomputed `filter_ratio` / `kAlpha` into `GpuHnswSearchParams`.
- VRAM cost: N=1M ⇒ 125 KB × pool_size(4) = 500 KB per segment. Negligible.

### 5.3 Kernel: two-tier beam + alpha gate (`GpuHnswSearchKernel.cuh`)
Mirror §3.1–3.2 inside `layer0_beam_search_kernel`:
- **Device bitset test:** `__device__ bool is_filtered(const uint8_t* b, uint32_t id)`
  (word/bit test; `b==nullptr ⇒ false`).
- **Split the beam.** Today one `ef` beam doubles as result+frontier via
  `is_expanded`. Add a per-slot `status` (valid/invalid) so:
  - valid (non-filtered) candidates occupy the **result** portion that feeds
    the top-k copy-out;
  - invalid (filtered) candidates are retained for **expansion only**, kept only
    while their distance beats the valid beam's worst — the GPU analog of
    `invalid_ns_` gated by `valid_ns_->at_search_back_dist()`.
  This touches `parallel_merge_into_result` / `bitonic_sort_staging`
  bookkeeping and the shared-memory layout (extra status array), so it needs a
  fresh smem-budget calc (§5.5) and racecheck run.
- **Alpha gate.** Carry `accumulated_alpha` as a per-block (per-query) scalar in
  shared memory; when a discovered neighbor is filtered, apply the exact
  `+= kAlpha; if (<1) skip; else -= 1` logic before admitting it to the invalid
  frontier. `kAlpha` is passed in from the host (`filter_ratio * 0.7`).
- **Upper layers unchanged.** `upper_layer_search_kernel` only routes; results
  come from layer 0. Leave greedy descent alone — deleted nodes still route.
- **Copy-out** (`GpuHnswSearchKernel.cuh:651`): emit only valid ids into the `k`
  outputs; pad with sentinels as today when fewer than `k` valid survive (the BF
  fallback in §5.4 then fills those queries).
- Thread `d_bitset` / `N` / `kAlpha` through **all 5** layer-0 specializations
  (`<float>`, `<half>`, `<__nv_bfloat16>`, `<int8,float>`, `<int8,int8,DP4A>`).
  The filter is dtype-independent (operates on ids).

### 5.4 Brute-force fallback kernel (§3.3 parity)
Add a device brute-force top-k over live rows (a distance kernel that skips
`is_filtered` ids + a top-k selection). GPU BF is embarrassingly parallel and
fast for a single segment.
- **Up-front trigger** (host, in knowhere `Search()`): if
  `filtered >= 0.93*N` or `k >= 0.5*live`, launch BF directly and skip the graph
  kernel. Matches `kHnswSearchKnnBFFilterThreshold` / `kHnswSearchBFTopkThreshold`.
- **Per-query trigger:** after the graph kernel, for any query whose valid-result
  count `< k` while `>= k` live rows exist, relaunch that query on the BF kernel.
  Matches `bf_search_needed()`. Honor a `disable_fallback_brute_force` equivalent.
- Distances use the same metric helpers (L2 / negated-IP / cosine via
  `d_inv_norms`) so scores are identical to the graph path and to CPU.

### 5.5 Shared-memory budget
Layer-0 smem is already `ef`-bound (`calc_layer0_smem_size`), with a clamp +
warning when `ef` exceeds the device budget. The extra per-slot `status` array
(and any beam split) increases per-`ef` cost, lowering the max fittable `ef`.
Re-derive `max_ef` and update the clamp/warning so heavy-filter searches don't
silently lose recall or fail to launch. If the split cost is prohibitive at large
`ef`, degrade gracefully to the BF path rather than clamp below `k`.

### 5.6 Concurrency (R4)
`d_bitset` is per scratch-pool slot and **read-only** in the kernel — no new write
hazards. The existing per-slot stream + scratch isolation covers concurrent
filtered searches; the reload path (`Deserialize` under `gpu_mutex_`) is
unchanged. Re-run the p1 concurrent-search + reload racecheck with filtering on.

## 6. Knowhere changes (`src/index/hnsw/faiss_hnsw.cc`)
- `GpuHnswIndexNode::Search()`: **remove** the reject guard; compute
  `filter_ratio` / `kAlpha` / BF-trigger from `bitset`; upload the bitset to the
  slot's `d_bitset`; pass params down; select graph vs BF path.
- `HasRawData()` / `GetVectorByIds()` unchanged (still `false` /
  `not_implemented`; vector output is served from raw field data by segcore, per
  the existing Failure-Modes analysis — unaffected by filtering).
- Add the ID-space assertion test (§5.1) and dtype × filter-ratio recall tests.

## 7. faiss changes
- `faiss/gpu/impl/GpuHnswTypes.h`: `d_bitset` + `bitset_bytes` in
  `GpuHnswSearchScratch`; `ef`/`kAlpha`/filter fields in `GpuHnswSearchParams`.
- `faiss/gpu/impl/GpuHnswSearch.cuh`: bitset upload + BF-vs-graph launch; thread
  params into all specializations.
- `faiss/gpu/impl/GpuHnswSearchKernel.cuh`: `is_filtered`, two-tier beam, alpha
  gate, filtered copy-out; new BF kernel.
- `faiss/gpu/GpuIndexHNSW.{h,cu}`: extend `searchHost` / `searchHostInt8` to
  accept the bitset + params.
- Re-vendor the changed GPU files into `knowhere/thirdparty/faiss` byte-identically
  (`cmp`-verified), as with every prior GPU change.

## 8. Milvus changes (docs/tests only — no logic)
- Flip the ⚠️ filtered-search callout and the **Failure Modes** row in
  [20260619-gpu-hnsw.md](20260619-gpu-hnsw.md) and the limitations note in
  [20260711-gpu-hnsw-ocq-removal.md](20260711-gpu-hnsw-ocq-removal.md) from
  "not supported" to "supported (CPU-parity; BF fallback at high filter ratio)".
- Flip the delete-then-query expectation in the `idx_gpu_hnsw*` Python specs from
  expecting `invalid_args` to expecting **deleted rows absent + non-deleted rows
  present** (the same assertion the CPU-HNSW baseline uses).
- Confirm (grep) no Milvus-side heuristic assumes GPU segments are delete-free
  (compaction / routing). Expected: none.

## 9. Testing & validation gates (all must pass before deploy)
- **G-ID:** GPU id == row offset == bitset index (§5.1) — blocking correctness.
- **Recall parity:** GPU vs CPU HNSW on the same data+bitset, delete ratios
  {0, 10, 50, 90, 95, 99}%, all 5 dtypes, L2/IP/COSINE. Filtered ids must never
  appear; recall on survivors within tolerance of CPU.
- **k-guarantee:** every query returns `k` valid results when `>= k` live rows
  exist (exercises the BF fallback).
- **Sanitizers:** memcheck / initcheck / racecheck (incl. the p1 concurrent +
  reload test with filtering on).
- **Milvus e2e:** create GPU_HNSW collection → seal + GPU-index → delete keys in
  the sealed segment → query → deleted rows absent (was `invalid_args`), against
  the CPU-HNSW baseline for side-by-side parity.
- **Perf:** measure the filtered-search throughput/recall vs the append-only
  baseline (v87) so the alpha-gate / BF-fallback cost is characterized.

## 10. Phasing, effort, risk
Single phase (parity is the requirement — no "v1 copy-out-only" shortcut, since
that fails R2/R3). Rough size **L–XL**, dominated by the two-tier beam +
shared-memory rework, the BF-fallback kernel, and the full recall/sanitizer gate
across 5 dtypes × delete ratios.

**Risks**
- Invalidates the validated v87 binary ⇒ full rebuild + re-gate + new immutable
  candidate tag (**v88**). Do not fold into v87.
- Shared-memory pressure may cap usable `ef` at high M (§5.5) — mitigated by
  degrading to BF rather than clamping below `k`.
- ID-space assumption (§5.1) is load-bearing — proven by G-ID before anything
  else.
- Alpha-gate tuning: `kAlpha = filter_ratio*0.7` and the 0.93/0.5 thresholds are
  copied verbatim from CPU for parity; revisit only with data.

## 11. Rollout
Land faiss + knowhere on `gpu-hnsw-faiss`, re-vendor, build immutable
`gpu-hnsw-v88`, run the full gate above, deploy-verify on `mpd_v2` with a
delete workload, benchmark vs v87, then scale to 0 / reset to the v83 baseline.
v84-r2 remains production; v88 becomes the filtered-search candidate.
