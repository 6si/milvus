# Decision Record: Remove the OCQ (Overflow Candidate Queue) from GPU HNSW

- **Date:** 2026-07-11
- **Status:** Accepted
- **Affects:** `6si/faiss` (`faiss/gpu/*`), `6si/knowhere` (vendored `thirdparty/faiss/faiss/gpu/*` + `src/index/hnsw/faiss_hnsw.cc`), `6si/milvus` design doc `20260619-gpu-hnsw.md`
- **Related PRs:** 6si/faiss#1, 6si/knowhere#2, 6si/milvus#6

## Context

The GPU HNSW search kernel shipped with an "Overflow Candidate Queue" (OCQ):
a per-query auxiliary buffer intended to hold runner-up candidates. The idea was
that when a beam-search iteration finds no unvisited neighbors to expand
(`num_parents == 0`), the walk could pull backup candidates from the overflow
queue instead of terminating early, recovering recall at a fixed `ef`. The
feature was described in the PR titles/descriptions and the design doc as the
distinguishing characteristic of the kernel.

## Problem

Code review established that the OCQ was **never functional — it was dead code**:

- `overflow_insert()` (the routine that would push runner-up candidates into the
  queue) had **no call site** anywhere in the kernel.
- `d_overflow_count[query_idx]` was only ever **set to 0** (init) and **read**
  (in the `num_parents == 0` fallback). Nothing ever incremented it.
- Therefore the fallback loop always saw an empty queue and did nothing. The
  kernel was, in effect, a plain parallel beam search.

Cost of keeping it: `overflow_factor` defaulted to `2`, so every search
allocated `nq * 2 * ef * (4+4+4)` bytes of scratch VRAM and issued a
per-search `cudaMemsetAsync` — all for a queue that was never populated. The
docs also made a correctness claim ("OCQ beam search", "a zero here would
silently drop recall") that did not reflect reality.

## Options considered

1. **Wire OCQ up** — implement `overflow_insert` into the staging/merge path so
   the queue is actually populated and consumed.
2. **Remove OCQ** — delete the overflow machinery, default to a plain beam
   search, and correct the docs. (Chosen.)

## Decision

**Remove the OCQ machinery.** Reasoning:

- **OCQ is a recall mechanism, not a speed one.** Wiring it up would *add*
  per-iteration work (maintaining a sorted overflow list) plus the VRAM scratch
  and memset it already paid for — i.e. slightly slower searches.
- **Its upside is bounded and usually small.** OCQ only does anything on
  iterations where the beam dead-ends (`num_parents == 0`) before `ef` is
  satisfied. On a well-connected HNSW graph with a reasonable `ef`, that is
  rare, so the recall benefit is typically negligible.
- **There is a simpler lever for the same goal.** The standard way to raise
  recall is to increase `ef`, which is already exposed as a search parameter.
  OCQ only helps in the narrow regime of a hard `ef` ceiling (VRAM/latency
  bound) where `ef` cannot be raised — not our situation.
- **Honesty.** Keeping non-functional code that the docs describe as active
  misleads reviewers and future maintainers.

If a future workload is proven to be `ef`-ceiling-bound and short on recall, OCQ
(or an equivalent) can be reintroduced as a real, benchmarked feature.

## Changes made

- **faiss / vendored faiss:** removed `SearchParametersGpuHNSW::overflow_factor`
  and `GpuHnswSearchParams::overflow_factor`; removed the `d_overflow_*` scratch
  fields, their allocation in `GpuHnswSearchScratch::ensure()` (signature dropped
  the `overflow_ef` parameter) and destructor frees; removed `overflow_insert()`,
  the overflow kernel parameters, the per-query overflow locals/init, and the
  dead `num_parents == 0` fallback; removed the `overflow_ef` computation and
  memset in the search host wrapper. Kernel is now a plain parallel beam search.
- **knowhere `faiss_hnsw.cc`:** dropped the explicit `gsp.overflow_factor = 2`
  in the search path (only `ef` is set now).
- **milvus design doc `20260619-gpu-hnsw.md`:** replaced "OCQ" references with
  "parallel beam-search kernel (recall tuned via `ef`)".

## Consequences

- Lower VRAM footprint per search and one fewer memset per search.
- No behavioral regression: the search path already ran with an empty (no-op)
  queue, so results are unchanged. Recall is tuned via `ef` as before.
- PR titles/descriptions and the design doc no longer claim OCQ behavior.
