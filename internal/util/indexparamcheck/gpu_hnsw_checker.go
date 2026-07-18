package indexparamcheck

import (
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
)

// gpuHnswMaxStagingCapacity is the largest per-block bitonic-merge staging
// capacity the GPU HNSW search kernel supports. The kernel pads
// search_width * max_degree0 (= search_width * 2*M) up to the next power of two
// and launches one thread per staging slot, so the padded capacity must fit in
// a single CUDA block (max 1024 threads). Even at the minimum search_width of
// 1, next_pow2(2*M) must not exceed this, otherwise the index can be built but
// never searched on the GPU.
const gpuHnswMaxStagingCapacity = 1024

// gpuHnswMaxM is the largest M the GPU search kernel can actually search: its
// layer-0 staging is next_pow2(2*M), which must fit in gpuHnswMaxStagingCapacity
// (a single CUDA block) at the minimum search_width of 1. 2*512 = 1024, so
// M > 512 would build fine but fail every GPU search. This is stricter than the
// CPU HNSW max (HNSWMaxM=2048); we advertise and enforce the honest GPU bound.
const gpuHnswMaxM = gpuHnswMaxStagingCapacity / 2

// gpuHnswChecker validates GPU_HNSW index parameters.
// knowhere's ValidateIndexParams returns 0 for GPU_HNSW (unimplemented config
// validation), so we validate M and efConstruction in Go. M and efConstruction
// are optional (knowhere fills defaults when omitted, matching CPU HNSW); we
// only range-check them when the caller supplies them.
type gpuHnswChecker struct {
	vecIndexChecker
}

func (c *gpuHnswChecker) CheckTrain(dataType schemapb.DataType, elementType schemapb.DataType, params map[string]string) error {
	if err := c.StaticCheck(dataType, elementType, params); err != nil {
		return err
	}
	if _, ok := params[EFConstruction]; ok {
		if !CheckIntByRange(params, EFConstruction, HNSWMinEfConstruction, HNSWMaxEfConstruction) {
			return errOutOfRange(params[EFConstruction], HNSWMinEfConstruction, HNSWMaxEfConstruction)
		}
	}
	if _, ok := params[HNSWM]; ok {
		// Enforce the GPU-specific max (gpuHnswMaxM=512): M > 512 stages more
		// than a single CUDA block can hold, so such an index can never be
		// searched on the GPU even though it would build.
		if !CheckIntByRange(params, HNSWM, HNSWMinM, gpuHnswMaxM) {
			return errOutOfRange(params[HNSWM], HNSWMinM, gpuHnswMaxM)
		}
	}
	if !CheckIntByRange(params, DIM, 1, 1<<31-1) {
		return errOutOfRange(params[DIM], 1, 1<<31-1)
	}
	return nil
}

func newGpuHnswChecker() IndexChecker {
	return &gpuHnswChecker{}
}
