package indexparamcheck

import (
	"fmt"
	"strconv"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// gpuHnswMaxStagingCapacity is the largest per-block bitonic-merge staging
// capacity the GPU HNSW search kernel supports. The kernel pads
// search_width * max_degree0 (= search_width * 2*M) up to the next power of two
// and launches one thread per staging slot, so the padded capacity must fit in
// a single CUDA block (max 1024 threads). Even at the minimum search_width of
// 1, next_pow2(2*M) must not exceed this, otherwise the index can be built but
// never searched on the GPU. search_width itself is a query-time parameter and
// is validated in the kernel (which emits a clear error), so here we only reject
// M values that are unusable regardless of search_width.
const gpuHnswMaxStagingCapacity = 1024

func nextPow2(x int) int {
	p := 1
	for p < x {
		p <<= 1
	}
	return p
}

// gpuHnswChecker validates GPU_HNSW index parameters.
// knowhere's ValidateIndexParams returns 0 for GPU_HNSW (unimplemented config validation),
// so we validate M and efConstruction ranges in Go.
type gpuHnswChecker struct {
	vecIndexChecker
}

func (c *gpuHnswChecker) CheckTrain(dataType schemapb.DataType, elementType schemapb.DataType, params map[string]string) error {
	if err := c.StaticCheck(dataType, elementType, params); err != nil {
		return err
	}
	if !CheckIntByRange(params, EFConstruction, HNSWMinEfConstruction, HNSWMaxEfConstruction) {
		return errOutOfRange(params[EFConstruction], HNSWMinEfConstruction, HNSWMaxEfConstruction)
	}
	if !CheckIntByRange(params, HNSWM, HNSWMinM, HNSWMaxM) {
		return errOutOfRange(params[HNSWM], HNSWMinM, HNSWMaxM)
	}
	if !CheckIntByRange(params, DIM, 1, 1<<31-1) {
		return errOutOfRange(params[DIM], 1, 1<<31-1)
	}
	// The GPU search kernel stages search_width*2*M candidates padded to the
	// next power of two, one thread per slot. Reject M values whose padded
	// layer-0 staging already exceeds a single CUDA block at search_width=1;
	// such an index would build fine but fail every GPU search.
	if m, err := strconv.Atoi(params[HNSWM]); err == nil {
		staging := nextPow2(2 * m)
		if staging > gpuHnswMaxStagingCapacity {
			return merr.WrapErrParameterInvalidMsg(
				fmt.Sprintf("GPU_HNSW M=%d is too large: layer-0 staging next_pow2(2*M)=%d "+
					"exceeds the max GPU search block size %d; use M<=%d",
					m, staging, gpuHnswMaxStagingCapacity, gpuHnswMaxStagingCapacity/2))
		}
	}
	return nil
}

func newGpuHnswChecker() IndexChecker {
	return &gpuHnswChecker{}
}
