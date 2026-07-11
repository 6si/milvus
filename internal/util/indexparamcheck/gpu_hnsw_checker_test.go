package indexparamcheck

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/util/vecindexmgr"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
)

// requireGpuHnswChecker returns the GPU_HNSW checker, or skips the test when the
// linked core was built without GPU_HNSW (no GPU/CUVS). A skip is honest; the
// previous silent `return` reported a PASS and gave false confidence.
func requireGpuHnswChecker(t *testing.T) IndexChecker {
	if !vecindexmgr.GetVecIndexMgrInstance().IsVecIndex("GPU_HNSW") {
		t.Skip("GPU_HNSW not registered in this build (no GPU/CUVS); skipping")
	}
	c, err := GetIndexCheckerMgrInstance().GetChecker("GPU_HNSW")
	require.NoError(t, err)
	require.NotNil(t, c)
	return c
}

func Test_gpuHnswChecker_routing(t *testing.T) {
	// GPU_HNSW / GPU_HNSW_SQ must resolve to the dedicated gpuHnswChecker so the
	// M/efConstruction range validation actually runs (knowhere's
	// ValidateIndexParams is a no-op for these types).
	for _, it := range []string{"GPU_HNSW", "GPU_HNSW_SQ"} {
		c, err := GetIndexCheckerMgrInstance().GetChecker(it)
		require.NoError(t, err)
		require.NotNil(t, c)
		_, ok := c.(*gpuHnswChecker)
		require.Truef(t, ok, "expected *gpuHnswChecker for %s, got %T", it, c)
	}
}

func Test_gpuHnswChecker_CheckTrain(t *testing.T) {
	validParams := map[string]string{
		DIM:            strconv.Itoa(128),
		HNSWM:          strconv.Itoa(16),
		EFConstruction: strconv.Itoa(200),
		Metric:         metric.L2,
	}

	invalidEfParamsMin := copyParams(validParams)
	invalidEfParamsMin[EFConstruction] = strconv.Itoa(HNSWMinEfConstruction - 1)

	invalidEfParamsMax := copyParams(validParams)
	invalidEfParamsMax[EFConstruction] = strconv.Itoa(HNSWMaxEfConstruction + 1)

	invalidMParamsMin := copyParams(validParams)
	invalidMParamsMin[HNSWM] = strconv.Itoa(HNSWMinM - 1)

	invalidMParamsMax := copyParams(validParams)
	invalidMParamsMax[HNSWM] = strconv.Itoa(HNSWMaxM + 1)

	p1 := map[string]string{
		DIM:            strconv.Itoa(128),
		HNSWM:          strconv.Itoa(16),
		EFConstruction: strconv.Itoa(200),
		Metric:         metric.L2,
	}
	p2 := map[string]string{
		DIM:            strconv.Itoa(128),
		HNSWM:          strconv.Itoa(16),
		EFConstruction: strconv.Itoa(200),
		Metric:         metric.IP,
	}
	p3 := map[string]string{
		DIM:            strconv.Itoa(128),
		HNSWM:          strconv.Itoa(16),
		EFConstruction: strconv.Itoa(200),
		Metric:         metric.COSINE,
	}
	// GPU_HNSW does NOT support binary metrics
	p4 := map[string]string{
		DIM:            strconv.Itoa(128),
		HNSWM:          strconv.Itoa(16),
		EFConstruction: strconv.Itoa(200),
		Metric:         metric.HAMMING,
	}
	p5 := map[string]string{
		DIM:            strconv.Itoa(128),
		HNSWM:          strconv.Itoa(16),
		EFConstruction: strconv.Itoa(200),
		Metric:         metric.JACCARD,
	}
	p6 := map[string]string{
		DIM:            strconv.Itoa(128),
		HNSWM:          strconv.Itoa(16),
		EFConstruction: strconv.Itoa(200),
		Metric:         metric.SUBSTRUCTURE,
	}
	p7 := map[string]string{
		DIM:            strconv.Itoa(128),
		HNSWM:          strconv.Itoa(16),
		EFConstruction: strconv.Itoa(200),
		Metric:         metric.SUPERSTRUCTURE,
	}
	// High dimensionality (384-d, typical for INT8 embeddings)
	p8 := map[string]string{
		DIM:            strconv.Itoa(384),
		HNSWM:          strconv.Itoa(32),
		EFConstruction: strconv.Itoa(200),
		Metric:         metric.COSINE,
	}

	cases := []struct {
		params   map[string]string
		errIsNil bool
	}{
		{validParams, true},
		{invalidEfParamsMin, false},
		{invalidEfParamsMax, false},
		{invalidMParamsMin, false},
		{invalidMParamsMax, false},
		{p1, true},
		{p2, true},
		{p3, true},
		{p4, false}, // HAMMING not supported for float vectors
		{p5, false}, // JACCARD not supported for float vectors
		{p6, false}, // SUBSTRUCTURE not supported
		{p7, false}, // SUPERSTRUCTURE not supported
		{p8, true},  // 384-d COSINE
	}

	c := requireGpuHnswChecker(t)
	for _, test := range cases {
		test.params[common.IndexTypeKey] = "GPU_HNSW"
		err := c.CheckTrain(schemapb.DataType_FloatVector, schemapb.DataType_None, test.params)
		if test.errIsNil {
			assert.NoError(t, err)
		} else {
			assert.Error(t, err)
		}
	}
}

func Test_gpuHnswChecker_CheckValidDataType(t *testing.T) {
	cases := []struct {
		dType    schemapb.DataType
		errIsNil bool
	}{
		{
			dType:    schemapb.DataType_Bool,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_Int8,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_Int16,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_Int32,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_Int64,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_Float,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_Double,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_String,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_VarChar,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_Array,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_JSON,
			errIsNil: false,
		},
		{
			dType:    schemapb.DataType_FloatVector,
			errIsNil: true,
		},
		// GPU_HNSW is registered in knowhere for FLOAT32, FP16, BF16 and INT8.
		// FP16/BF16 are uploaded to the GPU in their native 2-byte layout and
		// up-converted to fp32 per element inside the search kernel.
		{
			dType:    schemapb.DataType_Float16Vector,
			errIsNil: true,
		},
		{
			dType:    schemapb.DataType_BFloat16Vector,
			errIsNil: true,
		},
		{
			dType:    schemapb.DataType_Int8Vector,
			errIsNil: true,
		},
		// GPU_HNSW does NOT support BinaryVector
		{
			dType:    schemapb.DataType_BinaryVector,
			errIsNil: false,
		},
	}

	c := requireGpuHnswChecker(t)
	for _, test := range cases {
		err := c.CheckValidDataType("GPU_HNSW", &schemapb.FieldSchema{DataType: test.dType})
		if test.errIsNil {
			assert.NoError(t, err)
		} else {
			assert.Error(t, err)
		}
	}
}

func Test_gpuHnswChecker_SetDefaultMetricType(t *testing.T) {
	cases := []struct {
		dType      schemapb.DataType
		metricType string
	}{
		{
			dType:      schemapb.DataType_FloatVector,
			metricType: metric.COSINE,
		},
		{
			dType:      schemapb.DataType_Float16Vector,
			metricType: metric.COSINE,
		},
		{
			dType:      schemapb.DataType_BFloat16Vector,
			metricType: metric.COSINE,
		},
		{
			dType:      schemapb.DataType_Int8Vector,
			metricType: metric.COSINE,
		},
	}

	c := requireGpuHnswChecker(t)
	for _, test := range cases {
		p := map[string]string{
			DIM:            strconv.Itoa(128),
			HNSWM:          strconv.Itoa(16),
			EFConstruction: strconv.Itoa(200),
		}
		p[common.IndexTypeKey] = "GPU_HNSW"
		c.SetDefaultMetricTypeIfNotExist(test.dType, p)
		assert.Equal(t, test.metricType, p[common.MetricTypeKey])
	}
}
