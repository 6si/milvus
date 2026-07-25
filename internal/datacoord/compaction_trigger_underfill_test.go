package datacoord

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/lifetime"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// underFillExpectedSize is the per-segment target used by the under-fill tests
// (mirrors dataCoord.segment.maxSize = 32MB).
const underFillExpectedSize = int64(32 * 1024 * 1024)

// frac returns a runtime int64 fraction of the expected size (avoids the
// untyped-constant-to-int64 conversion error for non-integer results).
func frac(p float64) int64 { return int64(float64(underFillExpectedSize) * p) }

// newUnderFillSeg builds a flushed, sorted L1 segment whose raw size is rawBytes
// and which carries delRows deleted rows out of numRows (so its delete-adjusted
// residual size is rawBytes*(1-delRows/numRows)). Stats are set explicitly to
// make the size/residual deterministic.
func newUnderFillSeg(id, rawBytes, numRows, delRows int64) *SegmentInfo {
	return &SegmentInfo{SegmentInfo: &datapb.SegmentInfo{
		ID:            id,
		CollectionID:  2,
		PartitionID:   1,
		InsertChannel: "ch1",
		State:         commonpb.SegmentState_Flushed,
		Level:         datapb.SegmentLevel_L1,
		NumOfRows:     numRows,
		MaxRowNum:     numRows * 4,
		IsSorted:      true,
		Stats: &datapb.Statistics{
			InsertBinlogSize: rawBytes,
			DeleteNumRows:    delRows,
			DeltaBinlogCount: 1,
		},
	}}
}

// underFillTrigger returns a compactionTrigger wired with just enough state for
// generatePlans, with under-fill either enabled or disabled. Auto index upgrade
// is disabled so ShouldRebuildSegmentIndex does not touch meta.
func underFillTrigger(t *testing.T, enable bool) *compactionTrigger {
	pt := paramtable.Get()
	pt.Save(pt.DataCoordCfg.AutoUpgradeSegmentIndex.Key, "false")
	pt.Save(pt.DataCoordCfg.SegmentMaxSize.Key, "32")
	pt.Save(pt.DataCoordCfg.SegmentSmallProportion.Key, "0.5")
	pt.Save(pt.DataCoordCfg.SegmentCompactableProportion.Key, "0.85")
	pt.Save(pt.DataCoordCfg.SegmentExpansionRate.Key, "1.25")
	pt.Save(pt.DataCoordCfg.MinSegmentToMerge.Key, "3")
	pt.Save(pt.DataCoordCfg.SingleCompactionRatioThreshold.Key, "0.2")
	pt.Save(pt.DataCoordCfg.CompactionUnderFillEnable.Key, boolStr(enable))
	pt.Save(pt.DataCoordCfg.CompactionUnderFillGroupFactor.Key, "8")
	t.Cleanup(func() {
		pt.Reset(pt.DataCoordCfg.AutoUpgradeSegmentIndex.Key)
		pt.Reset(pt.DataCoordCfg.SegmentMaxSize.Key)
		pt.Reset(pt.DataCoordCfg.SegmentSmallProportion.Key)
		pt.Reset(pt.DataCoordCfg.SegmentCompactableProportion.Key)
		pt.Reset(pt.DataCoordCfg.SegmentExpansionRate.Key)
		pt.Reset(pt.DataCoordCfg.MinSegmentToMerge.Key)
		pt.Reset(pt.DataCoordCfg.SingleCompactionRatioThreshold.Key)
		pt.Reset(pt.DataCoordCfg.CompactionUnderFillEnable.Key)
		pt.Reset(pt.DataCoordCfg.CompactionUnderFillGroupFactor.Key)
	})
	return &compactionTrigger{
		inspector:   &spyCompactionInspector{t: t, spyChan: make(chan *datapb.CompactionPlan, 1)},
		closeCh:     lifetime.NewSafeChan(),
		testingOnly: true,
	}
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

func underFillSignal() *compactionSignal {
	return &compactionSignal{collectionID: 2, partitionID: 1, channel: "ch1"}
}

func coveredSegmentCount(plans []*typeutil.Pair[int64, []int64]) int {
	total := 0
	for _, p := range plans {
		total += len(p.B)
	}
	return total
}

// half-full segments in the [smallProportion, compactableProportion) dead-band
// are never merged today; the under-fill pass repacks them into fewer, fuller
// segments. This is the core reproduction of issue "many small segments hang
// around".
func TestGeneratePlans_UnderFill_ClosesDeadBand(t *testing.T) {
	// 10 segments at 0.6*expected raw size, no deletes -> residual 0.6*expected.
	// Not small (0.6 > 0.5) and not full (0.6 < 0.85): the frozen dead-band.
	mediumSize := frac(0.6)
	newMedium := func() []*SegmentInfo {
		segs := make([]*SegmentInfo, 0, 10)
		for i := int64(0); i < 10; i++ {
			segs = append(segs, newUnderFillSeg(1000+i, mediumSize, 6000, 0))
		}
		return segs
	}

	// Default behaviour: dead-band segments are frozen, no plans.
	off := underFillTrigger(t, false)
	plansOff := off.generatePlans(newMedium(), underFillSignal(), &compactTime{}, underFillExpectedSize)
	assert.Empty(t, plansOff, "default trigger must not touch dead-band segments")

	// With under-fill enabled: all 10 segments are repacked into a single group
	// that the datanode writer will split into ceil(10*0.6)=6 full segments.
	on := underFillTrigger(t, true)
	plansOn := on.generatePlans(newMedium(), underFillSignal(), &compactTime{}, underFillExpectedSize)
	assert.NotEmpty(t, plansOn, "under-fill pass must repack dead-band segments")
	assert.Equal(t, 10, coveredSegmentCount(plansOn), "all dead-band segments must be scheduled")
	assert.Len(t, plansOn, 1, "10 under-filled segments should form a single group under groupFactor=8")
}

// tiny-segment merging must be unchanged by the under-fill pass, and already-full
// segments must never be rewritten (no churn / no infinite compaction loop).
func TestGeneratePlans_UnderFill_NoRegression(t *testing.T) {
	on := underFillTrigger(t, true)

	// 5 already-full segments (0.95*expected): residual above compactableProportion.
	fullSize := frac(0.95)
	full := make([]*SegmentInfo, 0, 5)
	for i := int64(0); i < 5; i++ {
		full = append(full, newUnderFillSeg(2000+i, fullSize, 9500, 0))
	}
	plansFull := on.generatePlans(full, underFillSignal(), &compactTime{}, underFillExpectedSize)
	assert.Empty(t, plansFull, "full segments must not be repacked")

	// Small segments still merge via the ordinary path.
	smallSize := frac(0.1)
	small := make([]*SegmentInfo, 0, 10)
	for i := int64(0); i < 10; i++ {
		small = append(small, newUnderFillSeg(3000+i, smallSize, 1000, 0))
	}
	plansSmall := on.generatePlans(small, underFillSignal(), &compactTime{}, underFillExpectedSize)
	assert.NotEmpty(t, plansSmall, "small segments must still be merged")
	assert.Equal(t, 10, coveredSegmentCount(plansSmall))
}

// a group that would not reduce the segment count must be skipped: three
// near-full (0.8*expected) segments repack into three full segments -> no gain.
func TestGeneratePlans_UnderFill_SkipsNonReducingGroup(t *testing.T) {
	on := underFillTrigger(t, true)
	nearFull := frac(0.8) // under-full (<0.85) but near target
	segs := make([]*SegmentInfo, 0, 3)
	for i := int64(0); i < 3; i++ {
		segs = append(segs, newUnderFillSeg(4000+i, nearFull, 8000, 0))
	}
	plans := on.generatePlans(segs, underFillSignal(), &compactTime{}, underFillExpectedSize)
	assert.Empty(t, plans, "repacking 3 near-full segments into 3 full segments yields no benefit")
}

// frequent-upsert case: segments carrying moderate deletes (below the
// hasTooManyDeletions threshold) are frozen by default even though their live
// (residual) data is well under target. The residual-aware under-fill pass
// reclaims the deleted space and consolidates them.
func TestGeneratePlans_UnderFill_ReclaimsDeletedSpace(t *testing.T) {
	// raw 0.7*expected, 15% deleted -> residual 0.595*expected.
	// raw > small cutoff (not small), delete ratio 0.15 < 0.2 (not "too many
	// deletions"), residual < 0.85 (under-full).
	rawSize := frac(0.7)
	newUpsert := func() []*SegmentInfo {
		segs := make([]*SegmentInfo, 0, 10)
		for i := int64(0); i < 10; i++ {
			segs = append(segs, newUnderFillSeg(5000+i, rawSize, 10000, 1500))
		}
		return segs
	}

	off := underFillTrigger(t, false)
	plansOff := off.generatePlans(newUpsert(), underFillSignal(), &compactTime{}, underFillExpectedSize)
	assert.Empty(t, plansOff, "moderately-deleted segments are frozen by default")

	on := underFillTrigger(t, true)
	plansOn := on.generatePlans(newUpsert(), underFillSignal(), &compactTime{}, underFillExpectedSize)
	assert.NotEmpty(t, plansOn, "under-fill pass must reclaim deleted space")
	assert.Equal(t, 10, coveredSegmentCount(plansOn))
}

// isUnderFullSegment classification boundaries.
func TestIsUnderFullSegment(t *testing.T) {
	tr := underFillTrigger(t, true)
	exp := underFillExpectedSize

	small := newUnderFillSeg(1, frac(0.4), 1000, 0)
	assert.False(t, tr.isUnderFullSegment(small, exp), "small segments are handled elsewhere")

	medium := newUnderFillSeg(2, frac(0.6), 1000, 0)
	assert.True(t, tr.isUnderFullSegment(medium, exp))

	full := newUnderFillSeg(3, frac(0.9), 1000, 0)
	assert.False(t, tr.isUnderFullSegment(full, exp))

	// deletes push a raw-full segment below the compactable threshold.
	deleted := newUnderFillSeg(4, frac(0.9), 1000, 500)
	assert.True(t, tr.isUnderFullSegment(deleted, exp), "residual after deletes is under-full")
}

func TestUnderFillReducesCount(t *testing.T) {
	exp := underFillExpectedSize
	seg := func(id, sz int64) *SegmentInfo { return newUnderFillSeg(id, sz, 1000, 0) }

	// 3 x 0.6 = 1.8 -> 2 outputs < 3 inputs: reduces.
	assert.True(t, underFillReducesCount([]*SegmentInfo{
		seg(1, frac(0.6)), seg(2, frac(0.6)), seg(3, frac(0.6)),
	}, exp))

	// 3 x 0.8 = 2.4 -> 3 outputs == 3 inputs: no reduction.
	assert.False(t, underFillReducesCount([]*SegmentInfo{
		seg(1, frac(0.8)), seg(2, frac(0.8)), seg(3, frac(0.8)),
	}, exp))

	// single segment never reduces.
	assert.False(t, underFillReducesCount([]*SegmentInfo{seg(1, frac(0.6))}, exp))
}
