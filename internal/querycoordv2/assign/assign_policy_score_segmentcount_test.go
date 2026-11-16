// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package assign

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/milvus-io/milvus/internal/querycoordv2/session"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
)

// newSkewedCountPolicy builds a policy whose distribution has two nodes with
// EQUAL row count but UNEQUAL segment count — the exact steady state the
// row-only score-based balancer leaves behind after a scale-out:
//   - node 1: 4 segments x 2500 rows = 10000 rows
//   - node 2: 1 segment  x 10000 rows = 10000 rows
func newSkewedCountPolicy(t *testing.T) *ScoreBasedAssignPolicy {
	nodeManager := session.NewNodeManager()
	mockScheduler := task.NewMockScheduler(t)
	mockScheduler.EXPECT().GetSegmentTaskDeltaSnapshot(mock.Anything, mock.Anything).Return(task.NewSegmentTaskDeltaSnapshot(nil, nil)).Maybe()
	mockScheduler.EXPECT().GetChannelTaskDelta(mock.Anything, mock.Anything).Return(0).Maybe()
	dist := meta.NewDistributionManager(nodeManager)
	metaMgr := meta.NewMeta(nil, nil, nodeManager)

	for i := int64(1); i <= 2; i++ {
		nodeManager.Add(session.NewNodeInfo(session.ImmutableNodeInfo{
			NodeID:   i,
			Version:  common.Version,
			Address:  "localhost",
			Hostname: "node",
		}))
		nodeManager.Get(i).SetState(session.NodeStateNormal)
	}

	dist.SegmentDistManager.Update(1, []*meta.Segment{
		{SegmentInfo: &datapb.SegmentInfo{ID: 10, NumOfRows: 2500, CollectionID: 100}},
		{SegmentInfo: &datapb.SegmentInfo{ID: 11, NumOfRows: 2500, CollectionID: 100}},
		{SegmentInfo: &datapb.SegmentInfo{ID: 12, NumOfRows: 2500, CollectionID: 100}},
		{SegmentInfo: &datapb.SegmentInfo{ID: 13, NumOfRows: 2500, CollectionID: 100}},
	}...)
	dist.SegmentDistManager.Update(2, []*meta.Segment{
		{SegmentInfo: &datapb.SegmentInfo{ID: 20, NumOfRows: 10000, CollectionID: 100}},
	}...)

	return newScoreBasedAssignPolicy(nodeManager, mockScheduler, dist, metaMgr)
}

func withSegmentCountFactor(t *testing.T, v string) {
	key := paramtable.Get().QueryCoordCfg.ScoreBasedBalanceSegmentCountFactor.Key
	paramtable.Get().Save(key, v)
	t.Cleanup(func() { paramtable.Get().Reset(key) })
}

// TestScoreBasedBalance_SegmentCountFactorDisabled verifies the default (factor=0)
// behavior is unchanged: two nodes with equal rows but unequal segment counts get
// EQUAL scores, so the balancer has no signal to equalize segment count.
func TestScoreBasedBalance_SegmentCountFactorDisabled(t *testing.T) {
	paramtable.Init()
	// default is 0; assert explicitly to guard against config drift.
	assert.Equal(t, float64(0), paramtable.Get().QueryCoordCfg.ScoreBasedBalanceSegmentCountFactor.GetAsFloat())

	policy := newSkewedCountPolicy(t)
	items := policy.ConvertToNodeItemsBySegment(100, []int64{1, 2})

	// Equal rows -> equal current score, despite 4 vs 1 segments.
	assert.Equal(t, items[1].GetCurrentScore(), items[2].GetCurrentScore())
}

// TestScoreBasedBalance_SegmentCountFactorEnabled verifies that with factor>0 the
// node holding more segments scores strictly higher (even at equal rows), which is
// exactly the signal needed to move segments off the high-count node.
func TestScoreBasedBalance_SegmentCountFactorEnabled(t *testing.T) {
	paramtable.Init()
	withSegmentCountFactor(t, "100")

	policy := newSkewedCountPolicy(t)
	items := policy.ConvertToNodeItemsBySegment(100, []int64{1, 2})

	// node 1 has 4 segments, node 2 has 1 -> node 1 scores higher now.
	assert.Greater(t, items[1].GetCurrentScore(), items[2].GetCurrentScore())

	gf := paramtable.Get().QueryCoordCfg.GlobalRowCountFactor.GetAsFloat()
	// base score is identical (10000 rows each); the whole gap is the segment term:
	// node1 += 100*(4 + 4*gf), node2 += 100*(1 + 1*gf) -> gap = 100*3*(1+gf).
	expectedGap := 100.0 * 3.0 * (1 + gf)
	assert.InDelta(t, expectedGap, items[1].GetCurrentScore()-items[2].GetCurrentScore(), 1.0)
}

// TestScoreBasedBalance_CalculateSegmentScore covers the per-segment delta used when
// moving a segment: disabled -> rows*(1+gf); enabled -> plus factor*(1+gf).
func TestScoreBasedBalance_CalculateSegmentScore(t *testing.T) {
	paramtable.Init()
	policy := newSkewedCountPolicy(t)
	gf := paramtable.Get().QueryCoordCfg.GlobalRowCountFactor.GetAsFloat()
	seg := &meta.Segment{SegmentInfo: &datapb.SegmentInfo{ID: 1, NumOfRows: 1000, CollectionID: 100}}

	assert.InDelta(t, 1000.0*(1+gf), policy.CalculateSegmentScore(seg), 1e-6)

	withSegmentCountFactor(t, "100")
	assert.InDelta(t, 1000.0*(1+gf)+100.0*(1+gf), policy.CalculateSegmentScore(seg), 1e-6)
}

// TestScoreBasedBalance_AssignPrefersFewerSegmentsWhenEnabled verifies end-to-end that,
// with the factor enabled, a new segment is assigned to the node with fewer segments
// even though both nodes carry equal rows.
func TestScoreBasedBalance_AssignPrefersFewerSegmentsWhenEnabled(t *testing.T) {
	paramtable.Init()
	withSegmentCountFactor(t, "100")

	policy := newSkewedCountPolicy(t)
	segments := []*meta.Segment{
		{SegmentInfo: &datapb.SegmentInfo{ID: 1, NumOfRows: 500, CollectionID: 100}},
	}
	plans := policy.AssignSegment(context.Background(), 100, segments, []int64{1, 2}, false)

	assert.Equal(t, 1, len(plans))
	// node 2 has fewer segments -> should be the target.
	assert.Equal(t, int64(2), plans[0].To)
}
