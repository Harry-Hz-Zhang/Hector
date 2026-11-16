#include "hector/index/flat_sparse_index.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/concurrent_bitmap.h"
#include "hector/common/row_filter.h"
#include "hector/common/types.h"
#include "hector/index/sparse_index.h"
#include "tests/index/fake_accessors.h"

namespace hector {
namespace {

using ::hector::testing::FakeSparseAccessor;

constexpr uint32_t kMaxDim = 16;

// 行 0={1:1.0}、行 1={2:1.0}、行 2={1:0.5,2:3.0}。
FakeSparseAccessor MakeThreeRows() {
  return FakeSparseAccessor({
      SparseVector{{1}, {1.0F}},
      SparseVector{{2}, {1.0F}},
      SparseVector{{1, 2}, {0.5F, 3.0F}},
  });
}

SparseVectorView View(const SparseVector& v) {
  return SparseVectorView{v.indices, v.values};
}

void AddAll(FlatSparseIndex& index, RowId n) {
  for (RowId id = 0; id < n; ++id) {
    ASSERT_TRUE(index.Add(id).ok()) << id;
  }
}

std::vector<RowId> RowIds(const absl::StatusOr<std::vector<Hit>>& hits) {
  std::vector<RowId> ids;
  if (hits.ok()) {
    for (const Hit& h : *hits) {
      ids.push_back(h.row_id);
    }
  }
  return ids;
}

TEST(FlatSparseIndexTest, ScoresOnlyOverlappingRows) {
  const FakeSparseAccessor accessor = MakeThreeRows();
  FlatSparseIndex index(kMaxDim, &accessor);
  AddAll(index, 3);

  const SparseVector query{{1}, {2.0F}};
  const auto hits = index.Search(View(query), 5, SparseSearchParams{},
                                 RowFilter::AllowAll(3));
  ASSERT_TRUE(hits.ok()) << hits.status();
  ASSERT_EQ(hits->size(), 2U);
  EXPECT_EQ((*hits)[0].row_id, 0U);
  EXPECT_FLOAT_EQ((*hits)[0].distance, -2.0F);
  EXPECT_EQ((*hits)[1].row_id, 2U);
  EXPECT_FLOAT_EQ((*hits)[1].distance, -1.0F);
}

TEST(FlatSparseIndexTest, RespectsRowFilterAndTopK) {
  const FakeSparseAccessor accessor = MakeThreeRows();
  FlatSparseIndex index(kMaxDim, &accessor);
  AddAll(index, 3);
  ConcurrentBitmap tombstones(16);
  tombstones.Set(0);
  const RowFilter filter(3, &tombstones, nullptr);

  const SparseVector query{{1}, {2.0F}};
  const auto wand = index.Search(
      View(query), 1, SparseSearchParams{SparseAlgorithm::kWand}, filter);
  const auto automatic = index.Search(
      View(query), 1, SparseSearchParams{SparseAlgorithm::kAuto}, filter);
  ASSERT_TRUE(wand.ok()) << wand.status();
  ASSERT_EQ(wand->size(), 1U);
  EXPECT_EQ((*wand)[0].row_id, 2U);
  EXPECT_FLOAT_EQ((*wand)[0].distance, -1.0F);
  EXPECT_EQ(RowIds(automatic), RowIds(wand));
  EXPECT_EQ(accessor.get_count(0), 0);

  // 超出已发布行数的行不可见。
  EXPECT_TRUE(RowIds(index.Search(View(query), 5, SparseSearchParams{},
                                  RowFilter::AllowAll(2)))
                  .size() == 1);
}

TEST(FlatSparseIndexTest, BuildAndLoadOnlyOnEmptyIndex) {
  const FakeSparseAccessor accessor = MakeThreeRows();
  FlatSparseIndex index(kMaxDim, &accessor);
  ASSERT_TRUE(index.Add(0).ok());
  const std::vector<RowId> ids = {1, 2};
  EXPECT_EQ(index.Build(ids).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(index.Add(0).code(), absl::StatusCode::kAlreadyExists);
  ASSERT_TRUE(index.Add(2).ok());

  StringByteWriter out;
  ASSERT_TRUE(index.Save(out).ok());
  StringByteReader again(out.data());
  EXPECT_EQ(index.Load(again).code(), absl::StatusCode::kFailedPrecondition);

  FlatSparseIndex loaded(kMaxDim, &accessor);
  StringByteReader in(out.data());
  ASSERT_TRUE(loaded.Load(in).ok());
  EXPECT_EQ(loaded.Size(), index.Size());
  const SparseVector query{{1}, {2.0F}};
  EXPECT_EQ(RowIds(loaded.Search(View(query), 5, SparseSearchParams{},
                                 RowFilter::AllowAll(3))),
            (std::vector<RowId>{0, 2}));

  FlatSparseIndex built(kMaxDim, &accessor);
  const std::vector<RowId> all = {0, 1, 2};
  ASSERT_TRUE(built.Build(all).ok());
  EXPECT_EQ(built.Size(), 3U);

  std::string bad_version = out.data();
  bad_version[0] = 2;
  FlatSparseIndex wrong(kMaxDim, &accessor);
  StringByteReader bad_in(bad_version);
  EXPECT_EQ(wrong.Load(bad_in).code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(wrong.Size(), 0U);
}

TEST(FlatSparseIndexTest, RejectsBadQuery) {
  const FakeSparseAccessor accessor = MakeThreeRows();
  FlatSparseIndex index(kMaxDim, &accessor);
  AddAll(index, 3);
  const RowFilter filter = RowFilter::AllowAll(3);

  const SparseVector ok{{1}, {2.0F}};
  EXPECT_EQ(
      index.Search(View(ok), 0, SparseSearchParams{}, filter).status().code(),
      absl::StatusCode::kInvalidArgument);
  const SparseVector mismatched{{1, 2}, {2.0F}};
  EXPECT_EQ(index.Search(View(mismatched), 1, SparseSearchParams{}, filter)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  const SparseVector unsorted{{2, 1}, {1.0F, 1.0F}};
  EXPECT_EQ(index.Search(View(unsorted), 1, SparseSearchParams{}, filter)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  const SparseVector out_of_range{{kMaxDim}, {1.0F}};
  EXPECT_EQ(index.Search(View(out_of_range), 1, SparseSearchParams{}, filter)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(index.Add(3).code(), absl::StatusCode::kInvalidArgument);
}

TEST(FlatSparseIndexTest, NanScoreRanksLast) {
  // 同 FlatIndexTest.NanDistanceRanksLast：NaN 分数按 +inf 距离处理，排在最后。
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const FakeSparseAccessor accessor({
      SparseVector{{1}, {nan}},
      SparseVector{{1}, {1.0F}},
  });
  FlatSparseIndex index(kMaxDim, &accessor);
  AddAll(index, 2);

  const SparseVector query{{1}, {1.0F}};
  const auto hits = index.Search(View(query), 5, SparseSearchParams{},
                                 RowFilter::AllowAll(2));
  ASSERT_TRUE(hits.ok()) << hits.status();
  EXPECT_EQ(RowIds(hits), (std::vector<RowId>{1, 0}));
  EXPECT_EQ((*hits)[1].distance, std::numeric_limits<float>::infinity());
}

}  // namespace
}  // namespace hector
