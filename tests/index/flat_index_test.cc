#include "hector/index/flat_index.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/bitmap.h"
#include "hector/common/byte_io.h"
#include "hector/common/concurrent_bitmap.h"
#include "hector/common/row_filter.h"
#include "hector/common/types.h"
#include "hector/index/metric.h"
#include "hector/index/vector_index.h"
#include "tests/index/fake_accessors.h"

namespace hector {
namespace {

using ::hector::testing::FakeDenseAccessor;

struct ExpectedHit {
  RowId row_id;
  float distance;
};

void ExpectHits(const absl::StatusOr<std::vector<Hit>>& actual,
                const std::vector<ExpectedHit>& expected) {
  ASSERT_TRUE(actual.ok()) << actual.status();
  ASSERT_EQ(actual->size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ((*actual)[i].row_id, expected[i].row_id) << "at " << i;
    EXPECT_NEAR((*actual)[i].distance, expected[i].distance, 1e-6)
        << "at " << i;
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

void AddAll(FlatIndex& index, RowId n) {
  for (RowId id = 0; id < n; ++id) {
    ASSERT_TRUE(index.Add(id).ok()) << id;
  }
}

TEST(FlatIndexTest, L2SearchReturnsExactTopK) {
  FakeDenseAccessor accessor(2, {{0, 0}, {1, 0}, {3, 0}});
  FlatIndex index(2, MetricType::kL2, &accessor);
  AddAll(index, 3);

  const std::vector<float> query = {0.9F, 0};
  ExpectHits(index.Search(query, 2, SearchParams{}, RowFilter::AllowAll(3)),
             {{1, 0.01F}, {0, 0.81F}});
}

TEST(FlatIndexTest, InnerProductOrdersByNegativeDot) {
  FakeDenseAccessor accessor(2, {{0.5F, 0}, {2, 0}});
  FlatIndex index(2, MetricType::kInnerProduct, &accessor);
  AddAll(index, 2);

  const std::vector<float> query = {1, 0};
  ExpectHits(index.Search(query, 2, SearchParams{}, RowFilter::AllowAll(2)),
             {{1, -2.0F}, {0, -0.5F}});
}

TEST(FlatIndexTest, RespectsRowFilter) {
  FakeDenseAccessor accessor(2, {{0, 0}, {1, 0}, {2, 0}, {3, 0}});
  FlatIndex index(2, MetricType::kL2, &accessor);
  AddAll(index, 4);
  ConcurrentBitmap tombstones(16);
  tombstones.Set(0);
  Bitmap allow(4);
  allow.Set(0);
  allow.Set(1);
  allow.Set(3);
  accessor.ResetCounts();

  const std::vector<float> query = {0, 0};
  const auto hits =
      index.Search(query, 4, SearchParams{}, RowFilter(4, &tombstones, &allow));
  EXPECT_EQ(RowIds(hits), (std::vector<RowId>{1, 3}));
  EXPECT_EQ(accessor.get_count(0), 0);
  EXPECT_EQ(accessor.get_count(2), 0);
  EXPECT_EQ(accessor.get_count(1), 1);
  EXPECT_EQ(accessor.get_count(3), 1);
}

TEST(FlatIndexTest, ExcludesRowsBeyondPublishedCount) {
  FakeDenseAccessor accessor(1, {{0}, {1}, {2}, {3}, {4}});
  FlatIndex index(1, MetricType::kL2, &accessor);
  AddAll(index, 5);
  accessor.ResetCounts();

  // 查询靠近未发布的行 4，但只能看到 row_count=3 以内的行。
  const std::vector<float> query = {4};
  const auto hits =
      index.Search(query, 5, SearchParams{}, RowFilter::AllowAll(3));
  EXPECT_EQ(RowIds(hits), (std::vector<RowId>{2, 1, 0}));
  EXPECT_EQ(accessor.get_count(3), 0);
  EXPECT_EQ(accessor.get_count(4), 0);
}

TEST(FlatIndexTest, FewerRowsThanTopK) {
  FakeDenseAccessor accessor(2, {{0, 0}, {1, 1}});
  const std::vector<float> query = {0, 0};

  FlatIndex index(2, MetricType::kL2, &accessor);
  AddAll(index, 2);
  ExpectHits(index.Search(query, 10, SearchParams{}, RowFilter::AllowAll(2)),
             {{0, 0.0F}, {1, 2.0F}});

  FlatIndex empty(2, MetricType::kL2, &accessor);
  const auto none =
      empty.Search(query, 10, SearchParams{}, RowFilter::AllowAll(2));
  ASSERT_TRUE(none.ok()) << none.status();
  EXPECT_TRUE(none->empty());
}

TEST(FlatIndexTest, RejectsBadQueryTopKAndDuplicates) {
  FakeDenseAccessor accessor(4, {{0, 0, 0, 0}, {1, 1, 1, 1}});
  FlatIndex index(4, MetricType::kL2, &accessor);
  ASSERT_TRUE(index.Add(0).ok());

  const std::vector<float> short_query = {0, 0, 0};
  EXPECT_EQ(index.Search(short_query, 1, SearchParams{}, RowFilter::AllowAll(2))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  const std::vector<RowId> candidates = {0};
  EXPECT_EQ(index.SearchCandidates(short_query, 1, candidates).status().code(),
            absl::StatusCode::kInvalidArgument);

  const std::vector<float> query = {0, 0, 0, 0};
  EXPECT_EQ(index.Search(query, 0, SearchParams{}, RowFilter::AllowAll(2))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(index.SearchCandidates(query, 0, candidates).status().code(),
            absl::StatusCode::kInvalidArgument);

  EXPECT_EQ(index.Add(0).code(), absl::StatusCode::kAlreadyExists);
  // 访问器中不存在的行不能加入。
  EXPECT_EQ(index.Add(2).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(index.Size(), 1U);
}

TEST(FlatIndexTest, BuildOnlyOnEmptyIndex) {
  FakeDenseAccessor accessor(1, {{0}, {1}, {2}, {3}});
  FlatIndex index(1, MetricType::kL2, &accessor);
  const std::vector<RowId> ids = {0, 1, 2};
  ASSERT_TRUE(index.Build(ids).ok());
  EXPECT_EQ(index.Size(), 3U);
  EXPECT_EQ(index.Build(ids).code(), absl::StatusCode::kFailedPrecondition);

  FlatIndex added(1, MetricType::kL2, &accessor);
  ASSERT_TRUE(added.Add(3).ok());
  EXPECT_EQ(added.Build(ids).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(added.Size(), 1U);

  // 输入有问题时整体拒绝，索引保持为空。
  FlatIndex bad(1, MetricType::kL2, &accessor);
  const std::vector<RowId> duplicated = {0, 1, 0};
  EXPECT_EQ(bad.Build(duplicated).code(), absl::StatusCode::kInvalidArgument);
  const std::vector<RowId> out_of_range = {0, 9};
  EXPECT_EQ(bad.Build(out_of_range).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(bad.Size(), 0U);
}

TEST(FlatIndexTest, SearchCandidatesOnlyScoresGivenRows) {
  FakeDenseAccessor accessor(1, {{0}, {1}, {2}, {3}, {4}});
  FlatIndex index(1, MetricType::kL2, &accessor);
  AddAll(index, 5);
  accessor.ResetCounts();

  const std::vector<float> query = {4};
  const std::vector<RowId> candidates = {4, 2, 99};
  ExpectHits(index.SearchCandidates(query, 10, candidates),
             {{4, 0.0F}, {2, 4.0F}});
  EXPECT_EQ(accessor.total_gets(), 2);
}

TEST(FlatIndexTest, SaveLoadRoundTrip) {
  FakeDenseAccessor accessor(1, {{0}, {1}, {2}, {3}, {4}, {5}});
  FlatIndex index(1, MetricType::kL2, &accessor);
  ASSERT_TRUE(index.Add(0).ok());
  ASSERT_TRUE(index.Add(2).ok());
  ASSERT_TRUE(index.Add(5).ok());
  StringByteWriter out;
  ASSERT_TRUE(index.Save(out).ok());

  FlatIndex loaded(1, MetricType::kL2, &accessor);
  StringByteReader in(out.data());
  ASSERT_TRUE(loaded.Load(in).ok());
  EXPECT_EQ(loaded.Size(), 3U);
  const std::vector<float> query = {1.6F};
  const auto expected =
      index.Search(query, 10, SearchParams{}, RowFilter::AllowAll(6));
  const auto actual =
      loaded.Search(query, 10, SearchParams{}, RowFilter::AllowAll(6));
  ASSERT_TRUE(expected.ok());
  EXPECT_EQ(RowIds(actual), RowIds(expected));
  EXPECT_EQ(RowIds(actual), (std::vector<RowId>{2, 0, 5}));
  // 加载后重复检查仍然有效。
  EXPECT_EQ(loaded.Add(2).code(), absl::StatusCode::kAlreadyExists);

  // 版本号是开头的 u32（小端序）。
  std::string bad_version = out.data();
  bad_version[0] = 2;
  FlatIndex wrong(1, MetricType::kL2, &accessor);
  StringByteReader bad_in(bad_version);
  EXPECT_EQ(wrong.Load(bad_in).code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(wrong.Size(), 0U);

  // 截断的数据同样是 DataLoss。
  const std::string truncated = out.data().substr(0, out.data().size() - 1);
  StringByteReader truncated_in(truncated);
  EXPECT_EQ(wrong.Load(truncated_in).code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(wrong.Size(), 0U);

  StringByteReader again(out.data());
  EXPECT_EQ(loaded.Load(again).code(), absl::StatusCode::kFailedPrecondition);
}

TEST(FlatIndexTest, TiesBreakByRowIdAscending) {
  FakeDenseAccessor accessor(1, {{1}, {1}, {1}, {1}, {1}});
  FlatIndex index(1, MetricType::kL2, &accessor);
  // 倒序加入，迫使堆在距离相同时淘汰 row id 较大的一方。
  for (RowId id = 5; id-- > 0;) {
    ASSERT_TRUE(index.Add(id).ok());
  }
  const std::vector<float> query = {0};
  ExpectHits(index.Search(query, 3, SearchParams{}, RowFilter::AllowAll(5)),
             {{0, 1.0F}, {1, 1.0F}, {2, 1.0F}});
  const std::vector<RowId> candidates = {4, 3, 2, 1, 0};
  ExpectHits(index.SearchCandidates(query, 2, candidates),
             {{0, 1.0F}, {1, 1.0F}});
}

TEST(FlatIndexTest, NanDistanceRanksLast) {
  // NaN 不满足 HitLess 的严格弱序，索引把它按 +inf 处理，排在最后。core
  // 只写入有限值，但有限向量的距离在不做乘加融合的内核里也可能溢出成
  // inf + (-inf) = NaN；这里直接让访问器返回
  // NaN，结果与编译器是否融合乘加无关。
  const float nan = std::numeric_limits<float>::quiet_NaN();
  FakeDenseAccessor accessor(2, {{nan, 0}, {1, 0}, {0, 0}, {nan, 0}});
  FlatIndex index(2, MetricType::kInnerProduct, &accessor);
  AddAll(index, 4);
  const std::vector<float> query = {1, 0};

  const auto hits =
      index.Search(query, 4, SearchParams{}, RowFilter::AllowAll(4));
  ASSERT_TRUE(hits.ok()) << hits.status();
  EXPECT_EQ(RowIds(hits), (std::vector<RowId>{1, 2, 0, 3}));
  for (const Hit& h : *hits) {
    EXPECT_FALSE(std::isnan(h.distance)) << h.row_id;
  }
  EXPECT_EQ((*hits)[2].distance, std::numeric_limits<float>::infinity());

  EXPECT_EQ(
      RowIds(index.Search(query, 2, SearchParams{}, RowFilter::AllowAll(4))),
      (std::vector<RowId>{1, 2}));
  const std::vector<RowId> candidates = {3, 0, 2};
  EXPECT_EQ(RowIds(index.SearchCandidates(query, 2, candidates)),
            (std::vector<RowId>{2, 0}));
}

TEST(FlatIndexTest, ConcurrentSearchAndAdd) {
  constexpr RowId kInitial = 100;
  constexpr RowId kAdded = 1000;
  constexpr uint32_t kDim = 8;
  std::vector<std::vector<float>> rows;
  rows.reserve(kInitial + kAdded);
  for (RowId id = 0; id < kInitial + kAdded; ++id) {
    std::vector<float> row(kDim);
    for (uint32_t d = 0; d < kDim; ++d) {
      row[d] = static_cast<float>((id * 31 + d * 7) % 101);
    }
    rows.push_back(std::move(row));
  }
  FakeDenseAccessor accessor(kDim, std::move(rows));
  FlatIndex index(kDim, MetricType::kL2, &accessor);
  AddAll(index, kInitial);

  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&index, &failures, t] {
      const std::vector<float> query(kDim, static_cast<float>(t * 10));
      const RowFilter filter = RowFilter::AllowAll(kInitial + kAdded);
      for (int i = 0; i < 1000; ++i) {
        const auto hits = index.Search(query, 10, SearchParams{}, filter);
        if (!hits.ok() || hits->size() != 10) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  threads.emplace_back([&index, &failures] {
    for (RowId id = kInitial; id < kInitial + kAdded; ++id) {
      if (!index.Add(id).ok()) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });
  for (std::thread& t : threads) {
    t.join();
  }

  EXPECT_EQ(failures.load(), 0);
  EXPECT_EQ(index.Size(), static_cast<size_t>(kInitial + kAdded));
}

}  // namespace
}  // namespace hector
