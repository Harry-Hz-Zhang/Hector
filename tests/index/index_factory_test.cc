#include "hector/index/index_factory.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "hector/index/metric.h"
#include "tests/index/fake_accessors.h"

namespace hector {
namespace {

using ::hector::testing::FakeDenseAccessor;
using ::hector::testing::FakeSparseAccessor;
using ::testing::HasSubstr;

TEST(IndexFactoryTest, UnimplementedKindsAreRejected) {
  const FakeDenseAccessor dense(4, {});
  const FakeSparseAccessor sparse({});

  const auto hnsw = MakeVectorIndex(
      DenseIndexConfig{4, MetricType::kL2, DenseIndexKind::kHnsw, {}}, &dense);
  EXPECT_EQ(hnsw.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(std::string(hnsw.status().message()), HasSubstr("c2"));

  const auto inverted = MakeSparseIndex(
      SparseIndexConfig{100, SparseIndexKind::kInverted}, &sparse);
  EXPECT_EQ(inverted.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(std::string(inverted.status().message()), HasSubstr("c7"));

  const auto int8 = MakeVectorIndex(
      DenseIndexConfig{4, MetricType::kL2, DenseIndexKind::kFlat, {}, true},
      &dense);
  EXPECT_EQ(int8.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(std::string(int8.status().message()), HasSubstr("c6"));

  const auto flat = MakeVectorIndex(
      DenseIndexConfig{4, MetricType::kCosine, DenseIndexKind::kFlat, {}},
      &dense);
  ASSERT_TRUE(flat.ok()) << flat.status();
  ASSERT_NE(*flat, nullptr);
  EXPECT_LT((*flat)->MemoryBytes(), 1024U);

  const auto flat_sparse =
      MakeSparseIndex(SparseIndexConfig{100, SparseIndexKind::kFlat}, &sparse);
  ASSERT_TRUE(flat_sparse.ok()) << flat_sparse.status();
  ASSERT_NE(*flat_sparse, nullptr);
  EXPECT_LT((*flat_sparse)->MemoryBytes(), 1024U);
}

TEST(IndexFactoryTest, RejectsInvalidConfig) {
  const FakeDenseAccessor dense(4, {});
  const FakeSparseAccessor sparse({});
  const DenseIndexConfig flat{4, MetricType::kL2, DenseIndexKind::kFlat, {}};

  EXPECT_EQ(MakeVectorIndex(flat, nullptr).status().code(),
            absl::StatusCode::kInvalidArgument);
  const DenseIndexConfig zero_dim{
      0, MetricType::kL2, DenseIndexKind::kFlat, {}};
  EXPECT_EQ(MakeVectorIndex(zero_dim, &dense).status().code(),
            absl::StatusCode::kInvalidArgument);
  const DenseIndexConfig wrong_dim{
      8, MetricType::kL2, DenseIndexKind::kFlat, {}};
  EXPECT_EQ(MakeVectorIndex(wrong_dim, &dense).status().code(),
            absl::StatusCode::kInvalidArgument);

  EXPECT_EQ(
      MakeSparseIndex(SparseIndexConfig{100, SparseIndexKind::kFlat}, nullptr)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      MakeSparseIndex(SparseIndexConfig{0, SparseIndexKind::kFlat}, &sparse)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace hector
