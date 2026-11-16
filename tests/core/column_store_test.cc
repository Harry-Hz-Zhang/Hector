#include "hector/core/column_store.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "hector/common/byte_io.h"
#include "hector/common/types.h"
#include "hector/core/schema.h"
#include "hector/core/value.h"

namespace hector {
namespace {

std::vector<float> RowValue(RowId i) {
  const auto f = static_cast<float>(i);
  return {f, f + 0.25F, -f, f * 2.0F};
}

TEST(ColumnStoreTest, DenseChunksKeepAddressesStable) {
  DenseColumn column(4);
  EXPECT_EQ(column.dim(), 4U);
  EXPECT_EQ(column.size(), 0U);

  ASSERT_TRUE(column.Append(RowValue(0)).ok());
  const float* first = column.Get(0).data();
  for (RowId i = 1; i <= 5000; ++i) {
    ASSERT_TRUE(column.Append(RowValue(i)).ok()) << i;
  }
  EXPECT_EQ(column.Get(0).data(), first);
  EXPECT_EQ(column.size(), 5001U);
  const std::vector<float> expected = RowValue(4999);
  std::span<const float> got = column.Get(4999);
  ASSERT_EQ(got.size(), 4U);
  EXPECT_EQ(std::vector<float>(got.begin(), got.end()), expected);
  EXPECT_GT(column.MemoryBytes(), 5001U * 4U * sizeof(float) - 1);

  // 维度不符拒绝且不改变行数。
  EXPECT_EQ(column.Append(std::vector<float>{1.0F}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(column.size(), 5001U);

  StringByteWriter out;
  ASSERT_TRUE(column.Save(out).ok());
  DenseColumn loaded(4);
  StringByteReader in(out.data());
  ASSERT_TRUE(loaded.Load(in).ok());
  EXPECT_EQ(in.remaining(), 0U);
  ASSERT_EQ(loaded.size(), column.size());
  for (RowId i = 0; i < column.size(); ++i) {
    std::span<const float> a = column.Get(i);
    std::span<const float> b = loaded.Get(i);
    ASSERT_EQ(std::vector<float>(a.begin(), a.end()),
              std::vector<float>(b.begin(), b.end()))
        << i;
  }

  // 维度不同的列不能加载这份数据。
  DenseColumn wrong_dim(3);
  StringByteReader in2(out.data());
  EXPECT_EQ(wrong_dim.Load(in2).code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(wrong_dim.size(), 0U);
}

TEST(ColumnStoreTest, AppendBeyondCapacityFails) {
  DenseColumn column(4, 2048);
  for (RowId i = 0; i < 2048; ++i) {
    ASSERT_TRUE(column.Append(RowValue(i)).ok()) << i;
  }
  EXPECT_EQ(column.Append(RowValue(2048)).code(),
            absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(column.size(), 2048U);

  SparseColumn sparse(2);
  ASSERT_TRUE(sparse.Append(SparseVector{{1, 3}, {0.5F, 0.25F}}).ok());
  ASSERT_TRUE(sparse.Append(SparseVector{}).ok());
  EXPECT_EQ(sparse.Append(SparseVector{{1}, {1.0F}}).code(),
            absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(sparse.size(), 2U);
  EXPECT_EQ(sparse.Get(0).indices.size(), 2U);
  EXPECT_EQ(sparse.Get(0).indices[1], 3U);
  EXPECT_EQ(sparse.Get(0).values[1], 0.25F);
  EXPECT_TRUE(sparse.Get(1).indices.empty());

  ScalarColumn scalar(ScalarType::kString, 1);
  ASSERT_TRUE(scalar.Append(ScalarValue{std::string("x")}).ok());
  EXPECT_EQ(scalar.Append(ScalarValue{std::string("y")}).code(),
            absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(scalar.size(), 1U);
  EXPECT_EQ(std::get<std::string>(scalar.Get(0)), "x");

  // 类型不符拒绝。
  ScalarColumn ints(ScalarType::kInt64, 4);
  EXPECT_EQ(ints.Append(ScalarValue{1.0F}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ints.size(), 0U);

  // 稀疏与普通列的 Save/Load 往返。
  StringByteWriter sparse_out;
  ASSERT_TRUE(sparse.Save(sparse_out).ok());
  SparseColumn sparse_loaded(2);
  StringByteReader sparse_in(sparse_out.data());
  ASSERT_TRUE(sparse_loaded.Load(sparse_in).ok());
  ASSERT_EQ(sparse_loaded.size(), 2U);
  EXPECT_EQ(sparse_loaded.Get(0).indices[0], 1U);
  EXPECT_EQ(sparse_loaded.Get(0).values[0], 0.5F);

  ASSERT_TRUE(ints.Append(ScalarValue{int64_t{-7}}).ok());
  StringByteWriter scalar_out;
  ASSERT_TRUE(ints.Save(scalar_out).ok());
  ScalarColumn ints_loaded(ScalarType::kInt64, 4);
  StringByteReader scalar_in(scalar_out.data());
  ASSERT_TRUE(ints_loaded.Load(scalar_in).ok());
  ASSERT_EQ(ints_loaded.size(), 1U);
  EXPECT_EQ(std::get<int64_t>(ints_loaded.Get(0)), -7);
}

}  // namespace
}  // namespace hector
