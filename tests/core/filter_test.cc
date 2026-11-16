#include "hector/core/filter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "hector/common/bitmap.h"
#include "hector/common/types.h"
#include "hector/core/column_store.h"
#include "hector/core/schema.h"
#include "hector/core/value.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"

namespace hector {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

constexpr RowId kRows = 5;

std::vector<size_t> SetBits(const Bitmap& bitmap) {
  std::vector<size_t> bits;
  bitmap.ForEachSet([&bits](size_t i) { bits.push_back(i); });
  return bits;
}

Condition Cond(std::string column, CompareOp op,
               std::vector<ScalarValue> values) {
  return Condition{
      .column = std::move(column), .op = op, .values = std::move(values)};
}

// 列顺序：v（稠密
// dim=2，不能作为筛选列）、i（int64）、f（float）、s（string）。
class FilterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto schema = Schema::Create(
        "t",
        {ColumnDef{"v", DenseColumnSpec{.dim = 2,
                                        .metric = MetricType::kL2,
                                        .index = DenseIndexKind::kFlat,
                                        .hnsw = {},
                                        .int8_quantization = false}},
         ColumnDef{"i", ScalarType::kInt64}, ColumnDef{"f", ScalarType::kFloat},
         ColumnDef{"s", ScalarType::kString}});
    ASSERT_TRUE(schema.ok()) << schema.status();
    schema_ = std::make_unique<Schema>(*std::move(schema));

    for (int64_t v : {10, 20, 30, 20, 50}) {
      ASSERT_TRUE(ints_.Append(ScalarValue{v}).ok());
    }
    for (float v : {1.5F, -2.0F, 0.0F, 3.25F, 1.5F}) {
      ASSERT_TRUE(floats_.Append(ScalarValue{v}).ok());
    }
    // 按字节字典序："" < "a" < "ab" < "b" < "\xc3\xa9"（é 的 UTF-8 首字节 0xC3
    // 按无符号比较大于 'b'）。
    for (const char* v : {"a", "b", "ab", "\xc3\xa9", ""}) {
      ASSERT_TRUE(strings_.Append(ScalarValue{std::string(v)}).ok());
    }
  }

  std::vector<size_t> Eval(std::vector<Condition> conditions,
                           RowId row_count = kRows) const {
    const Filter filter{.conditions = std::move(conditions)};
    EXPECT_TRUE(ValidateFilter(filter, *schema_).ok());
    const std::vector<const ScalarColumn*> columns = {nullptr, &ints_, &floats_,
                                                      &strings_};
    const Bitmap allow = EvaluateFilter(filter, *schema_, columns, row_count);
    EXPECT_EQ(allow.size(), row_count);
    return SetBits(allow);
  }

  absl::Status Validate(Condition bad) const {
    // 第 0 个条件合法，出错的是第 1 个。
    const Filter filter{.conditions = {Cond("i", CompareOp::kEq, {int64_t{10}}),
                                       std::move(bad)}};
    return ValidateFilter(filter, *schema_);
  }

  std::unique_ptr<Schema> schema_;
  ScalarColumn ints_{ScalarType::kInt64, 16};
  ScalarColumn floats_{ScalarType::kFloat, 16};
  ScalarColumn strings_{ScalarType::kString, 16};
};

TEST_F(FilterTest, EvaluatesAllOperators) {
  // int64：10, 20, 30, 20, 50，与 20 比较。
  const ScalarValue i20{int64_t{20}};
  EXPECT_THAT(Eval({Cond("i", CompareOp::kEq, {i20})}), ElementsAre(1, 3));
  EXPECT_THAT(Eval({Cond("i", CompareOp::kNe, {i20})}), ElementsAre(0, 2, 4));
  EXPECT_THAT(Eval({Cond("i", CompareOp::kLt, {i20})}), ElementsAre(0));
  EXPECT_THAT(Eval({Cond("i", CompareOp::kLe, {i20})}), ElementsAre(0, 1, 3));
  EXPECT_THAT(Eval({Cond("i", CompareOp::kGt, {i20})}), ElementsAre(2, 4));
  EXPECT_THAT(Eval({Cond("i", CompareOp::kGe, {i20})}),
              ElementsAre(1, 2, 3, 4));
  EXPECT_THAT(Eval({Cond("i", CompareOp::kIn,
                         {int64_t{10}, int64_t{50}, int64_t{99}})}),
              ElementsAre(0, 4));

  // float：1.5, -2.0, 0.0, 3.25, 1.5，与 1.5 比较。
  const ScalarValue f15{1.5F};
  EXPECT_THAT(Eval({Cond("f", CompareOp::kEq, {f15})}), ElementsAre(0, 4));
  EXPECT_THAT(Eval({Cond("f", CompareOp::kNe, {f15})}), ElementsAre(1, 2, 3));
  EXPECT_THAT(Eval({Cond("f", CompareOp::kLt, {f15})}), ElementsAre(1, 2));
  EXPECT_THAT(Eval({Cond("f", CompareOp::kLe, {f15})}),
              ElementsAre(0, 1, 2, 4));
  EXPECT_THAT(Eval({Cond("f", CompareOp::kGt, {f15})}), ElementsAre(3));
  EXPECT_THAT(Eval({Cond("f", CompareOp::kGe, {f15})}), ElementsAre(0, 3, 4));
  EXPECT_THAT(Eval({Cond("f", CompareOp::kIn, {-2.0F, 3.25F})}),
              ElementsAre(1, 3));

  // string："a", "b", "ab", "\xc3\xa9", ""，与 "ab" 按字节字典序比较。
  const ScalarValue sab{std::string("ab")};
  EXPECT_THAT(Eval({Cond("s", CompareOp::kEq, {sab})}), ElementsAre(2));
  EXPECT_THAT(Eval({Cond("s", CompareOp::kNe, {sab})}),
              ElementsAre(0, 1, 3, 4));
  EXPECT_THAT(Eval({Cond("s", CompareOp::kLt, {sab})}), ElementsAre(0, 4));
  EXPECT_THAT(Eval({Cond("s", CompareOp::kLe, {sab})}), ElementsAre(0, 2, 4));
  EXPECT_THAT(Eval({Cond("s", CompareOp::kGt, {sab})}), ElementsAre(1, 3));
  EXPECT_THAT(Eval({Cond("s", CompareOp::kGe, {sab})}), ElementsAre(1, 2, 3));
  EXPECT_THAT(
      Eval({Cond("s", CompareOp::kIn,
                 {std::string("b"), std::string(""), std::string("zz")})}),
      ElementsAre(1, 4));

  // 条件之间是「并且」：i >= 20 为 {1,2,3,4}，s < "b" 为 {0,2,4}。
  EXPECT_THAT(Eval({Cond("i", CompareOp::kGe, {i20}),
                    Cond("s", CompareOp::kLt, {std::string("b")})}),
              ElementsAre(2, 4));

  // 空筛选全部置位；只计算 row id < row_count 的行。
  EXPECT_THAT(Eval({}), ElementsAreArray({0, 1, 2, 3, 4}));
  EXPECT_THAT(Eval({Cond("i", CompareOp::kEq, {i20})}, 3), ElementsAre(1));
  EXPECT_THAT(Eval({Cond("i", CompareOp::kEq, {i20})}, 0), IsEmpty());
}

TEST_F(FilterTest, RejectsInvalidConditions) {
  const std::vector<Condition> bad = {
      Cond("nope", CompareOp::kEq, {int64_t{1}}),           // 列不存在
      Cond("v", CompareOp::kEq, {int64_t{1}}),              // 向量列
      Cond("i", CompareOp::kEq, {std::string("2020")}),     // 类型不符
      Cond("f", CompareOp::kGt, {int64_t{1}}),              // 类型须完全一致
      Cond("i", CompareOp::kEq, {int64_t{1}, int64_t{2}}),  // 非 IN 给 2 个值
      Cond("i", CompareOp::kLt, {}),                        // 非 IN 给 0 个值
      Cond("s", CompareOp::kIn, {}),                        // IN 给 0 个值
  };
  for (const Condition& condition : bad) {
    const absl::Status status = Validate(condition);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument)
        << condition.column << ": " << status;
    EXPECT_THAT(status.message(), HasSubstr("conditions[1]"))
        << condition.column;
  }

  EXPECT_TRUE(ValidateFilter(Filter{}, *schema_).ok());
  EXPECT_TRUE(
      Validate(Cond("s", CompareOp::kIn, {std::string("a"), std::string("b")}))
          .ok());
}

}  // namespace
}  // namespace hector
