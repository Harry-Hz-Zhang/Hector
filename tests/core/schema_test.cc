#include "hector/core/schema.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"

namespace hector {
namespace {

using ::testing::HasSubstr;

ColumnDef Dense(std::string name, uint32_t dim, MetricType metric,
                HnswConfig hnsw = {.m = 0, .ef_construction = 0, .seed = 42}) {
  return ColumnDef{std::move(name),
                   DenseColumnSpec{.dim = dim,
                                   .metric = metric,
                                   .index = DenseIndexKind::kFlat,
                                   .hnsw = hnsw}};
}

ColumnDef Sparse(std::string name, uint32_t max_dim) {
  return ColumnDef{std::move(name), SparseColumnSpec{
                                        .max_dim = max_dim,
                                        .index = SparseIndexKind::kFlat,
                                    }};
}

ColumnDef Scalar(std::string name, ScalarType type) {
  return ColumnDef{std::move(name), type};
}

void ExpectSameSchema(const Schema& a, const Schema& b) {
  EXPECT_EQ(a.table_name(), b.table_name());
  ASSERT_EQ(a.columns().size(), b.columns().size());
  for (size_t i = 0; i < a.columns().size(); ++i) {
    SCOPED_TRACE(i);
    const ColumnDef& x = a.columns()[i];
    const ColumnDef& y = b.columns()[i];
    EXPECT_EQ(x.name, y.name);
    ASSERT_EQ(x.spec.index(), y.spec.index());
    if (const auto* dx = std::get_if<DenseColumnSpec>(&x.spec)) {
      const auto& dy = std::get<DenseColumnSpec>(y.spec);
      EXPECT_EQ(dx->dim, dy.dim);
      EXPECT_EQ(dx->metric, dy.metric);
      EXPECT_EQ(dx->index, dy.index);
      EXPECT_EQ(dx->hnsw.m, dy.hnsw.m);
      EXPECT_EQ(dx->hnsw.ef_construction, dy.hnsw.ef_construction);
      EXPECT_EQ(dx->hnsw.seed, dy.hnsw.seed);
      EXPECT_EQ(dx->int8_quantization, dy.int8_quantization);
    } else if (const auto* sx = std::get_if<SparseColumnSpec>(&x.spec)) {
      const auto& sy = std::get<SparseColumnSpec>(y.spec);
      EXPECT_EQ(sx->max_dim, sy.max_dim);
      EXPECT_EQ(sx->index, sy.index);
    } else {
      EXPECT_EQ(std::get<ScalarType>(x.spec), std::get<ScalarType>(y.spec));
    }
  }
}

TEST(SchemaTest, ValidSchemaFillsDefaults) {
  auto schema = Schema::Create(
      "docs", {Dense("emb", 8, MetricType::kCosine), Sparse("sp", 1000),
               Scalar("year", ScalarType::kInt64)});
  ASSERT_TRUE(schema.ok()) << schema.status();

  EXPECT_EQ(schema->table_name(), "docs");
  ASSERT_EQ(schema->columns().size(), 3U);
  const auto& dense = std::get<DenseColumnSpec>(schema->columns()[0].spec);
  EXPECT_EQ(dense.dim, 8U);
  EXPECT_EQ(dense.metric, MetricType::kCosine);
  EXPECT_EQ(dense.index, DenseIndexKind::kFlat);
  EXPECT_EQ(dense.hnsw.m, 16U);
  EXPECT_EQ(dense.hnsw.ef_construction, 200U);
  EXPECT_EQ(dense.hnsw.seed, 42U);
  EXPECT_FALSE(dense.int8_quantization);

  EXPECT_EQ(schema->FindColumn("emb"), 0U);
  EXPECT_EQ(schema->FindColumn("sp"), 1U);
  EXPECT_EQ(schema->FindColumn("year"), 2U);
  EXPECT_EQ(schema->FindColumn("nope"), std::nullopt);

  StringByteWriter out;
  ASSERT_TRUE(schema->Save(out).ok());
  StringByteReader in(out.data());
  auto loaded = Schema::Load(in);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  ExpectSameSchema(*schema, *loaded);
  EXPECT_EQ(in.remaining(), 0U);
  EXPECT_EQ(loaded->FindColumn("year"), 2U);

  // 截断的数据返回 DataLoss。
  const std::string truncated = out.data().substr(0, out.data().size() - 1);
  StringByteReader short_in(truncated);
  EXPECT_EQ(Schema::Load(short_in).status().code(),
            absl::StatusCode::kDataLoss);
}

struct InvalidSchemaCase {
  std::string label;
  std::string table_name;
  std::vector<ColumnDef> columns;
  std::string expected_substring;
};

// 让 ctest 用例名只显示标签，而不是对象的字节转储。
void PrintTo(const InvalidSchemaCase& c, std::ostream* os) { *os << c.label; }

class SchemaTest : public ::testing::TestWithParam<InvalidSchemaCase> {};

TEST_P(SchemaTest, RejectsInvalidSchemas) {
  const InvalidSchemaCase& c = GetParam();
  auto schema = Schema::Create(c.table_name, c.columns);
  ASSERT_FALSE(schema.ok());
  EXPECT_EQ(schema.status().code(), absl::StatusCode::kInvalidArgument)
      << schema.status();
  EXPECT_THAT(schema.status().message(), HasSubstr(c.expected_substring));
}

std::vector<ColumnDef> TooManyColumns() {
  std::vector<ColumnDef> columns = {Dense("emb", 4, MetricType::kL2)};
  for (int i = 0; i < 256; ++i) {
    columns.push_back(Scalar("c" + std::to_string(i), ScalarType::kInt64));
  }
  return columns;
}

std::vector<InvalidSchemaCase> InvalidCases() {
  return {
      {"TableNameStartsWithDigit",
       "1abc",
       {Dense("emb", 4, MetricType::kL2)},
       "1abc"},
      {"DuplicateColumnName",
       "t",
       {Dense("emb", 4, MetricType::kL2), Scalar("emb", ScalarType::kInt64)},
       "emb"},
      {"NoVectorColumn",
       "t",
       {Scalar("a", ScalarType::kInt64), Scalar("b", ScalarType::kString)},
       "vector column"},
      {"DimZero", "t", {Dense("emb0", 0, MetricType::kL2)}, "emb0"},
      {"Dim4097", "t", {Dense("emb4097", 4097, MetricType::kL2)}, "emb4097"},
      {"TooManyColumns", "t", TooManyColumns(), "257"},
      {"SparseMaxDimZero",
       "t",
       {Dense("emb", 4, MetricType::kL2), Sparse("sp0", 0)},
       "sp0"},
      {"HnswMTooSmall",
       "t",
       {Dense("emb", 4, MetricType::kL2,
              {.m = 1, .ef_construction = 200, .seed = 42})},
       "hnsw.m"},
      {"HnswMTooLarge",
       "t",
       {Dense("emb", 4, MetricType::kL2,
              {.m = 129, .ef_construction = 200, .seed = 42})},
       "hnsw.m"},
      {"EfConstructionBelowM",
       "t",
       {Dense("emb", 4, MetricType::kL2,
              {.m = 32, .ef_construction = 16, .seed = 42})},
       "hnsw.ef_construction"},
      {"EfConstructionTooLarge",
       "t",
       {Dense("emb", 4, MetricType::kL2,
              {.m = 16, .ef_construction = 4097, .seed = 42})},
       "hnsw.ef_construction"},
  };
}

INSTANTIATE_TEST_SUITE_P(
    Cases, SchemaTest, ::testing::ValuesIn(InvalidCases()),
    [](const ::testing::TestParamInfo<InvalidSchemaCase>& info) {
      return info.param.label;
    });

}  // namespace
}  // namespace hector
