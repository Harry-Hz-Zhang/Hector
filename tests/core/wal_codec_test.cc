#include "hector/core/wal_codec.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/types.h"
#include "hector/core/schema.h"
#include "hector/core/value.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"

namespace hector {
namespace {

using ::testing::ElementsAre;

Schema TestSchema() {
  auto schema = Schema::Create(
      "docs",
      {ColumnDef{
           "emb",
           DenseColumnSpec{.dim = 3,
                           .metric = MetricType::kCosine,
                           .index = DenseIndexKind::kFlat,
                           .hnsw = {.m = 8, .ef_construction = 50, .seed = 7},
                           .int8_quantization = false}},
       ColumnDef{"sp", SparseColumnSpec{.max_dim = 100,
                                        .index = SparseIndexKind::kFlat}},
       ColumnDef{"year", ScalarType::kInt64},
       ColumnDef{"score", ScalarType::kFloat},
       ColumnDef{"title", ScalarType::kString}});
  EXPECT_TRUE(schema.ok()) << schema.status();
  return *std::move(schema);
}

std::vector<RowInput> TestRows() {
  RowInput a;
  a.primary_key = "a";
  a.fields.emplace("emb", Value{std::vector<float>{0.6F, 0.8F, -0.0F}});
  a.fields.emplace("sp", Value{SparseVector{.indices = {2, 5, 99},
                                            .values = {0.2F, -0.5F, 7.0F}}});
  a.fields.emplace("year", Value{std::numeric_limits<int64_t>::min()});
  a.fields.emplace("score", Value{1.0e-38F});
  a.fields.emplace("title", Value{std::string("hello\0world", 11)});
  RowInput b;
  b.primary_key = std::string(256, 'b');
  b.fields.emplace("emb", Value{std::vector<float>{1.0F, 0.0F, 0.0F}});
  b.fields.emplace("sp", Value{SparseVector{}});
  b.fields.emplace("year", Value{int64_t{2024}});
  b.fields.emplace("score", Value{-3.5F});
  b.fields.emplace("title", Value{std::string()});
  return {std::move(a), std::move(b)};
}

void ExpectSchemaEq(const Schema& actual, const Schema& expected) {
  StringByteWriter a;
  StringByteWriter e;
  ASSERT_TRUE(actual.Save(a).ok());
  ASSERT_TRUE(expected.Save(e).ok());
  EXPECT_EQ(actual.table_name(), expected.table_name());
  EXPECT_EQ(a.data(), e.data());
}

// Value 的 == 对 float 按值比较；逐位比较 float 以覆盖 -0.0 与非规格化数。
bool SameBits(const Value& x, const Value& y) {
  if (x.index() != y.index()) {
    return false;
  }
  if (const auto* fx = std::get_if<float>(&x)) {
    return std::bit_cast<uint32_t>(*fx) ==
           std::bit_cast<uint32_t>(std::get<float>(y));
  }
  if (const auto* vx = std::get_if<std::vector<float>>(&x)) {
    const auto& vy = std::get<std::vector<float>>(y);
    if (vx->size() != vy.size()) {
      return false;
    }
    for (size_t i = 0; i < vx->size(); ++i) {
      if (std::bit_cast<uint32_t>((*vx)[i]) != std::bit_cast<uint32_t>(vy[i])) {
        return false;
      }
    }
    return true;
  }
  if (const auto* sx = std::get_if<SparseVector>(&x)) {
    const auto& sy = std::get<SparseVector>(y);
    return sx->indices == sy.indices && sx->values == sy.values;
  }
  if (const auto* ix = std::get_if<int64_t>(&x)) {
    return *ix == std::get<int64_t>(y);
  }
  return std::get<std::string>(x) == std::get<std::string>(y);
}

void ExpectRowsEq(const std::vector<RowInput>& actual,
                  const std::vector<RowInput>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i].primary_key, expected[i].primary_key) << "row " << i;
    ASSERT_EQ(actual[i].fields.size(), expected[i].fields.size())
        << "row " << i;
    for (const auto& [name, value] : expected[i].fields) {
      auto it = actual[i].fields.find(name);
      ASSERT_NE(it, actual[i].fields.end()) << "row " << i << " " << name;
      EXPECT_TRUE(SameBits(it->second, value)) << "row " << i << " " << name;
    }
  }
}

// 对 payload 去掉最后 1 字节后断言 decode 返回 DataLoss。
void ExpectTruncatedIsDataLoss(
    const std::string& payload,
    const std::function<absl::Status(std::string_view)>& decode) {
  ASSERT_FALSE(payload.empty());
  const std::string truncated = payload.substr(0, payload.size() - 1);
  EXPECT_EQ(decode(truncated).code(), absl::StatusCode::kDataLoss);
  // 多出的字节同样视为损坏。
  EXPECT_EQ(decode(payload + "x").code(), absl::StatusCode::kDataLoss);
}

TEST(WalCodecTest, RoundTripEveryRecordType) {
  EXPECT_EQ(static_cast<uint16_t>(WalRecordType::kCreateDatabase), 1);
  EXPECT_EQ(static_cast<uint16_t>(WalRecordType::kDropDatabase), 2);
  EXPECT_EQ(static_cast<uint16_t>(WalRecordType::kCreateTable), 3);
  EXPECT_EQ(static_cast<uint16_t>(WalRecordType::kDropTable), 4);
  EXPECT_EQ(static_cast<uint16_t>(WalRecordType::kInsert), 5);
  EXPECT_EQ(static_cast<uint16_t>(WalRecordType::kUpsert), 6);
  EXPECT_EQ(static_cast<uint16_t>(WalRecordType::kDelete), 7);

  // 1. CreateDatabase
  {
    auto payload = EncodeCreateDatabase("db_1");
    ASSERT_TRUE(payload.ok()) << payload.status();
    auto decoded = DecodeCreateDatabase(*payload);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(decoded->database, "db_1");
    ExpectTruncatedIsDataLoss(*payload, [](std::string_view p) {
      return DecodeCreateDatabase(p).status();
    });
  }
  // 2. DropDatabase
  {
    auto payload = EncodeDropDatabase("db_2");
    ASSERT_TRUE(payload.ok()) << payload.status();
    auto decoded = DecodeDropDatabase(*payload);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(decoded->database, "db_2");
    ExpectTruncatedIsDataLoss(*payload, [](std::string_view p) {
      return DecodeDropDatabase(p).status();
    });
  }
  // 3. CreateTable
  {
    const Schema schema = TestSchema();
    auto payload = EncodeCreateTable("db", schema);
    ASSERT_TRUE(payload.ok()) << payload.status();
    auto decoded = DecodeCreateTable(*payload);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(decoded->database, "db");
    ExpectSchemaEq(decoded->schema, schema);
    ExpectTruncatedIsDataLoss(*payload, [](std::string_view p) {
      return DecodeCreateTable(p).status();
    });
  }
  // 4. DropTable
  {
    auto payload = EncodeDropTable("db", "docs");
    ASSERT_TRUE(payload.ok()) << payload.status();
    auto decoded = DecodeDropTable(*payload);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(decoded->database, "db");
    EXPECT_EQ(decoded->table, "docs");
    ExpectTruncatedIsDataLoss(*payload, [](std::string_view p) {
      return DecodeDropTable(p).status();
    });
  }
  // 5. Insert
  {
    const std::vector<RowInput> rows = TestRows();
    auto payload = EncodeInsert("db", "docs", rows);
    ASSERT_TRUE(payload.ok()) << payload.status();
    auto decoded = DecodeInsert(*payload);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(decoded->database, "db");
    EXPECT_EQ(decoded->table, "docs");
    EXPECT_FALSE(decoded->upsert);
    ExpectRowsEq(decoded->rows, rows);
    ExpectTruncatedIsDataLoss(
        *payload, [](std::string_view p) { return DecodeInsert(p).status(); });
    // Insert 的 payload 不能当作 Upsert 解码。
    EXPECT_EQ(DecodeUpsert(*payload).status().code(),
              absl::StatusCode::kDataLoss);
  }
  // 6. Upsert
  {
    const std::vector<RowInput> rows = TestRows();
    auto payload = EncodeUpsert("db", "docs", rows);
    ASSERT_TRUE(payload.ok()) << payload.status();
    auto decoded = DecodeUpsert(*payload);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(decoded->database, "db");
    EXPECT_EQ(decoded->table, "docs");
    EXPECT_TRUE(decoded->upsert);
    ExpectRowsEq(decoded->rows, rows);
    ExpectTruncatedIsDataLoss(
        *payload, [](std::string_view p) { return DecodeUpsert(p).status(); });
    EXPECT_EQ(DecodeInsert(*payload).status().code(),
              absl::StatusCode::kDataLoss);
  }
  // 7. Delete
  {
    const std::vector<std::string> keys = {"a", std::string(256, 'z'), "c"};
    auto payload = EncodeDelete("db", "docs", keys);
    ASSERT_TRUE(payload.ok()) << payload.status();
    auto decoded = DecodeDelete(*payload);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
    EXPECT_EQ(decoded->database, "db");
    EXPECT_EQ(decoded->table, "docs");
    EXPECT_THAT(decoded->primary_keys,
                ElementsAre("a", std::string(256, 'z'), "c"));
    ExpectTruncatedIsDataLoss(
        *payload, [](std::string_view p) { return DecodeDelete(p).status(); });
  }
  // 空 payload 与超大计数都返回 DataLoss，不尝试分配。
  EXPECT_EQ(DecodeInsert("").status().code(), absl::StatusCode::kDataLoss);
  {
    StringByteWriter w;
    ASSERT_TRUE(w.WriteString("db").ok());
    ASSERT_TRUE(w.WriteString("docs").ok());
    ASSERT_TRUE(w.WriteU32(0xFFFFFFFFU).ok());
    EXPECT_EQ(DecodeDelete(w.data()).status().code(),
              absl::StatusCode::kDataLoss);
  }
}

}  // namespace
}  // namespace hector
