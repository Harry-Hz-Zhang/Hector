#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/types.h"
#include "hector/core/schema.h"
#include "hector/core/table.h"
#include "hector/core/value.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"
#include "tests/core/counting_index.h"

namespace hector {
namespace {

using ::hector::test_support::CountingIndexFactory;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

// 列顺序：emb（稠密 dim=2 COSINE）、sp（稀疏 max_dim=10）、year（int64）、
// score（float）、title（string）。
Schema DocsSchema(DenseIndexKind dense_index = DenseIndexKind::kFlat,
                  SparseIndexKind sparse_index = SparseIndexKind::kFlat,
                  bool int8 = false) {
  auto schema = Schema::Create(
      "docs",
      {ColumnDef{"emb", DenseColumnSpec{.dim = 2,
                                        .metric = MetricType::kCosine,
                                        .index = dense_index,
                                        .hnsw = {},
                                        .int8_quantization = int8}},
       ColumnDef{"sp", SparseColumnSpec{.max_dim = 10, .index = sparse_index}},
       ColumnDef{"year", ScalarType::kInt64},
       ColumnDef{"score", ScalarType::kFloat},
       ColumnDef{"title", ScalarType::kString}});
  EXPECT_TRUE(schema.ok()) << schema.status();
  return *std::move(schema);
}

RowInput Row(std::string pk, std::vector<float> emb = {1.0F, 0.0F},
             SparseVector sp = {}, int64_t year = 2020, float score = 0.5F,
             std::string title = "t") {
  RowInput row;
  row.primary_key = std::move(pk);
  row.fields.emplace("emb", Value{std::move(emb)});
  row.fields.emplace("sp", Value{std::move(sp)});
  row.fields.emplace("year", Value{year});
  row.fields.emplace("score", Value{score});
  row.fields.emplace("title", Value{std::move(title)});
  return row;
}

const Value* Field(const RowOutput& row, const std::string& name) {
  for (const auto& [n, v] : row.fields) {
    if (n == name) {
      return &v;
    }
  }
  return nullptr;
}

std::vector<std::string> FieldNames(const RowOutput& row) {
  std::vector<std::string> names;
  names.reserve(row.fields.size());
  for (const auto& [n, v] : row.fields) {
    names.push_back(n);
  }
  return names;
}

class TableWriteTest : public ::testing::Test {
 protected:
  void SetUp() override { table_ = MakeTable({}); }

  std::unique_ptr<Table> MakeTable(const TableOptions& options) {
    auto table = Table::Create(DocsSchema(), counting_.factory(), options);
    EXPECT_TRUE(table.ok()) << table.status();
    return table.ok() ? *std::move(table) : nullptr;
  }

  // 依次 Prepare（kOnline）、Apply、Publish。
  absl::Status Write(Table& table, std::vector<RowInput> rows,
                     bool upsert = false) {
    auto batch =
        table.PrepareInsert(std::move(rows), upsert, PrepareMode::kOnline);
    if (!batch.ok()) {
      return batch.status();
    }
    const Lsn lsn = ++lsn_;
    if (absl::Status s = table.ApplyInsert(*std::move(batch), lsn); !s.ok()) {
      return s;
    }
    table.Publish(lsn);
    return absl::OkStatus();
  }

  absl::Status Delete(Table& table, std::vector<std::string> pks) {
    auto ids = table.PrepareDelete(pks);
    if (!ids.ok()) {
      return ids.status();
    }
    const Lsn lsn = ++lsn_;
    if (absl::Status s = table.ApplyDelete(*ids, lsn); !s.ok()) {
      return s;
    }
    table.Publish(lsn);
    return absl::OkStatus();
  }

  std::optional<RowOutput> GetOne(const Table& table, const std::string& pk) {
    std::vector<std::string> missing;
    const std::vector<std::string> pks = {pk};
    auto rows = table.Get(pks, {}, &missing);
    EXPECT_TRUE(rows.ok()) << rows.status();
    if (!rows.ok() || rows->empty()) {
      return std::nullopt;
    }
    return (*rows)[0];
  }

  CountingIndexFactory counting_;
  std::unique_ptr<Table> table_;
  Lsn lsn_ = 0;
};

TEST_F(TableWriteTest, InsertThenGetReturnsNormalizedRow) {
  ASSERT_EQ(counting_.dense_count(), 1U);
  ASSERT_EQ(counting_.sparse_count(), 1U);
  ASSERT_TRUE(Write(*table_, {Row("p", {3.0F, 4.0F},
                                  SparseVector{{7, 2, 5}, {0.7F, 0.2F, 0.5F}},
                                  2021, 1.5F, "hello"),
                              Row("q"), Row("r")})
                  .ok());

  // ApplyInsert 对每个向量列的索引 Add 调用次数等于批内行数，Build 0 次。
  EXPECT_EQ(counting_.dense(0).add_calls(), 3);
  EXPECT_EQ(counting_.sparse(0).add_calls(), 3);
  EXPECT_THAT(counting_.dense(0).added_row_ids(), ElementsAre(0, 1, 2));
  EXPECT_EQ(counting_.total_build_calls(), 0);

  std::vector<std::string> missing;
  const std::vector<std::string> pks = {"p"};
  auto rows = table_->Get(pks, {}, &missing);
  ASSERT_TRUE(rows.ok()) << rows.status();
  EXPECT_THAT(missing, IsEmpty());
  ASSERT_EQ(rows->size(), 1U);
  const RowOutput& row = (*rows)[0];
  EXPECT_EQ(row.primary_key, "p");
  // 空 output_columns 返回全部列，按 Schema 顺序。
  EXPECT_THAT(FieldNames(row),
              ElementsAre("emb", "sp", "year", "score", "title"));
  const auto& emb = std::get<std::vector<float>>(*Field(row, "emb"));
  ASSERT_EQ(emb.size(), 2U);
  EXPECT_NEAR(emb[0], 0.6F, 1e-6);
  EXPECT_NEAR(emb[1], 0.8F, 1e-6);
  const auto& sp = std::get<SparseVector>(*Field(row, "sp"));
  EXPECT_THAT(sp.indices, ElementsAre(2, 5, 7));
  EXPECT_THAT(sp.values, ElementsAre(0.2F, 0.5F, 0.7F));
  EXPECT_EQ(std::get<int64_t>(*Field(row, "year")), 2021);
  EXPECT_EQ(std::get<float>(*Field(row, "score")), 1.5F);
  EXPECT_EQ(std::get<std::string>(*Field(row, "title")), "hello");

  // 指定 output_columns 时仍按 Schema 顺序返回，而不是请求顺序。
  const std::vector<std::string> cols = {"year", "emb"};
  const std::vector<std::string> two = {"r", "p"};
  auto projected = table_->Get(two, cols, &missing);
  ASSERT_TRUE(projected.ok()) << projected.status();
  ASSERT_EQ(projected->size(), 2U);
  EXPECT_EQ((*projected)[0].primary_key, "r");
  EXPECT_EQ((*projected)[1].primary_key, "p");
  EXPECT_THAT(FieldNames((*projected)[1]), ElementsAre("emb", "year"));

  // 不存在的列与空主键列表返回 InvalidArgument。
  const std::vector<std::string> bad_cols = {"nope"};
  EXPECT_EQ(table_->Get(pks, bad_cols, &missing).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(table_->Get({}, {}, &missing).status().code(),
            absl::StatusCode::kInvalidArgument);
  const std::vector<std::string> too_many(100001, "p");
  EXPECT_EQ(table_->Get(too_many, {}, &missing).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(TableWriteTest, ApplyWithoutPublishIsInvisible) {
  auto batch = table_->PrepareInsert({Row("a")}, false, PrepareMode::kOnline);
  ASSERT_TRUE(batch.ok()) << batch.status();
  ASSERT_TRUE(table_->ApplyInsert(*std::move(batch), 1).ok());

  std::vector<std::string> missing;
  const std::vector<std::string> pks = {"a"};
  auto rows = table_->Get(pks, {}, &missing);
  ASSERT_TRUE(rows.ok()) << rows.status();
  EXPECT_THAT(*rows, IsEmpty());
  EXPECT_THAT(missing, ElementsAre("a"));
  EXPECT_EQ(table_->Stats().live_rows, 0U);

  table_->Publish(1);
  missing.clear();
  rows = table_->Get(pks, {}, &missing);
  ASSERT_TRUE(rows.ok()) << rows.status();
  ASSERT_EQ(rows->size(), 1U);
  EXPECT_EQ((*rows)[0].primary_key, "a");
  EXPECT_THAT(missing, IsEmpty());
  EXPECT_EQ(table_->Stats().live_rows, 1U);
}

TEST_F(TableWriteTest, ReplayModeSkipsNormalization) {
  auto online = table_->PrepareInsert(
      {Row("a", {3.0F, 4.0F}, SparseVector{{7, 2}, {0.7F, 0.2F}})}, false,
      PrepareMode::kOnline);
  ASSERT_TRUE(online.ok()) << online.status();
  const auto online_emb =
      std::get<std::vector<float>>(online->rows[0].fields.at("emb"));

  auto replay =
      table_->PrepareInsert(online->rows, false, PrepareMode::kReplay);
  ASSERT_TRUE(replay.ok()) << replay.status();
  const auto replay_emb =
      std::get<std::vector<float>>(replay->rows[0].fields.at("emb"));
  ASSERT_EQ(replay_emb.size(), online_emb.size());
  for (size_t i = 0; i < online_emb.size(); ++i) {
    EXPECT_EQ(std::bit_cast<uint32_t>(replay_emb[i]),
              std::bit_cast<uint32_t>(online_emb[i]))
        << i;
  }
  EXPECT_THAT(std::get<SparseVector>(replay->rows[0].fields.at("sp")).indices,
              ElementsAre(2, 7));

  // kReplay 不做归一化：未归一化的输入原样保留。
  auto raw = table_->PrepareInsert({Row("b", {3.0F, 4.0F})}, false,
                                   PrepareMode::kReplay);
  ASSERT_TRUE(raw.ok()) << raw.status();
  EXPECT_THAT(std::get<std::vector<float>>(raw->rows[0].fields.at("emb")),
              ElementsAre(3.0F, 4.0F));

  // kReplay 照常校验：维度不符仍拒绝。
  EXPECT_EQ(table_
                ->PrepareInsert({Row("c", {1.0F, 0.0F, 0.0F})}, false,
                                PrepareMode::kReplay)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(TableWriteTest, InsertRejectsWholeBatchOnAnyBadRow) {
  auto batch =
      table_->PrepareInsert({Row("a"), Row("b"), Row("c", {1.0F, 2.0F, 3.0F})},
                            false, PrepareMode::kOnline);
  ASSERT_FALSE(batch.ok());
  EXPECT_EQ(batch.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(batch.status().message(), HasSubstr("rows[2]"));

  EXPECT_EQ(table_->Stats().live_rows, 0U);
  EXPECT_EQ(counting_.total_add_calls(), 0);

  // 列存储没有被写入：下一行仍拿到 row id 0。
  ASSERT_TRUE(Write(*table_, {Row("d")}).ok());
  const std::vector<std::string> pks = {"d"};
  auto ids = table_->PrepareDelete(pks);
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_THAT(*ids, ElementsAre(0));
}

TEST_F(TableWriteTest, InsertRejectsDuplicatePrimaryKeys) {
  ASSERT_TRUE(Write(*table_, {Row("a")}).ok());
  EXPECT_EQ(Write(*table_, {Row("a")}).code(),
            absl::StatusCode::kAlreadyExists);

  absl::Status dup = Write(*table_, {Row("b"), Row("b")});
  EXPECT_EQ(dup.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(dup.message(), HasSubstr("rows[1]"));
  // Upsert 的同批重复也是 InvalidArgument。
  EXPECT_EQ(Write(*table_, {Row("b"), Row("b")}, /*upsert=*/true).code(),
            absl::StatusCode::kInvalidArgument);

  EXPECT_EQ(Write(*table_, {Row("")}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Write(*table_, {Row(std::string(257, 'k'))}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(Write(*table_, {Row(std::string(256, 'k'))}).ok());
  EXPECT_EQ(table_->Stats().live_rows, 2U);
}

TEST_F(TableWriteTest, RejectsMissingExtraAndNonFiniteFields) {
  struct Case {
    std::string label;
    std::vector<RowInput> rows;
    std::string expected_substring;
  };
  std::vector<Case> cases;

  RowInput missing_year = Row("a");
  missing_year.fields.erase("year");
  cases.push_back({"missing year", {missing_year}, "year"});

  RowInput extra = Row("a");
  extra.fields.emplace("foo", Value{int64_t{1}});
  cases.push_back({"extra foo", {extra}, "foo"});

  RowInput wrong_type = Row("a");
  wrong_type.fields["year"] = Value{std::string("2020")};
  cases.push_back({"year as string", {wrong_type}, "year"});

  cases.push_back({"float NaN",
                   {Row("a", {1.0F, 0.0F}, {}, 2020,
                        std::numeric_limits<float>::quiet_NaN())},
                   "score"});
  cases.push_back({"dense inf",
                   {Row("a", {std::numeric_limits<float>::infinity(), 0.0F})},
                   "emb"});
  cases.push_back({"sparse index == max_dim",
                   {Row("a", {1.0F, 0.0F}, {{10}, {1.0F}})},
                   "sp"});
  cases.push_back({"sparse duplicate index",
                   {Row("a", {1.0F, 0.0F}, {{3, 1, 3}, {1.0F, 1.0F, 1.0F}})},
                   "sp"});
  cases.push_back({"sparse length mismatch",
                   {Row("a", {1.0F, 0.0F}, {{1, 2}, {1.0F}})},
                   "sp"});
  cases.push_back({"sparse NaN value",
                   {Row("a", {1.0F, 0.0F},
                        {{1}, {std::numeric_limits<float>::quiet_NaN()}})},
                   "sp"});
  cases.push_back({"cosine zero vector", {Row("a", {0.0F, 0.0F})}, "emb"});
  cases.push_back(
      {"string too long",
       {Row("a", {1.0F, 0.0F}, {}, 2020, 0.5F, std::string(65537, 'x'))},
       "title"});
  cases.push_back({"zero rows", {}, "rows"});

  std::vector<RowInput> too_many;
  too_many.reserve(100001);
  for (int i = 0; i < 100001; ++i) {
    too_many.push_back(Row("k" + std::to_string(i)));
  }
  cases.push_back({"100001 rows", std::move(too_many), "100000"});

  for (Case& c : cases) {
    SCOPED_TRACE(c.label);
    auto batch =
        table_->PrepareInsert(std::move(c.rows), false, PrepareMode::kOnline);
    ASSERT_FALSE(batch.ok());
    EXPECT_EQ(batch.status().code(), absl::StatusCode::kInvalidArgument)
        << batch.status();
    EXPECT_THAT(batch.status().message(), HasSubstr(c.expected_substring));
  }

  // 边界值可以通过：string 恰好 65536 字节、稀疏空向量。
  EXPECT_TRUE(Write(*table_, {Row("ok", {1.0F, 0.0F}, {}, 2020, 0.5F,
                                  std::string(65536, 'x'))})
                  .ok());
  EXPECT_EQ(counting_.total_add_calls(), 2);
}

TEST_F(TableWriteTest, UpsertTombstonesOldRow) {
  ASSERT_TRUE(Write(*table_, {Row("a", {1.0F, 0.0F}, {}, 2020)}).ok());
  EXPECT_EQ(table_->Stats().deleted_rows, 0U);

  auto batch = table_->PrepareInsert(
      {Row("a", {0.0F, 1.0F}, {}, 2024), Row("b")}, true, PrepareMode::kOnline);
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_TRUE(batch->upsert);
  EXPECT_THAT(batch->tombstoned_row_ids, ElementsAre(0));
  ASSERT_TRUE(table_->ApplyInsert(*std::move(batch), ++lsn_).ok());
  table_->Publish(lsn_);

  auto row = GetOne(*table_, "a");
  ASSERT_TRUE(row.has_value());
  EXPECT_EQ(std::get<int64_t>(*Field(*row, "year")), 2024);

  TableStats stats = table_->Stats();
  EXPECT_EQ(stats.live_rows, 2U);
  // 墓碑数的增加量等于被覆盖的行数（1）。
  EXPECT_EQ(stats.deleted_rows, 1U);
  EXPECT_EQ(counting_.dense(0).add_calls(), 3);

  // 只有一个主键时与规格场景一致：live_rows=1、deleted_rows=1、Add 2 次。
  CountingIndexFactory fresh;
  auto table = Table::Create(DocsSchema(), fresh.factory());
  ASSERT_TRUE(table.ok()) << table.status();
  ASSERT_TRUE(Write(**table, {Row("a", {1.0F, 0.0F}, {}, 2020)}).ok());
  ASSERT_TRUE(
      Write(**table, {Row("a", {1.0F, 0.0F}, {}, 2024)}, /*upsert=*/true).ok());
  auto upserted = GetOne(**table, "a");
  ASSERT_TRUE(upserted.has_value());
  EXPECT_EQ(std::get<int64_t>(*Field(*upserted, "year")), 2024);
  stats = (*table)->Stats();
  EXPECT_EQ(stats.live_rows, 1U);
  EXPECT_EQ(stats.deleted_rows, 1U);
  EXPECT_EQ(fresh.dense(0).add_calls(), 2);
  EXPECT_EQ(fresh.total_build_calls(), 0);

  // 统计按 Schema 列顺序列出每一列。
  ASSERT_EQ(stats.columns.size(), 5U);
  EXPECT_EQ(stats.columns[0].name, "emb");
  EXPECT_EQ(std::get<DenseIndexKind>(stats.columns[0].index),
            DenseIndexKind::kFlat);
  EXPECT_EQ(stats.columns[0].indexed_rows, 2U);
  EXPECT_GT(stats.columns[0].memory_bytes, 0U);
  EXPECT_EQ(std::get<SparseIndexKind>(stats.columns[1].index),
            SparseIndexKind::kFlat);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(stats.columns[2].index));
  EXPECT_EQ(stats.columns[4].name, "title");
}

TEST_F(TableWriteTest, DeleteSkipsUnknownAndAllowsReinsert) {
  ASSERT_TRUE(Write(*table_, {Row("a"), Row("b")}).ok());

  const std::vector<std::string> pks = {"a", "zzz", "a"};
  auto ids = table_->PrepareDelete(pks);
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_THAT(*ids, ElementsAre(0));

  ASSERT_TRUE(table_->ApplyDelete(*ids, ++lsn_).ok());
  // 墓碑在 Publish 之前不生效。
  EXPECT_EQ(table_->Stats().deleted_rows, 0U);
  table_->Publish(lsn_);

  std::vector<std::string> missing;
  const std::vector<std::string> get_pks = {"a", "b"};
  auto rows = table_->Get(get_pks, {}, &missing);
  ASSERT_TRUE(rows.ok()) << rows.status();
  ASSERT_EQ(rows->size(), 1U);
  EXPECT_EQ((*rows)[0].primary_key, "b");
  EXPECT_THAT(missing, ElementsAre("a"));
  EXPECT_EQ(table_->Stats().live_rows, 1U);
  EXPECT_EQ(table_->Stats().deleted_rows, 1U);

  EXPECT_TRUE(Write(*table_, {Row("a", {0.0F, 1.0F}, {}, 2030)}).ok());
  auto again = GetOne(*table_, "a");
  ASSERT_TRUE(again.has_value());
  EXPECT_EQ(std::get<int64_t>(*Field(*again, "year")), 2030);

  EXPECT_EQ(table_->PrepareDelete({}).status().code(),
            absl::StatusCode::kInvalidArgument);
  const std::vector<std::string> too_many(100001, "a");
  EXPECT_EQ(table_->PrepareDelete(too_many).status().code(),
            absl::StatusCode::kInvalidArgument);
  const std::vector<std::string> unknown = {"zzz"};
  auto none = table_->PrepareDelete(unknown);
  ASSERT_TRUE(none.ok()) << none.status();
  EXPECT_THAT(*none, IsEmpty());
}

TEST_F(TableWriteTest, PrepareRejectsBatchBeyondRowCapacity) {
  CountingIndexFactory counting;
  auto created = Table::Create(DocsSchema(), counting.factory(),
                               TableOptions{.max_rows = 4});
  ASSERT_TRUE(created.ok()) << created.status();
  Table& table = **created;
  ASSERT_TRUE(Write(table, {Row("a"), Row("b"), Row("c")}).ok());
  ASSERT_EQ(counting.dense(0).add_calls(), 3);

  auto batch =
      table.PrepareInsert({Row("d"), Row("e")}, false, PrepareMode::kOnline);
  EXPECT_EQ(batch.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(counting.dense(0).add_calls(), 3);
  EXPECT_EQ(counting.sparse(0).add_calls(), 3);

  EXPECT_TRUE(Write(table, {Row("d")}).ok());
  // 已写入的行（含墓碑行）计入上限：删除后仍然是满的。
  ASSERT_TRUE(Delete(table, {"a"}).ok());
  EXPECT_EQ(Write(table, {Row("e")}).code(),
            absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(table.Stats().live_rows, 3U);
}

TEST_F(TableWriteTest, CreateTableWithUnimplementedIndexFails) {
  EXPECT_EQ(Table::Create(DocsSchema(DenseIndexKind::kHnsw)).status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ(Table::Create(
                DocsSchema(DenseIndexKind::kFlat, SparseIndexKind::kInverted))
                .status()
                .code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ(Table::Create(DocsSchema(DenseIndexKind::kFlat,
                                     SparseIndexKind::kFlat, /*int8=*/true))
                .status()
                .code(),
            absl::StatusCode::kUnimplemented);

  StringByteWriter out;
  EXPECT_EQ(table_->Save(out).code(), absl::StatusCode::kUnimplemented);
  StringByteReader in(out.data());
  EXPECT_EQ(Table::Load(in).status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace hector
