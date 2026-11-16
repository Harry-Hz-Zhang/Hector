#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "hector/common/types.h"
#include "hector/core/engine.h"
#include "hector/core/schema.h"
#include "hector/core/search.h"
#include "hector/core/table.h"
#include "hector/core/value.h"
#include "hector/core/wal_codec.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"
#include "hector/storage/file_wal.h"
#include "hector/storage/wal.h"
#include "tests/core/fake_wal.h"

namespace hector {
namespace {

namespace fs = std::filesystem;

using ::hector::test_support::CallLog;
using ::hector::test_support::FakeSnapshotStore;
using ::hector::test_support::FakeWal;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr uint32_t kDim = 8;

// 列：emb（稠密 dim=kDim COSINE）、year（int64）。
Schema DocsSchema(std::string name = "docs") {
  auto schema = Schema::Create(
      std::move(name),
      {ColumnDef{"emb", DenseColumnSpec{.dim = kDim,
                                        .metric = MetricType::kCosine,
                                        .index = DenseIndexKind::kFlat,
                                        .hnsw = {},
                                        .int8_quantization = false}},
       ColumnDef{"year", ScalarType::kInt64}});
  EXPECT_TRUE(schema.ok()) << schema.status();
  return *std::move(schema);
}

std::vector<float> RandomVector(std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> v(kDim);
  for (float& x : v) {
    x = dist(rng);
  }
  return v;
}

RowInput Row(std::string pk, std::vector<float> emb, int64_t year = 2020) {
  RowInput row;
  row.primary_key = std::move(pk);
  row.fields.emplace("emb", Value{std::move(emb)});
  row.fields.emplace("year", Value{year});
  return row;
}

std::vector<RowInput> Rows(std::vector<RowInput> rows) { return rows; }

SearchRequest DenseQuery(std::vector<float> query, uint32_t top_k) {
  SearchRequest request;
  request.top_k = top_k;
  SearchRoute route;
  route.column = "emb";
  route.query = std::move(query);
  request.routes.push_back(std::move(route));
  request.output_columns = {"year"};
  return request;
}

std::vector<uint32_t> Bits(const std::vector<float>& v) {
  std::vector<uint32_t> bits;
  bits.reserve(v.size());
  for (float f : v) {
    bits.push_back(std::bit_cast<uint32_t>(f));
  }
  return bits;
}

// 读一行的某列；不存在时返回 nullopt。
std::optional<Value> GetField(const Engine& engine, const std::string& pk,
                              const std::string& column) {
  auto table = engine.GetTable("db", "docs");
  if (!table.ok()) {
    return std::nullopt;
  }
  const std::vector<std::string> keys = {pk};
  auto rows = (*table)->Get(keys, {}, nullptr);
  if (!rows.ok() || rows->size() != 1) {
    return std::nullopt;
  }
  for (const auto& [name, value] : (*rows)[0].fields) {
    if (name == column) {
      return value;
    }
  }
  return std::nullopt;
}

class EngineReplayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = fs::path(::testing::TempDir()) /
           absl::StrCat("engine_replay_test_", info->name());
    fs::remove_all(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  std::unique_ptr<Engine> MustOpen() {
    auto engine = Engine::Open(EngineOptions{.data_dir = dir_});
    EXPECT_TRUE(engine.ok()) << engine.status();
    return engine.ok() ? *std::move(engine) : nullptr;
  }

  fs::path dir_;
};

TEST_F(EngineReplayTest, RestartRestoresState) {
  std::mt19937 rng(1);
  Lsn lsn_before = 0;
  {
    std::unique_ptr<Engine> engine = MustOpen();
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->replayed_records(), 0U);
    ASSERT_TRUE(engine->CreateDatabase("db").ok());
    ASSERT_TRUE(engine->CreateTable("db", DocsSchema()).ok());
    ASSERT_TRUE(engine
                    ->Insert("db", "docs",
                             Rows({Row("a", RandomVector(rng), 2020),
                                   Row("b", RandomVector(rng), 2021),
                                   Row("c", RandomVector(rng), 2022)}))
                    .ok());
    ASSERT_TRUE(
        engine->Upsert("db", "docs", Rows({Row("a", RandomVector(rng), 2024)}))
            .ok());
    ASSERT_TRUE(engine->Delete("db", "docs", {"b"}).ok());
    ASSERT_TRUE(engine->CreateTable("db", DocsSchema("tmp")).ok());
    ASSERT_TRUE(engine->DropTable("db", "tmp").ok());
    lsn_before = engine->last_lsn();
    EXPECT_EQ(lsn_before, 7U);
  }

  std::unique_ptr<Engine> engine = MustOpen();
  ASSERT_NE(engine, nullptr);
  EXPECT_EQ(engine->replayed_records(), 7U);
  EXPECT_EQ(engine->last_lsn(), lsn_before);
  EXPECT_THAT(engine->ListDatabases(), ElementsAre("db"));
  auto tables = engine->ListTables("db");
  ASSERT_TRUE(tables.ok()) << tables.status();
  EXPECT_THAT(*tables, ElementsAre("docs"));

  auto table = engine->GetTable("db", "docs");
  ASSERT_TRUE(table.ok()) << table.status();
  const TableStats stats = (*table)->Stats();
  EXPECT_EQ(stats.live_rows, 2U);
  EXPECT_EQ(stats.deleted_rows, 2U);
  const std::optional<Value> a_year = GetField(*engine, "a", "year");
  ASSERT_TRUE(a_year.has_value());
  EXPECT_EQ(std::get<int64_t>(*a_year), 2024);
  const std::optional<Value> c_year = GetField(*engine, "c", "year");
  ASSERT_TRUE(c_year.has_value());
  EXPECT_EQ(std::get<int64_t>(*c_year), 2022);
  EXPECT_FALSE(GetField(*engine, "b", "year").has_value());

  // 重启后可以继续写，LSN 接着分配。
  auto more = engine->Insert("db", "docs", Rows({Row("d", RandomVector(rng))}));
  ASSERT_TRUE(more.ok()) << more.status();
  EXPECT_EQ(more->lsn, lsn_before + 1);
}

TEST_F(EngineReplayTest, SearchAfterRestartMatches) {
  constexpr int kRows = 50;
  std::mt19937 rng(42);
  std::vector<RowInput> rows;
  for (int i = 0; i < kRows; ++i) {
    rows.push_back(Row(absl::StrCat("p", i), RandomVector(rng), 2000 + i));
  }
  const std::vector<float> query = RandomVector(rng);

  absl::StatusOr<SearchResult> before = absl::UnknownError("unset");
  std::vector<std::vector<uint32_t>> vectors_before;
  {
    std::unique_ptr<Engine> engine = MustOpen();
    ASSERT_NE(engine, nullptr);
    ASSERT_TRUE(engine->CreateDatabase("db").ok());
    ASSERT_TRUE(engine->CreateTable("db", DocsSchema()).ok());
    ASSERT_TRUE(engine->Insert("db", "docs", rows).ok());
    auto table = engine->GetTable("db", "docs");
    ASSERT_TRUE(table.ok());
    before = (*table)->Search(DenseQuery(query, 20));
    ASSERT_TRUE(before.ok()) << before.status();
    for (int i = 0; i < kRows; ++i) {
      auto v = GetField(*engine, absl::StrCat("p", i), "emb");
      ASSERT_TRUE(v.has_value());
      vectors_before.push_back(Bits(std::get<std::vector<float>>(*v)));
    }
  }

  std::unique_ptr<Engine> engine = MustOpen();
  ASSERT_NE(engine, nullptr);
  auto table = engine->GetTable("db", "docs");
  ASSERT_TRUE(table.ok()) << table.status();
  auto after = (*table)->Search(DenseQuery(query, 20));
  ASSERT_TRUE(after.ok()) << after.status();

  ASSERT_EQ(after->rows.size(), before->rows.size());
  EXPECT_EQ(after->score_kind, before->score_kind);
  for (size_t i = 0; i < after->rows.size(); ++i) {
    EXPECT_EQ(after->rows[i].primary_key, before->rows[i].primary_key) << i;
    EXPECT_EQ(std::bit_cast<uint32_t>(after->rows[i].score),
              std::bit_cast<uint32_t>(before->rows[i].score))
        << i;
  }
  for (int i = 0; i < kRows; ++i) {
    auto v = GetField(*engine, absl::StrCat("p", i), "emb");
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(Bits(std::get<std::vector<float>>(*v)), vectors_before[i]) << i;
  }
}

TEST_F(EngineReplayTest, InvalidRecordFailsOpen) {
  {
    auto wal = OpenFileWal(dir_ / "wal");
    ASSERT_TRUE(wal.ok()) << wal.status();
    std::mt19937 rng(3);
    auto payload =
        EncodeInsert("db", "docs", Rows({Row("a", RandomVector(rng))}));
    ASSERT_TRUE(payload.ok()) << payload.status();
    auto lsn = (*wal)->Append(5, *payload);
    ASSERT_TRUE(lsn.ok()) << lsn.status();
    ASSERT_EQ(*lsn, 1U);
    ASSERT_TRUE((*wal)->SyncUpTo(*lsn).ok());
  }
  auto engine = Engine::Open(EngineOptions{.data_dir = dir_});
  EXPECT_EQ(engine.status().code(), absl::StatusCode::kDataLoss);
  EXPECT_THAT(engine.status().message(), HasSubstr("lsn 1"));

  // 无法解码的 payload 同样 DataLoss，消息含该 LSN。
  fs::remove_all(dir_);
  {
    auto wal = OpenFileWal(dir_ / "wal");
    ASSERT_TRUE(wal.ok()) << wal.status();
    auto db = EncodeCreateDatabase("db");
    ASSERT_TRUE(db.ok());
    ASSERT_TRUE((*wal)->Append(1, *db).ok());
    ASSERT_TRUE((*wal)->Append(3, "garbage").ok());
    ASSERT_TRUE((*wal)->Append(99, "").ok());
    ASSERT_TRUE((*wal)->SyncUpTo(3).ok());
  }
  engine = Engine::Open(EngineOptions{.data_dir = dir_});
  EXPECT_EQ(engine.status().code(), absl::StatusCode::kDataLoss);
  EXPECT_THAT(engine.status().message(), HasSubstr("lsn 2"));

  // 删除不存在主键的 Delete 记录：在线写入不会产生这种记录，回放时出现说明
  // WAL 与内存不一致，同样 DataLoss，消息含该 LSN。
  fs::remove_all(dir_);
  {
    auto wal = OpenFileWal(dir_ / "wal");
    ASSERT_TRUE(wal.ok()) << wal.status();
    auto db = EncodeCreateDatabase("db");
    auto schema = EncodeCreateTable("db", DocsSchema());
    const std::vector<std::string> keys = {"missing"};
    auto del = EncodeDelete("db", "docs", keys);
    ASSERT_TRUE(db.ok() && schema.ok() && del.ok());
    ASSERT_TRUE((*wal)->Append(1, *db).ok());
    ASSERT_TRUE((*wal)->Append(3, *schema).ok());
    ASSERT_TRUE((*wal)->Append(7, *del).ok());
    ASSERT_TRUE((*wal)->SyncUpTo(3).ok());
  }
  engine = Engine::Open(EngineOptions{.data_dir = dir_});
  EXPECT_EQ(engine.status().code(), absl::StatusCode::kDataLoss);
  EXPECT_THAT(engine.status().message(), HasSubstr("lsn 3"));
}

TEST_F(EngineReplayTest, SecondOpenOnSameDirIsRejected) {
  std::unique_ptr<Engine> a = MustOpen();
  ASSERT_NE(a, nullptr);
  EXPECT_TRUE(fs::exists(dir_ / "LOCK"));
  auto b = Engine::Open(EngineOptions{.data_dir = dir_});
  EXPECT_EQ(b.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(b.status().message(),
              HasSubstr("data_dir is locked by another process"));
  a.reset();
  auto c = Engine::Open(EngineOptions{.data_dir = dir_});
  EXPECT_TRUE(c.ok()) << c.status();
}

// ASSERT：回放走 Snapshot::Latest 之后的 Replay(0)，不调用 Append 与
// SyncUpTo，并以 kReplay 准备行（未归一化的向量原样读回，说明没有再归一化）。
TEST_F(EngineReplayTest, ReplaySkipsWalWritesAndNormalization) {
  // 建库、建表，再手工追加一条 COSINE 列向量未归一化的 Insert 记录。
  std::vector<float> raw(kDim, 0.0F);
  raw[0] = 3.0F;
  raw[1] = 4.0F;
  auto db = EncodeCreateDatabase("db");
  auto schema = EncodeCreateTable("db", DocsSchema());
  auto insert = EncodeInsert("db", "docs", Rows({Row("raw", raw)}));
  ASSERT_TRUE(db.ok() && schema.ok() && insert.ok());
  const std::vector<WalRecord> records = {
      WalRecord{.lsn = 1, .type = 1, .payload = *db},
      WalRecord{.lsn = 2, .type = 3, .payload = *schema},
      WalRecord{.lsn = 3, .type = 5, .payload = *insert}};

  auto log = std::make_shared<CallLog>();
  auto wal = std::make_unique<FakeWal>(log);
  wal->Preload(records);
  FakeWal* fake = wal.get();
  auto engine = Engine::OpenWith(std::move(wal),
                                 std::make_unique<FakeSnapshotStore>(log));
  ASSERT_TRUE(engine.ok()) << engine.status();
  EXPECT_EQ(fake->append_calls(), 0);
  EXPECT_EQ(fake->sync_calls(), 0);
  EXPECT_THAT(log->events(), ElementsAre("Latest", "Replay(0)"));
  EXPECT_EQ((*engine)->replayed_records(), 3U);
  EXPECT_EQ((*engine)->last_lsn(), 3U);

  auto value = GetField(**engine, "raw", "emb");
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(Bits(std::get<std::vector<float>>(*value)), Bits(raw));
  auto stats = (*engine)->GetTable("db", "docs");
  ASSERT_TRUE(stats.ok());
  EXPECT_EQ((*stats)->Stats().live_rows, 1U);
  EXPECT_THAT((*engine)->ListDatabases(), ElementsAre("db"));
}

}  // namespace
}  // namespace hector
