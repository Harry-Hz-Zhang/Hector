#include "hector/core/engine.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "hector/common/byte_io.h"
#include "hector/common/row_filter.h"
#include "hector/common/types.h"
#include "hector/common/vector_accessor.h"
#include "hector/core/schema.h"
#include "hector/core/search.h"
#include "hector/core/table.h"
#include "hector/core/value.h"
#include "hector/core/wal_codec.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"
#include "hector/index/vector_index.h"
#include "hector/storage/snapshot.h"
#include "tests/core/fake_wal.h"

namespace hector {
namespace {

using ::hector::test_support::FakeWal;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

constexpr uint16_t kTypeCreateDatabase = 1;
constexpr uint16_t kTypeCreateTable = 3;
constexpr uint16_t kTypeInsert = 5;
constexpr uint16_t kTypeUpsert = 6;
constexpr uint16_t kTypeDelete = 7;

// 列：emb（稠密 dim=2 COSINE）、year（int64）。
Schema DocsSchema(std::string name = "docs",
                  DenseIndexKind index = DenseIndexKind::kFlat) {
  auto schema = Schema::Create(
      std::move(name),
      {ColumnDef{"emb", DenseColumnSpec{.dim = 2,
                                        .metric = MetricType::kCosine,
                                        .index = index,
                                        .hnsw = {},
                                        .int8_quantization = false}},
       ColumnDef{"year", ScalarType::kInt64}});
  EXPECT_TRUE(schema.ok()) << schema.status();
  return *std::move(schema);
}

RowInput Row(std::string pk, int64_t year = 2020,
             std::vector<float> emb = {1.0F, 0.0F}) {
  RowInput row;
  row.primary_key = std::move(pk);
  row.fields.emplace("emb", Value{std::move(emb)});
  row.fields.emplace("year", Value{year});
  return row;
}

std::vector<RowInput> Rows(std::vector<RowInput> rows) { return rows; }

SearchRequest DenseQuery(std::vector<float> query, uint32_t top_k = 10) {
  SearchRequest request;
  request.top_k = top_k;
  SearchRoute route;
  route.column = "emb";
  route.query = std::move(query);
  request.routes.push_back(std::move(route));
  return request;
}

// Add 总是返回 Internal 的稠密索引，其余调用转发给真实索引。
class FailingAddIndex final : public VectorIndex {
 public:
  explicit FailingAddIndex(std::unique_ptr<VectorIndex> inner)
      : inner_(std::move(inner)) {}
  absl::Status Build(std::span<const RowId> row_ids) override {
    return inner_->Build(row_ids);
  }
  absl::Status Add(RowId /*row_id*/) override {
    return absl::InternalError("injected Add failure");
  }
  absl::StatusOr<std::vector<Hit>> Search(
      std::span<const float> query, uint32_t top_k, const SearchParams& params,
      const RowFilter& filter) const override {
    return inner_->Search(query, top_k, params, filter);
  }
  absl::StatusOr<std::vector<Hit>> SearchCandidates(
      std::span<const float> query, uint32_t top_k,
      std::span<const RowId> candidates) const override {
    return inner_->SearchCandidates(query, top_k, candidates);
  }
  size_t Size() const override { return inner_->Size(); }
  size_t MemoryBytes() const override { return inner_->MemoryBytes(); }
  absl::Status Save(ByteWriter& out) const override {
    return inner_->Save(out);
  }
  absl::Status Load(ByteReader& in) override { return inner_->Load(in); }

 private:
  std::unique_ptr<VectorIndex> inner_;
};

class EngineTest : public ::testing::Test {
 protected:
  void SetUp() override { engine_ = OpenEngine(DefaultIndexFactory()); }

  std::unique_ptr<Engine> OpenEngine(const IndexFactory& factory) {
    auto wal = std::make_unique<FakeWal>();
    wal_ = wal.get();
    auto engine =
        Engine::OpenWith(std::move(wal), MakeNullSnapshotStore(), factory);
    EXPECT_TRUE(engine.ok()) << engine.status();
    return engine.ok() ? *std::move(engine) : nullptr;
  }

  // 建库 db 与表 docs，返回表。
  std::shared_ptr<const Table> CreateDocs() {
    EXPECT_TRUE(engine_->CreateDatabase("db").ok());
    EXPECT_TRUE(engine_->CreateTable("db", DocsSchema()).ok());
    auto table = engine_->GetTable("db", "docs");
    EXPECT_TRUE(table.ok()) << table.status();
    return table.ok() ? *table : nullptr;
  }

  FakeWal* wal_ = nullptr;
  std::unique_ptr<Engine> engine_;
};

TEST_F(EngineTest, CatalogLifecycle) {
  ASSERT_TRUE(engine_->CreateDatabase("b").ok());
  ASSERT_TRUE(engine_->CreateDatabase("a").ok());
  EXPECT_THAT(engine_->ListDatabases(), ElementsAre("a", "b"));
  EXPECT_EQ(engine_->CreateDatabase("a").code(),
            absl::StatusCode::kAlreadyExists);
  EXPECT_EQ(engine_->CreateDatabase("1bad").code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(engine_->CreateDatabase(std::string(65, 'x')).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(engine_->CreateDatabase(std::string(64, 'x')).ok());
  EXPECT_EQ(engine_->CreateTable("nope", DocsSchema()).code(),
            absl::StatusCode::kNotFound);

  ASSERT_TRUE(engine_->CreateTable("a", DocsSchema("zeta")).ok());
  ASSERT_TRUE(engine_->CreateTable("a", DocsSchema("alpha")).ok());
  EXPECT_EQ(engine_->CreateTable("a", DocsSchema("alpha")).code(),
            absl::StatusCode::kAlreadyExists);
  auto tables = engine_->ListTables("a");
  ASSERT_TRUE(tables.ok()) << tables.status();
  EXPECT_THAT(*tables, ElementsAre("alpha", "zeta"));
  EXPECT_EQ(engine_->ListTables("nope").status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(engine_->GetTable("a", "nope").status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(engine_->DropTable("a", "nope").code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(engine_->DropDatabase("nope").code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(engine_->Insert("a", "nope", Rows({Row("k")})).status().code(),
            absl::StatusCode::kNotFound);

  // DropTable 期间持有 shared_ptr<const Table> 的检索仍然返回 OK。
  ASSERT_TRUE(engine_->Insert("a", "zeta", Rows({Row("k")})).ok());
  auto held = engine_->GetTable("a", "zeta");
  ASSERT_TRUE(held.ok()) << held.status();
  ASSERT_TRUE(engine_->DropTable("a", "zeta").ok());
  EXPECT_EQ(engine_->GetTable("a", "zeta").status().code(),
            absl::StatusCode::kNotFound);
  auto result = (*held)->Search(DenseQuery({1.0F, 0.0F}));
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->rows.size(), 1U);
  EXPECT_EQ(result->rows[0].primary_key, "k");

  // 删库连带删表。
  ASSERT_TRUE(engine_->DropDatabase("a").ok());
  EXPECT_THAT(engine_->ListDatabases(), ElementsAre("b", std::string(64, 'x')));
  EXPECT_EQ(engine_->GetTable("a", "alpha").status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(engine_->ListTables("a").status().code(),
            absl::StatusCode::kNotFound);
  // 同名库重建后是空库。
  ASSERT_TRUE(engine_->CreateDatabase("a").ok());
  auto empty = engine_->ListTables("a");
  ASSERT_TRUE(empty.ok());
  EXPECT_THAT(*empty, IsEmpty());
}

TEST_F(EngineTest, WriteOrderIsAppendSyncApplyPublish) {
  ASSERT_TRUE(engine_->CreateDatabase("db").ok());
  EXPECT_EQ(wal_->append_calls(), 1);
  EXPECT_EQ(wal_->sync_calls(), 1);
  ASSERT_TRUE(engine_->CreateTable("db", DocsSchema()).ok());
  EXPECT_EQ(wal_->append_calls(), 2);
  EXPECT_EQ(wal_->sync_calls(), 2);
  auto table = engine_->GetTable("db", "docs");
  ASSERT_TRUE(table.ok());

  // 在 Append 与 SyncUpTo 里读取表的 live_rows。
  std::vector<uint64_t> seen_in_append;
  std::vector<uint64_t> seen_in_sync;
  const std::shared_ptr<const Table> t = *table;
  wal_->set_on_append(
      [&](uint16_t) { seen_in_append.push_back(t->Stats().live_rows); });
  wal_->set_on_sync([&](Lsn) { seen_in_sync.push_back(t->Stats().live_rows); });

  auto inserted = engine_->Insert("db", "docs", Rows({Row("a"), Row("b")}));
  ASSERT_TRUE(inserted.ok()) << inserted.status();
  EXPECT_EQ(inserted->affected, 2U);
  EXPECT_EQ(t->Stats().live_rows, 2U);
  EXPECT_EQ(wal_->append_calls(), 3);
  EXPECT_EQ(wal_->sync_calls(), 3);

  auto upserted = engine_->Upsert("db", "docs", Rows({Row("a", 2024)}));
  ASSERT_TRUE(upserted.ok()) << upserted.status();
  EXPECT_EQ(upserted->affected, 1U);
  EXPECT_EQ(t->Stats().live_rows, 2U);
  EXPECT_EQ(t->Stats().deleted_rows, 1U);
  EXPECT_EQ(wal_->append_calls(), 4);
  EXPECT_EQ(wal_->sync_calls(), 4);

  auto deleted = engine_->Delete("db", "docs", {"b", "zzz"});
  ASSERT_TRUE(deleted.ok()) << deleted.status();
  EXPECT_EQ(deleted->affected, 1U);
  EXPECT_EQ(t->Stats().live_rows, 1U);
  EXPECT_EQ(wal_->append_calls(), 5);
  EXPECT_EQ(wal_->sync_calls(), 5);

  // Append 与 SyncUpTo 时内存尚未修改：读到的都是写入前的值。
  EXPECT_THAT(seen_in_append, ElementsAre(0U, 2U, 2U));
  EXPECT_THAT(seen_in_sync, ElementsAre(0U, 2U, 2U));

  const std::vector<test_support::AppendCall> appends = wal_->appends();
  ASSERT_EQ(appends.size(), 5U);
  EXPECT_EQ(appends[0].type, kTypeCreateDatabase);
  EXPECT_EQ(appends[1].type, kTypeCreateTable);
  EXPECT_EQ(appends[2].type, kTypeInsert);
  EXPECT_EQ(appends[3].type, kTypeUpsert);
  EXPECT_EQ(appends[4].type, kTypeDelete);
  EXPECT_EQ(inserted->lsn, appends[2].lsn);
  EXPECT_EQ(upserted->lsn, appends[3].lsn);
  EXPECT_EQ(deleted->lsn, appends[4].lsn);
  // 每次 SyncUpTo 的参数是刚分配的 LSN，且紧跟在对应的 Append 之后。
  EXPECT_THAT(wal_->synced_lsns(), ElementsAre(1U, 2U, 3U, 4U, 5U));
  EXPECT_THAT(
      wal_->log()->events(),
      ElementsAre("Replay(0)", "Append(1)", "SyncUpTo(1)", "Append(3)",
                  "SyncUpTo(2)", "Append(5)", "SyncUpTo(3)", "Append(6)",
                  "SyncUpTo(4)", "Append(7)", "SyncUpTo(5)"));
  EXPECT_EQ(engine_->last_lsn(), 5U);

  // 记录内容可以解码回写入的数据（Insert 保存归一化之后的行）。
  auto decoded = DecodeInsert(appends[2].payload);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(decoded->database, "db");
  EXPECT_EQ(decoded->table, "docs");
  ASSERT_EQ(decoded->rows.size(), 2U);
  EXPECT_EQ(decoded->rows[0].primary_key, "a");

  // 钩子引用了本函数的局部变量；析构时 Engine 还会调用 SyncUpTo。
  wal_->set_on_append(nullptr);
  wal_->set_on_sync(nullptr);
}

TEST_F(EngineTest, FailedValidationAppendsNothing) {
  std::shared_ptr<const Table> table = CreateDocs();
  ASSERT_NE(table, nullptr);
  const int appends_before = wal_->append_calls();
  const int syncs_before = wal_->sync_calls();

  // 第 2 行 dim 不符。
  auto bad =
      engine_->Insert("db", "docs", Rows({Row("a"), Row("b", 2020, {1.0F})}));
  EXPECT_EQ(bad.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(bad.status().message(), HasSubstr("rows[1]"));
  // 0 行。
  EXPECT_EQ(engine_->Insert("db", "docs", {}).status().code(),
            absl::StatusCode::kInvalidArgument);
  // 空主键列表。
  EXPECT_EQ(engine_->Delete("db", "docs", {}).status().code(),
            absl::StatusCode::kInvalidArgument);
  // 全部不存在的主键：不写 WAL，返回 affected=0、lsn=0。
  auto none = engine_->Delete("db", "docs", {"x", "y"});
  ASSERT_TRUE(none.ok()) << none.status();
  EXPECT_EQ(none->affected, 0U);
  EXPECT_EQ(none->lsn, 0U);
  // HNSW 尚未实现。
  EXPECT_EQ(
      engine_->CreateTable("db", DocsSchema("h", DenseIndexKind::kHnsw)).code(),
      absl::StatusCode::kUnimplemented);
  auto tables = engine_->ListTables("db");
  ASSERT_TRUE(tables.ok());
  EXPECT_THAT(*tables, ElementsAre("docs"));

  EXPECT_EQ(wal_->append_calls(), appends_before);
  EXPECT_EQ(wal_->sync_calls(), syncs_before);
  EXPECT_EQ(table->Stats().live_rows, 0U);
}

TEST_F(EngineTest, WalFailureLeavesMemoryUnchanged) {
  std::shared_ptr<const Table> table = CreateDocs();
  ASSERT_NE(table, nullptr);

  // Append 失败：返回该错误、内存不变、Engine 不进入失败态。
  wal_->FailNextAppend(absl::InternalError("append boom"));
  auto r1 = engine_->Insert("db", "docs", Rows({Row("a")}));
  EXPECT_EQ(r1.status().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(table->Stats().live_rows, 0U);
  EXPECT_EQ(wal_->sync_calls(), 2);  // 只有建库与建表的两次
  auto r2 = engine_->Insert("db", "docs", Rows({Row("a")}));
  ASSERT_TRUE(r2.ok()) << r2.status();
  EXPECT_EQ(table->Stats().live_rows, 1U);

  // SyncUpTo 失败：返回该错误、内存不变、Engine 进入失败态。
  wal_->FailNextSync(absl::InternalError("sync boom"));
  auto r3 = engine_->Insert("db", "docs", Rows({Row("b")}));
  EXPECT_EQ(r3.status().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(table->Stats().live_rows, 1U);
  const int appends_after_failure = wal_->append_calls();

  auto r4 = engine_->Insert("db", "docs", Rows({Row("c")}));
  EXPECT_EQ(r4.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(r4.status().message(), HasSubstr("engine is in failed state"));
  EXPECT_EQ(engine_->CreateDatabase("other").code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(engine_->Delete("db", "docs", {"a"}).status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(wal_->append_calls(), appends_after_failure);
  // 读照常。
  EXPECT_EQ(table->Stats().live_rows, 1U);
  EXPECT_THAT(engine_->ListDatabases(), ElementsAre("db"));
}

TEST_F(EngineTest, ApplyFailureEntersFailedState) {
  IndexFactory factory = DefaultIndexFactory();
  factory.dense = [](const DenseIndexConfig& config,
                     const DenseVectorAccessor* accessor)
      -> absl::StatusOr<std::unique_ptr<VectorIndex>> {
    auto inner = MakeVectorIndex(config, accessor);
    if (!inner.ok()) {
      return inner.status();
    }
    return std::make_unique<FailingAddIndex>(*std::move(inner));
  };
  engine_ = OpenEngine(factory);
  ASSERT_NE(engine_, nullptr);
  std::shared_ptr<const Table> table = CreateDocs();
  ASSERT_NE(table, nullptr);

  auto r1 = engine_->Insert("db", "docs", Rows({Row("a")}));
  EXPECT_EQ(r1.status().code(), absl::StatusCode::kInternal);
  // 记录已写入 WAL。
  EXPECT_EQ(wal_->appends().size(), 3U);

  auto r2 = engine_->Insert("db", "docs", Rows({Row("b")}));
  EXPECT_EQ(r2.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(r2.status().message(), HasSubstr("engine is in failed state"));
  EXPECT_EQ(wal_->appends().size(), 3U);

  auto search = table->Search(DenseQuery({1.0F, 0.0F}));
  ASSERT_TRUE(search.ok()) << search.status();
  EXPECT_THAT(search->rows, IsEmpty());
  auto again = engine_->GetTable("db", "docs");
  ASSERT_TRUE(again.ok());
  EXPECT_TRUE((*again)->Search(DenseQuery({0.0F, 1.0F})).ok());
}

TEST_F(EngineTest, CreateSnapshotIsUnimplemented) {
  EXPECT_EQ(engine_->CreateSnapshot().status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST_F(EngineTest, ConcurrentWritersAreSerialized) {
  std::shared_ptr<const Table> table = CreateDocs();
  ASSERT_NE(table, nullptr);
  // 在 Append 内让出时间片，放大并发写者交错的窗口。
  wal_->set_on_append([](uint16_t) { std::this_thread::yield(); });

  constexpr int kThreads = 4;
  constexpr int kRowsPerThread = 100;
  std::vector<std::vector<absl::StatusOr<WriteResult>>> results(kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([this, t, &results] {
      for (int i = 0; i < kRowsPerThread; ++i) {
        results[t].push_back(engine_->Insert(
            "db", "docs", Rows({Row(absl::StrCat("t", t, "_", i))})));
      }
    });
  }
  for (std::thread& th : threads) {
    th.join();
  }

  std::vector<Lsn> result_lsns;
  for (const auto& per_thread : results) {
    for (const auto& r : per_thread) {
      ASSERT_TRUE(r.ok()) << r.status();
      EXPECT_EQ(r->affected, 1U);
      result_lsns.push_back(r->lsn);
    }
  }
  std::sort(result_lsns.begin(), result_lsns.end());

  // 前 2 条是建库与建表；之后 400 条 LSN 连续无重复。
  const std::vector<test_support::AppendCall> appends = wal_->appends();
  constexpr size_t kTotal = 2 + (kThreads * kRowsPerThread);
  ASSERT_EQ(appends.size(), kTotal);
  for (size_t i = 0; i < appends.size(); ++i) {
    EXPECT_EQ(appends[i].lsn, i + 1);
  }
  ASSERT_EQ(result_lsns.size(), static_cast<size_t>(kThreads * kRowsPerThread));
  for (size_t i = 0; i < result_lsns.size(); ++i) {
    EXPECT_EQ(result_lsns[i], i + 3);
  }
  // 串行：打开时的 Replay(0) 之后，每次 Append 紧跟同一 LSN 的 SyncUpTo，
  // 没有交错。
  const std::vector<std::string> events = wal_->log()->events();
  ASSERT_EQ(events.size(), 1 + (2 * kTotal));
  EXPECT_EQ(events[0], "Replay(0)");
  for (size_t i = 0; i < kTotal; ++i) {
    EXPECT_THAT(events[1 + (2 * i)], HasSubstr("Append("));
    EXPECT_EQ(events[2 + (2 * i)], absl::StrCat("SyncUpTo(", i + 1, ")"));
  }
  EXPECT_EQ(table->Stats().live_rows,
            static_cast<uint64_t>(kThreads * kRowsPerThread));
}

}  // namespace
}  // namespace hector
