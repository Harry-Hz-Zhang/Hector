// 进程内集成测试：真实 Engine（临时数据目录）+ HectorServiceImpl + gRPC
// 服务端， 客户端通过 127.0.0.1 上系统分配的端口调用。

#include <gmock/gmock.h>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "hector/core/engine.h"
#include "hector/v1/hector.grpc.pb.h"
#include "hector/v1/hector.pb.h"
#include "spdlog/logger.h"
#include "spdlog/sinks/ringbuffer_sink.h"
#include "spdlog/spdlog.h"
#include "src/server/hector_service.h"

namespace hector::server {
namespace {

namespace fs = std::filesystem;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

constexpr int kDefaultMaxMessageBytes = 4 * 1024 * 1024;
constexpr int kSmallMaxMessageBytes = 1048576;

v1::Value Dense(std::vector<float> values) {
  v1::Value value;
  for (float v : values) {
    value.mutable_dense()->add_values(v);
  }
  return value;
}

v1::Value Sparse(std::vector<uint32_t> indices, std::vector<float> values) {
  v1::Value value;
  for (uint32_t i : indices) {
    value.mutable_sparse()->add_indices(i);
  }
  for (float v : values) {
    value.mutable_sparse()->add_values(v);
  }
  return value;
}

v1::Value Int64(int64_t v) {
  v1::Value value;
  value.set_int64_value(v);
  return value;
}

v1::Value String(std::string v) {
  v1::Value value;
  value.set_string_value(std::move(v));
  return value;
}

// 列：emb（稠密 dim=2 L2，index_type 未指定）、sp（稀疏 max_dim=16）、
// year（int64）、category（string）。
v1::TableSchema DocsSchema() {
  v1::TableSchema schema;
  schema.set_name("docs");
  v1::ColumnDef* emb = schema.add_columns();
  emb->set_name("emb");
  emb->mutable_dense()->set_dim(2);
  emb->mutable_dense()->set_metric(v1::METRIC_L2);
  v1::ColumnDef* sp = schema.add_columns();
  sp->set_name("sp");
  sp->mutable_sparse()->set_max_dim(16);
  v1::ColumnDef* year = schema.add_columns();
  year->set_name("year");
  year->mutable_scalar()->set_type(v1::SCALAR_TYPE_INT64);
  v1::ColumnDef* category = schema.add_columns();
  category->set_name("category");
  category->mutable_scalar()->set_type(v1::SCALAR_TYPE_STRING);
  return schema;
}

void AddDocRow(v1::InsertRequest& request, std::string pk,
               std::vector<float> emb, std::vector<uint32_t> sp_indices,
               std::vector<float> sp_values, int64_t year,
               std::string category) {
  v1::Row* row = request.add_rows();
  row->set_primary_key(std::move(pk));
  auto& fields = *row->mutable_fields();
  fields["emb"] = Dense(std::move(emb));
  fields["sp"] = Sparse(std::move(sp_indices), std::move(sp_values));
  fields["year"] = Int64(year);
  fields["category"] = String(std::move(category));
}

v1::SearchRoute DenseRoute(std::vector<float> query) {
  v1::SearchRoute route;
  route.set_column("emb");
  for (float v : query) {
    route.mutable_dense()->add_values(v);
  }
  return route;
}

v1::SearchRoute SparseRoute(std::vector<uint32_t> indices,
                            std::vector<float> values) {
  v1::SearchRoute route;
  route.set_column("sp");
  for (uint32_t i : indices) {
    route.mutable_sparse()->add_indices(i);
  }
  for (float v : values) {
    route.mutable_sparse()->add_values(v);
  }
  return route;
}

std::vector<std::string> HitKeys(const v1::SearchResponse& response) {
  std::vector<std::string> keys;
  for (const v1::SearchHit& hit : response.hits()) {
    keys.push_back(hit.primary_key());
  }
  return keys;
}

// 在作用域内把 spdlog 默认 logger 换成环形缓冲区，离开作用域时恢复。
class CapturedLogs {
 public:
  CapturedLogs()
      : sink_(std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(64)),
        previous_(spdlog::default_logger()) {
    spdlog::set_default_logger(std::make_shared<spdlog::logger>("test", sink_));
  }
  CapturedLogs(const CapturedLogs&) = delete;
  CapturedLogs& operator=(const CapturedLogs&) = delete;
  ~CapturedLogs() { spdlog::set_default_logger(previous_); }

  std::vector<std::string> Warnings() const {
    std::vector<std::string> warnings;
    for (const auto& message : sink_->last_raw()) {
      if (message.level == spdlog::level::warn) {
        warnings.emplace_back(message.payload.data(), message.payload.size());
      }
    }
    return warnings;
  }

 private:
  std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> sink_;
  std::shared_ptr<spdlog::logger> previous_;
};

class ServiceIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = fs::path(::testing::TempDir()) /
           absl::StrCat("service_integration_test_", info->name());
    fs::remove_all(dir_);
    auto engine = Engine::Open(EngineOptions{.data_dir = dir_});
    ASSERT_TRUE(engine.ok()) << engine.status();
    engine_ = *std::move(engine);
    service_ = std::make_unique<HectorServiceImpl>(*engine_);
  }

  void TearDown() override {
    if (server_ != nullptr) {
      server_->Shutdown();
      server_.reset();
    }
    service_.reset();
    engine_.reset();
    fs::remove_all(dir_);
  }

  // 启动服务端并创建客户端存根；客户端不设消息上限，只有服务端的上限生效。
  void StartWithLimit(int max_message_bytes) {
    auto running = StartServer(*service_, "127.0.0.1:0", max_message_bytes);
    ASSERT_TRUE(running.ok()) << running.status();
    server_ = std::move(running->server);
    ASSERT_THAT(running->address, Not(HasSubstr(":0")));
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(-1);
    args.SetMaxSendMessageSize(-1);
    stub_ = v1::HectorService::NewStub(grpc::CreateCustomChannel(
        running->address, grpc::InsecureChannelCredentials(), args));
  }

  grpc::Status CreateDocsTable() {
    v1::CreateDatabaseRequest db_request;
    db_request.set_name("db");
    v1::CreateDatabaseResponse db_response;
    grpc::ClientContext db_context;
    grpc::Status status =
        stub_->CreateDatabase(&db_context, db_request, &db_response);
    if (!status.ok()) {
      return status;
    }
    v1::CreateTableRequest request;
    request.set_database("db");
    *request.mutable_schema() = DocsSchema();
    v1::CreateTableResponse response;
    grpc::ClientContext context;
    return stub_->CreateTable(&context, request, &response);
  }

  grpc::Status Search(v1::SearchRequest request, v1::SearchResponse* response) {
    request.set_database("db");
    request.set_table("docs");
    grpc::ClientContext context;
    return stub_->Search(&context, request, response);
  }

  fs::path dir_;
  std::unique_ptr<Engine> engine_;
  std::unique_ptr<HectorServiceImpl> service_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<v1::HectorService::Stub> stub_;
};

TEST_F(ServiceIntegrationTest, EndToEndDenseSparseHybrid) {
  ASSERT_NO_FATAL_FAILURE(StartWithLimit(kDefaultMaxMessageBytes));
  ASSERT_TRUE(CreateDocsTable().ok());

  {
    v1::ListDatabasesResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->ListDatabases(&context, {}, &response).ok());
    EXPECT_THAT(response.names(), ElementsAre("db"));
  }
  {
    v1::ListTablesRequest request;
    request.set_database("db");
    v1::ListTablesResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->ListTables(&context, request, &response).ok());
    EXPECT_THAT(response.names(), ElementsAre("docs"));
  }
  {
    v1::DescribeTableRequest request;
    request.set_database("db");
    request.set_table("docs");
    v1::DescribeTableResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->DescribeTable(&context, request, &response).ok());
    ASSERT_EQ(response.schema().columns_size(), 4);
    EXPECT_EQ(response.schema().columns(0).dense().index_type(),
              v1::DENSE_INDEX_TYPE_FLAT);
    EXPECT_EQ(response.schema().columns(0).dense().hnsw().m(), 16U);
    EXPECT_EQ(response.schema().columns(1).sparse().index_type(),
              v1::SPARSE_INDEX_TYPE_FLAT);
    EXPECT_EQ(response.schema().columns(3).scalar().type(),
              v1::SCALAR_TYPE_STRING);
  }

  v1::InsertRequest insert;
  insert.set_database("db");
  insert.set_table("docs");
  AddDocRow(insert, "p0", {0, 0}, {1}, {1.0F}, 2020, "a");
  AddDocRow(insert, "p1", {1, 0}, {2}, {1.0F}, 2023, "b");
  AddDocRow(insert, "p2", {3, 0}, {2, 1}, {3.0F, 0.5F}, 2024, "a");
  {
    v1::InsertResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->Insert(&context, insert, &response).ok());
    EXPECT_EQ(response.inserted(), 3U);
    EXPECT_GT(response.lsn(), 0U);
  }

  // 单路稠密 L2。
  {
    v1::SearchRequest request;
    request.set_top_k(2);
    *request.add_routes() = DenseRoute({0.9F, 0.0F});
    v1::SearchResponse response;
    ASSERT_TRUE(Search(request, &response).ok());
    EXPECT_EQ(response.score_kind(), v1::SCORE_KIND_L2_DISTANCE);
    EXPECT_THAT(HitKeys(response), ElementsAre("p1", "p0"));
    EXPECT_NEAR(response.hits(0).score(), 0.01, 1e-6);
    EXPECT_NEAR(response.hits(1).score(), 0.81, 1e-6);
    // output_columns 为空时只返回主键与分数。
    EXPECT_THAT(response.hits(0).fields(), IsEmpty());
  }
  // 单路稀疏：只返回与查询共享位置的行。
  {
    v1::SearchRequest request;
    request.set_top_k(5);
    *request.add_routes() = SparseRoute({1}, {2.0F});
    v1::SearchResponse response;
    ASSERT_TRUE(Search(request, &response).ok());
    EXPECT_EQ(response.score_kind(), v1::SCORE_KIND_INNER_PRODUCT);
    EXPECT_THAT(HitKeys(response), ElementsAre("p0", "p2"));
    EXPECT_NEAR(response.hits(0).score(), 2.0, 1e-6);
    EXPECT_NEAR(response.hits(1).score(), 1.0, 1e-6);
  }
  // 两路 RRF（fusion 未指定，取 rrf_k=60）：稠密 [p1, p0, p2]，稀疏 [p0, p2]。
  {
    v1::SearchRequest request;
    request.set_top_k(3);
    *request.add_routes() = DenseRoute({0.9F, 0.0F});
    *request.add_routes() = SparseRoute({1}, {2.0F});
    v1::SearchResponse response;
    ASSERT_TRUE(Search(request, &response).ok());
    EXPECT_EQ(response.score_kind(), v1::SCORE_KIND_RRF);
    EXPECT_THAT(HitKeys(response), ElementsAre("p0", "p2", "p1"));
    EXPECT_NEAR(response.hits(0).score(), 1.0 / 62 + 1.0 / 61, 1e-6);
    EXPECT_NEAR(response.hits(1).score(), 1.0 / 63 + 1.0 / 62, 1e-6);
    EXPECT_NEAR(response.hits(2).score(), 1.0 / 61, 1e-6);
  }
  // 带筛选：year >= 2021 AND category IN ("a")，返回 category 列。
  {
    v1::SearchRequest request;
    request.set_top_k(3);
    *request.add_routes() = DenseRoute({0.9F, 0.0F});
    v1::Condition* year = request.mutable_filter()->add_conditions();
    year->set_column("year");
    year->set_op(v1::COMPARE_OP_GE);
    year->add_values()->set_int64_value(2021);
    v1::Condition* category = request.mutable_filter()->add_conditions();
    category->set_column("category");
    category->set_op(v1::COMPARE_OP_IN);
    category->add_values()->set_string_value("a");
    request.add_output_columns("category");
    v1::SearchResponse response;
    ASSERT_TRUE(Search(request, &response).ok());
    EXPECT_THAT(HitKeys(response), ElementsAre("p2"));
    ASSERT_EQ(response.hits(0).fields().size(), 1U);
    EXPECT_EQ(response.hits(0).fields().at("category").string_value(), "a");
  }
  // Get：output_columns 为空时返回全部列（含向量）。
  {
    v1::GetRequest request;
    request.set_database("db");
    request.set_table("docs");
    request.add_primary_keys("p2");
    request.add_primary_keys("nope");
    v1::GetResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->Get(&context, request, &response).ok());
    ASSERT_EQ(response.rows_size(), 1);
    const v1::Row& row = response.rows(0);
    EXPECT_EQ(row.primary_key(), "p2");
    EXPECT_EQ(row.fields().size(), 4U);
    EXPECT_THAT(row.fields().at("emb").dense().values(), ElementsAre(3, 0));
    // 稀疏向量按位置升序保存。
    EXPECT_THAT(row.fields().at("sp").sparse().indices(), ElementsAre(1, 2));
    EXPECT_THAT(row.fields().at("sp").sparse().values(), ElementsAre(0.5, 3));
    EXPECT_EQ(row.fields().at("year").int64_value(), 2024);
    EXPECT_EQ(row.fields().at("category").string_value(), "a");
    EXPECT_THAT(response.missing_primary_keys(), ElementsAre("nope"));
  }
  uint64_t delete_lsn = 0;
  {
    v1::DeleteRequest request;
    request.set_database("db");
    request.set_table("docs");
    request.add_primary_keys("p1");
    v1::DeleteResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->Delete(&context, request, &response).ok());
    EXPECT_EQ(response.deleted(), 1U);
    delete_lsn = response.lsn();
    EXPECT_GT(delete_lsn, 0U);
  }
  {
    v1::GetStatsRequest request;
    request.set_database("db");
    request.set_table("docs");
    v1::GetStatsResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->GetStats(&context, request, &response).ok());
    EXPECT_EQ(response.live_rows(), 2U);
    EXPECT_EQ(response.deleted_rows(), 1U);
    EXPECT_EQ(response.last_lsn(), delete_lsn);
    ASSERT_EQ(response.columns_size(), 4);
    EXPECT_EQ(response.columns(0).name(), "emb");
    EXPECT_EQ(response.columns(0).dense_index_type(),
              v1::DENSE_INDEX_TYPE_FLAT);
    EXPECT_EQ(response.columns(0).indexed_rows(), 3U);
    EXPECT_GT(response.columns(0).memory_bytes(), 0U);
    EXPECT_EQ(response.columns(1).sparse_index_type(),
              v1::SPARSE_INDEX_TYPE_FLAT);
    EXPECT_GT(response.columns(1).memory_bytes(), 0U);
    EXPECT_EQ(response.columns(2).index_case(), v1::ColumnStats::INDEX_NOT_SET);
  }
  {
    v1::UpsertRequest request;
    request.set_database("db");
    request.set_table("docs");
    *request.add_rows() = insert.rows(0);
    v1::UpsertResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->Upsert(&context, request, &response).ok());
    EXPECT_EQ(response.upserted(), 1U);
    EXPECT_GT(response.lsn(), delete_lsn);
  }
  {
    v1::DropTableRequest request;
    request.set_database("db");
    request.set_table("docs");
    v1::DropTableResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->DropTable(&context, request, &response).ok());
  }
  {
    v1::DropDatabaseRequest request;
    request.set_name("db");
    v1::DropDatabaseResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->DropDatabase(&context, request, &response).ok());
    v1::ListDatabasesResponse listed;
    grpc::ClientContext list_context;
    ASSERT_TRUE(stub_->ListDatabases(&list_context, {}, &listed).ok());
    EXPECT_THAT(listed.names(), IsEmpty());
  }
}

TEST_F(ServiceIntegrationTest, ErrorCodesReachClient) {
  // 用环形缓冲区接收日志，核对每个失败的 RPC 恰好产生 1 条 warn 日志。
  const CapturedLogs logs;

  ASSERT_NO_FATAL_FAILURE(StartWithLimit(kDefaultMaxMessageBytes));
  ASSERT_TRUE(CreateDocsTable().ok());

  {
    v1::CreateDatabaseRequest request;
    request.set_name("db");
    v1::CreateDatabaseResponse response;
    grpc::ClientContext context;
    EXPECT_EQ(stub_->CreateDatabase(&context, request, &response).error_code(),
              grpc::StatusCode::ALREADY_EXISTS);
  }
  {
    v1::DescribeTableRequest request;
    request.set_database("db");
    request.set_table("missing");
    v1::DescribeTableResponse response;
    grpc::ClientContext context;
    EXPECT_EQ(stub_->DescribeTable(&context, request, &response).error_code(),
              grpc::StatusCode::NOT_FOUND);
  }
  {
    v1::CreateTableRequest request;
    request.set_database("db");
    *request.mutable_schema() = DocsSchema();
    request.mutable_schema()->set_name("hnsw_docs");
    request.mutable_schema()
        ->mutable_columns(0)
        ->mutable_dense()
        ->set_index_type(v1::DENSE_INDEX_TYPE_HNSW);
    v1::CreateTableResponse response;
    grpc::ClientContext context;
    EXPECT_EQ(stub_->CreateTable(&context, request, &response).error_code(),
              grpc::StatusCode::UNIMPLEMENTED);
  }
  {
    v1::CreateSnapshotResponse response;
    grpc::ClientContext context;
    EXPECT_EQ(stub_->CreateSnapshot(&context, {}, &response).error_code(),
              grpc::StatusCode::UNIMPLEMENTED);
  }
  {
    v1::GetRequest request;
    request.set_database("db");
    request.set_table("docs");
    v1::GetResponse response;
    grpc::ClientContext context;
    EXPECT_EQ(stub_->Get(&context, request, &response).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
  }
  // 维度不符的写入：错误日志中不得出现向量数值。
  {
    v1::InsertRequest request;
    request.set_database("db");
    request.set_table("docs");
    AddDocRow(request, "bad", {123.456F, 789.25F, 1.5F}, {1}, {1.0F}, 2020,
              "a");
    v1::InsertResponse response;
    grpc::ClientContext context;
    EXPECT_EQ(stub_->Insert(&context, request, &response).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
  }

  const std::vector<std::string> warnings = logs.Warnings();
  ASSERT_EQ(warnings.size(), 6U);
  EXPECT_THAT(warnings[0], HasSubstr("CreateDatabase"));
  EXPECT_THAT(warnings[0], HasSubstr("ALREADY_EXISTS"));
  EXPECT_THAT(warnings[1], HasSubstr("DescribeTable"));
  EXPECT_THAT(warnings[1], HasSubstr("NOT_FOUND"));
  EXPECT_THAT(warnings[2], HasSubstr("CreateTable"));
  EXPECT_THAT(warnings[2], HasSubstr("UNIMPLEMENTED"));
  EXPECT_THAT(warnings[3], HasSubstr("CreateSnapshot"));
  EXPECT_THAT(warnings[3], HasSubstr("UNIMPLEMENTED"));
  EXPECT_THAT(warnings[4], HasSubstr("Get"));
  EXPECT_THAT(warnings[4], HasSubstr("INVALID_ARGUMENT"));
  EXPECT_THAT(warnings[5], HasSubstr("Insert"));
  EXPECT_THAT(warnings[5], HasSubstr("INVALID_ARGUMENT"));
  for (const std::string& warning : warnings) {
    EXPECT_THAT(warning, Not(HasSubstr("123.4")));
    EXPECT_THAT(warning, Not(HasSubstr("789.2")));
  }
}

TEST_F(ServiceIntegrationTest, MessageLimitsBothDirections) {
  ASSERT_NO_FATAL_FAILURE(StartWithLimit(kSmallMaxMessageBytes));
  v1::CreateDatabaseRequest db_request;
  db_request.set_name("db");
  v1::CreateDatabaseResponse db_response;
  grpc::ClientContext db_context;
  ASSERT_TRUE(
      stub_->CreateDatabase(&db_context, db_request, &db_response).ok());
  v1::CreateTableRequest table_request;
  table_request.set_database("db");
  v1::TableSchema* schema = table_request.mutable_schema();
  schema->set_name("big");
  v1::ColumnDef* emb = schema->add_columns();
  emb->set_name("emb");
  emb->mutable_dense()->set_dim(1024);
  emb->mutable_dense()->set_metric(v1::METRIC_L2);
  v1::CreateTableResponse table_response;
  grpc::ClientContext table_context;
  ASSERT_TRUE(
      stub_->CreateTable(&table_context, table_request, &table_response).ok());

  auto make_rows = [](size_t first, size_t count) {
    v1::InsertRequest request;
    request.set_database("db");
    request.set_table("big");
    for (size_t i = first; i < first + count; ++i) {
      v1::Row* row = request.add_rows();
      row->set_primary_key(absl::StrCat("k", i));
      v1::Value& value = (*row->mutable_fields())["emb"];
      for (int d = 0; d < 1024; ++d) {
        value.mutable_dense()->add_values(static_cast<float>(i + d));
      }
    }
    return request;
  };
  auto live_rows = [this] {
    v1::GetStatsRequest request;
    request.set_database("db");
    request.set_table("big");
    v1::GetStatsResponse response;
    grpc::ClientContext context;
    EXPECT_TRUE(stub_->GetStats(&context, request, &response).ok());
    return response.live_rows();
  };

  // 请求超限：约 2 MB 的 Insert。
  {
    const v1::InsertRequest request = make_rows(0, 500);
    ASSERT_GT(request.ByteSizeLong(), 2'000'000U);
    v1::InsertResponse response;
    grpc::ClientContext context;
    EXPECT_EQ(stub_->Insert(&context, request, &response).error_code(),
              grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(live_rows(), 0U);
  }
  // 响应超限：分 3 批写入 300 行，再一次 Get 全部主键（约 1.2 MB 响应）。
  for (size_t batch = 0; batch < 3; ++batch) {
    const v1::InsertRequest request = make_rows(batch * 100, 100);
    ASSERT_LT(request.ByteSizeLong(),
              static_cast<size_t>(kSmallMaxMessageBytes));
    v1::InsertResponse response;
    grpc::ClientContext context;
    ASSERT_TRUE(stub_->Insert(&context, request, &response).ok());
  }
  EXPECT_EQ(live_rows(), 300U);
  v1::GetRequest get;
  get.set_database("db");
  get.set_table("big");
  for (size_t i = 0; i < 300; ++i) {
    get.add_primary_keys(absl::StrCat("k", i));
  }
  v1::GetResponse response;
  grpc::ClientContext context;
  EXPECT_EQ(stub_->Get(&context, get, &response).error_code(),
            grpc::StatusCode::RESOURCE_EXHAUSTED);
}

}  // namespace
}  // namespace hector::server
