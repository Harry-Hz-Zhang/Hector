#include "src/server/hector_service.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "hector/common/status.h"
#include "hector/core/schema.h"
#include "hector/core/search.h"
#include "hector/core/table.h"
#include "hector/core/value.h"
#include "spdlog/spdlog.h"
#include "src/server/proto_convert.h"
#include "src/server/status_convert.h"

namespace hector::server {
namespace {

// 失败时打印 1 条 warn 日志（方法名、状态码与 core 给出的消息；core 的错误
// 消息只含字段路径、列名、主键与计数，不含向量数值），再转换为 gRPC 状态。
grpc::Status Finish(std::string_view method, const absl::Status& status) {
  if (!status.ok()) {
    spdlog::warn("rpc {} failed: {}: {}", method,
                 absl::StatusCodeToString(status.code()), status.message());
  }
  return ToGrpcStatus(status);
}

std::vector<std::string> ToVector(
    const google::protobuf::RepeatedPtrField<std::string>& values) {
  return {values.begin(), values.end()};
}

}  // namespace

HectorServiceImpl::HectorServiceImpl(Engine& engine) : engine_(engine) {}

grpc::Status HectorServiceImpl::CreateDatabase(
    grpc::ServerContext* /*context*/, const v1::CreateDatabaseRequest* request,
    v1::CreateDatabaseResponse* /*response*/) {
  return Finish("CreateDatabase", engine_.CreateDatabase(request->name()));
}

grpc::Status HectorServiceImpl::DropDatabase(
    grpc::ServerContext* /*context*/, const v1::DropDatabaseRequest* request,
    v1::DropDatabaseResponse* /*response*/) {
  return Finish("DropDatabase", engine_.DropDatabase(request->name()));
}

grpc::Status HectorServiceImpl::ListDatabases(
    grpc::ServerContext* /*context*/,
    const v1::ListDatabasesRequest* /*request*/,
    v1::ListDatabasesResponse* response) {
  for (std::string& name : engine_.ListDatabases()) {
    response->add_names(std::move(name));
  }
  return grpc::Status::OK;
}

grpc::Status HectorServiceImpl::CreateTable(
    grpc::ServerContext* /*context*/, const v1::CreateTableRequest* request,
    v1::CreateTableResponse* /*response*/) {
  const absl::Status status = [&]() -> absl::Status {
    HECTOR_ASSIGN_OR_RETURN(Schema schema, SchemaFromProto(request->schema()));
    return engine_.CreateTable(request->database(), std::move(schema));
  }();
  return Finish("CreateTable", status);
}

grpc::Status HectorServiceImpl::DropTable(grpc::ServerContext* /*context*/,
                                          const v1::DropTableRequest* request,
                                          v1::DropTableResponse* /*response*/) {
  return Finish("DropTable",
                engine_.DropTable(request->database(), request->table()));
}

grpc::Status HectorServiceImpl::ListTables(grpc::ServerContext* /*context*/,
                                           const v1::ListTablesRequest* request,
                                           v1::ListTablesResponse* response) {
  const absl::Status status = [&]() -> absl::Status {
    HECTOR_ASSIGN_OR_RETURN(std::vector<std::string> names,
                            engine_.ListTables(request->database()));
    for (std::string& name : names) {
      response->add_names(std::move(name));
    }
    return absl::OkStatus();
  }();
  return Finish("ListTables", status);
}

grpc::Status HectorServiceImpl::DescribeTable(
    grpc::ServerContext* /*context*/, const v1::DescribeTableRequest* request,
    v1::DescribeTableResponse* response) {
  const absl::Status status = [&]() -> absl::Status {
    HECTOR_ASSIGN_OR_RETURN(
        std::shared_ptr<const Table> table,
        engine_.GetTable(request->database(), request->table()));
    *response->mutable_schema() = SchemaToProto(table->schema());
    return absl::OkStatus();
  }();
  return Finish("DescribeTable", status);
}

grpc::Status HectorServiceImpl::Insert(grpc::ServerContext* /*context*/,
                                       const v1::InsertRequest* request,
                                       v1::InsertResponse* response) {
  const absl::Status status = [&]() -> absl::Status {
    HECTOR_ASSIGN_OR_RETURN(std::vector<RowInput> rows,
                            RowsFromProto(request->rows()));
    HECTOR_ASSIGN_OR_RETURN(
        WriteResult result,
        engine_.Insert(request->database(), request->table(), std::move(rows)));
    response->set_inserted(result.affected);
    response->set_lsn(result.lsn);
    return absl::OkStatus();
  }();
  return Finish("Insert", status);
}

grpc::Status HectorServiceImpl::Upsert(grpc::ServerContext* /*context*/,
                                       const v1::UpsertRequest* request,
                                       v1::UpsertResponse* response) {
  const absl::Status status = [&]() -> absl::Status {
    HECTOR_ASSIGN_OR_RETURN(std::vector<RowInput> rows,
                            RowsFromProto(request->rows()));
    HECTOR_ASSIGN_OR_RETURN(
        WriteResult result,
        engine_.Upsert(request->database(), request->table(), std::move(rows)));
    response->set_upserted(result.affected);
    response->set_lsn(result.lsn);
    return absl::OkStatus();
  }();
  return Finish("Upsert", status);
}

grpc::Status HectorServiceImpl::Delete(grpc::ServerContext* /*context*/,
                                       const v1::DeleteRequest* request,
                                       v1::DeleteResponse* response) {
  const absl::Status status = [&]() -> absl::Status {
    HECTOR_ASSIGN_OR_RETURN(
        WriteResult result,
        engine_.Delete(request->database(), request->table(),
                       ToVector(request->primary_keys())));
    response->set_deleted(result.affected);
    response->set_lsn(result.lsn);
    return absl::OkStatus();
  }();
  return Finish("Delete", status);
}

grpc::Status HectorServiceImpl::Get(grpc::ServerContext* /*context*/,
                                    const v1::GetRequest* request,
                                    v1::GetResponse* response) {
  const absl::Status status = [&]() -> absl::Status {
    HECTOR_ASSIGN_OR_RETURN(
        std::shared_ptr<const Table> table,
        engine_.GetTable(request->database(), request->table()));
    const std::vector<std::string> keys = ToVector(request->primary_keys());
    const std::vector<std::string> columns =
        ToVector(request->output_columns());
    std::vector<std::string> missing;
    HECTOR_ASSIGN_OR_RETURN(std::vector<RowOutput> rows,
                            table->Get(keys, columns, &missing));
    response->mutable_rows()->Reserve(static_cast<int>(rows.size()));
    for (const RowOutput& row : rows) {
      RowToProto(row, response->add_rows());
    }
    for (std::string& key : missing) {
      response->add_missing_primary_keys(std::move(key));
    }
    return absl::OkStatus();
  }();
  return Finish("Get", status);
}

grpc::Status HectorServiceImpl::Search(grpc::ServerContext* /*context*/,
                                       const v1::SearchRequest* request,
                                       v1::SearchResponse* response) {
  const absl::Status status = [&]() -> absl::Status {
    HECTOR_ASSIGN_OR_RETURN(
        std::shared_ptr<const Table> table,
        engine_.GetTable(request->database(), request->table()));
    HECTOR_ASSIGN_OR_RETURN(SearchRequest search,
                            SearchRequestFromProto(*request));
    HECTOR_ASSIGN_OR_RETURN(SearchResult result, table->Search(search));
    SearchResultToProto(result, response);
    return absl::OkStatus();
  }();
  return Finish("Search", status);
}

grpc::Status HectorServiceImpl::CreateSnapshot(
    grpc::ServerContext* /*context*/,
    const v1::CreateSnapshotRequest* /*request*/,
    v1::CreateSnapshotResponse* response) {
  const absl::Status status = [&]() -> absl::Status {
    HECTOR_ASSIGN_OR_RETURN(Lsn lsn, engine_.CreateSnapshot());
    response->set_lsn(lsn);
    return absl::OkStatus();
  }();
  return Finish("CreateSnapshot", status);
}

grpc::Status HectorServiceImpl::GetStats(grpc::ServerContext* /*context*/,
                                         const v1::GetStatsRequest* request,
                                         v1::GetStatsResponse* response) {
  const absl::Status status = [&]() -> absl::Status {
    HECTOR_ASSIGN_OR_RETURN(
        std::shared_ptr<const Table> table,
        engine_.GetTable(request->database(), request->table()));
    StatsToProto(table->Stats(), engine_.last_lsn(), response);
    return absl::OkStatus();
  }();
  return Finish("GetStats", status);
}

absl::StatusOr<RunningServer> StartServer(HectorServiceImpl& service,
                                          std::string_view listen_address,
                                          int max_message_bytes) {
  const std::string address(listen_address);
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.SetMaxReceiveMessageSize(max_message_bytes);
  builder.SetMaxSendMessageSize(max_message_bytes);
  // 不允许多个进程共用同一端口，端口被占用时绑定失败。
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  if (server == nullptr || selected_port == 0) {
    if (server != nullptr) {
      server->Shutdown();
    }
    return absl::UnavailableError(
        absl::StrCat("failed to bind listen_address=", address));
  }
  // 端口为 0 时由系统分配：把地址中的端口替换为实际端口。
  const size_t colon = address.rfind(':');
  std::string actual =
      colon == std::string::npos
          ? address
          : absl::StrCat(address.substr(0, colon + 1), selected_port);
  return RunningServer{.server = std::move(server),
                       .address = std::move(actual)};
}

}  // namespace hector::server
