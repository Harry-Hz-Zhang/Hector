#ifndef HECTOR_SERVER_HECTOR_SERVICE_H_
#define HECTOR_SERVER_HECTOR_SERVICE_H_

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/core/engine.h"
#include "hector/v1/hector.grpc.pb.h"
#include "hector/v1/hector.pb.h"

namespace hector::server {

// hector.v1.HectorService 的同步实现：把请求转换为 core 类型后只调用
// Engine 的公开方法（以及 Engine::GetTable 返回的只读 Table 的公开方法）。
// 每个失败的 RPC 打印 1 条 warn 日志，含方法名、状态码与错误消息，不含
// 向量数值。Engine 必须比本对象与使用它的 grpc::Server 活得久。
class HectorServiceImpl final : public v1::HectorService::Service {
 public:
  explicit HectorServiceImpl(Engine& engine);

  grpc::Status CreateDatabase(grpc::ServerContext* context,
                              const v1::CreateDatabaseRequest* request,
                              v1::CreateDatabaseResponse* response) override;
  grpc::Status DropDatabase(grpc::ServerContext* context,
                            const v1::DropDatabaseRequest* request,
                            v1::DropDatabaseResponse* response) override;
  grpc::Status ListDatabases(grpc::ServerContext* context,
                             const v1::ListDatabasesRequest* request,
                             v1::ListDatabasesResponse* response) override;
  grpc::Status CreateTable(grpc::ServerContext* context,
                           const v1::CreateTableRequest* request,
                           v1::CreateTableResponse* response) override;
  grpc::Status DropTable(grpc::ServerContext* context,
                         const v1::DropTableRequest* request,
                         v1::DropTableResponse* response) override;
  grpc::Status ListTables(grpc::ServerContext* context,
                          const v1::ListTablesRequest* request,
                          v1::ListTablesResponse* response) override;
  grpc::Status DescribeTable(grpc::ServerContext* context,
                             const v1::DescribeTableRequest* request,
                             v1::DescribeTableResponse* response) override;
  grpc::Status Insert(grpc::ServerContext* context,
                      const v1::InsertRequest* request,
                      v1::InsertResponse* response) override;
  grpc::Status Upsert(grpc::ServerContext* context,
                      const v1::UpsertRequest* request,
                      v1::UpsertResponse* response) override;
  grpc::Status Delete(grpc::ServerContext* context,
                      const v1::DeleteRequest* request,
                      v1::DeleteResponse* response) override;
  grpc::Status Get(grpc::ServerContext* context, const v1::GetRequest* request,
                   v1::GetResponse* response) override;
  grpc::Status Search(grpc::ServerContext* context,
                      const v1::SearchRequest* request,
                      v1::SearchResponse* response) override;
  grpc::Status CreateSnapshot(grpc::ServerContext* context,
                              const v1::CreateSnapshotRequest* request,
                              v1::CreateSnapshotResponse* response) override;
  grpc::Status GetStats(grpc::ServerContext* context,
                        const v1::GetStatsRequest* request,
                        v1::GetStatsResponse* response) override;

 private:
  Engine& engine_;
};

// 一个已启动的 gRPC 服务端与它实际监听的地址。
struct RunningServer {
  std::unique_ptr<grpc::Server> server;
  // listen_address 中的端口为 0 时替换为系统分配的实际端口。
  std::string address;
};

// 在 listen_address 上以明文方式启动服务端，max_message_bytes 同时作为
// 接收与发送的消息上限（design.md 第 6 节）。绑定失败返回 Unavailable，
// 消息含 listen_address。service 必须比返回的服务端活得久。
absl::StatusOr<RunningServer> StartServer(HectorServiceImpl& service,
                                          std::string_view listen_address,
                                          int max_message_bytes);

}  // namespace hector::server

#endif  // HECTOR_SERVER_HECTOR_SERVICE_H_
