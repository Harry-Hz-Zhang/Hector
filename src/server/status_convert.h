#ifndef HECTOR_SERVER_STATUS_CONVERT_H_
#define HECTOR_SERVER_STATUS_CONVERT_H_

#include <grpcpp/support/status.h>

#include "absl/status/status.h"

namespace hector::server {

// absl 与 gRPC 的标准状态码数值一致（design.md 第 6 节）：0 到 16 按数值
// 直接转换，其他值映射为 UNKNOWN；错误消息原样传递。
grpc::Status ToGrpcStatus(const absl::Status& status);

}  // namespace hector::server

#endif  // HECTOR_SERVER_STATUS_CONVERT_H_
