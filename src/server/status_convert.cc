#include "src/server/status_convert.h"

#include <string>

namespace hector::server {

grpc::Status ToGrpcStatus(const absl::Status& status) {
  if (status.ok()) {
    return grpc::Status::OK;
  }
  // raw_code 保留非标准码，code() 会把它们折叠成 kUnknown；这里显式检查范围。
  const int raw = status.raw_code();
  const grpc::StatusCode code =
      raw >= 0 && raw <= static_cast<int>(grpc::StatusCode::UNAUTHENTICATED)
          ? static_cast<grpc::StatusCode>(raw)
          : grpc::StatusCode::UNKNOWN;
  return {code, std::string(status.message())};
}

}  // namespace hector::server
