#include "src/server/status_convert.h"

#include <grpcpp/support/status.h>
#include <gtest/gtest.h>

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace hector::server {
namespace {

TEST(StatusConvertTest, MapsEveryCanonicalCode) {
  for (int code = 0; code <= 16; ++code) {
    const std::string message = absl::StrCat("message for code ", code);
    const absl::Status status(static_cast<absl::StatusCode>(code), message);
    const grpc::Status converted = ToGrpcStatus(status);
    EXPECT_EQ(static_cast<int>(converted.error_code()), code);
    // absl 的 OK 状态不保留消息。
    EXPECT_EQ(converted.error_message(), code == 0 ? "" : message);
  }

  const absl::Status nonstandard(static_cast<absl::StatusCode>(99), "odd");
  const grpc::Status converted = ToGrpcStatus(nonstandard);
  EXPECT_EQ(converted.error_code(), grpc::StatusCode::UNKNOWN);
  EXPECT_EQ(converted.error_message(), "odd");
}

}  // namespace
}  // namespace hector::server
