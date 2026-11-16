#include "hector/storage/snapshot.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/types.h"

namespace hector {
namespace {

TEST(SnapshotStoreTest, NullStoreIsInert) {
  std::unique_ptr<SnapshotStore> store = MakeNullSnapshotStore();
  ASSERT_NE(store, nullptr);

  absl::StatusOr<std::optional<Lsn>> latest = store->Latest();
  ASSERT_TRUE(latest.ok()) << latest.status();
  EXPECT_FALSE(latest->has_value());

  int calls = 0;
  absl::Status write = store->Write(1, [&calls](ByteWriter& /*out*/) {
    ++calls;
    return absl::OkStatus();
  });
  EXPECT_EQ(write.code(), absl::StatusCode::kUnimplemented);
  absl::Status read = store->Read(1, [&calls](ByteReader& /*in*/) {
    ++calls;
    return absl::OkStatus();
  });
  EXPECT_EQ(read.code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(calls, 0);
}

}  // namespace
}  // namespace hector
