#include "hector/storage/snapshot.h"

#include <functional>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/types.h"

namespace hector {
namespace {

// c0 不支持快照：从不调用回调，也从不报告已有快照。
class NullSnapshotStore final : public SnapshotStore {
 public:
  absl::Status Write(
      Lsn /*lsn*/,
      const std::function<absl::Status(ByteWriter&)>& /*body*/) override {
    return absl::UnimplementedError("snapshots are not supported in c0");
  }

  absl::StatusOr<std::optional<Lsn>> Latest() const override {
    return std::optional<Lsn>();
  }

  absl::Status Read(
      Lsn /*lsn*/,
      const std::function<absl::Status(ByteReader&)>& /*body*/) override {
    return absl::UnimplementedError("snapshots are not supported in c0");
  }
};

}  // namespace

std::unique_ptr<SnapshotStore> MakeNullSnapshotStore() {
  return std::make_unique<NullSnapshotStore>();
}

}  // namespace hector
