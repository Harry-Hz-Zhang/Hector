#ifndef HECTOR_STORAGE_SNAPSHOT_H_
#define HECTOR_STORAGE_SNAPSHOT_H_

#include <functional>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/types.h"

namespace hector {

// 快照存储。c0 只有 Null 实现，c5 提供基于文件的实现。
class SnapshotStore {
 public:
  virtual ~SnapshotStore() = default;

  // 写入覆盖到 lsn 为止的快照，内容由 body 写出。
  virtual absl::Status Write(
      Lsn lsn, const std::function<absl::Status(ByteWriter&)>& body) = 0;
  // 最新快照的 LSN；没有快照时为 nullopt。
  virtual absl::StatusOr<std::optional<Lsn>> Latest() const = 0;
  // 读取 lsn 对应的快照，内容交给 body 解析。
  virtual absl::Status Read(
      Lsn lsn, const std::function<absl::Status(ByteReader&)>& body) = 0;
};

// Write/Read 返回 Unimplemented 且不调用回调，Latest 返回 nullopt。
std::unique_ptr<SnapshotStore> MakeNullSnapshotStore();

}  // namespace hector

#endif  // HECTOR_STORAGE_SNAPSHOT_H_
