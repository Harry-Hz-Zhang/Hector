#ifndef HECTOR_STORAGE_WAL_H_
#define HECTOR_STORAGE_WAL_H_

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/types.h"

namespace hector {

// 一条 WAL 记录。type 与 payload 的含义由上层（Engine）定义。
struct WalRecord {
  Lsn lsn;
  uint16_t type;
  std::string payload;
};

// 回放访问者：返回非 OK 时回放立即停止，并把该状态原样返回。
using WalVisitor = std::function<absl::Status(const WalRecord&)>;

// 写前日志。LSN 从 1 开始逐条加 1。
class Wal {
 public:
  virtual ~Wal() = default;

  // 写入缓冲并分配下一个 LSN，不保证持久化。调用方必须串行调用 Append。
  virtual absl::StatusOr<Lsn> Append(uint16_t type,
                                     std::string_view payload) = 0;
  // 线程安全：返回时所有 LSN <= lsn 的记录都已持久化。c0 只 flush
  // 到操作系统；c5 用 F_FULLFSYNC，并让并发的调用方共享一次刷盘（批量提交）。
  virtual absl::Status SyncUpTo(Lsn lsn) = 0;
  // 按 LSN 升序回放 lsn > after 的记录。
  virtual absl::Status Replay(Lsn after, const WalVisitor& visit) = 0;
  // 关闭当前段，开新段；当前段没有记录时不做任何事。
  virtual absl::Status Rotate() = 0;
  // c0 返回 Unimplemented；c5 在快照后删除旧段。
  virtual absl::Status TruncateBefore(Lsn lsn) = 0;
  // 最后一条已分配的 LSN，没有记录时为 0。任意线程可调用。
  virtual Lsn last_lsn() const = 0;
};

}  // namespace hector

#endif  // HECTOR_STORAGE_WAL_H_
