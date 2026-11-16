#ifndef HECTOR_STORAGE_FILE_WAL_H_
#define HECTOR_STORAGE_FILE_WAL_H_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/storage/wal.h"

namespace hector {

struct FileWalOptions {
  // 测试注入点：非空时替代真实的文件写入（即替代 fwrite 这一步），
  // 返回非 OK 即模拟写失败。段头写入与每次 Append 写记录都经过它，
  // 因此「该次 Append 返回错误」可观察；SyncUpTo 的 fflush 不经过它。
  // 生产路径 Engine::Open 只调用不带 options 的重载。
  std::function<absl::Status(std::span<const std::byte>)> write_override;
};

// 打开 dir 下的分段 WAL（Engine 传入 <data_dir>/wal）。dir 不存在时创建；
// 没有任何段文件时创建首个段 wal-00000000000000000001.log。
// 段头或 LSN 校验失败、中间段不完整时返回 DataLoss，消息含段文件名；
// 最后一个段末尾的不完整记录被截断并打 warn 日志。
absl::StatusOr<std::unique_ptr<Wal>> OpenFileWal(
    const std::filesystem::path& dir);
absl::StatusOr<std::unique_ptr<Wal>> OpenFileWal(
    const std::filesystem::path& dir, const FileWalOptions& options);

}  // namespace hector

#endif  // HECTOR_STORAGE_FILE_WAL_H_
