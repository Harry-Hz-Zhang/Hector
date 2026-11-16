#ifndef HECTOR_CORE_ENGINE_H_
#define HECTOR_CORE_ENGINE_H_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/types.h"
#include "hector/core/database.h"
#include "hector/core/schema.h"
#include "hector/core/table.h"
#include "hector/core/value.h"
#include "hector/storage/snapshot.h"
#include "hector/storage/wal.h"

namespace hector {

struct EngineOptions {
  std::filesystem::path data_dir;
};

// 一次写入的结果：影响的行数与该写入的 LSN（没有写 WAL 时为 0）。
struct WriteResult {
  uint64_t affected;
  Lsn lsn;
};

// 库表目录、串行写路径与启动恢复（design.md 4.5）。
//
// 写路径：所有 DDL 与 DML 在写互斥锁内串行执行，顺序固定为 Prepare*(kOnline)
// 校验、Wal::Append、Wal::SyncUpTo、Apply*、Publish、返回。校验失败不写 WAL。
// Append 失败时返回该错误、内存不变，Engine 不进入失败态；SyncUpTo 失败或
// Apply 失败时返回该错误并进入失败态，之后所有写请求返回 FailedPrecondition
// （消息含 `engine is in failed state`），读请求照常。
//
// Engine 无法查询 Wal 的失败态，只记录自己的失败态。使用 FileWal 时，Append
// 写入失败会让 FileWal 自身进入失败态：Engine 仍不进入失败态，但之后的写入
// 会在 Append 处收到 FileWal 的 FailedPrecondition（消息来自 FileWal，不含
// `engine is in failed state`），内存同样不变。
//
// 读操作（List*、GetTable 以及对返回的 Table 调用 Get、Search、Stats）可以与
// 写操作并发。库名与表名不匹配 ^[A-Za-z_][A-Za-z0-9_]{0,63}$ 时返回
// InvalidArgument；库或表不存在返回 NotFound。
class Engine {
 public:
  // 在 <data_dir>/LOCK 上取 flock(LOCK_EX | LOCK_NB)（失败返回
  // FailedPrecondition，消息含 `data_dir is locked by another process`），
  // 打开 <data_dir>/wal 下的 FileWal，再按 4.5 的顺序恢复。data_dir 不存在时
  // 创建。回放中任何一条记录解码或校验失败返回 DataLoss，消息含该 LSN。
  static absl::StatusOr<std::unique_ptr<Engine>> Open(
      const EngineOptions& options);
  // 测试注入：不加锁文件，其余与 Open 相同（先 snapshots->Latest，再
  // wal->Replay）。
  static absl::StatusOr<std::unique_ptr<Engine>> OpenWith(
      std::unique_ptr<Wal> wal, std::unique_ptr<SnapshotStore> snapshots,
      const IndexFactory& factory = DefaultIndexFactory());

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  // Engine 自身未处于失败态时执行 Wal::SyncUpTo(last_lsn)（依据的是 Engine
  // 的失败态而不是 Wal 的；Wal 已处于自身失败态时 SyncUpTo 返回错误，只记
  // warn 日志），然后关闭 Wal 并释放 <data_dir>/LOCK（OpenWith 创建的引擎
  // 没有锁文件）。
  ~Engine();

  // 重名返回 AlreadyExists。
  absl::Status CreateDatabase(std::string_view name);
  // 连带删除库内所有表。
  absl::Status DropDatabase(std::string_view name);
  // 字典序。
  std::vector<std::string> ListDatabases() const;

  // 表名取 schema.table_name()；重名返回 AlreadyExists；HNSW、INVERTED、int8
  // 返回 Unimplemented（由 Table::Create 的索引工厂拒绝）。
  absl::Status CreateTable(std::string_view database, Schema schema);
  absl::Status DropTable(std::string_view database, std::string_view table);
  // 字典序。
  absl::StatusOr<std::vector<std::string>> ListTables(
      std::string_view database) const;
  // 返回的 shared_ptr 在 DropTable 或 DropDatabase 之后仍然有效。
  absl::StatusOr<std::shared_ptr<const Table>> GetTable(
      std::string_view database, std::string_view table) const;

  // affected 为写入的行数。
  absl::StatusOr<WriteResult> Insert(std::string_view database,
                                     std::string_view table,
                                     std::vector<RowInput> rows);
  absl::StatusOr<WriteResult> Upsert(std::string_view database,
                                     std::string_view table,
                                     std::vector<RowInput> rows);
  // affected 为实际删除的行数；一行都没删时不写 WAL，返回 {0, 0}。
  absl::StatusOr<WriteResult> Delete(std::string_view database,
                                     std::string_view table,
                                     std::vector<std::string> primary_keys);

  // c0 返回 Unimplemented。
  absl::StatusOr<Lsn> CreateSnapshot();

  // 最后一条已分配的 LSN（Wal::last_lsn）。
  Lsn last_lsn() const;
  // 打开时回放的 WAL 记录数。
  uint64_t replayed_records() const;

 private:
  Engine(std::unique_ptr<Wal> wal, std::unique_ptr<SnapshotStore> snapshots,
         IndexFactory factory, int lock_fd);

  // 构造引擎并恢复；失败时引擎被析构（释放锁文件）。
  static absl::StatusOr<std::unique_ptr<Engine>> Start(
      std::unique_ptr<Wal> wal, std::unique_ptr<SnapshotStore> snapshots,
      const IndexFactory& factory, int lock_fd);

  // 恢复：snapshots_->Latest()，再 wal_->Replay(0, …)。
  absl::Status Recover();
  // 解码一条记录并走与在线写入相同的 Do*（replay_lsn 为该记录的 LSN）。
  absl::Status ReplayRecord(const WalRecord& record);

  // 以下在持有 write_mu_ 时调用。replay_lsn 为 0 表示在线写入：校验通过后
  // 编码记录、Append、SyncUpTo，Apply 失败时进入失败态；非 0 表示回放该
  // LSN 的记录：跳过 Append 与 SyncUpTo，Prepare 使用 kReplay，错误原样返回。
  absl::Status CheckWritable() const;
  absl::Status DoCreateDatabase(std::string_view name, Lsn replay_lsn);
  absl::Status DoDropDatabase(std::string_view name, Lsn replay_lsn);
  absl::Status DoCreateTable(std::string_view database, Schema schema,
                             Lsn replay_lsn);
  absl::Status DoDropTable(std::string_view database, std::string_view table,
                           Lsn replay_lsn);
  absl::StatusOr<WriteResult> DoWriteRows(std::string_view database,
                                          std::string_view table,
                                          std::vector<RowInput> rows,
                                          bool upsert, Lsn replay_lsn);
  absl::StatusOr<WriteResult> DoDelete(
      std::string_view database, std::string_view table,
      std::span<const std::string> primary_keys, Lsn replay_lsn);
  // 在线写入：Append 并 SyncUpTo，返回新 LSN。Append 失败时原样返回；
  // SyncUpTo 失败时进入失败态并返回该错误。
  absl::StatusOr<Lsn> LogOnline(uint16_t type, std::string_view payload);
  // Apply 之后的收尾：在线写入失败时进入失败态。
  absl::Status CheckApplied(absl::Status applied, Lsn replay_lsn);
  void EnterFailedState(const absl::Status& cause);

  // 目录访问：查询持目录读锁，修改只在改映射的那一刻持目录写锁。
  bool HasDatabase(std::string_view name) const;
  absl::StatusOr<std::shared_ptr<Table>> FindTable(
      std::string_view database, std::string_view table) const;

  std::unique_ptr<Wal> wal_;
  std::unique_ptr<SnapshotStore> snapshots_;
  const IndexFactory factory_;
  // <data_dir>/LOCK 的文件描述符；OpenWith 创建的引擎为 -1。
  int lock_fd_;

  // 串行化所有写操作（DDL、DML 与回放）。
  std::mutex write_mu_;
  // 只由持有 write_mu_ 的写者读写；析构时读取。
  bool failed_ = false;
  // 只在 Open 期间写入。
  uint64_t replayed_records_ = 0;

  // 保护 databases_ 以及其中每个 Database 的表映射。不在 Append 与 SyncUpTo
  // 期间持有。
  mutable std::shared_mutex catalog_mu_;
  std::map<std::string, Database, std::less<>> databases_;
};

}  // namespace hector

#endif  // HECTOR_CORE_ENGINE_H_
