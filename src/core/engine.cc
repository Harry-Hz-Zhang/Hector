#include "hector/core/engine.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "hector/common/status.h"
#include "hector/common/types.h"
#include "hector/core/database.h"
#include "hector/core/schema.h"
#include "hector/core/table.h"
#include "hector/core/value.h"
#include "hector/core/wal_codec.h"
#include "hector/storage/file_wal.h"
#include "hector/storage/snapshot.h"
#include "hector/storage/wal.h"
#include "spdlog/spdlog.h"

namespace hector {
namespace {

// 与 Schema::Create 的表名、列名规则相同（design.md 第 5 节）。
constexpr size_t kMaxNameLength = 64;
constexpr std::string_view kNamePattern = "^[A-Za-z_][A-Za-z0-9_]{0,63}$";
constexpr std::string_view kLockFileName = "LOCK";
constexpr std::string_view kWalDirName = "wal";

bool IsNameStart(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

bool IsNameChar(char c) { return IsNameStart(c) || (c >= '0' && c <= '9'); }

bool IsValidName(std::string_view name) {
  if (name.empty() || name.size() > kMaxNameLength || !IsNameStart(name[0])) {
    return false;
  }
  for (char c : name) {
    if (!IsNameChar(c)) {
      return false;
    }
  }
  return true;
}

absl::Status ValidateName(std::string_view kind, std::string_view name) {
  if (!IsValidName(name)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kind, " name \"", name, "\" must match ", kNamePattern));
  }
  return absl::OkStatus();
}

absl::Status DatabaseNotFound(std::string_view name) {
  return absl::NotFoundError(absl::StrCat("database \"", name, "\" not found"));
}

uint16_t TypeOf(WalRecordType type) { return static_cast<uint16_t>(type); }

}  // namespace

Engine::Engine(std::unique_ptr<Wal> wal,
               std::unique_ptr<SnapshotStore> snapshots, IndexFactory factory,
               int lock_fd)
    : wal_(std::move(wal)),
      snapshots_(std::move(snapshots)),
      factory_(std::move(factory)),
      lock_fd_(lock_fd) {}

// 只有内存耗尽时 absl::Status::ToString 与 spdlog 格式化才会抛 bad_alloc；
// 核心代码本身不抛异常，此时终止进程可以接受。
// NOLINTNEXTLINE(bugprone-exception-escape)
Engine::~Engine() {
  if (wal_ != nullptr && !failed_) {
    if (absl::Status s = wal_->SyncUpTo(wal_->last_lsn()); !s.ok()) {
      spdlog::warn("engine shutdown: wal sync failed: {}", s.ToString());
    }
  }
  // 先关闭 Wal，再释放锁：另一个进程拿到锁时本进程已不再写段文件。
  wal_.reset();
  if (lock_fd_ >= 0) {
    ::flock(lock_fd_, LOCK_UN);
    ::close(lock_fd_);
  }
}

absl::StatusOr<std::unique_ptr<Engine>> Engine::Open(
    const EngineOptions& options) {
  const std::filesystem::path& dir = options.data_dir;
  if (dir.empty()) {
    return absl::InvalidArgumentError("data_dir must not be empty");
  }
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    return absl::ErrnoToStatus(
        ec.value(), absl::StrCat("cannot create data_dir ", dir.string()));
  }
  const std::filesystem::path lock_path = dir / kLockFileName;
  const int fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("cannot open lock file ", lock_path.string()));
  }
  if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
    const int err = errno;
    ::close(fd);
    if (err == EWOULDBLOCK) {
      return absl::FailedPreconditionError(absl::StrCat(
          "data_dir is locked by another process: ", dir.string()));
    }
    return absl::ErrnoToStatus(
        err, absl::StrCat("cannot lock ", lock_path.string()));
  }
  // 生产路径只使用不带 options 的重载。
  absl::StatusOr<std::unique_ptr<Wal>> wal = OpenFileWal(dir / kWalDirName);
  if (!wal.ok()) {
    ::flock(fd, LOCK_UN);
    ::close(fd);
    return wal.status();
  }
  return Start(*std::move(wal), MakeNullSnapshotStore(), DefaultIndexFactory(),
               fd);
}

absl::StatusOr<std::unique_ptr<Engine>> Engine::OpenWith(
    std::unique_ptr<Wal> wal, std::unique_ptr<SnapshotStore> snapshots,
    const IndexFactory& factory) {
  return Start(std::move(wal), std::move(snapshots), factory, -1);
}

absl::StatusOr<std::unique_ptr<Engine>> Engine::Start(
    std::unique_ptr<Wal> wal, std::unique_ptr<SnapshotStore> snapshots,
    const IndexFactory& factory, int lock_fd) {
  if (wal == nullptr || snapshots == nullptr) {
    if (lock_fd >= 0) {
      ::flock(lock_fd, LOCK_UN);
      ::close(lock_fd);
    }
    return absl::InvalidArgumentError("wal and snapshots must not be null");
  }
  // 构造函数是私有的，不能用 std::make_unique。从这里起锁文件由引擎持有。
  std::unique_ptr<Engine> engine(
      new Engine(std::move(wal), std::move(snapshots), factory, lock_fd));
  HECTOR_RETURN_IF_ERROR(engine->Recover());
  return engine;
}

absl::Status Engine::Recover() {
  HECTOR_ASSIGN_OR_RETURN(const std::optional<Lsn> snapshot,
                          snapshots_->Latest());
  if (snapshot.has_value()) {
    // c0 的 Table::Load 返回 Unimplemented；c5 在这里 Read 并恢复目录与表。
    return absl::UnimplementedError(
        absl::StrCat("snapshot recovery is not implemented in c0 (snapshot "
                     "at lsn ",
                     *snapshot, ")"));
  }
  const Lsn after = 0;
  std::lock_guard<std::mutex> lock(write_mu_);
  return wal_->Replay(after, [this](const WalRecord& record) {
    if (absl::Status s = ReplayRecord(record); !s.ok()) {
      return absl::DataLossError(absl::StrCat("wal record at lsn ", record.lsn,
                                              " (type ", record.type,
                                              "): ", s.message()));
    }
    ++replayed_records_;
    return absl::OkStatus();
  });
}

absl::Status Engine::ReplayRecord(const WalRecord& record) {
  if (record.lsn == 0) {
    return absl::DataLossError("lsn 0 is reserved");
  }
  switch (static_cast<WalRecordType>(record.type)) {
    case WalRecordType::kCreateDatabase: {
      HECTOR_ASSIGN_OR_RETURN(DatabaseRecord r,
                              DecodeCreateDatabase(record.payload));
      return DoCreateDatabase(r.database, record.lsn);
    }
    case WalRecordType::kDropDatabase: {
      HECTOR_ASSIGN_OR_RETURN(DatabaseRecord r,
                              DecodeDropDatabase(record.payload));
      return DoDropDatabase(r.database, record.lsn);
    }
    case WalRecordType::kCreateTable: {
      HECTOR_ASSIGN_OR_RETURN(CreateTableRecord r,
                              DecodeCreateTable(record.payload));
      return DoCreateTable(r.database, std::move(r.schema), record.lsn);
    }
    case WalRecordType::kDropTable: {
      HECTOR_ASSIGN_OR_RETURN(DropTableRecord r,
                              DecodeDropTable(record.payload));
      return DoDropTable(r.database, r.table, record.lsn);
    }
    case WalRecordType::kInsert:
    case WalRecordType::kUpsert: {
      const bool upsert = record.type == TypeOf(WalRecordType::kUpsert);
      HECTOR_ASSIGN_OR_RETURN(
          WriteRowsRecord r,
          upsert ? DecodeUpsert(record.payload) : DecodeInsert(record.payload));
      return DoWriteRows(r.database, r.table, std::move(r.rows), upsert,
                         record.lsn)
          .status();
    }
    case WalRecordType::kDelete: {
      HECTOR_ASSIGN_OR_RETURN(DeleteRecord r, DecodeDelete(record.payload));
      return DoDelete(r.database, r.table, r.primary_keys, record.lsn).status();
    }
  }
  return absl::DataLossError(absl::StrCat("unknown record type ", record.type));
}

absl::Status Engine::CheckWritable() const {
  if (failed_) {
    return absl::FailedPreconditionError(
        "engine is in failed state: writes are rejected until restart");
  }
  return absl::OkStatus();
}

absl::StatusOr<Lsn> Engine::LogOnline(uint16_t type, std::string_view payload) {
  // Append 失败：记录没有写入，内存不变，Engine 不进入失败态。
  HECTOR_ASSIGN_OR_RETURN(const Lsn lsn, wal_->Append(type, payload));
  if (absl::Status s = wal_->SyncUpTo(lsn); !s.ok()) {
    // 记录可能已部分落盘、重启后会被回放，内存不再与 WAL 一致。
    EnterFailedState(s);
    return s;
  }
  return lsn;
}

absl::Status Engine::CheckApplied(absl::Status applied, Lsn replay_lsn) {
  if (!applied.ok() && replay_lsn == 0) {
    EnterFailedState(applied);
  }
  return applied;
}

void Engine::EnterFailedState(const absl::Status& cause) {
  failed_ = true;
  spdlog::error("engine entered failed state: {}", cause.ToString());
}

bool Engine::HasDatabase(std::string_view name) const {
  std::shared_lock<std::shared_mutex> lock(catalog_mu_);
  return databases_.contains(name);
}

absl::StatusOr<std::shared_ptr<Table>> Engine::FindTable(
    std::string_view database, std::string_view table) const {
  std::shared_lock<std::shared_mutex> lock(catalog_mu_);
  auto it = databases_.find(database);
  if (it == databases_.end()) {
    return DatabaseNotFound(database);
  }
  return it->second.FindTable(table);
}

absl::Status Engine::DoCreateDatabase(std::string_view name, Lsn replay_lsn) {
  HECTOR_RETURN_IF_ERROR(ValidateName("database", name));
  if (HasDatabase(name)) {
    return absl::AlreadyExistsError(
        absl::StrCat("database \"", name, "\" already exists"));
  }
  if (replay_lsn == 0) {
    HECTOR_ASSIGN_OR_RETURN(const std::string payload,
                            EncodeCreateDatabase(name));
    HECTOR_RETURN_IF_ERROR(
        LogOnline(TypeOf(WalRecordType::kCreateDatabase), payload).status());
  }
  absl::Status applied = absl::OkStatus();
  {
    std::unique_lock<std::shared_mutex> lock(catalog_mu_);
    if (!databases_.try_emplace(std::string(name), std::string(name)).second) {
      applied = absl::AlreadyExistsError(
          absl::StrCat("database \"", name, "\" already exists"));
    }
  }
  return CheckApplied(std::move(applied), replay_lsn);
}

absl::Status Engine::DoDropDatabase(std::string_view name, Lsn replay_lsn) {
  HECTOR_RETURN_IF_ERROR(ValidateName("database", name));
  if (!HasDatabase(name)) {
    return DatabaseNotFound(name);
  }
  if (replay_lsn == 0) {
    HECTOR_ASSIGN_OR_RETURN(const std::string payload,
                            EncodeDropDatabase(name));
    HECTOR_RETURN_IF_ERROR(
        LogOnline(TypeOf(WalRecordType::kDropDatabase), payload).status());
  }
  absl::Status applied = absl::OkStatus();
  {
    std::unique_lock<std::shared_mutex> lock(catalog_mu_);
    auto it = databases_.find(name);
    if (it == databases_.end()) {
      applied = DatabaseNotFound(name);
    } else {
      databases_.erase(it);
    }
  }
  return CheckApplied(std::move(applied), replay_lsn);
}

absl::Status Engine::DoCreateTable(std::string_view database, Schema schema,
                                   Lsn replay_lsn) {
  HECTOR_RETURN_IF_ERROR(ValidateName("database", database));
  if (!HasDatabase(database)) {
    return DatabaseNotFound(database);
  }
  if (FindTable(database, schema.table_name()).ok()) {
    return absl::AlreadyExistsError(
        absl::StrCat("table \"", schema.table_name(),
                     "\" already exists in database \"", database, "\""));
  }
  // HNSW、INVERTED、int8 在这里由索引工厂拒绝，不写 WAL。
  HECTOR_ASSIGN_OR_RETURN(std::unique_ptr<Table> created,
                          Table::Create(std::move(schema), factory_));
  if (replay_lsn == 0) {
    HECTOR_ASSIGN_OR_RETURN(const std::string payload,
                            EncodeCreateTable(database, created->schema()));
    HECTOR_RETURN_IF_ERROR(
        LogOnline(TypeOf(WalRecordType::kCreateTable), payload).status());
  }
  absl::Status applied = absl::OkStatus();
  {
    std::unique_lock<std::shared_mutex> lock(catalog_mu_);
    auto it = databases_.find(database);
    applied =
        it == databases_.end()
            ? DatabaseNotFound(database)
            : it->second.AddTable(std::shared_ptr<Table>(std::move(created)));
  }
  return CheckApplied(std::move(applied), replay_lsn);
}

absl::Status Engine::DoDropTable(std::string_view database,
                                 std::string_view table, Lsn replay_lsn) {
  HECTOR_RETURN_IF_ERROR(ValidateName("database", database));
  HECTOR_RETURN_IF_ERROR(ValidateName("table", table));
  HECTOR_RETURN_IF_ERROR(FindTable(database, table).status());
  if (replay_lsn == 0) {
    HECTOR_ASSIGN_OR_RETURN(const std::string payload,
                            EncodeDropTable(database, table));
    HECTOR_RETURN_IF_ERROR(
        LogOnline(TypeOf(WalRecordType::kDropTable), payload).status());
  }
  absl::Status applied = absl::OkStatus();
  {
    std::unique_lock<std::shared_mutex> lock(catalog_mu_);
    auto it = databases_.find(database);
    applied = it == databases_.end() ? DatabaseNotFound(database)
                                     : it->second.RemoveTable(table);
  }
  return CheckApplied(std::move(applied), replay_lsn);
}

absl::StatusOr<WriteResult> Engine::DoWriteRows(std::string_view database,
                                                std::string_view table,
                                                std::vector<RowInput> rows,
                                                bool upsert, Lsn replay_lsn) {
  HECTOR_RETURN_IF_ERROR(ValidateName("database", database));
  HECTOR_RETURN_IF_ERROR(ValidateName("table", table));
  HECTOR_ASSIGN_OR_RETURN(const std::shared_ptr<Table> target,
                          FindTable(database, table));
  const PrepareMode mode =
      replay_lsn == 0 ? PrepareMode::kOnline : PrepareMode::kReplay;
  HECTOR_ASSIGN_OR_RETURN(PreparedBatch batch,
                          target->PrepareInsert(std::move(rows), upsert, mode));
  Lsn lsn = replay_lsn;
  if (replay_lsn == 0) {
    HECTOR_ASSIGN_OR_RETURN(const std::string payload,
                            upsert ? EncodeUpsert(database, table, batch.rows)
                                   : EncodeInsert(database, table, batch.rows));
    const WalRecordType type =
        upsert ? WalRecordType::kUpsert : WalRecordType::kInsert;
    HECTOR_ASSIGN_OR_RETURN(lsn, LogOnline(TypeOf(type), payload));
  }
  const uint64_t affected = batch.rows.size();
  HECTOR_RETURN_IF_ERROR(
      CheckApplied(target->ApplyInsert(std::move(batch), lsn), replay_lsn));
  target->Publish(lsn);
  return WriteResult{.affected = affected, .lsn = lsn};
}

absl::StatusOr<WriteResult> Engine::DoDelete(
    std::string_view database, std::string_view table,
    std::span<const std::string> primary_keys, Lsn replay_lsn) {
  HECTOR_RETURN_IF_ERROR(ValidateName("database", database));
  HECTOR_RETURN_IF_ERROR(ValidateName("table", table));
  HECTOR_ASSIGN_OR_RETURN(const std::shared_ptr<Table> target,
                          FindTable(database, table));
  HECTOR_ASSIGN_OR_RETURN(const std::vector<RowId> row_ids,
                          target->PrepareDelete(primary_keys));
  if (row_ids.empty()) {
    if (replay_lsn != 0) {
      // 在线写入只为删到行的 Delete 写 WAL；回放时一行都没删到，说明 WAL 与
      // 内存不一致（外层包装为含该 LSN 的 DataLoss）。
      return absl::FailedPreconditionError(
          absl::StrCat("delete record matches no live row in table \"", table,
                       "\" of database \"", database, "\""));
    }
    // 一行都没删：不写 WAL，返回 lsn=0。
    return WriteResult{.affected = 0, .lsn = 0};
  }
  Lsn lsn = replay_lsn;
  if (replay_lsn == 0) {
    HECTOR_ASSIGN_OR_RETURN(const std::string payload,
                            EncodeDelete(database, table, primary_keys));
    HECTOR_ASSIGN_OR_RETURN(lsn,
                            LogOnline(TypeOf(WalRecordType::kDelete), payload));
  }
  HECTOR_RETURN_IF_ERROR(
      CheckApplied(target->ApplyDelete(row_ids, lsn), replay_lsn));
  target->Publish(lsn);
  return WriteResult{.affected = row_ids.size(), .lsn = lsn};
}

absl::Status Engine::CreateDatabase(std::string_view name) {
  std::lock_guard<std::mutex> lock(write_mu_);
  HECTOR_RETURN_IF_ERROR(CheckWritable());
  return DoCreateDatabase(name, 0);
}

absl::Status Engine::DropDatabase(std::string_view name) {
  std::lock_guard<std::mutex> lock(write_mu_);
  HECTOR_RETURN_IF_ERROR(CheckWritable());
  return DoDropDatabase(name, 0);
}

std::vector<std::string> Engine::ListDatabases() const {
  std::shared_lock<std::shared_mutex> lock(catalog_mu_);
  std::vector<std::string> names;
  names.reserve(databases_.size());
  for (const auto& [name, database] : databases_) {
    names.push_back(name);
  }
  return names;
}

absl::Status Engine::CreateTable(std::string_view database, Schema schema) {
  std::lock_guard<std::mutex> lock(write_mu_);
  HECTOR_RETURN_IF_ERROR(CheckWritable());
  return DoCreateTable(database, std::move(schema), 0);
}

absl::Status Engine::DropTable(std::string_view database,
                               std::string_view table) {
  std::lock_guard<std::mutex> lock(write_mu_);
  HECTOR_RETURN_IF_ERROR(CheckWritable());
  return DoDropTable(database, table, 0);
}

absl::StatusOr<std::vector<std::string>> Engine::ListTables(
    std::string_view database) const {
  HECTOR_RETURN_IF_ERROR(ValidateName("database", database));
  std::shared_lock<std::shared_mutex> lock(catalog_mu_);
  auto it = databases_.find(database);
  if (it == databases_.end()) {
    return DatabaseNotFound(database);
  }
  return it->second.TableNames();
}

absl::StatusOr<std::shared_ptr<const Table>> Engine::GetTable(
    std::string_view database, std::string_view table) const {
  HECTOR_RETURN_IF_ERROR(ValidateName("database", database));
  HECTOR_RETURN_IF_ERROR(ValidateName("table", table));
  HECTOR_ASSIGN_OR_RETURN(std::shared_ptr<Table> found,
                          FindTable(database, table));
  return std::shared_ptr<const Table>(std::move(found));
}

absl::StatusOr<WriteResult> Engine::Insert(std::string_view database,
                                           std::string_view table,
                                           std::vector<RowInput> rows) {
  std::lock_guard<std::mutex> lock(write_mu_);
  HECTOR_RETURN_IF_ERROR(CheckWritable());
  return DoWriteRows(database, table, std::move(rows), /*upsert=*/false, 0);
}

absl::StatusOr<WriteResult> Engine::Upsert(std::string_view database,
                                           std::string_view table,
                                           std::vector<RowInput> rows) {
  std::lock_guard<std::mutex> lock(write_mu_);
  HECTOR_RETURN_IF_ERROR(CheckWritable());
  return DoWriteRows(database, table, std::move(rows), /*upsert=*/true, 0);
}

absl::StatusOr<WriteResult> Engine::Delete(
    std::string_view database, std::string_view table,
    std::vector<std::string> primary_keys) {
  std::lock_guard<std::mutex> lock(write_mu_);
  HECTOR_RETURN_IF_ERROR(CheckWritable());
  return DoDelete(database, table, primary_keys, 0);
}

absl::StatusOr<Lsn> Engine::CreateSnapshot() {
  return absl::UnimplementedError("CreateSnapshot is not implemented in c0");
}

Lsn Engine::last_lsn() const { return wal_->last_lsn(); }

uint64_t Engine::replayed_records() const { return replayed_records_; }

}  // namespace hector
