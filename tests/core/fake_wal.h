#ifndef HECTOR_TESTS_CORE_FAKE_WAL_H_
#define HECTOR_TESTS_CORE_FAKE_WAL_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "hector/common/byte_io.h"
#include "hector/common/types.h"
#include "hector/storage/snapshot.h"
#include "hector/storage/wal.h"

namespace hector {
namespace test_support {

// 跨对象的调用顺序记录（FakeWal 与 FakeSnapshotStore 共用），线程安全。
// 事件形如 "Latest"、"Replay(0)"、"Append(5)"、"SyncUpTo(3)"。
class CallLog {
 public:
  void Record(std::string event) {
    std::lock_guard<std::mutex> lock(mu_);
    events_.push_back(std::move(event));
  }
  std::vector<std::string> events() const {
    std::lock_guard<std::mutex> lock(mu_);
    return events_;
  }

 private:
  mutable std::mutex mu_;
  std::vector<std::string> events_;
};

// 一次成功的 Append：参数与分配到的 LSN。
struct AppendCall {
  uint16_t type;
  std::string payload;
  Lsn lsn;
};

// 内存 Wal：记录 Append 与 SyncUpTo 的调用次数、参数与先后顺序，可以一次性
// 注入失败，可以在 Append 与 SyncUpTo 内部执行测试提供的钩子；Replay 回放
// Preload 的记录与成功 Append 的记录。所有方法线程安全；钩子在不持有内部锁
// 的情况下执行。
class FakeWal final : public Wal {
 public:
  explicit FakeWal(std::shared_ptr<CallLog> log = std::make_shared<CallLog>())
      : log_(std::move(log)) {}

  // 预置记录；必须在任何 Append 之前调用，LSN 必须从 1 起逐条加 1。
  void Preload(std::vector<WalRecord> records) {
    std::lock_guard<std::mutex> lock(mu_);
    records_ = std::move(records);
    last_lsn_ = records_.empty() ? 0 : records_.back().lsn;
  }

  // 下一次 Append（或 SyncUpTo）返回 status 且不产生任何效果；只生效一次。
  void FailNextAppend(absl::Status status) {
    std::lock_guard<std::mutex> lock(mu_);
    append_error_ = std::move(status);
  }
  void FailNextSync(absl::Status status) {
    std::lock_guard<std::mutex> lock(mu_);
    sync_error_ = std::move(status);
  }

  // 在每次 Append（参数为 type）与 SyncUpTo（参数为 lsn）开始时执行。
  void set_on_append(std::function<void(uint16_t)> hook) {
    std::lock_guard<std::mutex> lock(mu_);
    on_append_ = std::move(hook);
  }
  void set_on_sync(std::function<void(Lsn)> hook) {
    std::lock_guard<std::mutex> lock(mu_);
    on_sync_ = std::move(hook);
  }

  absl::StatusOr<Lsn> Append(uint16_t type, std::string_view payload) override {
    std::function<void(uint16_t)> hook;
    {
      std::lock_guard<std::mutex> lock(mu_);
      ++append_calls_;
      hook = on_append_;
    }
    if (hook) {
      hook(type);
    }
    std::lock_guard<std::mutex> lock(mu_);
    log_->Record(absl::StrCat("Append(", type, ")"));
    if (append_error_.has_value()) {
      absl::Status error = *std::move(append_error_);
      append_error_.reset();
      return error;
    }
    const Lsn lsn = ++last_lsn_;
    records_.push_back(
        WalRecord{.lsn = lsn, .type = type, .payload = std::string(payload)});
    appends_.push_back(
        AppendCall{.type = type, .payload = std::string(payload), .lsn = lsn});
    return lsn;
  }

  absl::Status SyncUpTo(Lsn lsn) override {
    std::function<void(Lsn)> hook;
    {
      std::lock_guard<std::mutex> lock(mu_);
      ++sync_calls_;
      hook = on_sync_;
    }
    if (hook) {
      hook(lsn);
    }
    std::lock_guard<std::mutex> lock(mu_);
    log_->Record(absl::StrCat("SyncUpTo(", lsn, ")"));
    synced_lsns_.push_back(lsn);
    if (sync_error_.has_value()) {
      absl::Status error = *std::move(sync_error_);
      sync_error_.reset();
      return error;
    }
    return absl::OkStatus();
  }

  absl::Status Replay(Lsn after, const WalVisitor& visit) override {
    std::vector<WalRecord> records;
    {
      std::lock_guard<std::mutex> lock(mu_);
      log_->Record(absl::StrCat("Replay(", after, ")"));
      records = records_;
    }
    for (const WalRecord& record : records) {
      if (record.lsn <= after) {
        continue;
      }
      if (absl::Status s = visit(record); !s.ok()) {
        return s;
      }
    }
    return absl::OkStatus();
  }

  absl::Status Rotate() override { return absl::OkStatus(); }
  absl::Status TruncateBefore(Lsn /*lsn*/) override {
    return absl::UnimplementedError("FakeWal::TruncateBefore");
  }
  Lsn last_lsn() const override {
    std::lock_guard<std::mutex> lock(mu_);
    return last_lsn_;
  }

  // 调用次数包括注入失败的那一次。
  int append_calls() const {
    std::lock_guard<std::mutex> lock(mu_);
    return append_calls_;
  }
  int sync_calls() const {
    std::lock_guard<std::mutex> lock(mu_);
    return sync_calls_;
  }
  // 成功的 Append，按调用顺序。
  std::vector<AppendCall> appends() const {
    std::lock_guard<std::mutex> lock(mu_);
    return appends_;
  }
  // 每次 SyncUpTo 的参数，按调用顺序。
  std::vector<Lsn> synced_lsns() const {
    std::lock_guard<std::mutex> lock(mu_);
    return synced_lsns_;
  }
  // 预置记录加成功 Append 的记录，可以 Preload 到另一个 FakeWal 模拟重启。
  std::vector<WalRecord> records() const {
    std::lock_guard<std::mutex> lock(mu_);
    return records_;
  }
  const std::shared_ptr<CallLog>& log() const { return log_; }

 private:
  std::shared_ptr<CallLog> log_;

  mutable std::mutex mu_;
  std::vector<WalRecord> records_;
  Lsn last_lsn_ = 0;
  int append_calls_ = 0;
  int sync_calls_ = 0;
  std::vector<AppendCall> appends_;
  std::vector<Lsn> synced_lsns_;
  std::optional<absl::Status> append_error_;
  std::optional<absl::Status> sync_error_;
  std::function<void(uint16_t)> on_append_;
  std::function<void(Lsn)> on_sync_;
};

// 没有快照的 SnapshotStore：Latest 记录 "Latest" 并返回 nullopt。
class FakeSnapshotStore final : public SnapshotStore {
 public:
  explicit FakeSnapshotStore(std::shared_ptr<CallLog> log)
      : log_(std::move(log)) {}

  absl::Status Write(
      Lsn /*lsn*/,
      const std::function<absl::Status(ByteWriter&)>& /*body*/) override {
    log_->Record("Write");
    return absl::UnimplementedError("FakeSnapshotStore::Write");
  }
  absl::StatusOr<std::optional<Lsn>> Latest() const override {
    log_->Record("Latest");
    return std::optional<Lsn>();
  }
  absl::Status Read(
      Lsn /*lsn*/,
      const std::function<absl::Status(ByteReader&)>& /*body*/) override {
    log_->Record("Read");
    return absl::UnimplementedError("FakeSnapshotStore::Read");
  }

 private:
  std::shared_ptr<CallLog> log_;
};

}  // namespace test_support
}  // namespace hector

#endif  // HECTOR_TESTS_CORE_FAKE_WAL_H_
