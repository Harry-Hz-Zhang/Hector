#include "hector/storage/file_wal.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "hector/common/byte_io.h"
#include "hector/common/status.h"
#include "hector/common/types.h"
#include "hector/storage/wal.h"

namespace hector {
namespace {

namespace fs = std::filesystem;

// 段头：魔数 HWAL + u16 版本 1（小端）。
constexpr std::string_view kSegmentHeader("\x48\x57\x41\x4C\x01\x00", 6);
constexpr size_t kMagicSize = 4;
// 记录头：u32 payload 长度、u64 lsn、u16 type。
constexpr size_t kRecordHeaderSize = 4 + 8 + 2;
constexpr std::string_view kSegmentPrefix = "wal-";
constexpr std::string_view kSegmentSuffix = ".log";
constexpr size_t kLsnDigits = 20;

struct Segment {
  Lsn first_lsn;
  fs::path path;
};

std::string SegmentFileName(Lsn first_lsn) {
  const std::string digits = std::to_string(first_lsn);
  return absl::StrCat(kSegmentPrefix,
                      std::string(kLsnDigits - digits.size(), '0'), digits,
                      kSegmentSuffix);
}

// 文件名形如 wal-<20 位十进制>.log 时返回 true，并在 digits 中给出数字部分。
bool MatchSegmentName(std::string_view name, std::string_view& digits) {
  if (name.size() !=
          kSegmentPrefix.size() + kLsnDigits + kSegmentSuffix.size() ||
      !name.starts_with(kSegmentPrefix) || !name.ends_with(kSegmentSuffix)) {
    return false;
  }
  digits = name.substr(kSegmentPrefix.size(), kLsnDigits);
  return std::all_of(digits.begin(), digits.end(),
                     [](char c) { return c >= '0' && c <= '9'; });
}

// 列出 dir 下的段文件，按首条 LSN 升序。不符合命名的文件被忽略。
absl::StatusOr<std::vector<Segment>> ListSegments(const fs::path& dir) {
  std::vector<Segment> segments;
  std::error_code ec;
  fs::directory_iterator it(dir, ec);
  if (ec) {
    return absl::InternalError(
        absl::StrCat("cannot list wal dir ", dir.string(), ": ", ec.message()));
  }
  for (const fs::directory_entry& entry : it) {
    const std::string name = entry.path().filename().string();
    std::string_view digits;
    if (!entry.is_regular_file() || !MatchSegmentName(name, digits)) {
      continue;
    }
    Lsn first_lsn = 0;
    // 20 位十进制可能超出 u64，SimpleAtoi 溢出时返回 false。
    if (!absl::SimpleAtoi(digits, &first_lsn) || first_lsn == 0) {
      return absl::DataLossError(
          absl::StrCat("wal segment ", name, ": invalid first lsn in name"));
    }
    segments.push_back(Segment{first_lsn, entry.path()});
  }
  std::sort(segments.begin(), segments.end(),
            [](const Segment& a, const Segment& b) {
              return a.first_lsn < b.first_lsn;
            });
  return segments;
}

absl::StatusOr<std::string> ReadWholeFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return absl::InternalError(
        absl::StrCat("cannot open wal segment ", path.string()));
  }
  std::string data{std::istreambuf_iterator<char>(in),
                   std::istreambuf_iterator<char>()};
  if (in.bad()) {
    return absl::InternalError(
        absl::StrCat("cannot read wal segment ", path.string()));
  }
  return data;
}

// 解析一个段。next_lsn 输入该段首条记录应有的 LSN，输出下一条应有的 LSN。
// 完整记录逐条交给 on_record，它返回非 OK 时立即停止并返回该状态。
// valid_end 输出最后一条完整记录的结尾偏移；段头不完整时为 0。
// allow_torn_tail 为 false 时任何不完整的数据都是 DataLoss；为 true
// 时不完整的尾部（含只写了一部分的段头）不算错误，由调用方决定如何处理。
absl::Status ParseSegment(std::string_view name, std::string_view data,
                          bool allow_torn_tail, Lsn& next_lsn,
                          size_t& valid_end, const WalVisitor& on_record) {
  valid_end = 0;
  if (data.size() < kSegmentHeader.size()) {
    if (allow_torn_tail && kSegmentHeader.starts_with(data)) {
      return absl::OkStatus();
    }
    return absl::DataLossError(
        absl::StrCat("wal segment ", name, ": incomplete segment header"));
  }
  if (data.substr(0, kMagicSize) != kSegmentHeader.substr(0, kMagicSize)) {
    return absl::DataLossError(
        absl::StrCat("wal segment ", name, ": bad magic"));
  }
  if (data.substr(kMagicSize, kSegmentHeader.size() - kMagicSize) !=
      kSegmentHeader.substr(kMagicSize)) {
    return absl::DataLossError(
        absl::StrCat("wal segment ", name, ": unsupported version"));
  }

  size_t pos = kSegmentHeader.size();
  valid_end = pos;
  while (pos < data.size()) {
    const size_t remaining = data.size() - pos;
    bool complete = remaining >= kRecordHeaderSize;
    uint32_t length = 0;
    Lsn lsn = 0;
    uint16_t type = 0;
    if (complete) {
      StringByteReader reader(data.substr(pos, kRecordHeaderSize));
      HECTOR_ASSIGN_OR_RETURN(length, reader.ReadU32());
      HECTOR_ASSIGN_OR_RETURN(lsn, reader.ReadU64());
      HECTOR_ASSIGN_OR_RETURN(type, reader.ReadU16());
      complete = remaining - kRecordHeaderSize >= length;
    }
    if (!complete) {
      if (allow_torn_tail) {
        return absl::OkStatus();
      }
      return absl::DataLossError(absl::StrCat(
          "wal segment ", name, ": incomplete record at offset ", pos));
    }
    if (lsn != next_lsn) {
      return absl::DataLossError(absl::StrCat("wal segment ", name,
                                              ": lsn gap, expected ", next_lsn,
                                              " but found ", lsn));
    }
    const WalRecord record{
        lsn, type, std::string(data.substr(pos + kRecordHeaderSize, length))};
    pos += kRecordHeaderSize + length;
    valid_end = pos;
    ++next_lsn;
    HECTOR_RETURN_IF_ERROR(on_record(record));
  }
  return absl::OkStatus();
}

// 扫描结果：next_lsn 为最后一条完整记录的 LSN + 1；last_valid_end 与
// last_file_size 描述最后一个段，前者小于后者说明最后一段有不完整的尾部。
struct ScanResult {
  Lsn next_lsn = 0;
  size_t last_valid_end = 0;
  size_t last_file_size = 0;
};

// 依次读取并解析所有段（segments 非空）：每段的首条 LSN 必须紧接前一段，
// 只有最后一段允许不完整的尾部。
absl::StatusOr<ScanResult> ScanSegments(const std::vector<Segment>& segments,
                                        const WalVisitor& on_record) {
  ScanResult result;
  for (size_t i = 0; i < segments.size(); ++i) {
    const Segment& segment = segments[i];
    const std::string name = segment.path.filename().string();
    if (i == 0) {
      result.next_lsn = segment.first_lsn;
    } else if (segment.first_lsn != result.next_lsn) {
      return absl::DataLossError(
          absl::StrCat("wal segment ", name, ": lsn gap, expected first lsn ",
                       result.next_lsn, " but found ", segment.first_lsn));
    }
    HECTOR_ASSIGN_OR_RETURN(const std::string data,
                            ReadWholeFile(segment.path));
    const bool is_last = i + 1 == segments.size();
    size_t valid_end = 0;
    HECTOR_RETURN_IF_ERROR(ParseSegment(name, data, is_last, result.next_lsn,
                                        valid_end, on_record));
    if (is_last) {
      result.last_valid_end = valid_end;
      result.last_file_size = data.size();
    }
  }
  return result;
}

// 把 errno 转成状态。errno 为 0 时 ErrnoToStatus 会返回 OK，而调用方已判定
// 失败，因此这种情况返回 Internal，保证结果总是非 OK。
absl::Status ErrnoStatus(int error, std::string_view what,
                         const fs::path& path) {
  const std::string message = absl::StrCat(what, " ", path.string());
  if (error == 0) {
    return absl::InternalError(
        absl::StrCat(message, ": unknown error (errno=0)"));
  }
  return absl::ErrnoToStatus(error, message);
}

class FileWal final : public Wal {
 public:
  FileWal(fs::path dir, FileWalOptions options)
      : dir_(std::move(dir)), options_(std::move(options)) {}

  // 关闭当前段（fclose 会把缓冲里的记录交给操作系统）；析构时无法上报错误。
  ~FileWal() override {
    if (file_ != nullptr) {
      static_cast<void>(std::fclose(file_));
    }
  }

  FileWal(const FileWal&) = delete;
  FileWal& operator=(const FileWal&) = delete;

  // 校验已有段、截断最后一段的不完整尾部并打开当前段；只在发布前调用一次。
  absl::Status Init();

  absl::StatusOr<Lsn> Append(uint16_t type, std::string_view payload) override;
  absl::Status SyncUpTo(Lsn lsn) override;
  absl::Status Replay(Lsn after, const WalVisitor& visit) override;
  absl::Status Rotate() override;
  absl::Status TruncateBefore(Lsn lsn) override;
  Lsn last_lsn() const override {
    return last_lsn_.load(std::memory_order_acquire);
  }

 private:
  // 以下 *Locked 函数要求调用方持有 mu_（Init 期间对象尚未发布，可不加锁）。
  // 写入当前段：经 write_override 或 fwrite；失败时进入失败态。
  absl::Status WriteLocked(std::span<const std::byte> data);
  // 新建段文件（已存在则失败）并写入段头，成为当前段。
  absl::Status CreateSegmentLocked(Lsn first_lsn);
  absl::Status FailedStateError() const {
    return absl::FailedPreconditionError(
        absl::StrCat("wal ", dir_.string(), " is in failed state"));
  }

  const fs::path dir_;
  const FileWalOptions options_;

  std::mutex mu_;
  std::FILE* file_ = nullptr;      // 当前段，即 segments_.back()。
  std::vector<Segment> segments_;  // 按首条 LSN 升序。
  bool failed_ = false;
  Lsn synced_lsn_ = 0;  // 已 fflush 到操作系统的最大 LSN。
  // 写入只在持有 mu_ 时发生；last_lsn() 不加锁读取。
  std::atomic<Lsn> last_lsn_{0};
};

absl::Status FileWal::Init() {
  std::error_code ec;
  fs::create_directories(dir_, ec);
  if (ec) {
    return absl::InternalError(absl::StrCat("cannot create wal dir ",
                                            dir_.string(), ": ", ec.message()));
  }
  HECTOR_ASSIGN_OR_RETURN(segments_, ListSegments(dir_));
  if (segments_.empty()) {
    return CreateSegmentLocked(1);
  }

  HECTOR_ASSIGN_OR_RETURN(const ScanResult scan,
                          ScanSegments(segments_, [](const WalRecord&) {
                            return absl::OkStatus();
                          }));
  const Segment& last = segments_.back();
  if (scan.last_valid_end < scan.last_file_size) {
    fs::resize_file(last.path, scan.last_valid_end, ec);
    if (ec) {
      return absl::InternalError(absl::StrCat("cannot truncate wal segment ",
                                              last.path.string(), ": ",
                                              ec.message()));
    }
    spdlog::warn("wal segment {}: torn tail, truncated {} bytes",
                 last.path.filename().string(),
                 scan.last_file_size - scan.last_valid_end);
  }
  last_lsn_.store(scan.next_lsn - 1, std::memory_order_release);
  synced_lsn_ = scan.next_lsn - 1;

  // 段头不完整（例如开新段后、段头交给操作系统前崩溃）时已截断为空，重写段头。
  const bool rewrite_header = scan.last_valid_end == 0;
  file_ = std::fopen(last.path.c_str(), rewrite_header ? "wb" : "ab");
  if (file_ == nullptr) {
    return ErrnoStatus(errno, "cannot open wal segment", last.path);
  }
  if (rewrite_header) {
    return WriteLocked(std::as_bytes(std::span(kSegmentHeader)));
  }
  return absl::OkStatus();
}

absl::Status FileWal::WriteLocked(std::span<const std::byte> data) {
  absl::Status status;
  if (options_.write_override) {
    status = options_.write_override(data);
  } else if (std::fwrite(data.data(), 1, data.size(), file_) != data.size()) {
    status =
        ErrnoStatus(errno, "cannot write wal segment", segments_.back().path);
  }
  if (!status.ok()) {
    failed_ = true;
  }
  return status;
}

absl::Status FileWal::CreateSegmentLocked(Lsn first_lsn) {
  fs::path path = dir_ / SegmentFileName(first_lsn);
  file_ = std::fopen(path.c_str(), "wbx");
  if (file_ == nullptr) {
    failed_ = true;
    return ErrnoStatus(errno, "cannot create wal segment", path);
  }
  segments_.push_back(Segment{first_lsn, std::move(path)});
  return WriteLocked(std::as_bytes(std::span(kSegmentHeader)));
}

absl::StatusOr<Lsn> FileWal::Append(uint16_t type, std::string_view payload) {
  if (payload.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError(
        absl::StrCat("wal payload too large: ", payload.size(), " bytes"));
  }
  std::lock_guard<std::mutex> lock(mu_);
  if (failed_) {
    return FailedStateError();
  }
  const Lsn lsn = last_lsn_.load(std::memory_order_relaxed) + 1;
  // 整条记录先编码到内存，再一次写入，失败时不会只写出记录头。
  StringByteWriter record;
  HECTOR_RETURN_IF_ERROR(
      record.WriteU32(static_cast<uint32_t>(payload.size())));
  HECTOR_RETURN_IF_ERROR(record.WriteU64(lsn));
  HECTOR_RETURN_IF_ERROR(record.WriteU16(type));
  HECTOR_RETURN_IF_ERROR(record.WriteBytes(std::as_bytes(std::span(payload))));
  HECTOR_RETURN_IF_ERROR(WriteLocked(std::as_bytes(std::span(record.data()))));
  last_lsn_.store(lsn, std::memory_order_release);
  return lsn;
}

absl::Status FileWal::SyncUpTo(Lsn lsn) {
  std::lock_guard<std::mutex> lock(mu_);
  if (failed_) {
    return FailedStateError();
  }
  // 已被之前的刷盘（可能来自其他线程）覆盖；lsn == 0 也走这里。
  if (lsn <= synced_lsn_) {
    return absl::OkStatus();
  }
  if (std::fflush(file_) != 0) {
    failed_ = true;
    return ErrnoStatus(errno, "cannot flush wal segment",
                       segments_.back().path);
  }
  synced_lsn_ = last_lsn_.load(std::memory_order_relaxed);
  return absl::OkStatus();
}

absl::Status FileWal::Replay(Lsn after, const WalVisitor& visit) {
  std::vector<Segment> segments;
  Lsn upto = 0;
  {
    // 先把缓冲里的记录交给操作系统，回放才能读到全部已追加的记录。
    // 回调期间不持锁，访问者可以调用本对象的其他方法。
    std::lock_guard<std::mutex> lock(mu_);
    if (!failed_) {
      if (std::fflush(file_) != 0) {
        failed_ = true;
        return ErrnoStatus(errno, "cannot flush wal segment",
                           segments_.back().path);
      }
      synced_lsn_ = last_lsn_.load(std::memory_order_relaxed);
    }
    segments = segments_;
    upto = last_lsn_.load(std::memory_order_relaxed);
  }
  return ScanSegments(segments,
                      [after, upto, &visit](const WalRecord& record) {
                        if (record.lsn <= after || record.lsn > upto) {
                          return absl::OkStatus();
                        }
                        return visit(record);
                      })
      .status();
}

absl::Status FileWal::Rotate() {
  std::lock_guard<std::mutex> lock(mu_);
  if (failed_) {
    return FailedStateError();
  }
  const Lsn next_lsn = last_lsn_.load(std::memory_order_relaxed) + 1;
  if (segments_.back().first_lsn == next_lsn) {
    return absl::OkStatus();  // 当前段没有记录。
  }
  const int rc = std::fclose(file_);
  file_ = nullptr;
  if (rc != 0) {
    failed_ = true;
    return ErrnoStatus(errno, "cannot close wal segment",
                       segments_.back().path);
  }
  synced_lsn_ = next_lsn - 1;
  return CreateSegmentLocked(next_lsn);
}

absl::Status FileWal::TruncateBefore(Lsn /*lsn*/) {
  return absl::UnimplementedError("wal TruncateBefore is not supported in c0");
}

}  // namespace

absl::StatusOr<std::unique_ptr<Wal>> OpenFileWal(const fs::path& dir) {
  return OpenFileWal(dir, FileWalOptions{});
}

absl::StatusOr<std::unique_ptr<Wal>> OpenFileWal(
    const fs::path& dir, const FileWalOptions& options) {
  auto wal = std::make_unique<FileWal>(dir, options);
  HECTOR_RETURN_IF_ERROR(wal->Init());
  return wal;
}

}  // namespace hector
