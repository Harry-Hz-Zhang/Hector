#include "hector/storage/file_wal.h"

#include <gtest/gtest.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/types.h"
#include "hector/storage/wal.h"

namespace hector {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kFirstSegment = "wal-00000000000000000001.log";
constexpr std::string_view kThirdSegment = "wal-00000000000000000003.log";
constexpr std::string_view kFourthSegment = "wal-00000000000000000004.log";
constexpr size_t kSegmentHeaderSize = 6;
constexpr size_t kRecordHeaderSize = 14;

std::string ReadFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void WriteFile(const fs::path& path, const std::string& data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void AppendToFile(const fs::path& path, const std::string& data) {
  std::ofstream out(path, std::ios::binary | std::ios::app);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::vector<std::string> SegmentNames(const fs::path& dir) {
  std::vector<std::string> names;
  for (const fs::directory_entry& entry : fs::directory_iterator(dir)) {
    names.push_back(entry.path().filename().string());
  }
  std::sort(names.begin(), names.end());
  return names;
}

absl::StatusOr<std::vector<WalRecord>> Collect(Wal& wal, Lsn after) {
  std::vector<WalRecord> records;
  absl::Status status = wal.Replay(after, [&records](const WalRecord& r) {
    records.push_back(r);
    return absl::OkStatus();
  });
  if (!status.ok()) {
    return status;
  }
  return records;
}

// 把默认 logger 换成写入内存的 logger，析构时恢复原 logger。
class ScopedLogCapture {
 public:
  ScopedLogCapture() : previous_(spdlog::default_logger()) {
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(stream_);
    spdlog::set_default_logger(
        std::make_shared<spdlog::logger>(kName, std::move(sink)));
  }
  ~ScopedLogCapture() {
    spdlog::set_default_logger(previous_);
    spdlog::drop(kName);
  }
  ScopedLogCapture(const ScopedLogCapture&) = delete;
  ScopedLogCapture& operator=(const ScopedLogCapture&) = delete;

  std::string str() const { return stream_.str(); }

 private:
  static constexpr const char* kName = "file_wal_test_capture";
  std::ostringstream stream_;
  std::shared_ptr<spdlog::logger> previous_;
};

class FileWalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    root_ = fs::path(::testing::TempDir()) /
            (std::string("file_wal_test_") + info->name());
    fs::remove_all(root_);
    fs::create_directories(root_);
    dir_ = root_ / "wal";
  }

  void TearDown() override { fs::remove_all(root_); }

  std::unique_ptr<Wal> MustOpen(const fs::path& dir) {
    absl::StatusOr<std::unique_ptr<Wal>> wal = OpenFileWal(dir);
    EXPECT_TRUE(wal.ok()) << wal.status();
    return wal.ok() ? *std::move(wal) : nullptr;
  }

  // 在 dir 下写两条记录并关闭，返回段文件路径。
  fs::path WriteTwoRecords(const fs::path& dir) {
    std::unique_ptr<Wal> wal = MustOpen(dir);
    if (wal == nullptr) {
      return {};
    }
    EXPECT_TRUE(wal->Append(1, "first").ok());
    EXPECT_TRUE(wal->Append(2, "second").ok());
    EXPECT_TRUE(wal->SyncUpTo(2).ok());
    return dir / kFirstSegment;
  }

  fs::path root_;
  fs::path dir_;
};

TEST_F(FileWalTest, AppendAssignsIncreasingLsn) {
  std::unique_ptr<Wal> wal = MustOpen(dir_);
  ASSERT_NE(wal, nullptr);
  EXPECT_EQ(wal->last_lsn(), 0U);

  absl::StatusOr<Lsn> a = wal->Append(1, "a");
  absl::StatusOr<Lsn> b = wal->Append(2, "bb");
  absl::StatusOr<Lsn> c = wal->Append(3, "ccc");
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();
  ASSERT_TRUE(c.ok()) << c.status();
  EXPECT_EQ(*a, 1U);
  EXPECT_EQ(*b, 2U);
  EXPECT_EQ(*c, 3U);
  EXPECT_EQ(wal->last_lsn(), 3U);
  ASSERT_TRUE(wal->SyncUpTo(3).ok());

  const fs::path segment = dir_ / kFirstSegment;
  ASSERT_TRUE(fs::exists(segment));
  const std::string data = ReadFile(segment);
  ASSERT_GE(data.size(), kSegmentHeaderSize);
  EXPECT_EQ(data.substr(0, kSegmentHeaderSize),
            std::string("\x48\x57\x41\x4C\x01\x00", kSegmentHeaderSize));
}

TEST_F(FileWalTest, SyncUpToMakesRecordsReadable) {
  std::unique_ptr<Wal> wal = MustOpen(dir_);
  ASSERT_NE(wal, nullptr);
  ASSERT_TRUE(wal->Append(7, "x").ok());
  ASSERT_TRUE(wal->Append(8, "y").ok());
  ASSERT_TRUE(wal->SyncUpTo(2).ok());

  {
    std::unique_ptr<Wal> reader = MustOpen(dir_);
    ASSERT_NE(reader, nullptr);
    absl::StatusOr<std::vector<WalRecord>> records = Collect(*reader, 0);
    ASSERT_TRUE(records.ok()) << records.status();
    ASSERT_EQ(records->size(), 2U);
    EXPECT_EQ((*records)[0].payload, "x");
    EXPECT_EQ((*records)[1].payload, "y");
  }

  // SyncUpTo(0) 不刷盘：第三条记录仍留在用户态缓冲里，另开的实例看不到它。
  ASSERT_TRUE(wal->Append(9, "z").ok());
  EXPECT_TRUE(wal->SyncUpTo(0).ok());
  std::unique_ptr<Wal> reader = MustOpen(dir_);
  ASSERT_NE(reader, nullptr);
  EXPECT_EQ(reader->last_lsn(), 2U);
}

TEST_F(FileWalTest, ReplayAfterReopen) {
  {
    std::unique_ptr<Wal> wal = MustOpen(dir_);
    ASSERT_NE(wal, nullptr);
    ASSERT_TRUE(wal->Append(10, "alpha").ok());
    ASSERT_TRUE(wal->Append(11, "").ok());
    ASSERT_TRUE(wal->Append(12, std::string("g\0a", 3)).ok());
    ASSERT_TRUE(wal->SyncUpTo(3).ok());
  }

  std::unique_ptr<Wal> wal = MustOpen(dir_);
  ASSERT_NE(wal, nullptr);
  EXPECT_EQ(wal->last_lsn(), 3U);
  absl::StatusOr<std::vector<WalRecord>> all = Collect(*wal, 0);
  ASSERT_TRUE(all.ok()) << all.status();
  ASSERT_EQ(all->size(), 3U);
  EXPECT_EQ((*all)[0].lsn, 1U);
  EXPECT_EQ((*all)[0].type, 10U);
  EXPECT_EQ((*all)[0].payload, "alpha");
  EXPECT_EQ((*all)[1].lsn, 2U);
  EXPECT_EQ((*all)[1].type, 11U);
  EXPECT_EQ((*all)[1].payload, "");
  EXPECT_EQ((*all)[2].lsn, 3U);
  EXPECT_EQ((*all)[2].type, 12U);
  EXPECT_EQ((*all)[2].payload, std::string("g\0a", 3));

  absl::StatusOr<std::vector<WalRecord>> tail = Collect(*wal, 2);
  ASSERT_TRUE(tail.ok()) << tail.status();
  ASSERT_EQ(tail->size(), 1U);
  EXPECT_EQ((*tail)[0].lsn, 3U);

  // 重开后继续追加，LSN 接着 last_lsn + 1。
  absl::StatusOr<Lsn> next = wal->Append(13, "delta");
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_EQ(*next, 4U);
}

TEST_F(FileWalTest, RotateStartsNewSegment) {
  {
    std::unique_ptr<Wal> wal = MustOpen(dir_);
    ASSERT_NE(wal, nullptr);
    ASSERT_TRUE(wal->Append(1, "a").ok());
    ASSERT_TRUE(wal->Append(1, "b").ok());
    ASSERT_TRUE(wal->Rotate().ok());
    ASSERT_TRUE(wal->Rotate().ok());  // 当前段为空，不开新段。
    ASSERT_TRUE(wal->Append(1, "c").ok());
    ASSERT_TRUE(wal->SyncUpTo(3).ok());
    EXPECT_EQ(SegmentNames(dir_),
              (std::vector<std::string>{std::string(kFirstSegment),
                                        std::string(kThirdSegment)}));
  }

  {
    std::unique_ptr<Wal> wal = MustOpen(dir_);
    ASSERT_NE(wal, nullptr);
    absl::StatusOr<std::vector<WalRecord>> records = Collect(*wal, 0);
    ASSERT_TRUE(records.ok()) << records.status();
    ASSERT_EQ(records->size(), 3U);
    for (size_t i = 0; i < records->size(); ++i) {
      EXPECT_EQ((*records)[i].lsn, i + 1);
    }
    EXPECT_EQ((*records)[2].payload, "c");

    ASSERT_TRUE(wal->Rotate().ok());  // 第三段有记录：开第四段。
    EXPECT_EQ(SegmentNames(dir_).size(), 3U);
    ASSERT_TRUE(wal->Rotate().ok());
    ASSERT_TRUE(wal->Rotate().ok());
    EXPECT_EQ(SegmentNames(dir_).size(), 3U);
  }

  // 末尾是一个只有段头的空段：重开后 LSN 仍连续，新记录写进该段。
  {
    std::unique_ptr<Wal> wal = MustOpen(dir_);
    ASSERT_NE(wal, nullptr);
    EXPECT_EQ(wal->last_lsn(), 3U);
    absl::StatusOr<Lsn> lsn = wal->Append(1, "d");
    ASSERT_TRUE(lsn.ok()) << lsn.status();
    EXPECT_EQ(*lsn, 4U);
    ASSERT_TRUE(wal->SyncUpTo(4).ok());
  }
  EXPECT_GT(fs::file_size(dir_ / kFourthSegment), kSegmentHeaderSize);
  std::unique_ptr<Wal> wal = MustOpen(dir_);
  ASSERT_NE(wal, nullptr);
  absl::StatusOr<std::vector<WalRecord>> records = Collect(*wal, 0);
  ASSERT_TRUE(records.ok()) << records.status();
  ASSERT_EQ(records->size(), 4U);
  EXPECT_EQ((*records)[3].lsn, 4U);
  EXPECT_EQ((*records)[3].payload, "d");
}

TEST_F(FileWalTest, TruncatesTornTail) {
  const fs::path segment = WriteTwoRecords(dir_);
  const uintmax_t complete_size = fs::file_size(segment);
  AppendToFile(segment, std::string(5, '\xFF'));
  ASSERT_EQ(fs::file_size(segment), complete_size + 5);

  ScopedLogCapture capture;
  absl::StatusOr<std::unique_ptr<Wal>> wal = OpenFileWal(dir_);
  ASSERT_TRUE(wal.ok()) << wal.status();
  absl::StatusOr<std::vector<WalRecord>> records = Collect(**wal, 0);
  ASSERT_TRUE(records.ok()) << records.status();
  EXPECT_EQ(records->size(), 2U);
  EXPECT_EQ(fs::file_size(segment), complete_size);
  EXPECT_NE(capture.str().find("truncated 5 bytes"), std::string::npos)
      << capture.str();
}

TEST_F(FileWalTest, RejectsBadHeaderOrLsnGap) {
  const std::string name(kFirstSegment);

  const fs::path bad_magic_dir = root_ / "bad_magic";
  const fs::path bad_magic = WriteTwoRecords(bad_magic_dir);
  std::string data = ReadFile(bad_magic);
  ASSERT_GE(data.size(), kSegmentHeaderSize);
  data.replace(0, 4, "XXXX");
  WriteFile(bad_magic, data);
  absl::StatusOr<std::unique_ptr<Wal>> wal = OpenFileWal(bad_magic_dir);
  EXPECT_EQ(wal.status().code(), absl::StatusCode::kDataLoss) << wal.status();
  EXPECT_NE(wal.status().message().find(name), std::string_view::npos)
      << wal.status();

  const fs::path bad_version_dir = root_ / "bad_version";
  const fs::path bad_version = WriteTwoRecords(bad_version_dir);
  data = ReadFile(bad_version);
  ASSERT_GE(data.size(), kSegmentHeaderSize);
  data[4] = '\x02';
  WriteFile(bad_version, data);
  wal = OpenFileWal(bad_version_dir);
  EXPECT_EQ(wal.status().code(), absl::StatusCode::kDataLoss) << wal.status();
  EXPECT_NE(wal.status().message().find(name), std::string_view::npos)
      << wal.status();

  const fs::path gap_dir = root_ / "gap";
  const fs::path gap = WriteTwoRecords(gap_dir);
  data = ReadFile(gap);
  // 第二条记录的 LSN 字段：段头、第一条记录（头 + "first"）、长度字段之后。
  const size_t lsn_offset =
      kSegmentHeaderSize + kRecordHeaderSize + std::string("first").size() + 4;
  ASSERT_GT(data.size(), lsn_offset);
  ASSERT_EQ(static_cast<uint8_t>(data[lsn_offset]), 2U);
  data[lsn_offset] = '\x05';
  WriteFile(gap, data);
  wal = OpenFileWal(gap_dir);
  EXPECT_EQ(wal.status().code(), absl::StatusCode::kDataLoss) << wal.status();
  EXPECT_NE(wal.status().message().find(name), std::string_view::npos)
      << wal.status();
}

TEST_F(FileWalTest, RejectsCorruptMiddleSegment) {
  {
    std::unique_ptr<Wal> wal = MustOpen(dir_);
    ASSERT_NE(wal, nullptr);
    ASSERT_TRUE(wal->Append(1, "first").ok());
    ASSERT_TRUE(wal->Append(1, "second").ok());
    ASSERT_TRUE(wal->Rotate().ok());
    ASSERT_TRUE(wal->Append(1, "third").ok());
    ASSERT_TRUE(wal->SyncUpTo(3).ok());
  }
  const fs::path first = dir_ / kFirstSegment;
  fs::resize_file(first, fs::file_size(first) - 3);

  absl::StatusOr<std::unique_ptr<Wal>> wal = OpenFileWal(dir_);
  EXPECT_EQ(wal.status().code(), absl::StatusCode::kDataLoss) << wal.status();
  EXPECT_NE(wal.status().message().find(kFirstSegment), std::string_view::npos)
      << wal.status();
}

TEST_F(FileWalTest, WriteErrorEntersFailedState) {
  int writes = 0;
  bool fail = false;
  FileWalOptions options;
  options.write_override =
      [&writes, &fail](std::span<const std::byte> /*data*/) -> absl::Status {
    ++writes;
    if (fail) {
      return absl::InternalError("injected write failure");
    }
    return absl::OkStatus();
  };

  absl::StatusOr<std::unique_ptr<Wal>> opened = OpenFileWal(dir_, options);
  ASSERT_TRUE(opened.ok()) << opened.status();
  Wal& wal = **opened;
  EXPECT_EQ(writes, 1);  // 段头。

  ASSERT_TRUE(wal.Append(1, "ok").ok());
  EXPECT_EQ(writes, 2);
  ASSERT_TRUE(wal.SyncUpTo(1).ok());
  EXPECT_EQ(writes, 2);  // fflush 不经过 write_override。

  fail = true;
  absl::StatusOr<Lsn> failed = wal.Append(1, "boom");
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kInternal)
      << failed.status();
  EXPECT_EQ(writes, 3);

  fail = false;
  EXPECT_EQ(wal.Append(1, "after").status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(wal.SyncUpTo(1).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(writes, 3);
  EXPECT_EQ(wal.last_lsn(), 1U);
}

TEST_F(FileWalTest, VisitorErrorStopsReplay) {
  std::unique_ptr<Wal> wal = MustOpen(dir_);
  ASSERT_NE(wal, nullptr);
  ASSERT_TRUE(wal->Append(1, "a").ok());
  ASSERT_TRUE(wal->Append(1, "b").ok());
  ASSERT_TRUE(wal->Append(1, "c").ok());
  ASSERT_TRUE(wal->SyncUpTo(3).ok());

  int calls = 0;
  absl::Status status = wal->Replay(0, [&calls](const WalRecord& /*r*/) {
    ++calls;
    if (calls == 2) {
      return absl::InternalError("visitor failed");
    }
    return absl::OkStatus();
  });
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal) << status;
  EXPECT_EQ(status.message(), "visitor failed");
  EXPECT_EQ(calls, 2);
}

TEST_F(FileWalTest, TruncateBeforeIsUnimplemented) {
  std::unique_ptr<Wal> wal = MustOpen(dir_);
  ASSERT_NE(wal, nullptr);
  EXPECT_EQ(wal->TruncateBefore(1).code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace hector
