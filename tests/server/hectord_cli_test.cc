// 以子进程方式运行 hectord，核对启动参数校验、信号退出与数据目录锁。

#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"

extern char** environ;  // NOLINT(readability-redundant-declaration)

namespace hector::server {
namespace {

namespace fs = std::filesystem;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::steady_clock;
using ::testing::HasSubstr;
using ::testing::Not;

constexpr const char* kHectordPath = HECTORD_PATH;

// 一个 hectord 子进程：stdout 丢弃，stderr 通过管道读取。
// inherit_ignored_sigint 为 true 时，子进程启动时 SIGINT 的处置是 SIG_IGN，
// 模拟非交互 shell 以 `&` 在后台启动 hectord 的情形。
class Hectord {
 public:
  explicit Hectord(const std::vector<std::string>& flags,
                   bool inherit_ignored_sigint = false) {
    std::array<int, 2> pipe_fds{};
    if (pipe(pipe_fds.data()) != 0) {
      ADD_FAILURE() << "pipe failed: " << errno;
      return;
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null",
                                     O_WRONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipe_fds[0]);
    posix_spawn_file_actions_addclose(&actions, pipe_fds[1]);

    std::vector<std::string> args = {kHectordPath};
    args.insert(args.end(), flags.begin(), flags.end());
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (std::string& arg : args) {
      argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    // posix_spawn 把 SIG_IGN 的处置原样传给子进程；本进程只在 spawn 前后
    // 短暂改动，随即恢复。
    struct sigaction ignore{};
    struct sigaction previous{};
    ignore.sa_handler = SIG_IGN;
    if (inherit_ignored_sigint) {
      sigaction(SIGINT, &ignore, &previous);
    }
    const int rc = posix_spawn(&pid_, kHectordPath, &actions, nullptr,
                               argv.data(), environ);
    if (inherit_ignored_sigint) {
      sigaction(SIGINT, &previous, nullptr);
    }
    posix_spawn_file_actions_destroy(&actions);
    close(pipe_fds[1]);
    stderr_fd_ = pipe_fds[0];
    if (rc != 0) {
      ADD_FAILURE() << "posix_spawn failed: " << rc;
      pid_ = -1;
    }
  }

  Hectord(const Hectord&) = delete;
  Hectord& operator=(const Hectord&) = delete;

  ~Hectord() {
    if (pid_ > 0 && !exit_code_.has_value()) {
      kill(pid_, SIGKILL);
      int status = 0;
      waitpid(pid_, &status, 0);
    }
    if (stderr_fd_ >= 0) {
      close(stderr_fd_);
    }
  }

  // 读取 stderr，直到出现 needle 或超时；返回是否出现。
  bool WaitForStderr(const std::string& needle,
                     steady_clock::duration timeout) {
    const auto deadline = steady_clock::now() + timeout;
    while (stderr_.find(needle) == std::string::npos) {
      const auto left = std::chrono::duration_cast<milliseconds>(
          deadline - steady_clock::now());
      if (left.count() <= 0 || !ReadSome(static_cast<int>(left.count()))) {
        return stderr_.find(needle) != std::string::npos;
      }
    }
    return true;
  }

  // 等待进程退出；超时返回 nullopt。正常退出返回退出码，被信号终止返回
  // 128 + 信号编号。
  std::optional<int> WaitForExit(steady_clock::duration timeout) {
    const auto deadline = steady_clock::now() + timeout;
    while (!exit_code_.has_value()) {
      int status = 0;
      const pid_t done = waitpid(pid_, &status, WNOHANG);
      if (done == pid_) {
        exit_code_ =
            WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        break;
      }
      if (steady_clock::now() >= deadline) {
        return std::nullopt;
      }
      ReadSome(10);
    }
    while (ReadSome(0)) {
    }
    return exit_code_;
  }

  void Signal(int sig) const { kill(pid_, sig); }
  const std::string& stderr_text() const { return stderr_; }

 private:
  // 最多等待 timeout_ms 读一次；读到数据返回 true，超时或 EOF 返回 false。
  bool ReadSome(int timeout_ms) {
    if (stderr_fd_ < 0) {
      return false;
    }
    pollfd fd{.fd = stderr_fd_, .events = POLLIN, .revents = 0};
    if (poll(&fd, 1, timeout_ms) <= 0) {
      return false;
    }
    std::array<char, 4096> buffer{};
    const ssize_t n = read(stderr_fd_, buffer.data(), buffer.size());
    if (n <= 0) {
      return false;
    }
    stderr_.append(buffer.data(), static_cast<size_t>(n));
    return true;
  }

  pid_t pid_ = -1;
  int stderr_fd_ = -1;
  std::string stderr_;
  std::optional<int> exit_code_;
};

class HectordCliTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = fs::path(::testing::TempDir()) /
           absl::StrCat("hectord_cli_test_", info->name());
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  std::string DataDirFlag(const fs::path& path) const {
    return absl::StrCat("--data_dir=", path.string());
  }

  fs::path dir_;
};

constexpr auto kStartTimeout = seconds(20);
constexpr auto kExitTimeout = seconds(5);

TEST_F(HectordCliTest, RejectsInvalidFlags) {
  {
    Hectord hectord({DataDirFlag(dir_ / "data"), "--max_message_bytes=100"});
    EXPECT_EQ(hectord.WaitForExit(kStartTimeout), 1);
    EXPECT_THAT(hectord.stderr_text(), HasSubstr("max_message_bytes"));
  }
  {
    const fs::path file = dir_ / "plain_file";
    std::ofstream(file) << "not a directory";
    Hectord hectord({DataDirFlag(file), "--listen_address=127.0.0.1:0"});
    EXPECT_EQ(hectord.WaitForExit(kStartTimeout), 1);
    EXPECT_THAT(hectord.stderr_text(), HasSubstr("data_dir"));
  }
}

TEST_F(HectordCliTest, SigtermShutsDownCleanly) {
  const fs::path data = dir_ / "data";
  {
    Hectord hectord({DataDirFlag(data), "--listen_address=127.0.0.1:0"});
    ASSERT_TRUE(hectord.WaitForStderr("listening on", kStartTimeout))
        << hectord.stderr_text();
    EXPECT_THAT(hectord.stderr_text(), HasSubstr("data_dir="));
    EXPECT_THAT(hectord.stderr_text(), Not(HasSubstr("127.0.0.1:0 ")));
    hectord.Signal(SIGTERM);
    EXPECT_EQ(hectord.WaitForExit(kExitTimeout), 0) << hectord.stderr_text();
  }
  {
    Hectord hectord({DataDirFlag(data), "--listen_address=127.0.0.1:0"});
    ASSERT_TRUE(hectord.WaitForStderr("listening on", kStartTimeout))
        << hectord.stderr_text();
    EXPECT_THAT(hectord.stderr_text(), HasSubstr("replayed 0 records"));
    hectord.Signal(SIGTERM);
    EXPECT_EQ(hectord.WaitForExit(kExitTimeout), 0) << hectord.stderr_text();
  }
}

// 后台启动的 hectord 继承了被忽略的 SIGINT，显式发送 SIGINT 仍须正常退出。
TEST_F(HectordCliTest, SigintShutsDownEvenIfInheritedIgnored) {
  Hectord hectord({DataDirFlag(dir_ / "data"), "--listen_address=127.0.0.1:0"},
                  /*inherit_ignored_sigint=*/true);
  ASSERT_TRUE(hectord.WaitForStderr("listening on", kStartTimeout))
      << hectord.stderr_text();
  hectord.Signal(SIGINT);
  EXPECT_EQ(hectord.WaitForExit(kExitTimeout), 0) << hectord.stderr_text();
}

TEST_F(HectordCliTest, SecondInstanceOnSameDirFails) {
  const fs::path data = dir_ / "data";
  Hectord first({DataDirFlag(data), "--listen_address=127.0.0.1:0"});
  ASSERT_TRUE(first.WaitForStderr("listening on", kStartTimeout))
      << first.stderr_text();
  {
    Hectord second({DataDirFlag(data), "--listen_address=127.0.0.1:0"});
    EXPECT_EQ(second.WaitForExit(kStartTimeout), 1);
    EXPECT_THAT(second.stderr_text(), HasSubstr("data_dir is locked"));
  }
  first.Signal(SIGTERM);
  EXPECT_EQ(first.WaitForExit(kExitTimeout), 0) << first.stderr_text();
}

}  // namespace
}  // namespace hector::server
