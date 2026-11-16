// hectord：hector.v1 gRPC 服务端（design.md 第 7 节）。
//
// 启动顺序：解析并校验参数；屏蔽 SIGINT 与 SIGTERM（之后创建的线程都继承
// 屏蔽字）；打开 Engine；启动 gRPC 服务端；专用线程用 sigwait 等待信号，
// 收到后调用带 3 秒期限的 Shutdown。主线程在 Wait 返回后销毁服务端与 Engine
// （析构函数执行 Wal::SyncUpTo 并释放数据目录锁），以退出码 0 退出。
// 本进程不注册任何信号处理函数，因此不会在信号处理函数中调用 gRPC。

#include <pthread.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <thread>

#include "absl/base/config.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "hector/core/engine.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "src/server/hector_service.h"

ABSL_FLAG(std::string, data_dir, "./hector_data",
          "Data directory; created if missing, locked while hectord runs.");
ABSL_FLAG(std::string, listen_address, "127.0.0.1:50051",
          "host:port to listen on; port 0 lets the system choose.");
ABSL_FLAG(int64_t, max_message_bytes, 4194304,
          "Max gRPC message size for both receive and send, in [1048576, "
          "268435456].");

#ifdef ABSL_HAVE_ADDRESS_SANITIZER
// Homebrew 的 gRPC 与 protobuf 没有 ASan 插桩，容器溢出检查会误报
// （design.md 2.2）；直接启动 debug 版 hectord 时也关闭它。
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming)
extern "C" const char* __asan_default_options() {
  return "detect_container_overflow=0";
}
#endif

namespace {

constexpr int64_t kMinMessageBytes = 1048576;
constexpr int64_t kMaxMessageBytes = 268435456;
constexpr auto kShutdownGrace = std::chrono::seconds(3);

int Run() {
  const std::string data_dir = absl::GetFlag(FLAGS_data_dir);
  const std::string listen_address = absl::GetFlag(FLAGS_listen_address);
  const int64_t max_message_bytes = absl::GetFlag(FLAGS_max_message_bytes);
  if (max_message_bytes < kMinMessageBytes ||
      max_message_bytes > kMaxMessageBytes) {
    spdlog::error("invalid --max_message_bytes={}: must be in [{}, {}]",
                  max_message_bytes, kMinMessageBytes, kMaxMessageBytes);
    return 1;
  }

  // 处置为 SIG_IGN 的信号在产生时就被丢弃，sigwait 等不到（非交互 shell 以
  // `&` 启动的进程会继承被忽略的 SIGINT），因此先恢复为 SIG_DFL。这里不注册
  // 任何处理函数。
  struct sigaction default_action{};
  default_action.sa_handler = SIG_DFL;
  sigemptyset(&default_action.sa_mask);
  sigaction(SIGINT, &default_action, nullptr);
  sigaction(SIGTERM, &default_action, nullptr);
  // 在创建任何线程之前屏蔽，信号只会由下面的专用线程通过 sigwait 取走。
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);

  auto engine =
      hector::Engine::Open(hector::EngineOptions{.data_dir = data_dir});
  if (!engine.ok()) {
    spdlog::error("cannot open data_dir={}: {}", data_dir,
                  engine.status().ToString());
    return 1;
  }
  hector::server::HectorServiceImpl service(**engine);
  auto running = hector::server::StartServer(
      service, listen_address, static_cast<int>(max_message_bytes));
  if (!running.ok()) {
    spdlog::error("cannot start server: {}", running.status().ToString());
    return 1;
  }
  spdlog::info("hectord listening on {} data_dir={} replayed {} records",
               running->address, data_dir, (*engine)->replayed_records());

  grpc::Server* server = running->server.get();
  std::thread signal_thread([&signals, server] {
    int received = 0;
    sigwait(&signals, &received);
    spdlog::info("received signal {}, shutting down", received);
    server->Shutdown(std::chrono::system_clock::now() + kShutdownGrace);
  });
  server->Wait();
  signal_thread.join();

  running->server.reset();
  engine->reset();
  spdlog::info("hectord stopped");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("hectord: the Hector vector database server");
  absl::ParseCommandLine(argc, argv);
  // spdlog 与 std::thread 失败时抛异常：打印原因并以退出码 1 退出。
  try {
    // 日志统一写到 stderr。
    spdlog::set_default_logger(spdlog::stderr_color_mt("hectord"));
    return Run();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hectord: fatal: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "hectord: fatal: unknown exception\n");
  }
  return 1;
}
