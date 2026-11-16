# tests/storage：storage 层用例（FileWal、Null 快照），加入 hector_tests。
# file_wal_test.cc 直接使用 spdlog 捕获日志，因此显式链接 spdlog，不依赖 hector_core 的传递。
# hector_core 也 PUBLIC 链接 spdlog，macOS ld 会提示 duplicate libraries，这是不影响构建的提示性告警。

target_sources(hector_tests PRIVATE
  file_wal_test.cc
  snapshot_test.cc)
target_link_libraries(hector_tests PRIVATE spdlog::spdlog)
