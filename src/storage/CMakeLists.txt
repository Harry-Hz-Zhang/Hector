# hector_storage：分段文件 WAL 与快照存储。只依赖 hector_common；spdlog 仅内部使用。
add_library(hector_storage STATIC
  file_wal.cc
  snapshot.cc)
target_link_libraries(hector_storage
  PUBLIC hector_common
  PRIVATE absl::strings spdlog::spdlog)
hector_target_defaults(hector_storage)
