# hector_core：Schema、列存储、筛选、融合、Table、WAL 记录编解码与 Engine。依赖 index 与 storage 层。
add_library(hector_core STATIC
  column_store.cc
  database.cc
  engine.cc
  filter.cc
  fusion.cc
  schema.cc
  table.cc
  wal_codec.cc)
target_link_libraries(hector_core
  PUBLIC hector_index hector_storage absl::flat_hash_map spdlog::spdlog
  PRIVATE absl::strings)
hector_target_defaults(hector_core)
