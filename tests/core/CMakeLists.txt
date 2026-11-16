# tests/core：core 层用例，加入 hector_tests。
# counting_index.h 与 fake_wal.h 以仓库根为基准包含（"tests/core/counting_index.h"）。

target_sources(hector_tests PRIVATE
  column_store_test.cc
  engine_replay_test.cc
  engine_test.cc
  filter_test.cc
  fusion_test.cc
  schema_test.cc
  table_search_test.cc
  table_write_test.cc
  wal_codec_test.cc)
target_include_directories(hector_tests PRIVATE "${PROJECT_SOURCE_DIR}")
