# tests/index：index 层用例，加入 hector_tests。
# fake_accessors.h 以仓库根为基准包含（"tests/index/fake_accessors.h"）。

target_sources(hector_tests PRIVATE
  flat_index_test.cc
  flat_sparse_index_test.cc
  index_factory_test.cc
  metric_test.cc)
target_include_directories(hector_tests PRIVATE "${PROJECT_SOURCE_DIR}")
