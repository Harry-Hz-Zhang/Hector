# tests/common：common 层用例，加入 hector_tests。

target_sources(hector_tests PRIVATE
  bitmap_test.cc
  byte_io_test.cc
  concurrent_bitmap_test.cc
  row_filter_test.cc
  status_test.cc
  types_test.cc)
