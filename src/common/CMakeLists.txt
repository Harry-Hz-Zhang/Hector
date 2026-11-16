# hector_common：只依赖 abseil 的 status / statusor，不链接 spdlog、protobuf、gRPC。
add_library(hector_common STATIC
  bitmap.cc
  byte_io.cc
  concurrent_bitmap.cc
  row_filter.cc
  types.cc)
target_include_directories(hector_common PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(hector_common PUBLIC absl::status absl::statusor)
hector_target_defaults(hector_common)
