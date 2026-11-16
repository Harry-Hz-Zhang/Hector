cmake_minimum_required(VERSION 3.28)
project(Hector VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

option(HECTOR_ENABLE_ASAN_UBSAN "Enable ASan and UBSan for Hector targets" OFF)
option(HECTOR_ENABLE_TSAN "Enable TSan for Hector targets" OFF)
option(HECTOR_LAYERING_SELFTEST
  "Deliberately link hector_index to hector_storage to prove the layering check fails" OFF)

# 1. 选项互斥与工具检查：都在 find_package 与 FetchContent 之前，失败时不需要联网。
include(HectorSanitizers)
include(HectorTooling)
include(HectorLayering)

# 2. find_package（gRPC / protobuf / abseil），然后 FetchContent（GoogleTest / benchmark / spdlog）。
include(HectorDeps)

# 每个 Hector 目标统一调用：C++20、-mcpu=native（D22）、sanitizer；
# 自有源文件另加 -Wall -Wextra -Wpedantic -Werror，生成代码（GENERATED）不加。
function(hector_target_defaults target)
  cmake_parse_arguments(PARSE_ARGV 1 _arg "GENERATED" "" "")
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  target_compile_options(${target} PRIVATE -mcpu=native)
  if(NOT _arg_GENERATED)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  hector_apply_sanitizers(${target})
endfunction()

# 3. 子目录。可执行文件统一输出到构建目录根，例如 build/debug/hectord。
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
enable_testing()
add_subdirectory(proto)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(bench)

if(HECTOR_LAYERING_SELFTEST)
  target_link_libraries(hector_index PRIVATE hector_storage)
endif()

hector_add_quality_targets()
hector_check_layering()
