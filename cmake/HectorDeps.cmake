# 依赖获取（D12b）：
#   gRPC、protobuf、abseil 使用 Homebrew 当前版本，通过 find_package(... CONFIG) 引入；
#   GoogleTest、Google Benchmark、spdlog 通过 FetchContent 以固定 tag 获取，并以 SYSTEM 方式引入。

include(FetchContent)

find_package(absl CONFIG)
if(NOT absl_FOUND)
  message(FATAL_ERROR "abseil not found: run 'brew install abseil'")
endif()

find_package(Protobuf CONFIG)
if(NOT Protobuf_FOUND)
  message(FATAL_ERROR "protobuf not found: run 'brew install protobuf'")
endif()

find_package(gRPC CONFIG)
if(NOT gRPC_FOUND)
  message(FATAL_ERROR "gRPC not found: run 'brew install grpc'")
endif()

message(STATUS "Hector deps: gRPC ${gRPC_VERSION} (${gRPC_DIR})")
message(STATUS "Hector deps: protobuf ${Protobuf_VERSION} (${Protobuf_DIR})")
message(STATUS "Hector deps: abseil ${absl_VERSION} (${absl_DIR})")

# GoogleTest：不安装。
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
  GIT_SHALLOW TRUE
  SYSTEM)

# Google Benchmark：不构建自身测试，不安装，不对自身开启 -Werror。
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_Declare(benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.4
  GIT_SHALLOW TRUE
  SYSTEM)

# spdlog：使用自带的 fmt。
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.3
  GIT_SHALLOW TRUE
  SYSTEM)

FetchContent_MakeAvailable(googletest benchmark spdlog)
