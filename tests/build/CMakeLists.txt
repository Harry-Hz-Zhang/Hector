# tests/build：工程级检查（协议冒烟、配置、格式、静态检查）。

target_sources(hector_tests PRIVATE placeholder_test.cc)
target_sources(hector_server_tests PRIVATE proto_smoke_test.cc)

# ---------------------------------------------------------------------------
# ConfigureTest.*：在临时目录里重新配置本仓库，断言配置阶段的报错。
# 所有用例都复用 debug 构建已下载的依赖源码并断开 FetchContent 的网络更新，因此不联网；
# 不复用任何 *-build 目录。FAIL_REGULAR_EXPRESSION 保证配置确实失败。
# ---------------------------------------------------------------------------
set(_hector_configure_root "${CMAKE_CURRENT_BINARY_DIR}/configure")
set(_hector_configure_args
  -G "${CMAKE_GENERATOR}"
  "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  -DCMAKE_BUILD_TYPE=Debug
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${googletest_SOURCE_DIR}"
  "-DFETCHCONTENT_SOURCE_DIR_BENCHMARK=${benchmark_SOURCE_DIR}"
  "-DFETCHCONTENT_SOURCE_DIR_SPDLOG=${spdlog_SOURCE_DIR}"
)

function(hector_add_configure_test name expected_regex)
  set(_dir "${_hector_configure_root}/${name}")
  add_test(
    NAME ConfigureTest.${name}
    COMMAND "${CMAKE_COMMAND}" --fresh -S "${PROJECT_SOURCE_DIR}" -B "${_dir}"
            ${_hector_configure_args} ${ARGN})
  set_tests_properties(ConfigureTest.${name} PROPERTIES
    PASS_REGULAR_EXPRESSION "${expected_regex}"
    FAIL_REGULAR_EXPRESSION "Configuring done"
    LABELS configure)
endfunction()

hector_add_configure_test(SanitizersMutuallyExclusive
  "HECTOR_ENABLE_ASAN_UBSAN and HECTOR_ENABLE_TSAN are mutually exclusive"
  -DHECTOR_ENABLE_ASAN_UBSAN=ON -DHECTOR_ENABLE_TSAN=ON)

hector_add_configure_test(ClangTidyMissingFailsFast
  "clang-tidy not found: run 'brew install llvm'"
  -DHECTOR_ENABLE_ASAN_UBSAN=OFF -DHECTOR_ENABLE_TSAN=OFF
  -DHECTOR_CLANG_TIDY_EXE=/nonexistent)

hector_add_configure_test(MissingGrpcFailsWithHint
  "gRPC not found: run 'brew install grpc'"
  -DHECTOR_ENABLE_ASAN_UBSAN=OFF -DHECTOR_ENABLE_TSAN=OFF
  -DCMAKE_DISABLE_FIND_PACKAGE_gRPC=ON)

hector_add_configure_test(LayerLinksAreDownward
  "layering violation: hector_index -> hector_storage"
  -DHECTOR_ENABLE_ASAN_UBSAN=OFF -DHECTOR_ENABLE_TSAN=OFF
  -DHECTOR_LAYERING_SELFTEST=ON)

add_test(
  NAME ConfigureTest.GitignoreCoversLocalDirs
  COMMAND "${CMAKE_COMMAND}" "-DHECTOR_GITIGNORE=${PROJECT_SOURCE_DIR}/.gitignore"
          -P "${PROJECT_SOURCE_DIR}/cmake/check_gitignore.cmake")
set_tests_properties(ConfigureTest.GitignoreCoversLocalDirs PROPERTIES LABELS configure)

# ---------------------------------------------------------------------------
# FormatTest.*：与 hector_format_check 目标使用同一个文件清单与同一条命令。
# ---------------------------------------------------------------------------
add_test(
  NAME FormatTest.SourcesMatchClangFormat
  COMMAND "${HECTOR_CLANG_FORMAT_EXE}" --dry-run --Werror ${HECTOR_FORMAT_SOURCES}
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
# 断言输出为空：任何输出（包括格式告警）都判失败。
set_tests_properties(FormatTest.SourcesMatchClangFormat PROPERTIES
  FAIL_REGULAR_EXPRESSION "."
  LABELS quality)

add_test(
  NAME FormatTest.DetectsBadFixture
  COMMAND "${HECTOR_CLANG_FORMAT_EXE}" --dry-run --Werror
          "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/bad_format.cc"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
# 只有 --Werror 下的格式错误才会输出「<文件>:<行>:<列>: error:」，因此该匹配同时意味着退出码非 0。
set_tests_properties(FormatTest.DetectsBadFixture PROPERTIES
  PASS_REGULAR_EXPRESSION "bad_format\\.cc:[0-9]+:[0-9]+: error:"
  LABELS quality)

# ---------------------------------------------------------------------------
# TidyTest.*：运行 hector_tidy_check 目标；退出码必须为 0，且输出不含 warning:。
# ---------------------------------------------------------------------------
add_test(
  NAME TidyTest.SourcesPassClangTidy
  COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target hector_tidy_check)
set_tests_properties(TidyTest.SourcesPassClangTidy PROPERTIES
  FAIL_REGULAR_EXPRESSION "warning:"
  LABELS quality)
