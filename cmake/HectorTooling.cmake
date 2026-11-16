# clang-format 与 clang-tidy 都是必需工具（D13）。
# 本文件在 find_package 与 FetchContent 之前 include，因此缺工具时立即失败、不需要联网。

find_program(HECTOR_CLANG_FORMAT_EXE NAMES clang-format)
if(NOT HECTOR_CLANG_FORMAT_EXE OR NOT EXISTS "${HECTOR_CLANG_FORMAT_EXE}")
  message(FATAL_ERROR "clang-format not found: run 'brew install clang-format'")
endif()

# Homebrew 的 llvm 是 keg-only，clang-tidy 不在 PATH 中。
find_program(HECTOR_CLANG_TIDY_EXE NAMES clang-tidy
  PATHS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin)
if(NOT HECTOR_CLANG_TIDY_EXE OR NOT EXISTS "${HECTOR_CLANG_TIDY_EXE}")
  message(FATAL_ERROR "clang-tidy not found: run 'brew install llvm'")
endif()

# Homebrew 的 clang-tidy 搭配 Apple clang 时需要显式给出 SDK，否则找不到 libc++ 与系统头文件。
set(HECTOR_CLANG_TIDY_EXTRA_ARGS "")
if(APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE HECTOR_MACOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(HECTOR_MACOS_SDK_PATH)
    list(APPEND HECTOR_CLANG_TIDY_EXTRA_ARGS "--extra-arg=-isysroot${HECTOR_MACOS_SDK_PATH}")
  endif()
endif()

# 格式检查范围：include/ src/ tests/ bench/ 下的 .h 与 .cc，排除故意写坏的 fixtures。
file(GLOB_RECURSE HECTOR_FORMAT_SOURCES CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.cc"
  "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/*.cc"
  "${PROJECT_SOURCE_DIR}/tests/*.h" "${PROJECT_SOURCE_DIR}/tests/*.cc"
  "${PROJECT_SOURCE_DIR}/bench/*.h" "${PROJECT_SOURCE_DIR}/bench/*.cc")
list(FILTER HECTOR_FORMAT_SOURCES EXCLUDE REGEX "/tests/build/fixtures/")

# 静态检查范围：src/ 下所有 .cc，基于 compile_commands.json。
file(GLOB_RECURSE HECTOR_TIDY_SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cc")

# 在所有目标定义之后调用：定义 hector_format_check 与 hector_tidy_check。
function(hector_add_quality_targets)
  add_custom_target(hector_format_check
    COMMAND "${HECTOR_CLANG_FORMAT_EXE}" --dry-run --Werror ${HECTOR_FORMAT_SOURCES}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "clang-format --dry-run --Werror"
    VERBATIM)

  add_custom_target(hector_tidy_check
    COMMAND "${HECTOR_CLANG_TIDY_EXE}" -p "${PROJECT_BINARY_DIR}" --quiet
            ${HECTOR_CLANG_TIDY_EXTRA_ARGS} ${HECTOR_TIDY_SOURCES}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "clang-tidy (compile_commands.json in ${PROJECT_BINARY_DIR})"
    VERBATIM)
  # 服务层源文件会包含生成的协议头文件，先生成再检查。
  add_dependencies(hector_tidy_check hector_proto)
endfunction()
