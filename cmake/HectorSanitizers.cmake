# HECTOR_ENABLE_ASAN_UBSAN / HECTOR_ENABLE_TSAN：只作用于 Hector 自有目标（见 hector_target_defaults）。
# 本文件在 find_package 与 FetchContent 之前 include，因此互斥检查不需要联网。

if(HECTOR_ENABLE_ASAN_UBSAN AND HECTOR_ENABLE_TSAN)
  message(FATAL_ERROR
    "HECTOR_ENABLE_ASAN_UBSAN and HECTOR_ENABLE_TSAN are mutually exclusive")
endif()

set(HECTOR_SANITIZER_FLAGS "")
if(HECTOR_ENABLE_ASAN_UBSAN)
  set(HECTOR_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
elseif(HECTOR_ENABLE_TSAN)
  set(HECTOR_SANITIZER_FLAGS -fsanitize=thread)
endif()

# 给单个目标加 sanitizer 的编译与链接选项。
function(hector_apply_sanitizers target)
  if(HECTOR_SANITIZER_FLAGS)
    target_compile_options(${target} PRIVATE ${HECTOR_SANITIZER_FLAGS})
    target_link_options(${target} PRIVATE ${HECTOR_SANITIZER_FLAGS})
  endif()
  if(HECTOR_ENABLE_ASAN_UBSAN)
    # abseil 头文件在检测到 ASan 时开启 swisstable 的 generation 校验，改变 raw_hash_set
    # 的内存布局；Homebrew 的 abseil 库没有用 ASan 编译，二者不一致会在首次插入时崩溃。
    # NDEBUG_SANITIZER 是 abseil 提供的关闭开关，必须对所有 Hector 编译单元一致定义
    # （含生成代码），否则同一个 inline 函数在不同编译单元里布局不同（ODR 违规）。
    # 升级 abseil 时复查。
    target_compile_definitions(${target} PRIVATE NDEBUG_SANITIZER)
  endif()
endfunction()
