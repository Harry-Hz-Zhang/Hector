# 两个 GoogleTest 可执行目标：
#   hector_tests         不链接 gRPC，包含 tests/{common,index,storage,core} 与 tests/build/placeholder_test.cc；
#                        tsan 预设只构建并运行它。
#   hector_server_tests  链接 hector_server（含 gRPC），包含 tests/build/proto_smoke_test.cc 与 tests/server。
include(GoogleTest)

add_executable(hector_tests)
target_link_libraries(hector_tests PRIVATE hector_core GTest::gtest_main GTest::gmock)
hector_target_defaults(hector_tests)

add_executable(hector_server_tests)
target_link_libraries(hector_server_tests PRIVATE hector_server GTest::gtest_main)
hector_target_defaults(hector_server_tests)

add_subdirectory(build)
add_subdirectory(common)
add_subdirectory(index)
add_subdirectory(storage)
add_subdirectory(core)
add_subdirectory(server)

gtest_discover_tests(hector_tests)
gtest_discover_tests(hector_server_tests)
