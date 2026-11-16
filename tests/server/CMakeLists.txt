# tests/server：服务层用例，加入 hector_server_tests。

target_sources(hector_server_tests PRIVATE
  hectord_cli_test.cc
  proto_convert_test.cc
  service_integration_test.cc
  status_convert_test.cc)
# 服务层的内部头文件（src/server/*.h）以仓库根为基准包含（"src/server/hector_service.h"）。
target_include_directories(hector_server_tests PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(hector_server_tests PRIVATE GTest::gmock)

# HectordCliTest 以子进程方式启动 hectord，先保证它已构建。
add_dependencies(hector_server_tests hectord)
target_compile_definitions(hector_server_tests PRIVATE
  "HECTORD_PATH=\"$<TARGET_FILE:hectord>\"")
