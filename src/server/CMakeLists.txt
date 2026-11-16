# hector_server：hector.v1 gRPC 服务实现、proto 与 core 互转、状态码映射。只链接 hector_core 与 hector_proto。
add_library(hector_server STATIC
  hector_service.cc
  proto_convert.cc
  status_convert.cc)
# 源文件以仓库根为基准包含本层头文件（"src/server/hector_service.h"）。
target_include_directories(hector_server PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(hector_server PUBLIC hector_core hector_proto)
hector_target_defaults(hector_server)

# hectord：启动参数解析、信号处理与服务端生命周期。
add_executable(hectord main.cc)
target_link_libraries(hectord PRIVATE hector_server absl::flags absl::flags_parse absl::flags_usage)
hector_target_defaults(hectord)
