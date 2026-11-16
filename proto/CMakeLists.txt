# hector_proto：由 proto/hector/v1/hector.proto 生成的 C++ 消息与 gRPC 服务代码。
# 生成代码不加 -Werror，头文件目录以 SYSTEM 方式暴露，生成代码中的告警不影响 Hector 自有目标。

set(_hector_proto_root "${CMAKE_CURRENT_SOURCE_DIR}")
set(_hector_proto_file "${_hector_proto_root}/hector/v1/hector.proto")
set(_hector_proto_out "${CMAKE_CURRENT_BINARY_DIR}/gen")
set(_hector_proto_generated
  "${_hector_proto_out}/hector/v1/hector.pb.h"
  "${_hector_proto_out}/hector/v1/hector.pb.cc"
  "${_hector_proto_out}/hector/v1/hector.grpc.pb.h"
  "${_hector_proto_out}/hector/v1/hector.grpc.pb.cc")

add_custom_command(
  OUTPUT ${_hector_proto_generated}
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${_hector_proto_out}"
  COMMAND protobuf::protoc
          "--proto_path=${_hector_proto_root}"
          "--cpp_out=${_hector_proto_out}"
          "--grpc_out=${_hector_proto_out}"
          "--plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
          "${_hector_proto_file}"
  DEPENDS "${_hector_proto_file}" protobuf::protoc gRPC::grpc_cpp_plugin
  COMMENT "Generating C++ and gRPC code from hector/v1/hector.proto"
  VERBATIM)

add_library(hector_proto STATIC ${_hector_proto_generated})
target_include_directories(hector_proto SYSTEM PUBLIC "${_hector_proto_out}")
target_link_libraries(hector_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
hector_target_defaults(hector_proto GENERATED)

# protobuf 在 ABSL_HAVE_ADDRESS_SANITIZER 下默认启用 PROTOBUF_MESSAGE_GLOBALS，改变生成代码中全局对象的布局；
# Homebrew 的 libprotobuf 没有 ASan 插桩、按普通布局编译，两者混用会在序列化时崩溃
# （TcParser::VerifyHasBitConsistency 中 BUS）。开启 ASan 时让所有包含 protobuf 头文件的目标退回普通布局。
if(HECTOR_ENABLE_ASAN_UBSAN)
  target_compile_definitions(hector_proto PUBLIC PROTOBUF_MESSAGE_GLOBALS_TEMPORARY_OPTOUT)
endif()
