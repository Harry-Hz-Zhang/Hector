# 分层：common <- {index, storage} <- core <- server。依赖方向由 hector_check_layering() 在配置阶段检查。
add_subdirectory(common)
add_subdirectory(index)
add_subdirectory(storage)
add_subdirectory(core)
add_subdirectory(server)
