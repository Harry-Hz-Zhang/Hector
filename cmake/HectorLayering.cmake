# hector_check_layering()：在配置阶段末尾检查 Hector 库之间的直接链接方向。
# 每个受检目标只允许链接下表列出的 hector_* 目标；第三方目标不受限制。
# 发现违反方向的链接时报 FATAL_ERROR：layering violation: <目标> -> <目标>。

set(_hector_layer_allowed_hector_common "")
set(_hector_layer_allowed_hector_index hector_common)
set(_hector_layer_allowed_hector_storage hector_common)
set(_hector_layer_allowed_hector_core hector_common hector_index hector_storage)
set(_hector_layer_allowed_hector_proto "")
set(_hector_layer_allowed_hector_server hector_core hector_proto)
set(_hector_layer_allowed_hectord hector_server)

set(HECTOR_LAYERED_TARGETS
  hector_common hector_index hector_storage hector_core hector_proto hector_server hectord)

function(hector_check_layering)
  foreach(_target IN LISTS HECTOR_LAYERED_TARGETS)
    if(NOT TARGET ${_target})
      message(FATAL_ERROR "layering check: target ${_target} is not defined")
    endif()
    get_target_property(_links ${_target} LINK_LIBRARIES)
    if(NOT _links)
      continue()
    endif()
    foreach(_dep IN LISTS _links)
      if(NOT _dep MATCHES "^hector")
        continue()
      endif()
      if(NOT _dep IN_LIST _hector_layer_allowed_${_target})
        message(FATAL_ERROR "layering violation: ${_target} -> ${_dep}")
      endif()
    endforeach()
  endforeach()
endfunction()
