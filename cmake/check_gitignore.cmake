# 用法：cmake [-DHECTOR_GITIGNORE=<路径>] -P cmake/check_gitignore.cmake
# 检查 .gitignore 是否逐行包含本机目录的忽略规则；缺任一行时以非 0 退出。

cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED HECTOR_GITIGNORE)
  get_filename_component(HECTOR_GITIGNORE "${CMAKE_CURRENT_LIST_DIR}/../.gitignore" ABSOLUTE)
endif()

if(NOT EXISTS "${HECTOR_GITIGNORE}")
  message(FATAL_ERROR "gitignore check: ${HECTOR_GITIGNORE} does not exist")
endif()

set(_hector_required_ignores
  ".codegraph/"
  "build/"
  "clients/python/.venv/"
  "clients/python/hector_client/_gen/"
  "__pycache__/"
  "hector_data/"
)

file(STRINGS "${HECTOR_GITIGNORE}" _hector_raw_lines ENCODING UTF-8)
set(_hector_lines "")
foreach(_line IN LISTS _hector_raw_lines)
  string(STRIP "${_line}" _line)
  list(APPEND _hector_lines "${_line}")
endforeach()
set(_hector_missing "")
foreach(_entry IN LISTS _hector_required_ignores)
  list(FIND _hector_lines "${_entry}" _index)
  if(_index EQUAL -1)
    list(APPEND _hector_missing "${_entry}")
  endif()
endforeach()

if(_hector_missing)
  list(JOIN _hector_missing ", " _hector_missing_text)
  message(FATAL_ERROR "gitignore check: missing entries: ${_hector_missing_text}")
endif()
message(STATUS "gitignore check: all required entries present")
