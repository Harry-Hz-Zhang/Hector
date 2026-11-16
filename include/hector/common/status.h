#ifndef HECTOR_COMMON_STATUS_H_
#define HECTOR_COMMON_STATUS_H_

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

// HECTOR_RETURN_IF_ERROR(expr)：expr 求值为 absl::Status；非 OK
// 时原样从当前函数返回。
#define HECTOR_RETURN_IF_ERROR(expr)                                    \
  do {                                                                  \
    if (::absl::Status hector_status_ = (expr); !hector_status_.ok()) { \
      return hector_status_;                                            \
    }                                                                   \
  } while (false)

// HECTOR_ASSIGN_OR_RETURN(lhs, expr)：expr 求值为 absl::StatusOr<T>；
// OK 时把值移动赋给 lhs（lhs 可以是声明，如 `auto x`），否则返回其状态。
// 宏展开为多条语句，不能用在不带花括号的 if/else 分支里。
#define HECTOR_ASSIGN_OR_RETURN(lhs, expr) \
  HECTOR_ASSIGN_OR_RETURN_IMPL_(           \
      HECTOR_STATUS_CONCAT_(hector_statusor_, __LINE__), lhs, expr)

// statusor 是标识符、lhs 可能是声明，二者都不能加括号。
// NOLINTBEGIN(bugprone-macro-parentheses)
#define HECTOR_ASSIGN_OR_RETURN_IMPL_(statusor, lhs, expr) \
  auto statusor = (expr);                                  \
  if (!statusor.ok()) {                                    \
    return std::move(statusor).status();                   \
  }                                                        \
  lhs = *std::move(statusor)
// NOLINTEND(bugprone-macro-parentheses)

#define HECTOR_STATUS_CONCAT_INNER_(a, b) a##b
#define HECTOR_STATUS_CONCAT_(a, b) HECTOR_STATUS_CONCAT_INNER_(a, b)

#endif  // HECTOR_COMMON_STATUS_H_
