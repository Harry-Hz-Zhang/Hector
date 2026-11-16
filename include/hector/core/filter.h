#ifndef HECTOR_CORE_FILTER_H_
#define HECTOR_CORE_FILTER_H_

#include <span>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "hector/common/bitmap.h"
#include "hector/common/types.h"
#include "hector/core/column_store.h"
#include "hector/core/schema.h"
#include "hector/core/value.h"

namespace hector {

// NOLINTNEXTLINE(performance-enum-size): 声明与 design.md 4.4 一致
enum class CompareOp { kEq, kNe, kLt, kLe, kGt, kGe, kIn };

// 一个筛选条件：column op values。非 IN 恰好 1 个值，IN 至少 1 个值。
struct Condition {
  std::string column;
  CompareOp op;
  std::vector<ScalarValue> values;
};

// 条件之间只有「并且」；空表示不过滤。
struct Filter {
  std::vector<Condition> conditions;
};

// 校验规则见 design.md 第 5 节：列存在且是普通列；值类型与列类型完全一致；
// 非 IN 恰好 1 个值，IN 至少 1 个值。失败返回 InvalidArgument，消息含
// conditions[i]。
absl::Status ValidateFilter(const Filter& filter, const Schema& schema);

// 前置条件：已通过 ValidateFilter。columns 按 Schema 列顺序给出，非普通列
// 位置为 nullptr。只计算 row id < row_count 的行，返回大小为 row_count 的
// allow 位图；filter 为空时全部置位。string 列按字节字典序比较。
Bitmap EvaluateFilter(const Filter& filter, const Schema& schema,
                      std::span<const ScalarColumn* const> columns,
                      RowId row_count);

}  // namespace hector

#endif  // HECTOR_CORE_FILTER_H_
