#include "hector/core/filter.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "hector/common/bitmap.h"
#include "hector/common/types.h"
#include "hector/core/column_store.h"
#include "hector/core/schema.h"
#include "hector/core/value.h"

namespace hector {
namespace {

absl::Status ConditionError(size_t i, std::string_view column,
                            std::string_view detail) {
  return absl::InvalidArgumentError(absl::StrCat(
      "filter.conditions[", i, "] on column \"", column, "\": ", detail));
}

// ScalarValue 的下标与 ScalarType 的取值一一对应：int64、float、string。
size_t AlternativeOf(ScalarType type) {
  switch (type) {
    case ScalarType::kInt64:
      return 0;
    case ScalarType::kFloat:
      return 1;
    case ScalarType::kString:
      return 2;
  }
  return std::variant_npos;
}

std::string_view TypeName(ScalarType type) {
  switch (type) {
    case ScalarType::kInt64:
      return "int64";
    case ScalarType::kFloat:
      return "float";
    case ScalarType::kString:
      return "string";
  }
  return "unknown";
}

// 前置条件：a 与 b 持有同一种类型。string 的 operator< 按 char_traits<char>
// 比较，即按无符号字节的字典序。
template <typename T>
bool CompareTyped(const T& a, CompareOp op, const T& b) {
  switch (op) {
    case CompareOp::kEq:
    case CompareOp::kIn:
      return a == b;
    case CompareOp::kNe:
      return a != b;
    case CompareOp::kLt:
      return a < b;
    case CompareOp::kLe:
      return a <= b;
    case CompareOp::kGt:
      return a > b;
    case CompareOp::kGe:
      return a >= b;
  }
  return false;
}

bool Compare(const ScalarValue& cell, CompareOp op, const ScalarValue& value) {
  return std::visit(
      [op, &value](const auto& c) {
        using T = std::decay_t<decltype(c)>;
        return CompareTyped(c, op, std::get<T>(value));
      },
      cell);
}

bool Matches(const ScalarValue& cell, const Condition& condition) {
  if (condition.op == CompareOp::kIn) {
    for (const ScalarValue& value : condition.values) {
      if (Compare(cell, CompareOp::kEq, value)) {
        return true;
      }
    }
    return false;
  }
  return Compare(cell, condition.op, condition.values.front());
}

}  // namespace

absl::Status ValidateFilter(const Filter& filter, const Schema& schema) {
  for (size_t i = 0; i < filter.conditions.size(); ++i) {
    const Condition& condition = filter.conditions[i];
    const std::optional<size_t> column = schema.FindColumn(condition.column);
    if (!column.has_value()) {
      return ConditionError(i, condition.column, "unknown column");
    }
    const auto* type = std::get_if<ScalarType>(&schema.columns()[*column].spec);
    if (type == nullptr) {
      return ConditionError(i, condition.column,
                            "only scalar columns can be filtered");
    }
    if (condition.op == CompareOp::kIn) {
      if (condition.values.empty()) {
        return ConditionError(i, condition.column,
                              "IN requires at least 1 value");
      }
    } else if (condition.values.size() != 1) {
      return ConditionError(
          i, condition.column,
          absl::StrCat("comparison requires exactly 1 value, got ",
                       condition.values.size()));
    }
    for (size_t v = 0; v < condition.values.size(); ++v) {
      if (condition.values[v].index() != AlternativeOf(*type)) {
        return ConditionError(
            i, condition.column,
            absl::StrCat("values[", v, "] must be ", TypeName(*type)));
      }
    }
  }
  return absl::OkStatus();
}

Bitmap EvaluateFilter(const Filter& filter, const Schema& schema,
                      std::span<const ScalarColumn* const> columns,
                      RowId row_count) {
  Bitmap allow(row_count);
  for (RowId id = 0; id < row_count; ++id) {
    allow.Set(id);
  }
  // 条件之间是「并且」：逐个条件清掉不满足的行。
  for (const Condition& condition : filter.conditions) {
    const std::optional<size_t> index = schema.FindColumn(condition.column);
    if (!index.has_value() || *index >= columns.size() ||
        columns[*index] == nullptr) {
      // 违反前置条件（未经 ValidateFilter）：宁可全部排除，也不放行。
      return Bitmap(row_count);
    }
    const ScalarColumn& column = *columns[*index];
    for (RowId id = 0; id < row_count; ++id) {
      if (allow.Test(id) && !Matches(column.Get(id), condition)) {
        allow.Clear(id);
      }
    }
  }
  return allow;
}

}  // namespace hector
