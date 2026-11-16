#ifndef HECTOR_CORE_VALUE_H_
#define HECTOR_CORE_VALUE_H_

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "hector/common/types.h"

namespace hector {

// 一个字段的值：普通列的 int64 / float / string，或稠密、稀疏向量。
using Value =
    std::variant<int64_t, float, std::string, std::vector<float>, SparseVector>;
// 普通列的值。
using ScalarValue = std::variant<int64_t, float, std::string>;

// 写入的一行：必须给出 Schema 中的全部列，不多不少。
struct RowInput {
  std::string primary_key;
  absl::flat_hash_map<std::string, Value> fields;
};

// 读出的一行：fields 按 Schema 列顺序排列。
struct RowOutput {
  std::string primary_key;
  std::vector<std::pair<std::string, Value>> fields;
};

}  // namespace hector

#endif  // HECTOR_CORE_VALUE_H_
