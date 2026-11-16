#ifndef HECTOR_CORE_SEARCH_H_
#define HECTOR_CORE_SEARCH_H_

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "hector/common/types.h"
#include "hector/core/filter.h"
#include "hector/core/fusion.h"
#include "hector/core/value.h"
#include "hector/index/sparse_index.h"

namespace hector {

// 一路检索：在 column 上用 query 检索。查询类型必须与列类型一致。
struct SearchRoute {
  std::string column;
  std::variant<std::vector<float>, SparseVector> query;
  // 0 表示 max(100, top_k)；非 0 时不小于 top_k、不超过 10000。
  // 单路检索时同样校验，但不使用。
  uint32_t candidate_limit = 0;
  // 稠密路；0 表示 64，不超过 10000。
  uint32_t ef_search = 0;
  // 稠密路且开启 int8 时生效；0 表示自动，不超过 10000。
  uint32_t rerank_candidates = 0;
  // 稀疏路；kAuto 表示由索引自行选择。
  SparseAlgorithm sparse_algorithm = SparseAlgorithm::kAuto;
};

struct SearchRequest {
  // 1 到 1024。
  uint32_t top_k = 10;
  // 1 到 8 路；同一列可以出现在多路中。
  std::vector<SearchRoute> routes;
  Filter filter;
  // 多路检索时使用。
  FusionParams fusion;
  // 只能是普通列；为空时只返回主键与分数。
  std::vector<std::string> output_columns;
};

// 单路稠密 L2 为 kL2Distance（升序）；单路稠密 IP/COSINE 与单路稀疏为
// kInnerProduct（降序）；多路为 kRrf（降序）。
// NOLINTNEXTLINE(performance-enum-size): 声明与 design.md 4.4 一致
enum class ScoreKind { kL2Distance, kInnerProduct, kRrf };

struct SearchResultRow {
  std::string primary_key;
  float score;
  // 请求的普通列，按 Schema 列顺序。
  std::vector<std::pair<std::string, Value>> fields;
};

struct SearchResult {
  ScoreKind score_kind;
  std::vector<SearchResultRow> rows;
};

}  // namespace hector

#endif  // HECTOR_CORE_SEARCH_H_
