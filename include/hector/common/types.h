#ifndef HECTOR_COMMON_TYPES_H_
#define HECTOR_COMMON_TYPES_H_

#include <cstdint>
#include <span>
#include <vector>

namespace hector {

// 表内连续编号，从 0 开始。
using RowId = uint32_t;
// 日志序号，从 1 开始；0 表示「无」。
using Lsn = uint64_t;

// 单表行数上限：16,777,216 行（c0 设计默认）。
inline constexpr RowId kMaxRowsPerTable = 1U << 24U;

// 一条检索命中。distance 越小越好：L2 为平方距离，IP/COSINE/稀疏为 -内积。
struct Hit {
  RowId row_id;
  float distance;
};

// distance 升序；distance 相同时按 row_id 升序。
bool HitLess(const Hit& a, const Hit& b);

// 稀疏向量的只读视图：indices 与 values 等长。
struct SparseVectorView {
  std::span<const uint32_t> indices;
  std::span<const float> values;
};

// 持有数据的稀疏向量：indices 与 values 等长。
struct SparseVector {
  std::vector<uint32_t> indices;
  std::vector<float> values;
};

}  // namespace hector

#endif  // HECTOR_COMMON_TYPES_H_
