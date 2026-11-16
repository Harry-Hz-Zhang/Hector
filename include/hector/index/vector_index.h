#ifndef HECTOR_INDEX_VECTOR_INDEX_H_
#define HECTOR_INDEX_VECTOR_INDEX_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/row_filter.h"
#include "hector/common/types.h"

namespace hector {

struct SearchParams {
  // HNSW 使用；Table 只负责把请求中的 0 换成 64，由索引自己取
  // max(ef_search, top_k)。
  uint32_t ef_search = 64;
  // int8 索引使用；0 表示自动。
  uint32_t rerank_candidates = 0;
};

// 稠密向量索引。实现必须线程安全：Search 可以与 Search、Add 并发。
class VectorIndex {
 public:
  virtual ~VectorIndex() = default;

  // 批量构建。只能在空索引上调用，否则返回 FailedPrecondition。
  virtual absl::Status Build(std::span<const RowId> row_ids) = 0;
  // 加入一行，从访问器读向量；重复加入返回 AlreadyExists。
  // 任何索引都必须允许在 Build 之前调用 Add。Add 与 Add、Add 与 Search
  // 都可以并发调用。
  virtual absl::Status Add(RowId row_id) = 0;
  // 返回 filter 允许的行中最好的 top_k 条，按 HitLess 排序。
  virtual absl::StatusOr<std::vector<Hit>> Search(
      std::span<const float> query, uint32_t top_k, const SearchParams& params,
      const RowFilter& filter) const = 0;
  // 对给定候选集做精确检索（c4 回退）。不在索引中的 row id 静默跳过；
  // 墓碑与筛选由调用方事先排除。
  virtual absl::StatusOr<std::vector<Hit>> SearchCandidates(
      std::span<const float> query, uint32_t top_k,
      std::span<const RowId> candidates) const = 0;
  virtual size_t Size() const = 0;
  // 派生结构的字节数，不含原始向量。
  virtual size_t MemoryBytes() const = 0;
  virtual absl::Status Save(ByteWriter& out) const = 0;
  // 只能在空索引上调用。
  virtual absl::Status Load(ByteReader& in) = 0;
};

}  // namespace hector

#endif  // HECTOR_INDEX_VECTOR_INDEX_H_
