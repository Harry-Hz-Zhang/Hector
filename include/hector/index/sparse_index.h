#ifndef HECTOR_INDEX_SPARSE_INDEX_H_
#define HECTOR_INDEX_SPARSE_INDEX_H_

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

// NOLINTNEXTLINE(performance-enum-size): 声明与 design.md 4.2 一致
enum class SparseAlgorithm { kAuto, kTermAtATime, kWand };

// Flat 忽略。
struct SparseSearchParams {
  SparseAlgorithm algorithm = SparseAlgorithm::kAuto;
};

// 稀疏向量索引。形状与 VectorIndex 对称，Build/Add/Load 的约定相同。
class SparseIndex {
 public:
  virtual ~SparseIndex() = default;

  virtual absl::Status Build(std::span<const RowId> row_ids) = 0;
  virtual absl::Status Add(RowId row_id) = 0;
  // 按内积打分，distance 为 -内积；只返回与查询至少共享一个位置的行。
  virtual absl::StatusOr<std::vector<Hit>> Search(
      SparseVectorView query, uint32_t top_k, const SparseSearchParams& params,
      const RowFilter& filter) const = 0;
  virtual size_t Size() const = 0;
  virtual size_t MemoryBytes() const = 0;
  virtual absl::Status Save(ByteWriter& out) const = 0;
  virtual absl::Status Load(ByteReader& in) = 0;
};

}  // namespace hector

#endif  // HECTOR_INDEX_SPARSE_INDEX_H_
