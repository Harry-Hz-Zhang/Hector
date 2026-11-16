#ifndef HECTOR_INDEX_FLAT_SPARSE_INDEX_H_
#define HECTOR_INDEX_FLAT_SPARSE_INDEX_H_

#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <span>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/bitmap.h"
#include "hector/common/byte_io.h"
#include "hector/common/row_filter.h"
#include "hector/common/types.h"
#include "hector/common/vector_accessor.h"
#include "hector/index/sparse_index.h"

namespace hector {

// 暴力检索的稀疏索引：只保存已加入的 row id，向量通过访问器读取。
// 要求访问器返回的每行 indices 严格升序（由 core 写入时保证）。
// 读写由 std::shared_mutex 保护：Search 持读锁，Add/Build/Load 持写锁。
class FlatSparseIndex final : public SparseIndex {
 public:
  // accessor 不归索引所有，调用方保证它比索引活得久。
  FlatSparseIndex(uint32_t max_dim, const SparseVectorAccessor* accessor);

  absl::Status Build(std::span<const RowId> row_ids) override;
  absl::Status Add(RowId row_id) override;
  // 查询的 indices 必须严格升序且小于 max_dim。Flat 忽略 params。
  absl::StatusOr<std::vector<Hit>> Search(
      SparseVectorView query, uint32_t top_k, const SparseSearchParams& params,
      const RowFilter& filter) const override;
  size_t Size() const override;
  size_t MemoryBytes() const override;
  // 编码与 FlatIndex 相同：u32 版本=1、u32 行数、各 row id（u32）。
  absl::Status Save(ByteWriter& out) const override;
  absl::Status Load(ByteReader& in) override;

 private:
  uint32_t max_dim_;
  const SparseVectorAccessor* accessor_;

  mutable std::shared_mutex mu_;
  std::vector<RowId> row_ids_;  // 加入顺序
  Bitmap present_;              // present_.Test(id) 当且仅当 id 已加入
};

}  // namespace hector

#endif  // HECTOR_INDEX_FLAT_SPARSE_INDEX_H_
