#ifndef HECTOR_INDEX_FLAT_INDEX_H_
#define HECTOR_INDEX_FLAT_INDEX_H_

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
#include "hector/index/metric.h"
#include "hector/index/vector_index.h"

namespace hector {

// 暴力检索的稠密索引：只保存已加入的 row id，向量通过访问器读取。
// 读写由 std::shared_mutex 保护：Search 持读锁，Add/Build/Load 持写锁。
class FlatIndex final : public VectorIndex {
 public:
  // accessor 不归索引所有，调用方保证它比索引活得久；accessor->dim() == dim，
  // metric 是合法取值（工厂负责校验）。
  FlatIndex(uint32_t dim, MetricType metric,
            const DenseVectorAccessor* accessor);

  // row id 必须小于 accessor->size() 且互不重复，否则返回 InvalidArgument，
  // 索引保持不变。
  absl::Status Build(std::span<const RowId> row_ids) override;
  // row id 不小于 accessor->size() 时返回 InvalidArgument。
  absl::Status Add(RowId row_id) override;
  // Flat 是精确检索，忽略 params。NaN 距离按 +inf 处理。
  absl::StatusOr<std::vector<Hit>> Search(
      std::span<const float> query, uint32_t top_k, const SearchParams& params,
      const RowFilter& filter) const override;
  absl::StatusOr<std::vector<Hit>> SearchCandidates(
      std::span<const float> query, uint32_t top_k,
      std::span<const RowId> candidates) const override;
  size_t Size() const override;
  size_t MemoryBytes() const override;
  // 编码：u32 版本=1、u32 行数、各 row id（u32）。
  absl::Status Save(ByteWriter& out) const override;
  // 版本不符、数据截断或 row id 非法（越界、重复）返回 DataLoss，索引保持为空。
  absl::Status Load(ByteReader& in) override;

 private:
  absl::Status CheckQuery(std::span<const float> query, uint32_t top_k) const;
  // 前置条件：id 已加入索引。
  Hit Score(std::span<const float> query, RowId id) const;

  uint32_t dim_;
  DistanceFn distance_;
  const DenseVectorAccessor* accessor_;

  mutable std::shared_mutex mu_;
  std::vector<RowId> row_ids_;  // 加入顺序
  Bitmap present_;              // present_.Test(id) 当且仅当 id 已加入
};

}  // namespace hector

#endif  // HECTOR_INDEX_FLAT_INDEX_H_
