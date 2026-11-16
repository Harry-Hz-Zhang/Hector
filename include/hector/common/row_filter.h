#ifndef HECTOR_COMMON_ROW_FILTER_H_
#define HECTOR_COMMON_ROW_FILTER_H_

#include <cstddef>
#include <optional>

#include "hector/common/bitmap.h"
#include "hector/common/concurrent_bitmap.h"
#include "hector/common/types.h"

namespace hector {

// 索引看到的唯一筛选接口。不持有位图：调用方保证两个指针在检索期间有效。
class RowFilter {
 public:
  static RowFilter AllowAll(RowId row_count);

  // 两个指针都可以为空。
  RowFilter(RowId row_count, const ConcurrentBitmap* tombstones,
            const Bitmap* allow);

  // id < row_count，且未打墓碑，且 allow 为空或 allow->Test(id)。
  bool Allows(RowId id) const;
  // 由 Table 在检索开始时用 acquire 读到的已发布行数。
  RowId row_count() const;
  // 有 allow 时返回其 Count()，否则 nullopt。
  std::optional<size_t> AllowedCountHint() const;

 private:
  RowId row_count_;
  const ConcurrentBitmap* tombstones_;
  const Bitmap* allow_;
};

}  // namespace hector

#endif  // HECTOR_COMMON_ROW_FILTER_H_
