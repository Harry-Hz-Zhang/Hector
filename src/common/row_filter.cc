#include "hector/common/row_filter.h"

#include <cstddef>
#include <optional>

#include "hector/common/bitmap.h"
#include "hector/common/concurrent_bitmap.h"
#include "hector/common/types.h"

namespace hector {

RowFilter RowFilter::AllowAll(RowId row_count) {
  return RowFilter(row_count, nullptr, nullptr);
}

RowFilter::RowFilter(RowId row_count, const ConcurrentBitmap* tombstones,
                     const Bitmap* allow)
    : row_count_(row_count), tombstones_(tombstones), allow_(allow) {}

bool RowFilter::Allows(RowId id) const {
  if (id >= row_count_) {
    return false;
  }
  if (tombstones_ != nullptr && tombstones_->Test(id)) {
    return false;
  }
  return allow_ == nullptr || allow_->Test(id);
}

RowId RowFilter::row_count() const { return row_count_; }

std::optional<size_t> RowFilter::AllowedCountHint() const {
  if (allow_ == nullptr) {
    return std::nullopt;
  }
  return allow_->Count();
}

}  // namespace hector
