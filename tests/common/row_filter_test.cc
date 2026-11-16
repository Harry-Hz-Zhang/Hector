#include "hector/common/row_filter.h"

#include <gtest/gtest.h>

#include <optional>

#include "hector/common/bitmap.h"
#include "hector/common/concurrent_bitmap.h"

namespace hector {
namespace {

TEST(RowFilterTest, CombinesTombstoneAndAllow) {
  ConcurrentBitmap tombstones(16);
  tombstones.Set(1);
  Bitmap allow(4);
  allow.Set(1);
  allow.Set(2);

  RowFilter filter(4, &tombstones, &allow);
  EXPECT_EQ(filter.row_count(), 4U);
  EXPECT_FALSE(filter.Allows(0));
  EXPECT_FALSE(filter.Allows(1));
  EXPECT_TRUE(filter.Allows(2));
  EXPECT_FALSE(filter.Allows(3));
  EXPECT_FALSE(filter.Allows(4));
  EXPECT_EQ(filter.AllowedCountHint(), std::optional<size_t>(2));

  // 只有墓碑、没有 allow 时：未打墓碑且在 row_count 内的行都可见。
  RowFilter tombstone_only(4, &tombstones, nullptr);
  EXPECT_TRUE(tombstone_only.Allows(0));
  EXPECT_FALSE(tombstone_only.Allows(1));
  EXPECT_TRUE(tombstone_only.Allows(3));
  EXPECT_EQ(tombstone_only.AllowedCountHint(), std::nullopt);
}

TEST(RowFilterTest, AllowAllHasNoHint) {
  const RowFilter filter = RowFilter::AllowAll(3);
  EXPECT_EQ(filter.row_count(), 3U);
  EXPECT_TRUE(filter.Allows(0));
  EXPECT_TRUE(filter.Allows(1));
  EXPECT_TRUE(filter.Allows(2));
  EXPECT_FALSE(filter.Allows(3));
  EXPECT_EQ(filter.AllowedCountHint(), std::nullopt);
}

}  // namespace
}  // namespace hector
