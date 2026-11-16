#include "hector/common/types.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <type_traits>
#include <vector>

#include "hector/common/vector_accessor.h"

namespace hector {
namespace {

static_assert(std::is_abstract_v<DenseVectorAccessor>);
static_assert(std::is_abstract_v<SparseVectorAccessor>);
static_assert(kMaxRowsPerTable == 16777216U);

TEST(TypesTest, HitLessBreaksTiesByRowId) {
  std::vector<Hit> hits = {{5, 1.0F}, {2, 1.0F}, {9, 0.5F}};
  std::sort(hits.begin(), hits.end(), HitLess);
  ASSERT_EQ(hits.size(), 3U);
  EXPECT_EQ(hits[0].row_id, 9U);
  EXPECT_EQ(hits[1].row_id, 2U);
  EXPECT_EQ(hits[2].row_id, 5U);
  EXPECT_FALSE(HitLess(hits[1], hits[1]));
}

}  // namespace
}  // namespace hector
