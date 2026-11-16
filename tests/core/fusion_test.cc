#include "hector/core/fusion.h"

#include <gtest/gtest.h>

#include <vector>

#include "hector/common/types.h"

namespace hector {
namespace {

constexpr RowId kA = 0;
constexpr RowId kB = 1;
constexpr RowId kC = 2;

TEST(FusionTest, RrfMatchesHandComputed) {
  // 稠密路 [a, b]，稀疏路 [b, c]；distance 只决定各路内部的排名，不参与融合。
  const std::vector<std::vector<Hit>> lists = {
      {Hit{.row_id = kA, .distance = 0.1F},
       Hit{.row_id = kB, .distance = 0.2F}},
      {Hit{.row_id = kB, .distance = -5.0F},
       Hit{.row_id = kC, .distance = -1.0F}}};

  const std::vector<FusedHit> fused =
      Fuse(lists, FusionParams{.method = FusionMethod::kRrf, .rrf_k = 60}, 3);

  ASSERT_EQ(fused.size(), 3U);
  EXPECT_EQ(fused[0].row_id, kB);
  EXPECT_NEAR(fused[0].score, 1.0 / 62 + 1.0 / 61, 1e-6);
  EXPECT_EQ(fused[1].row_id, kA);
  EXPECT_NEAR(fused[1].score, 1.0 / 61, 1e-6);
  EXPECT_EQ(fused[2].row_id, kC);
  EXPECT_NEAR(fused[2].score, 1.0 / 62, 1e-6);

  // rrf_k 为 0 时按默认 60 计算。
  const std::vector<FusedHit> defaulted =
      Fuse(lists, FusionParams{.method = FusionMethod::kRrf, .rrf_k = 0}, 3);
  ASSERT_EQ(defaulted.size(), 3U);
  for (size_t i = 0; i < defaulted.size(); ++i) {
    EXPECT_EQ(defaulted[i].row_id, fused[i].row_id);
    EXPECT_EQ(defaulted[i].score, fused[i].score);
  }
}

TEST(FusionTest, TiesBrokenByRowIdAndTruncatedToTopK) {
  constexpr RowId kX = 3;
  constexpr RowId kY = 7;
  const FusionParams params{.method = FusionMethod::kRrf, .rrf_k = 60};

  // 两种输入顺序都应让 row id 较小的 x 排在前面。
  const std::vector<std::vector<Hit>> x_first = {
      {Hit{.row_id = kX, .distance = 0.0F}},
      {Hit{.row_id = kY, .distance = 0.0F}}};
  const std::vector<std::vector<Hit>> y_first = {
      {Hit{.row_id = kY, .distance = 0.0F}},
      {Hit{.row_id = kX, .distance = 0.0F}}};
  for (const auto& lists : {x_first, y_first}) {
    const std::vector<FusedHit> fused = Fuse(lists, params, 10);
    ASSERT_EQ(fused.size(), 2U);
    EXPECT_EQ(fused[0].row_id, kX);
    EXPECT_EQ(fused[1].row_id, kY);
    EXPECT_EQ(fused[0].score, fused[1].score);

    const std::vector<FusedHit> top1 = Fuse(lists, params, 1);
    ASSERT_EQ(top1.size(), 1U);
    EXPECT_EQ(top1[0].row_id, kX);
  }

  const std::vector<std::vector<Hit>> empty_lists = {{}, {}};
  EXPECT_TRUE(Fuse(empty_lists, params, 5).empty());
  EXPECT_TRUE(Fuse({}, params, 5).empty());
}

}  // namespace
}  // namespace hector
