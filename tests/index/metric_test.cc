#include "hector/index/metric.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "absl/status/status.h"

namespace hector {
namespace {

TEST(MetricTest, ScalarKernelsMatchHandComputed) {
  const std::vector<float> a = {1, 2, 3};
  const std::vector<float> b = {4, 5, 6};
  EXPECT_FLOAT_EQ(L2SquaredScalar(a.data(), b.data(), 3), 27.0F);
  EXPECT_FLOAT_EQ(InnerProductScalar(a.data(), b.data(), 3), 32.0F);

  // dim=1：(2-5)^2 = 9，2*5 = 10。
  const float x1 = 2;
  const float y1 = 5;
  EXPECT_FLOAT_EQ(L2SquaredScalar(&x1, &y1, 1), 9.0F);
  EXPECT_FLOAT_EQ(InnerProductScalar(&x1, &y1, 1), 10.0F);

  // dim=7：x = (1..7)，y = 全 1 → L2 = Σ (i-1)^2 = 0+1+4+9+16+25+36 = 91，
  // IP = Σ i = 28。
  const std::vector<float> x7 = {1, 2, 3, 4, 5, 6, 7};
  const std::vector<float> y7 = {1, 1, 1, 1, 1, 1, 1};
  EXPECT_FLOAT_EQ(L2SquaredScalar(x7.data(), y7.data(), 7), 91.0F);
  EXPECT_FLOAT_EQ(InnerProductScalar(x7.data(), y7.data(), 7), 28.0F);

  // GetDistanceFn：L2 为平方距离，IP 与 COSINE 为负内积。
  EXPECT_FLOAT_EQ(GetDistanceFn(MetricType::kL2)(a.data(), b.data(), 3), 27.0F);
  EXPECT_FLOAT_EQ(
      GetDistanceFn(MetricType::kInnerProduct)(a.data(), b.data(), 3), -32.0F);
  EXPECT_FLOAT_EQ(GetDistanceFn(MetricType::kCosine)(a.data(), b.data(), 3),
                  -32.0F);
}

TEST(MetricTest, NormalizeRejectsZeroAndNonFinite) {
  std::vector<float> v = {3, 4};
  ASSERT_TRUE(NormalizeInPlace(v).ok());
  EXPECT_NEAR(v[0], 0.6F, 1e-6);
  EXPECT_NEAR(v[1], 0.8F, 1e-6);

  std::vector<float> zero = {0, 0};
  EXPECT_EQ(NormalizeInPlace(zero).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(zero, (std::vector<float>{0, 0}));

  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> with_nan = {1, nan, 2};
  EXPECT_EQ(NormalizeInPlace(with_nan).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(with_nan[0], 1.0F);
  EXPECT_TRUE(std::isnan(with_nan[1]));
  EXPECT_EQ(with_nan[2], 2.0F);

  std::vector<float> with_inf = {1, std::numeric_limits<float>::infinity()};
  EXPECT_EQ(NormalizeInPlace(with_inf).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(with_inf[0], 1.0F);

  // 平方和超出 float 范围的有限向量仍能正确归一化。
  const float big = std::numeric_limits<float>::max();
  std::vector<float> large = {big, big};
  ASSERT_TRUE(NormalizeInPlace(large).ok());
  EXPECT_NEAR(large[0], std::sqrt(0.5F), 1e-6);
  EXPECT_NEAR(large[1], std::sqrt(0.5F), 1e-6);
}

}  // namespace
}  // namespace hector
