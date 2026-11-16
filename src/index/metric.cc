#include "hector/index/metric.h"

#include <cmath>
#include <cstddef>
#include <span>

#include "absl/status/status.h"

namespace hector {
namespace {

float NegativeInnerProductScalar(const float* a, const float* b, size_t dim) {
  return -InnerProductScalar(a, b, dim);
}

}  // namespace

float L2SquaredScalar(const float* a, const float* b, size_t dim) {
  float sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

float InnerProductScalar(const float* a, const float* b, size_t dim) {
  float sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

DistanceFn GetDistanceFn(MetricType m) {
  switch (m) {
    case MetricType::kL2:
      return &L2SquaredScalar;
    case MetricType::kInnerProduct:
    case MetricType::kCosine:
      return &NegativeInnerProductScalar;
  }
  return nullptr;
}

absl::Status NormalizeInPlace(std::span<float> v) {
  // 用 double 累加：有限 float 的平方和不会溢出 double（4096 维时约 1e80）。
  double sum = 0;
  for (const float x : v) {
    if (!std::isfinite(x)) {
      return absl::InvalidArgumentError("vector contains a non-finite value");
    }
    sum += static_cast<double>(x) * static_cast<double>(x);
  }
  if (sum == 0) {
    return absl::InvalidArgumentError("cannot normalize a zero vector");
  }
  const double norm = std::sqrt(sum);
  for (float& x : v) {
    x = static_cast<float>(static_cast<double>(x) / norm);
  }
  return absl::OkStatus();
}

}  // namespace hector
