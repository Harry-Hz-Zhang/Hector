#ifndef HECTOR_INDEX_METRIC_H_
#define HECTOR_INDEX_METRIC_H_

#include <cstddef>
#include <span>

#include "absl/status/status.h"

namespace hector {

// NOLINTNEXTLINE(performance-enum-size): 声明与 design.md 4.2 一致
enum class MetricType { kL2, kInnerProduct, kCosine };

// 返回「越小越好」的距离。
using DistanceFn = float (*)(const float* a, const float* b, size_t dim);

// 平方欧氏距离 Σ (a[i] - b[i])^2。
float L2SquaredScalar(const float* a, const float* b, size_t dim);
// 内积 Σ a[i] * b[i]。
float InnerProductScalar(const float* a, const float* b, size_t dim);

// kL2 返回 L2 平方；kInnerProduct / kCosine 返回负内积。
// 不认识的取值返回 nullptr。c1 在此切换 NEON 实现。
DistanceFn GetDistanceFn(MetricType m);

// 把 v 归一化为单位长度。范数为 0 或含非有限值时返回 InvalidArgument，且不修改
// v。
absl::Status NormalizeInPlace(std::span<float> v);

}  // namespace hector

#endif  // HECTOR_INDEX_METRIC_H_
