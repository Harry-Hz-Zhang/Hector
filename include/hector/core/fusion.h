#ifndef HECTOR_CORE_FUSION_H_
#define HECTOR_CORE_FUSION_H_

#include <cstdint>
#include <span>
#include <vector>

#include "hector/common/types.h"

namespace hector {

// NOLINTNEXTLINE(performance-enum-size): 声明与 design.md 4.4 一致
enum class FusionMethod { kRrf };

struct FusionParams {
  FusionMethod method = FusionMethod::kRrf;
  // 0 表示默认 60。
  uint32_t rrf_k = 60;
};

// score 越大越好。
struct FusedHit {
  RowId row_id;
  float score;
};

// 融合多路检索结果。每一路按 HitLess 排好序，rank 从 1 开始；RRF 分数为
// Σ 1 / (rrf_k + rank)，某行只出现在部分路中时只累加出现的那几路。
// 返回按分数降序、同分按 row id 升序的前 top_k 条。
std::vector<FusedHit> Fuse(std::span<const std::vector<Hit>> ranked_lists,
                           const FusionParams& params, uint32_t top_k);

}  // namespace hector

#endif  // HECTOR_CORE_FUSION_H_
