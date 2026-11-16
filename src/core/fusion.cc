#include "hector/core/fusion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "hector/common/types.h"

namespace hector {
namespace {

constexpr uint32_t kDefaultRrfK = 60;

}  // namespace

std::vector<FusedHit> Fuse(std::span<const std::vector<Hit>> ranked_lists,
                           const FusionParams& params, uint32_t top_k) {
  const double rrf_k = params.rrf_k == 0 ? kDefaultRrfK : params.rrf_k;
  // 用 double 累加，最后才转换为 float。
  absl::flat_hash_map<RowId, double> scores;
  for (const std::vector<Hit>& list : ranked_lists) {
    for (size_t i = 0; i < list.size(); ++i) {
      scores[list[i].row_id] += 1.0 / (rrf_k + static_cast<double>(i + 1));
    }
  }

  std::vector<FusedHit> fused;
  fused.reserve(scores.size());
  for (const auto& [row_id, score] : scores) {
    fused.push_back(
        FusedHit{.row_id = row_id, .score = static_cast<float>(score)});
  }
  // 分数降序，同分按 row id 升序；只需要排出前 top_k 条。
  const auto better = [](const FusedHit& a, const FusedHit& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.row_id < b.row_id;
  };
  const size_t keep = std::min<size_t>(top_k, fused.size());
  std::partial_sort(fused.begin(),
                    fused.begin() + static_cast<std::ptrdiff_t>(keep),
                    fused.end(), better);
  fused.resize(keep);
  return fused;
}

}  // namespace hector
