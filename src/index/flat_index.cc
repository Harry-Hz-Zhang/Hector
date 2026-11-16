#include "hector/index/flat_index.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/bitmap.h"
#include "hector/common/byte_io.h"
#include "hector/common/row_filter.h"
#include "hector/common/status.h"
#include "hector/common/types.h"
#include "hector/common/vector_accessor.h"
#include "hector/index/metric.h"
#include "hector/index/vector_index.h"

namespace hector {
namespace {

constexpr uint32_t kFormatVersion = 1;

// 有限向量的距离也可能溢出成 NaN（例如 inf + (-inf)）。NaN 不满足 HitLess
// 的严格弱序，交给堆或排序是未定义行为，因此按 +inf 处理，排在最后。
float SanitizeDistance(float d) {
  return std::isnan(d) ? std::numeric_limits<float>::infinity() : d;
}

// 保留按 HitLess 最好的 k 条；堆顶是当前保留的最差一条。
class TopK {
 public:
  TopK(uint32_t k, size_t candidates) : k_(k) {
    heap_.reserve(std::min<size_t>(k, candidates));
  }

  void Push(const Hit& hit) {
    if (heap_.size() < k_) {
      heap_.push_back(hit);
      std::push_heap(heap_.begin(), heap_.end(), HitLess);
    } else if (HitLess(hit, heap_.front())) {
      std::pop_heap(heap_.begin(), heap_.end(), HitLess);
      heap_.back() = hit;
      std::push_heap(heap_.begin(), heap_.end(), HitLess);
    }
  }

  // 按 HitLess 升序返回。
  std::vector<Hit> Take() && {
    std::sort_heap(heap_.begin(), heap_.end(), HitLess);
    return std::move(heap_);
  }

 private:
  size_t k_;
  std::vector<Hit> heap_;
};

absl::StatusOr<uint32_t> ReadU32OrDataLoss(ByteReader& in) {
  auto value = in.ReadU32();
  if (!value.ok()) {
    return absl::DataLossError("flat index data is truncated");
  }
  return *value;
}

}  // namespace

FlatIndex::FlatIndex(uint32_t dim, MetricType metric,
                     const DenseVectorAccessor* accessor)
    : dim_(dim), distance_(GetDistanceFn(metric)), accessor_(accessor) {}

absl::Status FlatIndex::CheckQuery(std::span<const float> query,
                                   uint32_t top_k) const {
  if (query.size() != dim_) {
    return absl::InvalidArgumentError("query dimension does not match index");
  }
  if (top_k == 0) {
    return absl::InvalidArgumentError("top_k must be positive");
  }
  return absl::OkStatus();
}

Hit FlatIndex::Score(std::span<const float> query, RowId id) const {
  const std::span<const float> row = accessor_->Get(id);
  return Hit{id, SanitizeDistance(distance_(query.data(), row.data(), dim_))};
}

absl::Status FlatIndex::Build(std::span<const RowId> row_ids) {
  std::unique_lock lock(mu_);
  if (!row_ids_.empty()) {
    return absl::FailedPreconditionError("Build requires an empty index");
  }
  const RowId available = accessor_->size();
  Bitmap present;
  std::vector<RowId> ids;
  ids.reserve(row_ids.size());
  for (const RowId id : row_ids) {
    if (id >= available) {
      return absl::InvalidArgumentError("row id is not in the accessor");
    }
    if (present.Test(id)) {
      return absl::InvalidArgumentError("duplicate row id in Build");
    }
    if (present.size() <= id) {
      present.Resize(static_cast<size_t>(id) + 1);
    }
    present.Set(id);
    ids.push_back(id);
  }
  row_ids_ = std::move(ids);
  present_ = std::move(present);
  return absl::OkStatus();
}

absl::Status FlatIndex::Add(RowId row_id) {
  std::unique_lock lock(mu_);
  if (row_id >= accessor_->size()) {
    return absl::InvalidArgumentError("row id is not in the accessor");
  }
  if (present_.Test(row_id)) {
    return absl::AlreadyExistsError("row id is already in the index");
  }
  if (present_.size() <= row_id) {
    present_.Resize(static_cast<size_t>(row_id) + 1);
  }
  present_.Set(row_id);
  row_ids_.push_back(row_id);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Hit>> FlatIndex::Search(
    std::span<const float> query, uint32_t top_k,
    const SearchParams& /*params*/, const RowFilter& filter) const {
  HECTOR_RETURN_IF_ERROR(CheckQuery(query, top_k));
  std::shared_lock lock(mu_);
  TopK top(top_k, row_ids_.size());
  for (const RowId id : row_ids_) {
    // 先判断可见性再读向量：被排除的行（含未发布的行）不访问访问器。
    if (filter.Allows(id)) {
      top.Push(Score(query, id));
    }
  }
  return std::move(top).Take();
}

absl::StatusOr<std::vector<Hit>> FlatIndex::SearchCandidates(
    std::span<const float> query, uint32_t top_k,
    std::span<const RowId> candidates) const {
  HECTOR_RETURN_IF_ERROR(CheckQuery(query, top_k));
  // 去重，避免同一行出现两次。
  std::vector<RowId> unique(candidates.begin(), candidates.end());
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());

  std::shared_lock lock(mu_);
  TopK top(top_k, unique.size());
  for (const RowId id : unique) {
    if (present_.Test(id)) {
      top.Push(Score(query, id));
    }
  }
  return std::move(top).Take();
}

size_t FlatIndex::Size() const {
  std::shared_lock lock(mu_);
  return row_ids_.size();
}

size_t FlatIndex::MemoryBytes() const {
  std::shared_lock lock(mu_);
  return row_ids_.capacity() * sizeof(RowId) + (present_.size() + 7) / 8;
}

absl::Status FlatIndex::Save(ByteWriter& out) const {
  std::shared_lock lock(mu_);
  HECTOR_RETURN_IF_ERROR(out.WriteU32(kFormatVersion));
  HECTOR_RETURN_IF_ERROR(out.WriteU32(static_cast<uint32_t>(row_ids_.size())));
  for (const RowId id : row_ids_) {
    HECTOR_RETURN_IF_ERROR(out.WriteU32(id));
  }
  return absl::OkStatus();
}

absl::Status FlatIndex::Load(ByteReader& in) {
  std::unique_lock lock(mu_);
  if (!row_ids_.empty()) {
    return absl::FailedPreconditionError("Load requires an empty index");
  }
  HECTOR_ASSIGN_OR_RETURN(const uint32_t version, ReadU32OrDataLoss(in));
  if (version != kFormatVersion) {
    return absl::DataLossError("unsupported flat index format version");
  }
  HECTOR_ASSIGN_OR_RETURN(const uint32_t count, ReadU32OrDataLoss(in));
  if (count > kMaxRowsPerTable) {
    return absl::DataLossError("flat index row count is out of range");
  }
  Bitmap present;
  std::vector<RowId> ids;
  ids.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    HECTOR_ASSIGN_OR_RETURN(const RowId id, ReadU32OrDataLoss(in));
    if (id >= kMaxRowsPerTable || present.Test(id)) {
      return absl::DataLossError("flat index contains an invalid row id");
    }
    if (present.size() <= id) {
      present.Resize(static_cast<size_t>(id) + 1);
    }
    present.Set(id);
    ids.push_back(id);
  }
  row_ids_ = std::move(ids);
  present_ = std::move(present);
  return absl::OkStatus();
}

}  // namespace hector
