#include "hector/index/index_factory.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/vector_accessor.h"
#include "hector/index/flat_index.h"
#include "hector/index/flat_sparse_index.h"
#include "hector/index/metric.h"
#include "hector/index/sparse_index.h"
#include "hector/index/vector_index.h"

namespace hector {

absl::StatusOr<std::unique_ptr<VectorIndex>> MakeVectorIndex(
    const DenseIndexConfig& config, const DenseVectorAccessor* accessor) {
  if (accessor == nullptr) {
    return absl::InvalidArgumentError("dense index requires an accessor");
  }
  if (config.kind == DenseIndexKind::kHnsw) {
    return absl::UnimplementedError(
        "HNSW index is not implemented yet (planned for c2)");
  }
  if (config.int8_quantization) {
    return absl::UnimplementedError(
        "int8 quantization is not implemented yet (planned for c6)");
  }
  if (config.kind != DenseIndexKind::kFlat) {
    return absl::InvalidArgumentError("unknown dense index kind");
  }
  if (config.dim == 0 || accessor->dim() != config.dim) {
    return absl::InvalidArgumentError(
        "dense index dim must be positive and match the accessor");
  }
  if (GetDistanceFn(config.metric) == nullptr) {
    return absl::InvalidArgumentError("unknown metric type");
  }
  return std::make_unique<FlatIndex>(config.dim, config.metric, accessor);
}

absl::StatusOr<std::unique_ptr<SparseIndex>> MakeSparseIndex(
    const SparseIndexConfig& config, const SparseVectorAccessor* accessor) {
  if (accessor == nullptr) {
    return absl::InvalidArgumentError("sparse index requires an accessor");
  }
  if (config.kind == SparseIndexKind::kInverted) {
    return absl::UnimplementedError(
        "inverted sparse index is not implemented yet (planned for c7)");
  }
  if (config.kind != SparseIndexKind::kFlat) {
    return absl::InvalidArgumentError("unknown sparse index kind");
  }
  if (config.max_dim == 0) {
    return absl::InvalidArgumentError("sparse index max_dim must be positive");
  }
  return std::make_unique<FlatSparseIndex>(config.max_dim, accessor);
}

}  // namespace hector
