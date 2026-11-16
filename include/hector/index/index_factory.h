#ifndef HECTOR_INDEX_INDEX_FACTORY_H_
#define HECTOR_INDEX_INDEX_FACTORY_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "hector/common/vector_accessor.h"
#include "hector/index/metric.h"
#include "hector/index/sparse_index.h"
#include "hector/index/vector_index.h"

namespace hector {

// core 把 ColumnDef 翻译成这里的配置，index 不认识 core 的类型。
// NOLINTNEXTLINE(performance-enum-size): 声明与 design.md 4.2 一致
enum class DenseIndexKind { kFlat, kHnsw };
// NOLINTNEXTLINE(performance-enum-size): 声明与 design.md 4.2 一致
enum class SparseIndexKind { kFlat, kInverted };

struct HnswConfig {
  uint32_t m = 16;
  uint32_t ef_construction = 200;
  uint64_t seed = 42;
};

struct DenseIndexConfig {
  uint32_t dim;
  MetricType metric;
  DenseIndexKind kind;
  HnswConfig hnsw;
  bool int8_quantization = false;
};

struct SparseIndexConfig {
  uint32_t max_dim;
  SparseIndexKind kind;
};

// 访问器不归索引所有：调用方保证它比索引活得久。
// c0：kHnsw、int8_quantization=true 返回 Unimplemented，消息分别含 c2、c6。
absl::StatusOr<std::unique_ptr<VectorIndex>> MakeVectorIndex(
    const DenseIndexConfig& config, const DenseVectorAccessor* accessor);
// c0：kInverted 返回 Unimplemented，消息含 c7。
absl::StatusOr<std::unique_ptr<SparseIndex>> MakeSparseIndex(
    const SparseIndexConfig& config, const SparseVectorAccessor* accessor);

}  // namespace hector

#endif  // HECTOR_INDEX_INDEX_FACTORY_H_
