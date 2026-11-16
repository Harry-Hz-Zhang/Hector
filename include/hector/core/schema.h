#ifndef HECTOR_CORE_SCHEMA_H_
#define HECTOR_CORE_SCHEMA_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"

namespace hector {

// NOLINTNEXTLINE(performance-enum-size): 声明与 design.md 4.4 一致
enum class ScalarType { kInt64, kFloat, kString };

struct DenseColumnSpec {
  uint32_t dim;
  MetricType metric;
  DenseIndexKind index;
  // m 为 0 时 Schema::Create 取 16，ef_construction 为 0 时取 200。
  HnswConfig hnsw;
  bool int8_quantization = false;
};

// 相似度固定为内积。
struct SparseColumnSpec {
  uint32_t max_dim;
  SparseIndexKind index;
};

struct ColumnDef {
  std::string name;
  std::variant<DenseColumnSpec, SparseColumnSpec, ScalarType> spec;
};

// 表结构：创建后不可变，可以在线程间共享只读访问。
class Schema {
 public:
  // 校验见 design.md 第 5 节：表名与列名匹配 ^[A-Za-z_][A-Za-z0-9_]{0,63}$；
  // 1 到 256 列、至少 1 个向量列、列名不重复；稠密 dim 在 1 到 4096；
  // HNSW 参数先把 0 换成默认值（m=16、ef_construction=200），再校验 m 在
  // 2 到 128、ef_construction 在 m 到 4096；稀疏 max_dim 至少为 1。
  // 失败返回 InvalidArgument，消息含出错的表名、列名或字段名。
  // HNSW、INVERTED、int8 在这里合法，由 Table::Create 的索引工厂拒绝。
  static absl::StatusOr<Schema> Create(std::string table_name,
                                       std::vector<ColumnDef> columns);

  const std::string& table_name() const;
  const std::vector<ColumnDef>& columns() const;
  std::optional<size_t> FindColumn(std::string_view name) const;

  // WAL 与快照共用。编码：u32 版本=1、表名、u32 列数，每列为列名、u8 类别
  // （0 稠密、1 稀疏、2 普通）与该类别的字段，枚举以 u8 保存。
  absl::Status Save(ByteWriter& out) const;
  // 数据截断、版本不符或解码出的 Schema 不合法时返回 DataLoss。
  static absl::StatusOr<Schema> Load(ByteReader& in);

 private:
  Schema(std::string table_name, std::vector<ColumnDef> columns);

  std::string table_name_;
  std::vector<ColumnDef> columns_;
  absl::flat_hash_map<std::string, size_t> column_index_;
};

}  // namespace hector

#endif  // HECTOR_CORE_SCHEMA_H_
