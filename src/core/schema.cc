#include "hector/core/schema.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "hector/common/byte_io.h"
#include "hector/common/status.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"

namespace hector {
namespace {

constexpr uint32_t kFormatVersion = 1;
constexpr size_t kMaxNameLength = 64;
constexpr size_t kMaxColumns = 256;
constexpr uint32_t kMaxDenseDim = 4096;
constexpr uint32_t kDefaultHnswM = 16;
constexpr uint32_t kDefaultHnswEfConstruction = 200;
constexpr uint32_t kMinHnswM = 2;
constexpr uint32_t kMaxHnswM = 128;
constexpr uint32_t kMaxHnswEfConstruction = 4096;

constexpr uint8_t kKindDense = 0;
constexpr uint8_t kKindSparse = 1;
constexpr uint8_t kKindScalar = 2;

constexpr std::string_view kNamePattern = "^[A-Za-z_][A-Za-z0-9_]{0,63}$";

bool IsNameStart(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

bool IsNameChar(char c) { return IsNameStart(c) || (c >= '0' && c <= '9'); }

bool IsValidName(std::string_view name) {
  if (name.empty() || name.size() > kMaxNameLength || !IsNameStart(name[0])) {
    return false;
  }
  for (char c : name) {
    if (!IsNameChar(c)) {
      return false;
    }
  }
  return true;
}

bool IsValidMetric(MetricType m) {
  return m == MetricType::kL2 || m == MetricType::kInnerProduct ||
         m == MetricType::kCosine;
}

bool IsValidDenseIndex(DenseIndexKind k) {
  return k == DenseIndexKind::kFlat || k == DenseIndexKind::kHnsw;
}

bool IsValidSparseIndex(SparseIndexKind k) {
  return k == SparseIndexKind::kFlat || k == SparseIndexKind::kInverted;
}

bool IsValidScalarType(ScalarType t) {
  return t == ScalarType::kInt64 || t == ScalarType::kFloat ||
         t == ScalarType::kString;
}

absl::Status ColumnError(std::string_view column, std::string_view detail) {
  return absl::InvalidArgumentError(
      absl::StrCat("column \"", column, "\": ", detail));
}

// 先把 HNSW 参数的 0 换成默认值，再校验范围。
absl::Status NormalizeDense(std::string_view name, DenseColumnSpec& spec) {
  if (spec.dim == 0 || spec.dim > kMaxDenseDim) {
    return ColumnError(name, absl::StrCat("dim must be in [1, ", kMaxDenseDim,
                                          "], got ", spec.dim));
  }
  if (!IsValidMetric(spec.metric)) {
    return ColumnError(name, "metric must be L2, IP or COSINE");
  }
  if (!IsValidDenseIndex(spec.index)) {
    return ColumnError(name, "index must be FLAT or HNSW");
  }
  if (spec.hnsw.m == 0) {
    spec.hnsw.m = kDefaultHnswM;
  }
  if (spec.hnsw.ef_construction == 0) {
    spec.hnsw.ef_construction = kDefaultHnswEfConstruction;
  }
  if (spec.hnsw.m < kMinHnswM || spec.hnsw.m > kMaxHnswM) {
    return ColumnError(
        name, absl::StrCat("hnsw.m must be in [", kMinHnswM, ", ", kMaxHnswM,
                           "], got ", spec.hnsw.m));
  }
  if (spec.hnsw.ef_construction < spec.hnsw.m ||
      spec.hnsw.ef_construction > kMaxHnswEfConstruction) {
    return ColumnError(
        name, absl::StrCat("hnsw.ef_construction must be in [hnsw.m=",
                           spec.hnsw.m, ", ", kMaxHnswEfConstruction, "], got ",
                           spec.hnsw.ef_construction));
  }
  return absl::OkStatus();
}

absl::Status ValidateSparse(std::string_view name,
                            const SparseColumnSpec& spec) {
  if (spec.max_dim == 0) {
    return ColumnError(name, "max_dim must be at least 1");
  }
  if (!IsValidSparseIndex(spec.index)) {
    return ColumnError(name, "index must be FLAT or INVERTED");
  }
  return absl::OkStatus();
}

absl::Status DataLoss(std::string_view what) {
  return absl::DataLossError(absl::StrCat("schema data: ", what));
}

// 读取失败统一转为 DataLoss。
template <typename T>
absl::StatusOr<T> OrDataLoss(absl::StatusOr<T> value) {
  if (!value.ok()) {
    return DataLoss("truncated");
  }
  return value;
}

absl::StatusOr<ColumnDef> LoadColumn(ByteReader& in) {
  ColumnDef column;
  HECTOR_ASSIGN_OR_RETURN(column.name, OrDataLoss(in.ReadString()));
  HECTOR_ASSIGN_OR_RETURN(const uint8_t kind, OrDataLoss(in.ReadU8()));
  switch (kind) {
    case kKindDense: {
      DenseColumnSpec spec{};
      HECTOR_ASSIGN_OR_RETURN(spec.dim, OrDataLoss(in.ReadU32()));
      HECTOR_ASSIGN_OR_RETURN(const uint8_t metric, OrDataLoss(in.ReadU8()));
      HECTOR_ASSIGN_OR_RETURN(const uint8_t index, OrDataLoss(in.ReadU8()));
      HECTOR_ASSIGN_OR_RETURN(spec.hnsw.m, OrDataLoss(in.ReadU32()));
      HECTOR_ASSIGN_OR_RETURN(spec.hnsw.ef_construction,
                              OrDataLoss(in.ReadU32()));
      HECTOR_ASSIGN_OR_RETURN(spec.hnsw.seed, OrDataLoss(in.ReadU64()));
      HECTOR_ASSIGN_OR_RETURN(const uint8_t int8, OrDataLoss(in.ReadU8()));
      spec.metric = static_cast<MetricType>(metric);
      spec.index = static_cast<DenseIndexKind>(index);
      spec.int8_quantization = int8 != 0;
      column.spec = spec;
      break;
    }
    case kKindSparse: {
      SparseColumnSpec spec{};
      HECTOR_ASSIGN_OR_RETURN(spec.max_dim, OrDataLoss(in.ReadU32()));
      HECTOR_ASSIGN_OR_RETURN(const uint8_t index, OrDataLoss(in.ReadU8()));
      spec.index = static_cast<SparseIndexKind>(index);
      column.spec = spec;
      break;
    }
    case kKindScalar: {
      HECTOR_ASSIGN_OR_RETURN(const uint8_t type, OrDataLoss(in.ReadU8()));
      column.spec = static_cast<ScalarType>(type);
      break;
    }
    default:
      return DataLoss(absl::StrCat("unknown column kind ", kind));
  }
  return column;
}

}  // namespace

Schema::Schema(std::string table_name, std::vector<ColumnDef> columns)
    : table_name_(std::move(table_name)), columns_(std::move(columns)) {
  column_index_.reserve(columns_.size());
  for (size_t i = 0; i < columns_.size(); ++i) {
    column_index_.emplace(columns_[i].name, i);
  }
}

absl::StatusOr<Schema> Schema::Create(std::string table_name,
                                      std::vector<ColumnDef> columns) {
  if (!IsValidName(table_name)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "table name \"", table_name, "\" must match ", kNamePattern));
  }
  if (columns.empty() || columns.size() > kMaxColumns) {
    return absl::InvalidArgumentError(
        absl::StrCat("table \"", table_name, "\" must have 1 to ", kMaxColumns,
                     " columns, got ", columns.size()));
  }
  absl::flat_hash_map<std::string_view, bool> seen;
  bool has_vector = false;
  for (ColumnDef& column : columns) {
    if (!IsValidName(column.name)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "column name \"", column.name, "\" must match ", kNamePattern));
    }
    if (!seen.emplace(column.name, true).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate column name \"", column.name, "\""));
    }
    if (auto* dense = std::get_if<DenseColumnSpec>(&column.spec)) {
      HECTOR_RETURN_IF_ERROR(NormalizeDense(column.name, *dense));
      has_vector = true;
    } else if (const auto* sparse =
                   std::get_if<SparseColumnSpec>(&column.spec)) {
      HECTOR_RETURN_IF_ERROR(ValidateSparse(column.name, *sparse));
      has_vector = true;
    } else if (!IsValidScalarType(std::get<ScalarType>(column.spec))) {
      return ColumnError(column.name, "type must be int64, float or string");
    }
  }
  if (!has_vector) {
    return absl::InvalidArgumentError(absl::StrCat(
        "table \"", table_name, "\" must have at least one vector column"));
  }
  return Schema(std::move(table_name), std::move(columns));
}

const std::string& Schema::table_name() const { return table_name_; }

const std::vector<ColumnDef>& Schema::columns() const { return columns_; }

std::optional<size_t> Schema::FindColumn(std::string_view name) const {
  auto it = column_index_.find(name);
  if (it == column_index_.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::Status Schema::Save(ByteWriter& out) const {
  HECTOR_RETURN_IF_ERROR(out.WriteU32(kFormatVersion));
  HECTOR_RETURN_IF_ERROR(out.WriteString(table_name_));
  HECTOR_RETURN_IF_ERROR(out.WriteU32(static_cast<uint32_t>(columns_.size())));
  for (const ColumnDef& column : columns_) {
    HECTOR_RETURN_IF_ERROR(out.WriteString(column.name));
    if (const auto* dense = std::get_if<DenseColumnSpec>(&column.spec)) {
      HECTOR_RETURN_IF_ERROR(out.WriteU8(kKindDense));
      HECTOR_RETURN_IF_ERROR(out.WriteU32(dense->dim));
      HECTOR_RETURN_IF_ERROR(out.WriteU8(static_cast<uint8_t>(dense->metric)));
      HECTOR_RETURN_IF_ERROR(out.WriteU8(static_cast<uint8_t>(dense->index)));
      HECTOR_RETURN_IF_ERROR(out.WriteU32(dense->hnsw.m));
      HECTOR_RETURN_IF_ERROR(out.WriteU32(dense->hnsw.ef_construction));
      HECTOR_RETURN_IF_ERROR(out.WriteU64(dense->hnsw.seed));
      HECTOR_RETURN_IF_ERROR(out.WriteU8(dense->int8_quantization ? 1 : 0));
    } else if (const auto* sparse =
                   std::get_if<SparseColumnSpec>(&column.spec)) {
      HECTOR_RETURN_IF_ERROR(out.WriteU8(kKindSparse));
      HECTOR_RETURN_IF_ERROR(out.WriteU32(sparse->max_dim));
      HECTOR_RETURN_IF_ERROR(out.WriteU8(static_cast<uint8_t>(sparse->index)));
    } else {
      HECTOR_RETURN_IF_ERROR(out.WriteU8(kKindScalar));
      HECTOR_RETURN_IF_ERROR(
          out.WriteU8(static_cast<uint8_t>(std::get<ScalarType>(column.spec))));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Schema> Schema::Load(ByteReader& in) {
  HECTOR_ASSIGN_OR_RETURN(const uint32_t version, OrDataLoss(in.ReadU32()));
  if (version != kFormatVersion) {
    return DataLoss(absl::StrCat("unsupported version ", version));
  }
  HECTOR_ASSIGN_OR_RETURN(std::string table_name, OrDataLoss(in.ReadString()));
  HECTOR_ASSIGN_OR_RETURN(const uint32_t count, OrDataLoss(in.ReadU32()));
  if (count > kMaxColumns) {
    return DataLoss(absl::StrCat("too many columns: ", count));
  }
  std::vector<ColumnDef> columns;
  columns.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    HECTOR_ASSIGN_OR_RETURN(ColumnDef column, LoadColumn(in));
    columns.push_back(std::move(column));
  }
  auto schema = Create(std::move(table_name), std::move(columns));
  if (!schema.ok()) {
    return DataLoss(schema.status().message());
  }
  return schema;
}

}  // namespace hector
