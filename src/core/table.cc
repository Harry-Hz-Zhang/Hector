#include "hector/core/table.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "hector/common/bitmap.h"
#include "hector/common/byte_io.h"
#include "hector/common/row_filter.h"
#include "hector/common/status.h"
#include "hector/common/types.h"
#include "hector/core/column_store.h"
#include "hector/core/filter.h"
#include "hector/core/fusion.h"
#include "hector/core/schema.h"
#include "hector/core/search.h"
#include "hector/core/value.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"
#include "hector/index/sparse_index.h"
#include "hector/index/vector_index.h"

namespace hector {
namespace {

constexpr size_t kMaxBatchRows = 100000;
constexpr size_t kMaxPrimaryKeys = 100000;
constexpr size_t kMaxPrimaryKeyBytes = 256;
constexpr size_t kMaxStringBytes = 65536;

// 行级错误：消息以 rows[i] 开头，列级错误再接 .列名。
absl::Status RowError(size_t row, std::string_view detail) {
  return absl::InvalidArgumentError(absl::StrCat("rows[", row, "]: ", detail));
}

absl::Status FieldError(size_t row, std::string_view column,
                        std::string_view detail) {
  return absl::InvalidArgumentError(
      absl::StrCat("rows[", row, "].", column, ": ", detail));
}

bool AllFinite(std::span<const float> values) {
  return std::all_of(values.begin(), values.end(),
                     [](float f) { return std::isfinite(f); });
}

bool AllZero(std::span<const float> values) {
  return std::all_of(values.begin(), values.end(),
                     [](float f) { return f == 0.0F; });
}

bool StrictlyAscending(std::span<const uint32_t> indices) {
  return std::adjacent_find(indices.begin(), indices.end(),
                            [](uint32_t a, uint32_t b) { return a >= b; }) ==
         indices.end();
}

// 按位置升序重排 indices 与 values（稳定排序，重复位置保持输入顺序）。
void SortSparse(SparseVector& v) {
  if (std::is_sorted(v.indices.begin(), v.indices.end())) {
    return;
  }
  std::vector<size_t> order(v.indices.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&v](size_t a, size_t b) {
    return v.indices[a] < v.indices[b];
  });
  SparseVector sorted;
  sorted.indices.reserve(order.size());
  sorted.values.reserve(order.size());
  for (size_t i : order) {
    sorted.indices.push_back(v.indices[i]);
    sorted.values.push_back(v.values[i]);
  }
  v = std::move(sorted);
}

absl::Status ValidateDense(size_t row, const ColumnDef& column,
                           const DenseColumnSpec& spec, Value& value,
                           PrepareMode mode) {
  auto* vec = std::get_if<std::vector<float>>(&value);
  if (vec == nullptr) {
    return FieldError(row, column.name, "expected a dense vector");
  }
  if (vec->size() != spec.dim) {
    return FieldError(row, column.name,
                      absl::StrCat("dense vector must have ", spec.dim,
                                   " values, got ", vec->size()));
  }
  if (!AllFinite(*vec)) {
    return FieldError(row, column.name, "dense values must be finite");
  }
  if (spec.metric == MetricType::kCosine) {
    if (AllZero(*vec)) {
      return FieldError(row, column.name,
                        "COSINE vector must not be a zero vector");
    }
    // kReplay：WAL 中已是归一化后的数据，再归一化一次会改变末位。
    if (mode == PrepareMode::kOnline) {
      if (absl::Status s = NormalizeInPlace(*vec); !s.ok()) {
        return FieldError(row, column.name, s.message());
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateSparse(size_t row, const ColumnDef& column,
                            const SparseColumnSpec& spec, Value& value,
                            PrepareMode mode) {
  auto* vec = std::get_if<SparseVector>(&value);
  if (vec == nullptr) {
    return FieldError(row, column.name, "expected a sparse vector");
  }
  if (vec->indices.size() != vec->values.size()) {
    return FieldError(
        row, column.name,
        absl::StrCat("sparse vector has ", vec->indices.size(), " indices but ",
                     vec->values.size(), " values"));
  }
  for (uint32_t index : vec->indices) {
    if (index >= spec.max_dim) {
      return FieldError(
          row, column.name,
          absl::StrCat("sparse index ", index, " must be less than max_dim ",
                       spec.max_dim));
    }
  }
  if (!AllFinite(vec->values)) {
    return FieldError(row, column.name, "sparse values must be finite");
  }
  // kReplay：WAL 中已排序，跳过排序；下面的检查同时拒绝乱序与重复。
  if (mode == PrepareMode::kOnline) {
    SortSparse(*vec);
  }
  if (!StrictlyAscending(vec->indices)) {
    return FieldError(row, column.name,
                      mode == PrepareMode::kOnline
                          ? "sparse vector has duplicate indices"
                          : "sparse indices must be strictly ascending");
  }
  return absl::OkStatus();
}

absl::Status ValidateScalar(size_t row, const ColumnDef& column,
                            ScalarType type, const Value& value) {
  switch (type) {
    case ScalarType::kInt64:
      if (!std::holds_alternative<int64_t>(value)) {
        return FieldError(row, column.name, "expected an int64 value");
      }
      return absl::OkStatus();
    case ScalarType::kFloat: {
      const auto* f = std::get_if<float>(&value);
      if (f == nullptr) {
        return FieldError(row, column.name, "expected a float value");
      }
      if (!std::isfinite(*f)) {
        return FieldError(row, column.name, "float value must be finite");
      }
      return absl::OkStatus();
    }
    case ScalarType::kString: {
      const auto* s = std::get_if<std::string>(&value);
      if (s == nullptr) {
        return FieldError(row, column.name, "expected a string value");
      }
      if (s->size() > kMaxStringBytes) {
        return FieldError(
            row, column.name,
            absl::StrCat("string must be at most ", kMaxStringBytes,
                         " bytes, got ", s->size()));
      }
      return absl::OkStatus();
    }
  }
  return FieldError(row, column.name, "unknown scalar type");
}

// 校验一行的字段：必须恰好给出 Schema 的全部列；kOnline 下就地归一化与排序。
absl::Status ValidateFields(const Schema& schema, size_t row, RowInput& input,
                            PrepareMode mode) {
  for (const ColumnDef& column : schema.columns()) {
    auto it = input.fields.find(column.name);
    if (it == input.fields.end()) {
      return RowError(row,
                      absl::StrCat("missing column \"", column.name, "\""));
    }
    Value& value = it->second;
    if (const auto* dense = std::get_if<DenseColumnSpec>(&column.spec)) {
      HECTOR_RETURN_IF_ERROR(ValidateDense(row, column, *dense, value, mode));
    } else if (const auto* sparse =
                   std::get_if<SparseColumnSpec>(&column.spec)) {
      HECTOR_RETURN_IF_ERROR(ValidateSparse(row, column, *sparse, value, mode));
    } else {
      HECTOR_RETURN_IF_ERROR(ValidateScalar(
          row, column, std::get<ScalarType>(column.spec), value));
    }
  }
  if (input.fields.size() != schema.columns().size()) {
    for (const auto& [name, value] : input.fields) {
      if (!schema.FindColumn(name).has_value()) {
        return RowError(row, absl::StrCat("unknown column \"", name, "\""));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidatePrimaryKey(size_t row, const std::string& pk) {
  if (pk.empty() || pk.size() > kMaxPrimaryKeyBytes) {
    return RowError(
        row, absl::StrCat("primary key must be 1 to ", kMaxPrimaryKeyBytes,
                          " bytes, got ", pk.size()));
  }
  return absl::OkStatus();
}

// ApplyInsert 的前置条件检查：每一列都存在且形状与列存储一致，保证随后的
// 追加不会中途失败。
bool MatchesColumns(const Schema& schema, const RowInput& row) {
  if (row.fields.size() != schema.columns().size()) {
    return false;
  }
  for (const ColumnDef& column : schema.columns()) {
    auto it = row.fields.find(column.name);
    if (it == row.fields.end()) {
      return false;
    }
    const Value& value = it->second;
    if (const auto* dense = std::get_if<DenseColumnSpec>(&column.spec)) {
      const auto* vec = std::get_if<std::vector<float>>(&value);
      if (vec == nullptr || vec->size() != dense->dim) {
        return false;
      }
    } else if (std::holds_alternative<SparseColumnSpec>(column.spec)) {
      const auto* vec = std::get_if<SparseVector>(&value);
      if (vec == nullptr || vec->indices.size() != vec->values.size()) {
        return false;
      }
    } else {
      switch (std::get<ScalarType>(column.spec)) {
        case ScalarType::kInt64:
          if (!std::holds_alternative<int64_t>(value)) {
            return false;
          }
          break;
        case ScalarType::kFloat:
          if (!std::holds_alternative<float>(value)) {
            return false;
          }
          break;
        case ScalarType::kString:
          if (!std::holds_alternative<std::string>(value)) {
            return false;
          }
          break;
      }
    }
  }
  return true;
}

// 前置条件：value 持有普通列的值。
ScalarValue ToScalar(Value&& value) {
  if (auto* i = std::get_if<int64_t>(&value)) {
    return *i;
  }
  if (auto* f = std::get_if<float>(&value)) {
    return *f;
  }
  return std::get<std::string>(std::move(value));
}

Value ToValue(const ScalarValue& value) {
  return std::visit([](const auto& v) { return Value{v}; }, value);
}

constexpr uint32_t kMaxTopK = 1024;
constexpr size_t kMaxRoutes = 8;
constexpr uint32_t kMaxCandidateLimit = 10000;
constexpr uint32_t kDefaultCandidateLimit = 100;
constexpr uint32_t kMaxEfSearch = 10000;
constexpr uint32_t kDefaultEfSearch = 64;
constexpr uint32_t kMaxRerankCandidates = 10000;

absl::Status RouteError(size_t route, std::string_view detail) {
  return absl::InvalidArgumentError(
      absl::StrCat("routes[", route, "]: ", detail));
}

// 已校验的一路检索：查询已复制，COSINE 查询已归一化，稀疏查询已排序。
struct PreparedRoute {
  size_t column;
  bool dense;
  std::vector<float> dense_query;
  SparseVector sparse_query;
};

absl::Status PrepareDenseQuery(size_t route, const DenseColumnSpec& spec,
                               std::vector<float>& query) {
  if (query.size() != spec.dim) {
    return RouteError(route, absl::StrCat("dense query must have ", spec.dim,
                                          " values, got ", query.size()));
  }
  if (!AllFinite(query)) {
    return RouteError(route, "dense query values must be finite");
  }
  if (spec.metric == MetricType::kCosine) {
    if (AllZero(query)) {
      return RouteError(route, "COSINE query must not be a zero vector");
    }
    if (absl::Status s = NormalizeInPlace(query); !s.ok()) {
      return RouteError(route, s.message());
    }
  }
  return absl::OkStatus();
}

absl::Status PrepareSparseQuery(size_t route, const SparseColumnSpec& spec,
                                SparseVector& query) {
  if (query.indices.size() != query.values.size()) {
    return RouteError(
        route, absl::StrCat("sparse query has ", query.indices.size(),
                            " indices but ", query.values.size(), " values"));
  }
  for (uint32_t index : query.indices) {
    if (index >= spec.max_dim) {
      return RouteError(
          route, absl::StrCat("sparse index ", index,
                              " must be less than max_dim ", spec.max_dim));
    }
  }
  if (!AllFinite(query.values)) {
    return RouteError(route, "sparse query values must be finite");
  }
  SortSparse(query);
  if (!StrictlyAscending(query.indices)) {
    return RouteError(route, "sparse query has duplicate indices");
  }
  return absl::OkStatus();
}

// 校验一路检索并复制、规范化查询；不调用索引。
absl::StatusOr<PreparedRoute> PrepareRoute(const Schema& schema, size_t i,
                                           const SearchRoute& route,
                                           uint32_t top_k) {
  if (route.candidate_limit != 0 &&
      (route.candidate_limit < top_k ||
       route.candidate_limit > kMaxCandidateLimit)) {
    return RouteError(
        i, absl::StrCat("candidate_limit must be 0 or in [top_k=", top_k, ", ",
                        kMaxCandidateLimit, "], got ", route.candidate_limit));
  }
  if (route.ef_search > kMaxEfSearch) {
    return RouteError(i, absl::StrCat("ef_search must be at most ",
                                      kMaxEfSearch, ", got ", route.ef_search));
  }
  if (route.rerank_candidates > kMaxRerankCandidates) {
    return RouteError(i, absl::StrCat("rerank_candidates must be at most ",
                                      kMaxRerankCandidates, ", got ",
                                      route.rerank_candidates));
  }
  const std::optional<size_t> column = schema.FindColumn(route.column);
  if (!column.has_value()) {
    return RouteError(i, absl::StrCat("unknown column \"", route.column, "\""));
  }
  const ColumnDef& def = schema.columns()[*column];
  PreparedRoute prepared{
      .column = *column, .dense = false, .dense_query = {}, .sparse_query = {}};
  if (const auto* dense = std::get_if<DenseColumnSpec>(&def.spec)) {
    const auto* query = std::get_if<std::vector<float>>(&route.query);
    if (query == nullptr) {
      return RouteError(i, absl::StrCat("column \"", def.name,
                                        "\" is dense and needs a dense query"));
    }
    prepared.dense = true;
    prepared.dense_query = *query;
    HECTOR_RETURN_IF_ERROR(PrepareDenseQuery(i, *dense, prepared.dense_query));
  } else if (const auto* sparse = std::get_if<SparseColumnSpec>(&def.spec)) {
    const auto* query = std::get_if<SparseVector>(&route.query);
    if (query == nullptr) {
      return RouteError(i,
                        absl::StrCat("column \"", def.name,
                                     "\" is sparse and needs a sparse query"));
    }
    prepared.sparse_query = *query;
    HECTOR_RETURN_IF_ERROR(
        PrepareSparseQuery(i, *sparse, prepared.sparse_query));
  } else {
    return RouteError(
        i, absl::StrCat("column \"", def.name, "\" is not a vector column"));
  }
  return prepared;
}

// 校验 Search.output_columns：只能是普通列。返回按 Schema 顺序去重的列下标。
absl::StatusOr<std::vector<size_t>> ResolveSearchOutputColumns(
    const Schema& schema, std::span<const std::string> names) {
  std::vector<size_t> columns;
  columns.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    const std::optional<size_t> c = schema.FindColumn(names[i]);
    if (!c.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "output_columns[", i, "]: unknown column \"", names[i], "\""));
    }
    if (!std::holds_alternative<ScalarType>(schema.columns()[*c].spec)) {
      return absl::InvalidArgumentError(
          absl::StrCat("output_columns[", i, "]: column \"", names[i],
                       "\" is a vector column; search only returns scalar "
                       "columns"));
    }
    columns.push_back(*c);
  }
  std::sort(columns.begin(), columns.end());
  columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
  return columns;
}

}  // namespace

IndexFactory DefaultIndexFactory() {
  return IndexFactory{.dense = MakeVectorIndex, .sparse = MakeSparseIndex};
}

Table::Table(Schema schema, RowId max_rows)
    : schema_(std::move(schema)),
      max_rows_(max_rows),
      primary_keys_(ScalarType::kString, max_rows),
      tombstones_(max_rows) {}

absl::StatusOr<std::unique_ptr<Table>> Table::Create(
    Schema schema, const IndexFactory& factory, const TableOptions& options) {
  if (options.max_rows == 0 || options.max_rows > kMaxRowsPerTable) {
    return absl::InvalidArgumentError(absl::StrCat("max_rows must be in [1, ",
                                                   kMaxRowsPerTable, "], got ",
                                                   options.max_rows));
  }
  if (!factory.dense || !factory.sparse) {
    return absl::InvalidArgumentError("index factory must be fully set");
  }
  // 构造函数是私有的，不能用 std::make_unique。
  std::unique_ptr<Table> table(new Table(std::move(schema), options.max_rows));
  const RowId capacity = options.max_rows;
  for (const ColumnDef& column : table->schema_.columns()) {
    ColumnSlot slot;
    if (const auto* dense = std::get_if<DenseColumnSpec>(&column.spec)) {
      slot.dense = std::make_unique<DenseColumn>(dense->dim, capacity);
      const DenseIndexConfig config{
          .dim = dense->dim,
          .metric = dense->metric,
          .kind = dense->index,
          .hnsw = dense->hnsw,
          .int8_quantization = dense->int8_quantization};
      HECTOR_ASSIGN_OR_RETURN(slot.dense_index,
                              factory.dense(config, slot.dense.get()));
      if (slot.dense_index == nullptr) {
        return absl::InternalError(absl::StrCat(
            "index factory returned null for \"", column.name, "\""));
      }
    } else if (const auto* sparse =
                   std::get_if<SparseColumnSpec>(&column.spec)) {
      slot.sparse = std::make_unique<SparseColumn>(capacity);
      const SparseIndexConfig config{.max_dim = sparse->max_dim,
                                     .kind = sparse->index};
      HECTOR_ASSIGN_OR_RETURN(slot.sparse_index,
                              factory.sparse(config, slot.sparse.get()));
      if (slot.sparse_index == nullptr) {
        return absl::InternalError(absl::StrCat(
            "index factory returned null for \"", column.name, "\""));
      }
    } else {
      slot.scalar = std::make_unique<ScalarColumn>(
          std::get<ScalarType>(column.spec), capacity);
    }
    table->columns_.push_back(std::move(slot));
  }
  return table;
}

const Schema& Table::schema() const { return schema_; }

RowId Table::written_rows() const { return primary_keys_.size(); }

absl::StatusOr<PreparedBatch> Table::PrepareInsert(std::vector<RowInput> rows,
                                                   bool upsert,
                                                   PrepareMode mode) const {
  if (rows.empty() || rows.size() > kMaxBatchRows) {
    return absl::InvalidArgumentError(
        absl::StrCat("write batch must have 1 to ", kMaxBatchRows,
                     " rows, got ", rows.size()));
  }
  const RowId written = written_rows();
  if (rows.size() > static_cast<size_t>(max_rows_ - written)) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "table \"", schema_.table_name(), "\" has ", written,
        " rows written (including deleted rows); adding ", rows.size(),
        " rows would exceed the limit of ", max_rows_));
  }

  PreparedBatch batch;
  batch.upsert = upsert;
  absl::flat_hash_set<std::string_view> batch_keys;
  batch_keys.reserve(rows.size());
  {
    std::shared_lock<std::shared_mutex> lock(pk_mu_);
    for (size_t i = 0; i < rows.size(); ++i) {
      const std::string& pk = rows[i].primary_key;
      HECTOR_RETURN_IF_ERROR(ValidatePrimaryKey(i, pk));
      if (!batch_keys.insert(pk).second) {
        return RowError(
            i, absl::StrCat("duplicate primary key \"", pk, "\" in batch"));
      }
      auto existing = pk_to_row_.find(pk);
      if (existing != pk_to_row_.end()) {
        if (!upsert) {
          return absl::AlreadyExistsError(absl::StrCat(
              "rows[", i, "]: primary key \"", pk, "\" already exists"));
        }
        batch.tombstoned_row_ids.push_back(existing->second);
      }
    }
  }
  for (size_t i = 0; i < rows.size(); ++i) {
    HECTOR_RETURN_IF_ERROR(ValidateFields(schema_, i, rows[i], mode));
  }
  batch.rows = std::move(rows);
  return batch;
}

absl::Status Table::ApplyInsert(PreparedBatch batch, Lsn lsn) {
  const RowId start = written_rows();
  const size_t count = batch.rows.size();
  // 前置条件检查放在任何写入之前，保证失败时状态不变。
  if (count > static_cast<size_t>(max_rows_ - start)) {
    return absl::FailedPreconditionError(
        "ApplyInsert batch exceeds the row capacity; call PrepareInsert first");
  }
  for (const RowInput& row : batch.rows) {
    if (!MatchesColumns(schema_, row)) {
      return absl::FailedPreconditionError(
          "ApplyInsert batch does not match the schema; call PrepareInsert "
          "first");
    }
  }

  // 先写列存储：索引的 Add 要求 row id 小于访问器的 size()。
  const std::vector<ColumnDef>& defs = schema_.columns();
  for (RowInput& row : batch.rows) {
    for (size_t c = 0; c < defs.size(); ++c) {
      Value& value = row.fields.find(defs[c].name)->second;
      ColumnSlot& slot = columns_[c];
      if (slot.dense != nullptr) {
        HECTOR_RETURN_IF_ERROR(
            slot.dense->Append(std::get<std::vector<float>>(value)));
      } else if (slot.sparse != nullptr) {
        HECTOR_RETURN_IF_ERROR(
            slot.sparse->Append(std::get<SparseVector>(std::move(value))));
      } else {
        HECTOR_RETURN_IF_ERROR(slot.scalar->Append(ToScalar(std::move(value))));
      }
    }
    HECTOR_RETURN_IF_ERROR(primary_keys_.Append(ScalarValue{row.primary_key}));
  }
  for (ColumnSlot& slot : columns_) {
    for (size_t r = 0; r < count; ++r) {
      const auto id = static_cast<RowId>(start + r);
      if (slot.dense_index != nullptr) {
        HECTOR_RETURN_IF_ERROR(slot.dense_index->Add(id));
      } else if (slot.sparse_index != nullptr) {
        HECTOR_RETURN_IF_ERROR(slot.sparse_index->Add(id));
      }
    }
  }
  {
    std::unique_lock<std::shared_mutex> lock(pk_mu_);
    for (size_t r = 0; r < count; ++r) {
      pk_to_row_.insert_or_assign(std::move(batch.rows[r].primary_key),
                                  static_cast<RowId>(start + r));
    }
  }
  pending_.push_back(
      PendingChange{.lsn = lsn,
                    .row_count = static_cast<RowId>(start + count),
                    .tombstones = std::move(batch.tombstoned_row_ids)});
  return absl::OkStatus();
}

absl::StatusOr<std::vector<RowId>> Table::PrepareDelete(
    std::span<const std::string> primary_keys) const {
  if (primary_keys.empty() || primary_keys.size() > kMaxPrimaryKeys) {
    return absl::InvalidArgumentError(
        absl::StrCat("delete requires 1 to ", kMaxPrimaryKeys,
                     " primary keys, got ", primary_keys.size()));
  }
  std::vector<RowId> row_ids;
  absl::flat_hash_set<RowId> seen;
  std::shared_lock<std::shared_mutex> lock(pk_mu_);
  for (const std::string& pk : primary_keys) {
    auto it = pk_to_row_.find(pk);
    if (it != pk_to_row_.end() && seen.insert(it->second).second) {
      row_ids.push_back(it->second);
    }
  }
  return row_ids;
}

absl::Status Table::ApplyDelete(std::span<const RowId> row_ids, Lsn lsn) {
  const RowId written = written_rows();
  for (RowId id : row_ids) {
    if (id >= written) {
      return absl::FailedPreconditionError(
          absl::StrCat("ApplyDelete row id ", id,
                       " is not written; call PrepareDelete "
                       "first"));
    }
  }
  {
    std::unique_lock<std::shared_mutex> lock(pk_mu_);
    for (RowId id : row_ids) {
      const auto& pk = std::get<std::string>(primary_keys_.Get(id));
      auto it = pk_to_row_.find(pk);
      if (it != pk_to_row_.end() && it->second == id) {
        pk_to_row_.erase(it);
      }
    }
  }
  pending_.push_back(PendingChange{
      .lsn = lsn,
      .row_count = written,
      .tombstones = std::vector<RowId>(row_ids.begin(), row_ids.end())});
  return absl::OkStatus();
}

void Table::Publish(Lsn up_to) {
  while (!pending_.empty() && pending_.front().lsn <= up_to) {
    const PendingChange& change = pending_.front();
    // 先发布新行数，再设置墓碑：Upsert 的新行先于旧行的墓碑可见。
    if (change.row_count > published_rows_.load(std::memory_order_relaxed)) {
      published_rows_.store(change.row_count, std::memory_order_release);
    }
    for (RowId id : change.tombstones) {
      tombstones_.Set(id);
    }
    pending_.pop_front();
  }
}

Value Table::ReadValue(size_t column, RowId row) const {
  const ColumnSlot& slot = columns_[column];
  if (slot.dense != nullptr) {
    std::span<const float> v = slot.dense->Get(row);
    return std::vector<float>(v.begin(), v.end());
  }
  if (slot.sparse != nullptr) {
    const SparseVectorView v = slot.sparse->Get(row);
    return SparseVector{
        .indices = std::vector<uint32_t>(v.indices.begin(), v.indices.end()),
        .values = std::vector<float>(v.values.begin(), v.values.end())};
  }
  return ToValue(slot.scalar->Get(row));
}

absl::StatusOr<std::vector<RowOutput>> Table::Get(
    std::span<const std::string> primary_keys,
    std::span<const std::string> output_columns,
    std::vector<std::string>* missing) const {
  if (primary_keys.empty() || primary_keys.size() > kMaxPrimaryKeys) {
    return absl::InvalidArgumentError(
        absl::StrCat("get requires 1 to ", kMaxPrimaryKeys,
                     " primary keys, got ", primary_keys.size()));
  }
  std::vector<size_t> columns;
  if (output_columns.empty()) {
    columns.resize(schema_.columns().size());
    std::iota(columns.begin(), columns.end(), 0);
  } else {
    for (size_t i = 0; i < output_columns.size(); ++i) {
      std::optional<size_t> c = schema_.FindColumn(output_columns[i]);
      if (!c.has_value()) {
        return absl::InvalidArgumentError(
            absl::StrCat("output_columns[", i, "]: unknown column \"",
                         output_columns[i], "\""));
      }
      columns.push_back(*c);
    }
    // 返回的列按 Schema 顺序排列，重复的列只返回一次。
    std::sort(columns.begin(), columns.end());
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
  }

  // 未找到的主键记为 kNotFound；kMaxRowsPerTable 不超过 2^24，不会与真实 row id
  // 冲突。
  constexpr RowId kNotFound = std::numeric_limits<RowId>::max();
  std::vector<RowId> found(primary_keys.size(), kNotFound);
  {
    std::shared_lock<std::shared_mutex> lock(pk_mu_);
    for (size_t i = 0; i < primary_keys.size(); ++i) {
      auto it = pk_to_row_.find(primary_keys[i]);
      if (it != pk_to_row_.end()) {
        found[i] = it->second;
      }
    }
  }
  // acquire 与 Publish 的 release 配对：row id < published 的列数据都已可见。
  const RowId published = published_rows_.load(std::memory_order_acquire);

  std::vector<RowOutput> rows;
  for (size_t i = 0; i < primary_keys.size(); ++i) {
    const RowId id = found[i];
    // kNotFound 必然不小于 published。
    if (id >= published || tombstones_.Test(id)) {
      if (missing != nullptr) {
        missing->push_back(primary_keys[i]);
      }
      continue;
    }
    RowOutput row;
    row.primary_key = primary_keys[i];
    row.fields.reserve(columns.size());
    for (size_t c : columns) {
      row.fields.emplace_back(schema_.columns()[c].name, ReadValue(c, id));
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

absl::StatusOr<SearchResult> Table::Search(const SearchRequest& request) const {
  // acquire 与 Publish 的 release 配对：row id < published 的列数据都已可见。
  const RowId published = published_rows_.load(std::memory_order_acquire);

  // 先完成全部校验，再调用任何索引。
  const uint32_t top_k = request.top_k;
  if (top_k == 0 || top_k > kMaxTopK) {
    return absl::InvalidArgumentError(
        absl::StrCat("top_k must be in [1, ", kMaxTopK, "], got ", top_k));
  }
  if (request.routes.empty() || request.routes.size() > kMaxRoutes) {
    return absl::InvalidArgumentError(absl::StrCat("search requires 1 to ",
                                                   kMaxRoutes, " routes, got ",
                                                   request.routes.size()));
  }
  std::vector<PreparedRoute> routes;
  routes.reserve(request.routes.size());
  for (size_t i = 0; i < request.routes.size(); ++i) {
    HECTOR_ASSIGN_OR_RETURN(PreparedRoute route,
                            PrepareRoute(schema_, i, request.routes[i], top_k));
    routes.push_back(std::move(route));
  }
  HECTOR_ASSIGN_OR_RETURN(
      const std::vector<size_t> output_columns,
      ResolveSearchOutputColumns(schema_, request.output_columns));
  HECTOR_RETURN_IF_ERROR(ValidateFilter(request.filter, schema_));

  // 没有条件时不构造 allow 位图，RowFilter 只排除墓碑与未发布行，
  // 与全部置位的 allow 位图等价。
  std::optional<Bitmap> allow;
  if (!request.filter.conditions.empty()) {
    std::vector<const ScalarColumn*> scalars(columns_.size(), nullptr);
    for (size_t c = 0; c < columns_.size(); ++c) {
      scalars[c] = columns_[c].scalar.get();
    }
    allow = EvaluateFilter(request.filter, schema_, scalars, published);
  }
  const RowFilter filter(published, &tombstones_,
                         allow.has_value() ? &*allow : nullptr);

  // 单路取 top_k；多路各取实际 candidate_limit。
  const bool single = routes.size() == 1;
  std::vector<std::vector<Hit>> lists;
  lists.reserve(routes.size());
  for (size_t i = 0; i < routes.size(); ++i) {
    const SearchRoute& route = request.routes[i];
    const uint32_t limit = single
                               ? top_k
                               : (route.candidate_limit == 0
                                      ? std::max(kDefaultCandidateLimit, top_k)
                                      : route.candidate_limit);
    const ColumnSlot& slot = columns_[routes[i].column];
    if (routes[i].dense) {
      const SearchParams params{.ef_search = route.ef_search == 0
                                                 ? kDefaultEfSearch
                                                 : route.ef_search,
                                .rerank_candidates = route.rerank_candidates};
      HECTOR_ASSIGN_OR_RETURN(std::vector<Hit> hits,
                              slot.dense_index->Search(routes[i].dense_query,
                                                       limit, params, filter));
      lists.push_back(std::move(hits));
    } else {
      const SparseVector& query = routes[i].sparse_query;
      const SparseSearchParams params{.algorithm = route.sparse_algorithm};
      HECTOR_ASSIGN_OR_RETURN(
          std::vector<Hit> hits,
          slot.sparse_index->Search(SparseVectorView{.indices = query.indices,
                                                     .values = query.values},
                                    limit, params, filter));
      lists.push_back(std::move(hits));
    }
  }

  // 统一成「行、分数」序列：单路把距离换回分数，多路交给 Fuse。
  SearchResult result;
  std::vector<FusedHit> ranked;
  if (single) {
    const ColumnDef& def = schema_.columns()[routes[0].column];
    const auto* dense = std::get_if<DenseColumnSpec>(&def.spec);
    const bool l2 = dense != nullptr && dense->metric == MetricType::kL2;
    result.score_kind = l2 ? ScoreKind::kL2Distance : ScoreKind::kInnerProduct;
    ranked.reserve(lists[0].size());
    for (const Hit& hit : lists[0]) {
      ranked.push_back(FusedHit{.row_id = hit.row_id,
                                .score = l2 ? hit.distance : -hit.distance});
    }
  } else {
    result.score_kind = ScoreKind::kRrf;
    ranked = Fuse(lists, request.fusion, top_k);
  }

  result.rows.reserve(ranked.size());
  for (const FusedHit& hit : ranked) {
    SearchResultRow row;
    row.primary_key = std::get<std::string>(primary_keys_.Get(hit.row_id));
    row.score = hit.score;
    row.fields.reserve(output_columns.size());
    for (size_t c : output_columns) {
      row.fields.emplace_back(schema_.columns()[c].name,
                              ToValue(columns_[c].scalar->Get(hit.row_id)));
    }
    result.rows.push_back(std::move(row));
  }
  return result;
}

TableStats Table::Stats() const {
  // 先读墓碑数再读行数：墓碑只会打在已发布的行上，因此 deleted <= published。
  const size_t deleted = tombstones_.Count();
  const RowId published = published_rows_.load(std::memory_order_acquire);
  TableStats stats{};
  stats.deleted_rows = deleted;
  stats.live_rows = published >= deleted ? published - deleted : 0;
  const std::vector<ColumnDef>& defs = schema_.columns();
  stats.columns.reserve(defs.size());
  for (size_t c = 0; c < defs.size(); ++c) {
    const ColumnSlot& slot = columns_[c];
    ColumnStatsRow row{.name = defs[c].name,
                       .index = std::monostate{},
                       .indexed_rows = 0,
                       .memory_bytes = 0};
    if (slot.dense != nullptr) {
      row.index = std::get<DenseColumnSpec>(defs[c].spec).index;
      row.indexed_rows = slot.dense_index->Size();
      row.memory_bytes =
          slot.dense->MemoryBytes() + slot.dense_index->MemoryBytes();
    } else if (slot.sparse != nullptr) {
      row.index = std::get<SparseColumnSpec>(defs[c].spec).index;
      row.indexed_rows = slot.sparse_index->Size();
      row.memory_bytes =
          slot.sparse->MemoryBytes() + slot.sparse_index->MemoryBytes();
    } else {
      row.memory_bytes = slot.scalar->MemoryBytes();
    }
    stats.columns.push_back(std::move(row));
  }
  return stats;
}

absl::Status Table::Save(ByteWriter& /*out*/) const {
  return absl::UnimplementedError(
      "table snapshot is not implemented yet (planned for c5)");
}

absl::StatusOr<std::unique_ptr<Table>> Table::Load(
    ByteReader& /*in*/, const IndexFactory& /*factory*/) {
  return absl::UnimplementedError(
      "table snapshot is not implemented yet (planned for c5)");
}

}  // namespace hector
