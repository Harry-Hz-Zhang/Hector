#include "src/server/proto_convert.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "hector/common/status.h"
#include "hector/core/filter.h"
#include "hector/core/fusion.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"
#include "hector/index/sparse_index.h"

namespace hector::server {
namespace {

constexpr uint64_t kDefaultHnswSeed = 42;
constexpr uint32_t kDefaultRrfK = 60;

// 消息格式：<字段路径>: <说明>。
absl::Status FieldError(std::string_view path, std::string_view detail) {
  return absl::InvalidArgumentError(absl::StrCat(path, ": ", detail));
}

std::string Indexed(std::string_view prefix, int i) {
  return absl::StrCat(prefix, "[", i, "]");
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

absl::StatusOr<MetricType> MetricFromProto(v1::Metric metric,
                                           std::string_view path) {
  switch (metric) {
    case v1::METRIC_L2:
      return MetricType::kL2;
    case v1::METRIC_IP:
      return MetricType::kInnerProduct;
    case v1::METRIC_COSINE:
      return MetricType::kCosine;
    case v1::METRIC_UNSPECIFIED:
      return FieldError(path, "unspecified");
    default:
      return FieldError(path, absl::StrCat("unknown value ", metric));
  }
}

v1::Metric MetricToProto(MetricType metric) {
  switch (metric) {
    case MetricType::kL2:
      return v1::METRIC_L2;
    case MetricType::kInnerProduct:
      return v1::METRIC_IP;
    case MetricType::kCosine:
      return v1::METRIC_COSINE;
  }
  return v1::METRIC_UNSPECIFIED;
}

absl::StatusOr<DenseIndexKind> DenseIndexFromProto(v1::DenseIndexType type,
                                                   std::string_view path) {
  switch (type) {
    case v1::DENSE_INDEX_TYPE_UNSPECIFIED:
    case v1::DENSE_INDEX_TYPE_FLAT:
      return DenseIndexKind::kFlat;
    case v1::DENSE_INDEX_TYPE_HNSW:
      return DenseIndexKind::kHnsw;
    default:
      return FieldError(path, absl::StrCat("unknown value ", type));
  }
}

v1::DenseIndexType DenseIndexToProto(DenseIndexKind kind) {
  switch (kind) {
    case DenseIndexKind::kFlat:
      return v1::DENSE_INDEX_TYPE_FLAT;
    case DenseIndexKind::kHnsw:
      return v1::DENSE_INDEX_TYPE_HNSW;
  }
  return v1::DENSE_INDEX_TYPE_UNSPECIFIED;
}

absl::StatusOr<SparseIndexKind> SparseIndexFromProto(v1::SparseIndexType type,
                                                     std::string_view path) {
  switch (type) {
    case v1::SPARSE_INDEX_TYPE_UNSPECIFIED:
    case v1::SPARSE_INDEX_TYPE_FLAT:
      return SparseIndexKind::kFlat;
    case v1::SPARSE_INDEX_TYPE_INVERTED:
      return SparseIndexKind::kInverted;
    default:
      return FieldError(path, absl::StrCat("unknown value ", type));
  }
}

v1::SparseIndexType SparseIndexToProto(SparseIndexKind kind) {
  switch (kind) {
    case SparseIndexKind::kFlat:
      return v1::SPARSE_INDEX_TYPE_FLAT;
    case SparseIndexKind::kInverted:
      return v1::SPARSE_INDEX_TYPE_INVERTED;
  }
  return v1::SPARSE_INDEX_TYPE_UNSPECIFIED;
}

absl::StatusOr<ScalarType> ScalarTypeFromProto(v1::ScalarType type,
                                               std::string_view path) {
  switch (type) {
    case v1::SCALAR_TYPE_INT64:
      return ScalarType::kInt64;
    case v1::SCALAR_TYPE_FLOAT:
      return ScalarType::kFloat;
    case v1::SCALAR_TYPE_STRING:
      return ScalarType::kString;
    case v1::SCALAR_TYPE_UNSPECIFIED:
      return FieldError(path, "unspecified");
    default:
      return FieldError(path, absl::StrCat("unknown value ", type));
  }
}

v1::ScalarType ScalarTypeToProto(ScalarType type) {
  switch (type) {
    case ScalarType::kInt64:
      return v1::SCALAR_TYPE_INT64;
    case ScalarType::kFloat:
      return v1::SCALAR_TYPE_FLOAT;
    case ScalarType::kString:
      return v1::SCALAR_TYPE_STRING;
  }
  return v1::SCALAR_TYPE_UNSPECIFIED;
}

absl::StatusOr<DenseColumnSpec> DenseFromProto(
    const v1::DenseVectorColumn& proto, const std::string& path) {
  HECTOR_ASSIGN_OR_RETURN(MetricType metric,
                          MetricFromProto(proto.metric(), path + ".metric"));
  HECTOR_ASSIGN_OR_RETURN(
      DenseIndexKind index,
      DenseIndexFromProto(proto.index_type(), path + ".index_type"));
  // m 与 ef_construction 原样传 0，由 Schema::Create 换成默认值。
  const HnswConfig hnsw{
      .m = proto.hnsw().m(),
      .ef_construction = proto.hnsw().ef_construction(),
      .seed = proto.hnsw().has_seed() ? proto.hnsw().seed() : kDefaultHnswSeed};
  return DenseColumnSpec{.dim = proto.dim(),
                         .metric = metric,
                         .index = index,
                         .hnsw = hnsw,
                         .int8_quantization = proto.int8_quantization()};
}

absl::StatusOr<ColumnDef> ColumnFromProto(const v1::ColumnDef& proto,
                                          const std::string& path) {
  ColumnDef column{.name = proto.name(), .spec = {}};
  switch (proto.kind_case()) {
    case v1::ColumnDef::kDense: {
      HECTOR_ASSIGN_OR_RETURN(column.spec,
                              DenseFromProto(proto.dense(), path + ".dense"));
      return column;
    }
    case v1::ColumnDef::kSparse: {
      HECTOR_ASSIGN_OR_RETURN(
          SparseIndexKind index,
          SparseIndexFromProto(proto.sparse().index_type(),
                               path + ".sparse.index_type"));
      column.spec =
          SparseColumnSpec{.max_dim = proto.sparse().max_dim(), .index = index};
      return column;
    }
    case v1::ColumnDef::kScalar: {
      HECTOR_ASSIGN_OR_RETURN(
          column.spec,
          ScalarTypeFromProto(proto.scalar().type(), path + ".scalar.type"));
      return column;
    }
    case v1::ColumnDef::KIND_NOT_SET:
      break;
  }
  return FieldError(path + ".kind", "unspecified");
}

// ---------------------------------------------------------------------------
// 值
// ---------------------------------------------------------------------------

SparseVector SparseFromProto(const v1::SparseVector& proto) {
  return SparseVector{
      .indices = {proto.indices().begin(), proto.indices().end()},
      .values = {proto.values().begin(), proto.values().end()}};
}

absl::StatusOr<Value> ValueFromProto(const v1::Value& proto,
                                     std::string_view path) {
  switch (proto.kind_case()) {
    case v1::Value::kInt64Value:
      return Value(proto.int64_value());
    case v1::Value::kFloatValue:
      return Value(proto.float_value());
    case v1::Value::kStringValue:
      return Value(proto.string_value());
    case v1::Value::kDense:
      return Value(std::vector<float>(proto.dense().values().begin(),
                                      proto.dense().values().end()));
    case v1::Value::kSparse:
      return Value(SparseFromProto(proto.sparse()));
    case v1::Value::KIND_NOT_SET:
      break;
  }
  return FieldError(path, "value is empty");
}

void ValueToProto(const Value& value, v1::Value* proto) {
  struct Visitor {
    v1::Value* proto;
    void operator()(int64_t v) const { proto->set_int64_value(v); }
    void operator()(float v) const { proto->set_float_value(v); }
    void operator()(const std::string& v) const { proto->set_string_value(v); }
    void operator()(const std::vector<float>& v) const {
      proto->mutable_dense()->mutable_values()->Add(v.begin(), v.end());
    }
    void operator()(const SparseVector& v) const {
      v1::SparseVector* sparse = proto->mutable_sparse();
      sparse->mutable_indices()->Add(v.indices.begin(), v.indices.end());
      sparse->mutable_values()->Add(v.values.begin(), v.values.end());
    }
  };
  std::visit(Visitor{proto}, value);
}

void FieldsToProto(const std::vector<std::pair<std::string, Value>>& fields,
                   google::protobuf::Map<std::string, v1::Value>* proto) {
  for (const auto& [name, value] : fields) {
    ValueToProto(value, &(*proto)[name]);
  }
}

absl::StatusOr<ScalarValue> ScalarValueFromProto(const v1::ScalarValue& proto,
                                                 std::string_view path) {
  switch (proto.kind_case()) {
    case v1::ScalarValue::kInt64Value:
      return ScalarValue(proto.int64_value());
    case v1::ScalarValue::kFloatValue:
      return ScalarValue(proto.float_value());
    case v1::ScalarValue::kStringValue:
      return ScalarValue(proto.string_value());
    case v1::ScalarValue::KIND_NOT_SET:
      break;
  }
  return FieldError(path, "value is empty");
}

// ---------------------------------------------------------------------------
// 检索
// ---------------------------------------------------------------------------

absl::StatusOr<CompareOp> CompareOpFromProto(v1::CompareOp op,
                                             std::string_view path) {
  switch (op) {
    case v1::COMPARE_OP_EQ:
      return CompareOp::kEq;
    case v1::COMPARE_OP_NE:
      return CompareOp::kNe;
    case v1::COMPARE_OP_LT:
      return CompareOp::kLt;
    case v1::COMPARE_OP_LE:
      return CompareOp::kLe;
    case v1::COMPARE_OP_GT:
      return CompareOp::kGt;
    case v1::COMPARE_OP_GE:
      return CompareOp::kGe;
    case v1::COMPARE_OP_IN:
      return CompareOp::kIn;
    case v1::COMPARE_OP_UNSPECIFIED:
      return FieldError(path, "unspecified");
    default:
      return FieldError(path, absl::StrCat("unknown value ", op));
  }
}

absl::StatusOr<SparseAlgorithm> SparseAlgorithmFromProto(
    v1::SparseAlgorithm algorithm, std::string_view path) {
  switch (algorithm) {
    case v1::SPARSE_ALGORITHM_UNSPECIFIED:
      return SparseAlgorithm::kAuto;
    case v1::SPARSE_ALGORITHM_TERM_AT_A_TIME:
      return SparseAlgorithm::kTermAtATime;
    case v1::SPARSE_ALGORITHM_WAND:
      return SparseAlgorithm::kWand;
    default:
      return FieldError(path, absl::StrCat("unknown value ", algorithm));
  }
}

absl::StatusOr<SearchRoute> RouteFromProto(const v1::SearchRoute& proto,
                                           const std::string& path) {
  SearchRoute route;
  route.column = proto.column();
  switch (proto.query_case()) {
    case v1::SearchRoute::kDense:
      route.query = std::vector<float>(proto.dense().values().begin(),
                                       proto.dense().values().end());
      break;
    case v1::SearchRoute::kSparse:
      route.query = SparseFromProto(proto.sparse());
      break;
    case v1::SearchRoute::QUERY_NOT_SET:
      return FieldError(path + ".query", "unspecified");
  }
  route.candidate_limit = proto.candidate_limit();
  route.ef_search = proto.ef_search();
  route.rerank_candidates = proto.rerank_candidates();
  HECTOR_ASSIGN_OR_RETURN(route.sparse_algorithm,
                          SparseAlgorithmFromProto(proto.sparse_algorithm(),
                                                   path + ".sparse_algorithm"));
  return route;
}

absl::StatusOr<Condition> ConditionFromProto(const v1::Condition& proto,
                                             const std::string& path) {
  Condition condition;
  condition.column = proto.column();
  HECTOR_ASSIGN_OR_RETURN(condition.op,
                          CompareOpFromProto(proto.op(), path + ".op"));
  condition.values.reserve(proto.values_size());
  for (int j = 0; j < proto.values_size(); ++j) {
    HECTOR_ASSIGN_OR_RETURN(
        ScalarValue value,
        ScalarValueFromProto(proto.values(j), Indexed(path + ".values", j)));
    condition.values.push_back(std::move(value));
  }
  return condition;
}

absl::StatusOr<FusionParams> FusionFromProto(const v1::FusionSpec& proto) {
  FusionParams params;
  switch (proto.method()) {
    case v1::FUSION_METHOD_UNSPECIFIED:
    case v1::FUSION_METHOD_RRF:
      params.method = FusionMethod::kRrf;
      break;
    default:
      return FieldError("fusion.method",
                        absl::StrCat("unknown value ", proto.method()));
  }
  params.rrf_k = proto.rrf_k() == 0 ? kDefaultRrfK : proto.rrf_k();
  return params;
}

v1::ScoreKind ScoreKindToProto(ScoreKind kind) {
  switch (kind) {
    case ScoreKind::kL2Distance:
      return v1::SCORE_KIND_L2_DISTANCE;
    case ScoreKind::kInnerProduct:
      return v1::SCORE_KIND_INNER_PRODUCT;
    case ScoreKind::kRrf:
      return v1::SCORE_KIND_RRF;
  }
  return v1::SCORE_KIND_UNSPECIFIED;
}

}  // namespace

absl::StatusOr<Schema> SchemaFromProto(const v1::TableSchema& proto) {
  std::vector<ColumnDef> columns;
  columns.reserve(proto.columns_size());
  for (int i = 0; i < proto.columns_size(); ++i) {
    HECTOR_ASSIGN_OR_RETURN(
        ColumnDef column,
        ColumnFromProto(proto.columns(i), Indexed("schema.columns", i)));
    columns.push_back(std::move(column));
  }
  return Schema::Create(proto.name(), std::move(columns));
}

v1::TableSchema SchemaToProto(const Schema& schema) {
  v1::TableSchema proto;
  proto.set_name(schema.table_name());
  for (const ColumnDef& column : schema.columns()) {
    v1::ColumnDef* target = proto.add_columns();
    target->set_name(column.name);
    if (const auto* dense = std::get_if<DenseColumnSpec>(&column.spec)) {
      v1::DenseVectorColumn* out = target->mutable_dense();
      out->set_dim(dense->dim);
      out->set_metric(MetricToProto(dense->metric));
      out->set_index_type(DenseIndexToProto(dense->index));
      out->mutable_hnsw()->set_m(dense->hnsw.m);
      out->mutable_hnsw()->set_ef_construction(dense->hnsw.ef_construction);
      out->mutable_hnsw()->set_seed(dense->hnsw.seed);
      out->set_int8_quantization(dense->int8_quantization);
    } else if (const auto* sparse =
                   std::get_if<SparseColumnSpec>(&column.spec)) {
      target->mutable_sparse()->set_max_dim(sparse->max_dim);
      target->mutable_sparse()->set_index_type(
          SparseIndexToProto(sparse->index));
    } else {
      target->mutable_scalar()->set_type(
          ScalarTypeToProto(std::get<ScalarType>(column.spec)));
    }
  }
  return proto;
}

absl::StatusOr<std::vector<RowInput>> RowsFromProto(
    const google::protobuf::RepeatedPtrField<v1::Row>& rows) {
  std::vector<RowInput> converted;
  converted.reserve(rows.size());
  for (int i = 0; i < rows.size(); ++i) {
    const v1::Row& source = rows.Get(i);
    RowInput row;
    row.primary_key = source.primary_key();
    row.fields.reserve(source.fields().size());
    for (const auto& [name, value] : source.fields()) {
      HECTOR_ASSIGN_OR_RETURN(
          Value field,
          ValueFromProto(value, absl::StrCat(Indexed("rows", i), ".fields[\"",
                                             name, "\"]")));
      row.fields.emplace(name, std::move(field));
    }
    converted.push_back(std::move(row));
  }
  return converted;
}

void RowToProto(const RowOutput& row, v1::Row* proto) {
  proto->set_primary_key(row.primary_key);
  FieldsToProto(row.fields, proto->mutable_fields());
}

absl::StatusOr<SearchRequest> SearchRequestFromProto(
    const v1::SearchRequest& proto) {
  SearchRequest request;
  request.top_k = proto.top_k();
  request.routes.reserve(proto.routes_size());
  for (int i = 0; i < proto.routes_size(); ++i) {
    HECTOR_ASSIGN_OR_RETURN(
        SearchRoute route,
        RouteFromProto(proto.routes(i), Indexed("routes", i)));
    request.routes.push_back(std::move(route));
  }
  const v1::Filter& filter = proto.filter();
  request.filter.conditions.reserve(filter.conditions_size());
  for (int i = 0; i < filter.conditions_size(); ++i) {
    HECTOR_ASSIGN_OR_RETURN(
        Condition condition,
        ConditionFromProto(filter.conditions(i),
                           Indexed("filter.conditions", i)));
    request.filter.conditions.push_back(std::move(condition));
  }
  HECTOR_ASSIGN_OR_RETURN(request.fusion, FusionFromProto(proto.fusion()));
  request.output_columns.assign(proto.output_columns().begin(),
                                proto.output_columns().end());
  return request;
}

void SearchResultToProto(const SearchResult& result,
                         v1::SearchResponse* proto) {
  proto->set_score_kind(ScoreKindToProto(result.score_kind));
  proto->mutable_hits()->Reserve(static_cast<int>(result.rows.size()));
  for (const SearchResultRow& row : result.rows) {
    v1::SearchHit* hit = proto->add_hits();
    hit->set_primary_key(row.primary_key);
    hit->set_score(row.score);
    FieldsToProto(row.fields, hit->mutable_fields());
  }
}

void StatsToProto(const TableStats& stats, Lsn last_lsn,
                  v1::GetStatsResponse* proto) {
  proto->set_live_rows(stats.live_rows);
  proto->set_deleted_rows(stats.deleted_rows);
  proto->set_last_lsn(last_lsn);
  for (const ColumnStatsRow& column : stats.columns) {
    v1::ColumnStats* out = proto->add_columns();
    out->set_name(column.name);
    if (const auto* dense = std::get_if<DenseIndexKind>(&column.index)) {
      out->set_dense_index_type(DenseIndexToProto(*dense));
    } else if (const auto* sparse =
                   std::get_if<SparseIndexKind>(&column.index)) {
      out->set_sparse_index_type(SparseIndexToProto(*sparse));
    }
    // indexed_rows 取自索引的 Size()：含已 Add 但尚未发布的行，也含已打墓碑
    // 的行（c0 不从索引中移除墓碑行）。
    out->set_indexed_rows(column.indexed_rows);
    out->set_memory_bytes(column.memory_bytes);
  }
}

}  // namespace hector::server
