#ifndef HECTOR_SERVER_PROTO_CONVERT_H_
#define HECTOR_SERVER_PROTO_CONVERT_H_

#include <google/protobuf/repeated_ptr_field.h>

#include <vector>

#include "absl/status/statusor.h"
#include "hector/common/types.h"
#include "hector/core/schema.h"
#include "hector/core/search.h"
#include "hector/core/table.h"
#include "hector/core/value.h"
#include "hector/v1/hector.pb.h"

namespace hector::server {

// proto 与 core 类型互转（design.md 3.2、3.3、第 6 节）。proto 字段非法时
// 返回 InvalidArgument，消息以字段路径开头，例如
// `schema.columns[0].dense.metric: unspecified`、
// `rows[3].fields["emb"]: value is empty`。取值范围等其余校验由 core 负责。

// 零值分工：index_type 未指定转为 kFlat，hnsw.seed 缺省转为 42；hnsw.m 与
// ef_construction 原样传 0，由 Schema::Create 换成默认值；metric 与普通列
// type 未指定时报错。
absl::StatusOr<Schema> SchemaFromProto(const v1::TableSchema& proto);
// 回显填好默认值的 Schema。
v1::TableSchema SchemaToProto(const Schema& schema);

absl::StatusOr<std::vector<RowInput>> RowsFromProto(
    const google::protobuf::RepeatedPtrField<v1::Row>& rows);
void RowToProto(const RowOutput& row, v1::Row* proto);

// FUSION_METHOD_UNSPECIFIED 转为 kRrf，rrf_k 为 0 时转为 60；
// SPARSE_ALGORITHM_UNSPECIFIED 转为 kAuto。
absl::StatusOr<SearchRequest> SearchRequestFromProto(
    const v1::SearchRequest& proto);
void SearchResultToProto(const SearchResult& result, v1::SearchResponse* proto);

// last_lsn 是整个引擎的最后一条 WAL 记录的 LSN，不是本表的。
void StatsToProto(const TableStats& stats, Lsn last_lsn,
                  v1::GetStatsResponse* proto);

}  // namespace hector::server

#endif  // HECTOR_SERVER_PROTO_CONVERT_H_
