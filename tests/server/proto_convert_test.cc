#include "src/server/proto_convert.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <variant>

#include "absl/status/status.h"
#include "hector/core/schema.h"
#include "hector/index/index_factory.h"
#include "hector/v1/hector.pb.h"

namespace hector::server {
namespace {

using ::testing::HasSubstr;

TEST(ProtoConvertTest, SchemaRoundTripWithDefaults) {
  v1::TableSchema proto;
  proto.set_name("docs");
  v1::ColumnDef* column = proto.add_columns();
  column->set_name("emb");
  column->mutable_dense()->set_dim(8);
  column->mutable_dense()->set_metric(v1::METRIC_COSINE);

  auto schema = SchemaFromProto(proto);
  ASSERT_TRUE(schema.ok()) << schema.status();
  ASSERT_EQ(schema->columns().size(), 1U);
  const auto* dense = std::get_if<DenseColumnSpec>(&schema->columns()[0].spec);
  ASSERT_NE(dense, nullptr);
  EXPECT_EQ(dense->dim, 8U);
  EXPECT_EQ(dense->metric, MetricType::kCosine);
  EXPECT_EQ(dense->index, DenseIndexKind::kFlat);
  EXPECT_EQ(dense->hnsw.seed, 42U);

  const v1::TableSchema back = SchemaToProto(*schema);
  EXPECT_EQ(back.name(), "docs");
  ASSERT_EQ(back.columns_size(), 1);
  ASSERT_TRUE(back.columns(0).has_dense());
  const v1::DenseVectorColumn& echoed = back.columns(0).dense();
  EXPECT_EQ(echoed.dim(), 8U);
  EXPECT_EQ(echoed.metric(), v1::METRIC_COSINE);
  EXPECT_EQ(echoed.index_type(), v1::DENSE_INDEX_TYPE_FLAT);
  EXPECT_EQ(echoed.hnsw().m(), 16U);
  EXPECT_EQ(echoed.hnsw().ef_construction(), 200U);
  EXPECT_EQ(echoed.hnsw().seed(), 42U);

  column->mutable_dense()->set_metric(v1::METRIC_UNSPECIFIED);
  auto unspecified = SchemaFromProto(proto);
  EXPECT_EQ(unspecified.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(unspecified.status().message(),
              HasSubstr("schema.columns[0].dense.metric"));
}

TEST(ProtoConvertTest, EmptyValueReportsFieldPath) {
  google::protobuf::RepeatedPtrField<v1::Row> rows;
  v1::Row* row = rows.Add();
  row->set_primary_key("a");
  (*row->mutable_fields())["emb"];  // oneof 未设置

  auto converted = RowsFromProto(rows);
  EXPECT_EQ(converted.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(converted.status().message(),
              HasSubstr("rows[0].fields[\"emb\"]"));
}

}  // namespace
}  // namespace hector::server
