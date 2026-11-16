// 冒烟测试：hector.v1 的生成代码能在 C++20 下编译、使用，并覆盖全部 RPC。

#include <google/protobuf/descriptor.h>
#include <google/protobuf/util/message_differencer.h>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "hector/v1/hector.grpc.pb.h"
#include "hector/v1/hector.pb.h"

namespace hector::v1 {
namespace {

using Service = HectorService::Service;

template <typename Request, typename Response>
using UnaryHandler = ::grpc::Status (Service::*)(::grpc::ServerContext*,
                                                 const Request*, Response*);

// 任一 RPC 缺失或签名不符时，下面的 static_assert 会让本文件编译失败。
static_assert(std::is_same_v<
              decltype(&Service::CreateDatabase),
              UnaryHandler<CreateDatabaseRequest, CreateDatabaseResponse>>);
static_assert(
    std::is_same_v<decltype(&Service::DropDatabase),
                   UnaryHandler<DropDatabaseRequest, DropDatabaseResponse>>);
static_assert(
    std::is_same_v<decltype(&Service::ListDatabases),
                   UnaryHandler<ListDatabasesRequest, ListDatabasesResponse>>);
static_assert(
    std::is_same_v<decltype(&Service::CreateTable),
                   UnaryHandler<CreateTableRequest, CreateTableResponse>>);
static_assert(
    std::is_same_v<decltype(&Service::DropTable),
                   UnaryHandler<DropTableRequest, DropTableResponse>>);
static_assert(
    std::is_same_v<decltype(&Service::ListTables),
                   UnaryHandler<ListTablesRequest, ListTablesResponse>>);
static_assert(
    std::is_same_v<decltype(&Service::DescribeTable),
                   UnaryHandler<DescribeTableRequest, DescribeTableResponse>>);
static_assert(std::is_same_v<decltype(&Service::Insert),
                             UnaryHandler<InsertRequest, InsertResponse>>);
static_assert(std::is_same_v<decltype(&Service::Upsert),
                             UnaryHandler<UpsertRequest, UpsertResponse>>);
static_assert(std::is_same_v<decltype(&Service::Delete),
                             UnaryHandler<DeleteRequest, DeleteResponse>>);
static_assert(std::is_same_v<decltype(&Service::Get),
                             UnaryHandler<GetRequest, GetResponse>>);
static_assert(std::is_same_v<decltype(&Service::Search),
                             UnaryHandler<SearchRequest, SearchResponse>>);
static_assert(std::is_same_v<
              decltype(&Service::CreateSnapshot),
              UnaryHandler<CreateSnapshotRequest, CreateSnapshotResponse>>);
static_assert(std::is_same_v<decltype(&Service::GetStats),
                             UnaryHandler<GetStatsRequest, GetStatsResponse>>);

constexpr int kRpcCount = 14;

SearchRequest MakeFullSearchRequest() {
  SearchRequest request;
  request.set_database("demo_db");
  request.set_table("docs");
  request.set_top_k(10);

  SearchRoute* dense_route = request.add_routes();
  dense_route->set_column("emb");
  for (float v : {0.1F, -0.2F, 0.3F, 0.4F}) {
    dense_route->mutable_dense()->add_values(v);
  }
  dense_route->set_candidate_limit(100);
  dense_route->set_ef_search(64);
  dense_route->set_rerank_candidates(50);

  SearchRoute* sparse_route = request.add_routes();
  sparse_route->set_column("bm25");
  SparseVector* sparse = sparse_route->mutable_sparse();
  sparse->add_indices(3);
  sparse->add_values(1.5F);
  sparse->add_indices(17);
  sparse->add_values(0.25F);
  sparse_route->set_sparse_algorithm(SPARSE_ALGORITHM_WAND);

  Filter* filter = request.mutable_filter();
  Condition* category = filter->add_conditions();
  category->set_column("category");
  category->set_op(COMPARE_OP_IN);
  category->add_values()->set_string_value("news");
  category->add_values()->set_string_value("blog");
  Condition* year = filter->add_conditions();
  year->set_column("year");
  year->set_op(COMPARE_OP_GE);
  year->add_values()->set_int64_value(2020);
  Condition* rating = filter->add_conditions();
  rating->set_column("rating");
  rating->set_op(COMPARE_OP_LT);
  rating->add_values()->set_float_value(4.5F);

  request.mutable_fusion()->set_method(FUSION_METHOD_RRF);
  request.mutable_fusion()->set_rrf_k(60);
  request.add_output_columns("category");
  request.add_output_columns("year");
  return request;
}

TEST(ProtoSmokeTest, GeneratedTypesRoundTrip) {
  const SearchRequest original = MakeFullSearchRequest();
  ASSERT_EQ(original.routes_size(), 2);
  ASSERT_EQ(original.filter().conditions_size(), 3);

  std::string bytes;
  ASSERT_TRUE(original.SerializeToString(&bytes));

  SearchRequest parsed;
  EXPECT_TRUE(parsed.ParseFromString(bytes));
  EXPECT_TRUE(
      google::protobuf::util::MessageDifferencer::Equals(original, parsed));
  EXPECT_EQ(parsed.routes(0).rerank_candidates(), 50U);
  EXPECT_EQ(parsed.routes(1).sparse_algorithm(), SPARSE_ALGORITHM_WAND);
  EXPECT_EQ(parsed.fusion().method(), FUSION_METHOD_RRF);
}

TEST(ProtoSmokeTest, ServiceStubDeclaresAllRpcs) {
  const google::protobuf::ServiceDescriptor* service =
      google::protobuf::DescriptorPool::generated_pool()->FindServiceByName(
          std::string(HectorService::service_full_name()));
  ASSERT_NE(service, nullptr);
  EXPECT_EQ(service->full_name(), "hector.v1.HectorService");
  EXPECT_EQ(service->method_count(), kRpcCount);
}

TEST(ProtoSmokeTest, EnumZeroValuesAreUnspecified) {
  EXPECT_EQ(static_cast<int>(METRIC_UNSPECIFIED), 0);
  EXPECT_EQ(static_cast<int>(DENSE_INDEX_TYPE_UNSPECIFIED), 0);
  EXPECT_EQ(static_cast<int>(SPARSE_INDEX_TYPE_UNSPECIFIED), 0);
  EXPECT_EQ(static_cast<int>(SCALAR_TYPE_UNSPECIFIED), 0);
  EXPECT_EQ(static_cast<int>(COMPARE_OP_UNSPECIFIED), 0);
  EXPECT_EQ(static_cast<int>(FUSION_METHOD_UNSPECIFIED), 0);
  EXPECT_EQ(static_cast<int>(SPARSE_ALGORITHM_UNSPECIFIED), 0);
  EXPECT_EQ(static_cast<int>(SCORE_KIND_UNSPECIFIED), 0);
}

}  // namespace
}  // namespace hector::v1
