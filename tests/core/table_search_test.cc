#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/types.h"
#include "hector/core/filter.h"
#include "hector/core/fusion.h"
#include "hector/core/schema.h"
#include "hector/core/search.h"
#include "hector/core/table.h"
#include "hector/core/value.h"
#include "hector/index/index_factory.h"
#include "hector/index/metric.h"
#include "hector/index/sparse_index.h"
#include "tests/core/counting_index.h"

namespace hector {
namespace {

using ::hector::test_support::CountingIndexFactory;
using ::hector::test_support::IndexCallLog;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

// 列顺序：emb（稠密 dim=2）、sp（稀疏 max_dim=10）、year（int64）、
// category（string）。
Schema SearchSchema(MetricType metric) {
  auto schema = Schema::Create(
      "docs",
      {ColumnDef{"emb", DenseColumnSpec{.dim = 2,
                                        .metric = metric,
                                        .index = DenseIndexKind::kFlat,
                                        .hnsw = {},
                                        .int8_quantization = false}},
       ColumnDef{"sp", SparseColumnSpec{.max_dim = 10,
                                        .index = SparseIndexKind::kFlat}},
       ColumnDef{"year", ScalarType::kInt64},
       ColumnDef{"category", ScalarType::kString}});
  EXPECT_TRUE(schema.ok()) << schema.status();
  return *std::move(schema);
}

RowInput Row(std::string pk, std::vector<float> emb, SparseVector sp = {},
             int64_t year = 2020, std::string category = "a") {
  RowInput row;
  row.primary_key = std::move(pk);
  row.fields.emplace("emb", Value{std::move(emb)});
  row.fields.emplace("sp", Value{std::move(sp)});
  row.fields.emplace("year", Value{year});
  row.fields.emplace("category", Value{std::move(category)});
  return row;
}

SearchRoute DenseRoute(std::vector<float> query, std::string column = "emb") {
  SearchRoute route;
  route.column = std::move(column);
  route.query = std::move(query);
  return route;
}

SearchRoute SparseRoute(SparseVector query, std::string column = "sp") {
  SearchRoute route;
  route.column = std::move(column);
  route.query = std::move(query);
  return route;
}

SearchRequest Request(uint32_t top_k, std::vector<SearchRoute> routes) {
  SearchRequest request;
  request.top_k = top_k;
  request.routes = std::move(routes);
  return request;
}

std::vector<std::string> Keys(const SearchResult& result) {
  std::vector<std::string> keys;
  keys.reserve(result.rows.size());
  for (const SearchResultRow& row : result.rows) {
    keys.push_back(row.primary_key);
  }
  return keys;
}

std::vector<std::string> FieldNames(const SearchResultRow& row) {
  std::vector<std::string> names;
  names.reserve(row.fields.size());
  for (const auto& [name, value] : row.fields) {
    names.push_back(name);
  }
  return names;
}

class TableSearchTest : public ::testing::Test {
 protected:
  // 每次创建的表在 counting_ 中依次占用 dense(i) 与 sparse(i)。
  std::unique_ptr<Table> MakeTable(MetricType metric) {
    auto table = Table::Create(SearchSchema(metric), counting_.factory());
    EXPECT_TRUE(table.ok()) << table.status();
    return table.ok() ? *std::move(table) : nullptr;
  }

  // 依次 Prepare（kOnline）、Apply、Publish。
  absl::Status Write(Table& table, std::vector<RowInput> rows) {
    auto batch =
        table.PrepareInsert(std::move(rows), false, PrepareMode::kOnline);
    if (!batch.ok()) {
      return batch.status();
    }
    const Lsn lsn = ++lsn_;
    if (absl::Status s = table.ApplyInsert(*std::move(batch), lsn); !s.ok()) {
      return s;
    }
    table.Publish(lsn);
    return absl::OkStatus();
  }

  absl::Status Delete(Table& table, std::vector<std::string> pks) {
    auto ids = table.PrepareDelete(pks);
    if (!ids.ok()) {
      return ids.status();
    }
    const Lsn lsn = ++lsn_;
    if (absl::Status s = table.ApplyDelete(*ids, lsn); !s.ok()) {
      return s;
    }
    table.Publish(lsn);
    return absl::OkStatus();
  }

  CountingIndexFactory counting_;
  Lsn lsn_ = 0;
};

TEST_F(TableSearchTest, SingleDenseRouteScoresAndKind) {
  std::unique_ptr<Table> l2 = MakeTable(MetricType::kL2);
  ASSERT_NE(l2, nullptr);
  ASSERT_TRUE(Write(*l2, {Row("p0", {0.0F, 0.0F}), Row("p1", {1.0F, 0.0F}),
                          Row("p2", {3.0F, 0.0F})})
                  .ok());

  const std::vector<float> query = {0.9F, 0.0F};
  auto result = l2->Search(Request(2, {DenseRoute(query)}));
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->score_kind, ScoreKind::kL2Distance);
  EXPECT_THAT(Keys(*result), ElementsAre("p1", "p0"));
  ASSERT_EQ(result->rows.size(), 2U);
  EXPECT_NEAR(result->rows[0].score, 0.01, 1e-6);
  EXPECT_NEAR(result->rows[1].score, 0.81, 1e-6);
  // 分数就是索引返回的平方距离本身，没有经过融合。
  const std::vector<float> p1 = {1.0F, 0.0F};
  const std::vector<float> p0 = {0.0F, 0.0F};
  EXPECT_EQ(result->rows[0].score, L2SquaredScalar(query.data(), p1.data(), 2));
  EXPECT_EQ(result->rows[1].score, L2SquaredScalar(query.data(), p0.data(), 2));
  // 单路检索时索引收到的 top_k 等于请求的 top_k。
  EXPECT_THAT(counting_.dense(0).search_top_ks(), ElementsAre(2));
  EXPECT_EQ(counting_.sparse(0).search_calls(), 0);

  // 行数少于 top_k 时返回全部行。
  auto all = l2->Search(Request(10, {DenseRoute(query)}));
  ASSERT_TRUE(all.ok()) << all.status();
  EXPECT_THAT(Keys(*all), ElementsAre("p1", "p0", "p2"));

  std::unique_ptr<Table> cosine = MakeTable(MetricType::kCosine);
  ASSERT_NE(cosine, nullptr);
  ASSERT_TRUE(
      Write(*cosine,
            {Row("c0", {0.0F, 2.0F}, SparseVector{{1}, {1.0F}}),
             Row("c1", {4.0F, 3.0F}, SparseVector{{2}, {1.0F}}),
             Row("c2", {5.0F, 0.0F}, SparseVector{{1, 3}, {0.5F, 9.0F}})})
          .ok());
  // 查询 (3,0) 先归一化为 (1,0)：内积依次为 c2=1、c1=0.8、c0=0。
  auto cos_result = cosine->Search(Request(3, {DenseRoute({3.0F, 0.0F})}));
  ASSERT_TRUE(cos_result.ok()) << cos_result.status();
  EXPECT_EQ(cos_result->score_kind, ScoreKind::kInnerProduct);
  EXPECT_THAT(Keys(*cos_result), ElementsAre("c2", "c1", "c0"));
  ASSERT_EQ(cos_result->rows.size(), 3U);
  EXPECT_NEAR(cos_result->rows[0].score, 1.0, 1e-6);
  EXPECT_NEAR(cos_result->rows[1].score, 0.8, 1e-6);
  EXPECT_NEAR(cos_result->rows[2].score, 0.0, 1e-6);
  EXPECT_GE(cos_result->rows[0].score, cos_result->rows[1].score);
  EXPECT_GE(cos_result->rows[1].score, cos_result->rows[2].score);

  // 单路稀疏检索：分数为内积、降序，只返回与查询共享位置的行。
  auto sparse = cosine->Search(Request(5, {SparseRoute({{1}, {2.0F}})}));
  ASSERT_TRUE(sparse.ok()) << sparse.status();
  EXPECT_EQ(sparse->score_kind, ScoreKind::kInnerProduct);
  EXPECT_THAT(Keys(*sparse), ElementsAre("c0", "c2"));
  ASSERT_EQ(sparse->rows.size(), 2U);
  EXPECT_FLOAT_EQ(sparse->rows[0].score, 2.0F);
  EXPECT_FLOAT_EQ(sparse->rows[1].score, 1.0F);
  EXPECT_THAT(counting_.sparse(1).search_top_ks(), ElementsAre(5));
}

TEST_F(TableSearchTest, FilterAndTombstoneExcluded) {
  std::unique_ptr<Table> table = MakeTable(MetricType::kL2);
  ASSERT_NE(table, nullptr);
  const SparseVector sp{{1}, {1.0F}};
  // y 满足筛选条件，只因墓碑被排除；u 只因 category 被排除。
  ASSERT_TRUE(Write(*table, {Row("x", {0.0F, 0.0F}, sp, 2020, "a"),
                             Row("y", {0.1F, 0.0F}, sp, 2023, "a"),
                             Row("z", {0.2F, 0.0F}, sp, 2024, "a"),
                             Row("u", {0.3F, 0.0F}, sp, 2022, "b")})
                  .ok());
  ASSERT_TRUE(Delete(*table, {"y"}).ok());
  // v 满足筛选条件，但只 Apply、未 Publish，对检索不可见。
  auto pending = table->PrepareInsert({Row("v", {0.0F, 0.0F}, sp, 2025, "a")},
                                      false, PrepareMode::kOnline);
  ASSERT_TRUE(pending.ok()) << pending.status();
  ASSERT_TRUE(table->ApplyInsert(*std::move(pending), ++lsn_).ok());

  Filter filter;
  filter.conditions = {
      Condition{
          .column = "year", .op = CompareOp::kGe, .values = {int64_t{2021}}},
      Condition{.column = "category",
                .op = CompareOp::kIn,
                .values = {std::string("a")}}};

  SearchRequest dense = Request(10, {DenseRoute({0.0F, 0.0F})});
  dense.filter = filter;
  auto dense_result = table->Search(dense);
  ASSERT_TRUE(dense_result.ok()) << dense_result.status();
  EXPECT_THAT(Keys(*dense_result), ElementsAre("z"));

  SearchRequest hybrid =
      Request(10, {DenseRoute({0.0F, 0.0F}), SparseRoute(sp)});
  hybrid.filter = filter;
  auto hybrid_result = table->Search(hybrid);
  ASSERT_TRUE(hybrid_result.ok()) << hybrid_result.status();
  EXPECT_THAT(Keys(*hybrid_result), ElementsAre("z"));

  // 不筛选时仍排除墓碑行 y 与未发布行 v。
  auto unfiltered = table->Search(Request(10, {DenseRoute({0.0F, 0.0F})}));
  ASSERT_TRUE(unfiltered.ok()) << unfiltered.status();
  EXPECT_THAT(Keys(*unfiltered), ElementsAre("x", "z", "u"));
}

TEST_F(TableSearchTest, HybridSearchUsesCandidateLimitPerRoute) {
  std::unique_ptr<Table> table = MakeTable(MetricType::kL2);
  ASSERT_NE(table, nullptr);
  std::vector<RowInput> rows;
  for (int i = 0; i < 8; ++i) {
    rows.push_back(Row("r" + std::to_string(i), {static_cast<float>(i), 0.0F},
                       SparseVector{{1}, {static_cast<float>(8 - i)}}));
  }
  ASSERT_TRUE(Write(*table, std::move(rows)).ok());

  SearchRoute dense = DenseRoute({0.0F, 0.0F});
  dense.candidate_limit = 5;
  SearchRoute sparse = SparseRoute({{1}, {1.0F}});
  sparse.candidate_limit = 5;
  sparse.sparse_algorithm = SparseAlgorithm::kWand;
  auto result = table->Search(Request(2, {dense, sparse}));
  ASSERT_TRUE(result.ok()) << result.status();

  EXPECT_EQ(counting_.dense(0).search_calls(), 1);
  EXPECT_THAT(counting_.dense(0).search_top_ks(), ElementsAre(5));
  EXPECT_EQ(counting_.sparse(0).search_calls(), 1);
  EXPECT_THAT(counting_.sparse(0).search_top_ks(), ElementsAre(5));
  ASSERT_EQ(counting_.sparse(0).sparse_search_params().size(), 1U);
  EXPECT_EQ(counting_.sparse(0).sparse_search_params()[0].algorithm,
            SparseAlgorithm::kWand);

  EXPECT_EQ(result->score_kind, ScoreKind::kRrf);
  // 两路排名都是 r0、r1、r2…，融合后 r0 = 2/61，r1 = 2/62。
  EXPECT_THAT(Keys(*result), ElementsAre("r0", "r1"));
  ASSERT_EQ(result->rows.size(), 2U);
  EXPECT_NEAR(result->rows[0].score, 2.0 / 61, 1e-6);
  EXPECT_NEAR(result->rows[1].score, 2.0 / 62, 1e-6);
}

TEST_F(TableSearchTest, DefaultsAppliedWhenZero) {
  std::unique_ptr<Table> table = MakeTable(MetricType::kL2);
  ASSERT_NE(table, nullptr);
  ASSERT_TRUE(Write(*table, {Row("a", {0.0F, 0.0F}, SparseVector{{1}, {1.0F}}),
                             Row("b", {1.0F, 0.0F}, SparseVector{{1}, {2.0F}})})
                  .ok());
  const IndexCallLog& dense_log = counting_.dense(0);
  const IndexCallLog& sparse_log = counting_.sparse(0);
  const SparseVector sp{{1}, {1.0F}};

  // 多路、candidate_limit=0：每路取 max(100, top_k)。
  ASSERT_TRUE(
      table->Search(Request(150, {DenseRoute({0.0F, 0.0F}), SparseRoute(sp)}))
          .ok());
  ASSERT_TRUE(
      table->Search(Request(10, {DenseRoute({0.0F, 0.0F}), SparseRoute(sp)}))
          .ok());
  EXPECT_THAT(dense_log.search_top_ks(), ElementsAre(150, 100));
  EXPECT_THAT(sparse_log.search_top_ks(), ElementsAre(150, 100));

  // ef_search=0 换成 64；非 0 原样传给索引，由索引自己取 max。
  SearchRoute explicit_ef = DenseRoute({0.0F, 0.0F});
  explicit_ef.ef_search = 200;
  explicit_ef.rerank_candidates = 30;
  ASSERT_TRUE(table->Search(Request(10, {explicit_ef})).ok());
  const std::vector<SearchParams> params = dense_log.search_params();
  ASSERT_EQ(params.size(), 3U);
  EXPECT_EQ(params[0].ef_search, 64U);
  EXPECT_EQ(params[0].rerank_candidates, 0U);
  EXPECT_EQ(params[1].ef_search, 64U);
  EXPECT_EQ(params[2].ef_search, 200U);
  EXPECT_EQ(params[2].rerank_candidates, 30U);

  // 单路检索：candidate_limit 只校验不使用，索引收到的是 top_k。
  SearchRoute single = DenseRoute({0.0F, 0.0F});
  single.candidate_limit = 20;
  ASSERT_TRUE(table->Search(Request(7, {single})).ok());
  EXPECT_EQ(dense_log.search_top_ks().back(), 7U);

  // rrf_k=0 按 60 融合：两行在两路中排名相同，a = 2/61。
  SearchRequest rrf =
      Request(1, {DenseRoute({0.0F, 0.0F}), SparseRoute({{1}, {-1.0F}})});
  rrf.fusion.rrf_k = 0;
  auto fused = table->Search(rrf);
  ASSERT_TRUE(fused.ok()) << fused.status();
  ASSERT_EQ(fused->rows.size(), 1U);
  EXPECT_EQ(fused->rows[0].primary_key, "a");
  EXPECT_NEAR(fused->rows[0].score, 2.0 / 61, 1e-6);
}

TEST_F(TableSearchTest, RejectsInvalidRequests) {
  std::unique_ptr<Table> table = MakeTable(MetricType::kCosine);
  ASSERT_NE(table, nullptr);
  ASSERT_TRUE(
      Write(*table, {Row("a", {1.0F, 0.0F}, SparseVector{{1}, {1.0F}})}).ok());

  // 合法的两路请求；每个用例只破坏一处。
  const auto base = [] {
    return Request(5, {DenseRoute({1.0F, 0.0F}), SparseRoute({{1}, {1.0F}})});
  };
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<std::pair<std::string, std::function<void(SearchRequest&)>>>
      cases = {
          {"top_k=0", [](SearchRequest& r) { r.top_k = 0; }},
          {"top_k=1025", [](SearchRequest& r) { r.top_k = 1025; }},
          {"candidate_limit<top_k",
           [](SearchRequest& r) { r.routes[0].candidate_limit = 4; }},
          {"candidate_limit=10001",
           [](SearchRequest& r) { r.routes[1].candidate_limit = 10001; }},
          {"single route candidate_limit<top_k",
           [](SearchRequest& r) {
             r.routes.pop_back();
             r.routes[0].candidate_limit = 4;
           }},
          {"ef_search=10001",
           [](SearchRequest& r) { r.routes[0].ef_search = 10001; }},
          {"rerank_candidates=10001",
           [](SearchRequest& r) { r.routes[0].rerank_candidates = 10001; }},
          {"0 routes", [](SearchRequest& r) { r.routes.clear(); }},
          {"9 routes",
           [](SearchRequest& r) {
             while (r.routes.size() < 9) {
               r.routes.push_back(r.routes[0]);
             }
           }},
          {"unknown column",
           [](SearchRequest& r) { r.routes[1].column = "nope"; }},
          {"scalar column",
           [](SearchRequest& r) { r.routes[1].column = "year"; }},
          {"sparse query on dense column",
           [](SearchRequest& r) {
             r.routes[0].query = SparseVector{{1}, {1.0F}};
           }},
          {"dense query on sparse column",
           [](SearchRequest& r) {
             r.routes[1].query = std::vector<float>{1.0F, 0.0F};
           }},
          {"zero COSINE query",
           [](SearchRequest& r) {
             r.routes[0].query = std::vector<float>{0.0F, 0.0F};
           }},
          {"dense dim mismatch",
           [](SearchRequest& r) {
             r.routes[0].query = std::vector<float>{1.0F};
           }},
          {"dense NaN",
           [nan](SearchRequest& r) {
             r.routes[0].query = std::vector<float>{nan, 1.0F};
           }},
          {"sparse index >= max_dim",
           [](SearchRequest& r) {
             r.routes[1].query = SparseVector{{10}, {1.0F}};
           }},
          {"sparse duplicate index",
           [](SearchRequest& r) {
             r.routes[1].query = SparseVector{{2, 2}, {1.0F, 1.0F}};
           }},
          {"sparse length mismatch",
           [](SearchRequest& r) {
             r.routes[1].query = SparseVector{{1, 2}, {1.0F}};
           }},
          {"output vector column",
           [](SearchRequest& r) { r.output_columns = {"emb"}; }},
          {"output unknown column",
           [](SearchRequest& r) { r.output_columns = {"category", "nope"}; }},
          {"invalid filter",
           [](SearchRequest& r) {
             r.filter.conditions = {Condition{.column = "year",
                                              .op = CompareOp::kEq,
                                              .values = {std::string("2020")}}};
           }},
  };
  for (const auto& [name, mutate] : cases) {
    SearchRequest request = base();
    mutate(request);
    EXPECT_EQ(table->Search(request).status().code(),
              absl::StatusCode::kInvalidArgument)
        << name;
  }
  // 请求非法时任何索引的 Search 都没有被调用。
  EXPECT_EQ(counting_.dense(0).search_calls(), 0);
  EXPECT_EQ(counting_.sparse(0).search_calls(), 0);

  // 对照：未破坏的请求合法，且两路索引各被调用 1 次。
  ASSERT_TRUE(table->Search(base()).ok());
  EXPECT_EQ(counting_.dense(0).search_calls(), 1);
  EXPECT_EQ(counting_.sparse(0).search_calls(), 1);
}

TEST_F(TableSearchTest, OutputColumnsAndSameColumnTwice) {
  std::unique_ptr<Table> table = MakeTable(MetricType::kL2);
  ASSERT_NE(table, nullptr);
  ASSERT_TRUE(Write(*table, {Row("a", {0.0F, 0.0F}, {}, 2021, "news"),
                             Row("b", {1.0F, 0.0F}, {}, 2022, "blog")})
                  .ok());

  auto bare = table->Search(Request(2, {DenseRoute({0.0F, 0.0F})}));
  ASSERT_TRUE(bare.ok()) << bare.status();
  ASSERT_EQ(bare->rows.size(), 2U);
  for (const SearchResultRow& row : bare->rows) {
    EXPECT_THAT(row.fields, IsEmpty());
  }

  SearchRequest with_category = Request(2, {DenseRoute({0.0F, 0.0F})});
  with_category.output_columns = {"category"};
  auto projected = table->Search(with_category);
  ASSERT_TRUE(projected.ok()) << projected.status();
  ASSERT_EQ(projected->rows.size(), 2U);
  EXPECT_THAT(FieldNames(projected->rows[0]), ElementsAre("category"));
  EXPECT_EQ(std::get<std::string>(projected->rows[0].fields[0].second), "news");
  EXPECT_EQ(std::get<std::string>(projected->rows[1].fields[0].second), "blog");

  // 返回的列按 Schema 顺序排列。
  SearchRequest two_columns = Request(1, {DenseRoute({0.0F, 0.0F})});
  two_columns.output_columns = {"category", "year"};
  auto both = table->Search(two_columns);
  ASSERT_TRUE(both.ok()) << both.status();
  ASSERT_EQ(both->rows.size(), 1U);
  EXPECT_THAT(FieldNames(both->rows[0]), ElementsAre("year", "category"));
  EXPECT_EQ(std::get<int64_t>(both->rows[0].fields[0].second), 2021);

  // 同一稠密列出现在两路中是合法的，结果按 RRF 融合。
  auto twice = table->Search(
      Request(2, {DenseRoute({0.0F, 0.0F}), DenseRoute({1.0F, 0.0F})}));
  ASSERT_TRUE(twice.ok()) << twice.status();
  EXPECT_EQ(twice->score_kind, ScoreKind::kRrf);
  // a、b 各在一路排第 1、另一路排第 2，分数相同，按 row id 升序。
  EXPECT_THAT(Keys(*twice), ElementsAre("a", "b"));
}

TEST_F(TableSearchTest, ConcurrentSearchDuringInsert) {
  std::unique_ptr<Table> table = MakeTable(MetricType::kL2);
  ASSERT_NE(table, nullptr);
  const auto make_row = [](int i) {
    return Row("r" + std::to_string(i),
               {static_cast<float>(i % 17), static_cast<float>(i % 5)},
               SparseVector{{static_cast<uint32_t>(i % 10)}, {1.0F}},
               2000 + (i % 30), i % 2 == 0 ? "a" : "b");
  };
  constexpr int kInitialRows = 50;
  constexpr int kInsertedRows = 500;
  std::vector<RowInput> initial;
  for (int i = 0; i < kInitialRows; ++i) {
    initial.push_back(make_row(i));
  }
  ASSERT_TRUE(Write(*table, std::move(initial)).ok());

  std::atomic<bool> done{false};
  std::atomic<int> searches{0};
  std::atomic<int> failures{0};
  const auto reader = [&](int seed) {
    int iteration = 0;
    while (!done.load(std::memory_order_acquire) || iteration == 0) {
      const float q = static_cast<float>((seed + iteration) % 17);
      SearchRequest request =
          iteration % 2 == 0
              ? Request(10, {DenseRoute({q, 0.0F})})
              : Request(10,
                        {DenseRoute({q, 1.0F}),
                         SparseRoute({{static_cast<uint32_t>(seed)}, {1.0F}})});
      if (iteration % 3 == 0) {
        request.filter.conditions = {
            Condition{.column = "category",
                      .op = CompareOp::kEq,
                      .values = {std::string(seed % 2 == 0 ? "a" : "b")}}};
        request.output_columns = {"year"};
      }
      auto result = table->Search(request);
      if (!result.ok()) {
        failures.fetch_add(1, std::memory_order_relaxed);
      } else {
        std::vector<std::string> keys = Keys(*result);
        if (!keys.empty()) {
          std::vector<std::string> missing;
          auto rows = table->Get(keys, {}, &missing);
          if (!rows.ok() || !missing.empty() || rows->size() != keys.size()) {
            failures.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
      searches.fetch_add(1, std::memory_order_relaxed);
      ++iteration;
    }
  };

  std::vector<std::thread> readers;
  readers.reserve(4);
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back(reader, t);
  }
  std::thread writer([&] {
    for (int i = kInitialRows; i < kInitialRows + kInsertedRows; ++i) {
      auto batch =
          table->PrepareInsert({make_row(i)}, false, PrepareMode::kOnline);
      if (!batch.ok() || !table->ApplyInsert(*std::move(batch), ++lsn_).ok()) {
        failures.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      table->Publish(lsn_);
    }
    done.store(true, std::memory_order_release);
  });
  writer.join();
  for (std::thread& t : readers) {
    t.join();
  }

  EXPECT_EQ(failures.load(), 0);
  EXPECT_GE(searches.load(), 4);
  EXPECT_EQ(table->Stats().live_rows,
            static_cast<uint64_t>(kInitialRows + kInsertedRows));
}

}  // namespace
}  // namespace hector
