// Flat 稠密索引的检索基准。参数：行数、维度。

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "hector/common/row_filter.h"
#include "hector/common/types.h"
#include "hector/common/vector_accessor.h"
#include "hector/index/flat_index.h"
#include "hector/index/metric.h"
#include "hector/index/vector_index.h"

namespace {

using hector::DenseVectorAccessor;
using hector::FlatIndex;
using hector::MetricType;
using hector::RowFilter;
using hector::RowId;
using hector::SearchParams;

// 连续内存中的行主序向量；core 的列存储在 Task 5 才有，这里先用简单实现。
class InMemoryDenseAccessor final : public DenseVectorAccessor {
 public:
  InMemoryDenseAccessor(uint32_t dim, std::vector<float> data)
      : dim_(dim), data_(std::move(data)) {}

  uint32_t dim() const override { return dim_; }
  RowId size() const override {
    return static_cast<RowId>(data_.size() / dim_);
  }
  std::span<const float> Get(RowId id) const override {
    return std::span<const float>(data_).subspan(static_cast<size_t>(id) * dim_,
                                                 dim_);
  }

 private:
  uint32_t dim_;
  std::vector<float> data_;
};

std::vector<float> RandomFloats(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> out(n);
  for (float& x : out) {
    x = dist(rng);
  }
  return out;
}

void BM_FlatSearch(benchmark::State& state) {
  const auto rows = static_cast<RowId>(state.range(0));
  const auto dim = static_cast<uint32_t>(state.range(1));
  const InMemoryDenseAccessor accessor(
      dim, RandomFloats(static_cast<size_t>(rows) * dim, 42));
  FlatIndex index(dim, MetricType::kL2, &accessor);
  for (RowId id = 0; id < rows; ++id) {
    if (!index.Add(id).ok()) {
      state.SkipWithError("Add failed");
      return;
    }
  }
  const std::vector<float> query = RandomFloats(dim, 7);
  const RowFilter filter = RowFilter::AllowAll(rows);

  for ([[maybe_unused]] auto _ : state) {
    auto hits = index.Search(query, 10, SearchParams{}, filter);
    if (!hits.ok()) {
      state.SkipWithError("Search failed");
      return;
    }
    benchmark::DoNotOptimize(hits);
  }
  state.SetItemsProcessed(state.iterations() * rows);
}
BENCHMARK(BM_FlatSearch)->Args({1000, 128})->Args({10000, 128});

}  // namespace
