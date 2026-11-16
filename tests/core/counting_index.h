#ifndef HECTOR_TESTS_CORE_COUNTING_INDEX_H_
#define HECTOR_TESTS_CORE_COUNTING_INDEX_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/row_filter.h"
#include "hector/common/types.h"
#include "hector/common/vector_accessor.h"
#include "hector/core/table.h"
#include "hector/index/index_factory.h"
#include "hector/index/sparse_index.h"
#include "hector/index/vector_index.h"

namespace hector {
namespace test_support {

// 一个索引的调用记录。计数器是原子类型：并发用例中多个检索线程同时计数，
// tsan 下不能报竞争；参数列表由互斥锁保护。
class IndexCallLog {
 public:
  int add_calls() const { return add_calls_.load(std::memory_order_relaxed); }
  int build_calls() const {
    return build_calls_.load(std::memory_order_relaxed);
  }
  int search_calls() const {
    return search_calls_.load(std::memory_order_relaxed);
  }

  std::vector<RowId> added_row_ids() const {
    std::lock_guard<std::mutex> lock(mu_);
    return added_row_ids_;
  }
  std::vector<RowId> built_row_ids() const {
    std::lock_guard<std::mutex> lock(mu_);
    return built_row_ids_;
  }
  // 每次 Search 收到的 top_k，按调用顺序。
  std::vector<uint32_t> search_top_ks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return search_top_ks_;
  }
  // 稠密索引每次 Search 收到的 SearchParams，按调用顺序。
  std::vector<SearchParams> search_params() const {
    std::lock_guard<std::mutex> lock(mu_);
    return search_params_;
  }
  // 稀疏索引每次 Search 收到的 SparseSearchParams，按调用顺序。
  std::vector<SparseSearchParams> sparse_search_params() const {
    std::lock_guard<std::mutex> lock(mu_);
    return sparse_search_params_;
  }

  void RecordAdd(RowId id) {
    add_calls_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mu_);
    added_row_ids_.push_back(id);
  }
  void RecordBuild(std::span<const RowId> ids) {
    build_calls_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mu_);
    built_row_ids_.insert(built_row_ids_.end(), ids.begin(), ids.end());
  }
  void RecordSearch(uint32_t top_k, const SearchParams& params) {
    search_calls_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mu_);
    search_top_ks_.push_back(top_k);
    search_params_.push_back(params);
  }
  void RecordSparseSearch(uint32_t top_k, const SparseSearchParams& params) {
    search_calls_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mu_);
    search_top_ks_.push_back(top_k);
    sparse_search_params_.push_back(params);
  }

 private:
  std::atomic<int> add_calls_{0};
  std::atomic<int> build_calls_{0};
  std::atomic<int> search_calls_{0};

  mutable std::mutex mu_;
  std::vector<RowId> added_row_ids_;
  std::vector<RowId> built_row_ids_;
  std::vector<uint32_t> search_top_ks_;
  std::vector<SearchParams> search_params_;
  std::vector<SparseSearchParams> sparse_search_params_;
};

// 包装真实稠密索引，转发每个调用并记录次数与参数。
class CountingVectorIndex final : public VectorIndex {
 public:
  CountingVectorIndex(std::unique_ptr<VectorIndex> inner,
                      std::shared_ptr<IndexCallLog> log)
      : inner_(std::move(inner)), log_(std::move(log)) {}

  absl::Status Build(std::span<const RowId> row_ids) override {
    log_->RecordBuild(row_ids);
    return inner_->Build(row_ids);
  }
  absl::Status Add(RowId row_id) override {
    log_->RecordAdd(row_id);
    return inner_->Add(row_id);
  }
  absl::StatusOr<std::vector<Hit>> Search(
      std::span<const float> query, uint32_t top_k, const SearchParams& params,
      const RowFilter& filter) const override {
    log_->RecordSearch(top_k, params);
    return inner_->Search(query, top_k, params, filter);
  }
  absl::StatusOr<std::vector<Hit>> SearchCandidates(
      std::span<const float> query, uint32_t top_k,
      std::span<const RowId> candidates) const override {
    return inner_->SearchCandidates(query, top_k, candidates);
  }
  size_t Size() const override { return inner_->Size(); }
  size_t MemoryBytes() const override { return inner_->MemoryBytes(); }
  absl::Status Save(ByteWriter& out) const override {
    return inner_->Save(out);
  }
  absl::Status Load(ByteReader& in) override { return inner_->Load(in); }

 private:
  std::unique_ptr<VectorIndex> inner_;
  std::shared_ptr<IndexCallLog> log_;
};

// 包装真实稀疏索引，转发每个调用并记录次数与参数。
class CountingSparseIndex final : public SparseIndex {
 public:
  CountingSparseIndex(std::unique_ptr<SparseIndex> inner,
                      std::shared_ptr<IndexCallLog> log)
      : inner_(std::move(inner)), log_(std::move(log)) {}

  absl::Status Build(std::span<const RowId> row_ids) override {
    log_->RecordBuild(row_ids);
    return inner_->Build(row_ids);
  }
  absl::Status Add(RowId row_id) override {
    log_->RecordAdd(row_id);
    return inner_->Add(row_id);
  }
  absl::StatusOr<std::vector<Hit>> Search(
      SparseVectorView query, uint32_t top_k, const SparseSearchParams& params,
      const RowFilter& filter) const override {
    log_->RecordSparseSearch(top_k, params);
    return inner_->Search(query, top_k, params, filter);
  }
  size_t Size() const override { return inner_->Size(); }
  size_t MemoryBytes() const override { return inner_->MemoryBytes(); }
  absl::Status Save(ByteWriter& out) const override {
    return inner_->Save(out);
  }
  absl::Status Load(ByteReader& in) override { return inner_->Load(in); }

 private:
  std::unique_ptr<SparseIndex> inner_;
  std::shared_ptr<IndexCallLog> log_;
};

// 通过 IndexFactory 注入 counting 索引。内部用 MakeVectorIndex /
// MakeSparseIndex 创建真实索引，因此 HNSW 等仍然返回 Unimplemented。
// Table 按 Schema 列顺序创建索引，所以 dense(i) / sparse(i) 是第 i 个
// 稠密列 / 稀疏列的记录。本对象必须比由 factory() 创建的 Table 活得久；
// Table::Create 只能在一个线程中调用。
class CountingIndexFactory {
 public:
  IndexFactory factory() {
    IndexFactory f;
    f.dense = [this](const DenseIndexConfig& config,
                     const DenseVectorAccessor* accessor)
        -> absl::StatusOr<std::unique_ptr<VectorIndex>> {
      auto inner = MakeVectorIndex(config, accessor);
      if (!inner.ok()) {
        return inner.status();
      }
      auto log = std::make_shared<IndexCallLog>();
      dense_logs_.push_back(log);
      return std::make_unique<CountingVectorIndex>(*std::move(inner),
                                                   std::move(log));
    };
    f.sparse = [this](const SparseIndexConfig& config,
                      const SparseVectorAccessor* accessor)
        -> absl::StatusOr<std::unique_ptr<SparseIndex>> {
      auto inner = MakeSparseIndex(config, accessor);
      if (!inner.ok()) {
        return inner.status();
      }
      auto log = std::make_shared<IndexCallLog>();
      sparse_logs_.push_back(log);
      return std::make_unique<CountingSparseIndex>(*std::move(inner),
                                                   std::move(log));
    };
    return f;
  }

  size_t dense_count() const { return dense_logs_.size(); }
  size_t sparse_count() const { return sparse_logs_.size(); }
  const IndexCallLog& dense(size_t i) const { return *dense_logs_.at(i); }
  const IndexCallLog& sparse(size_t i) const { return *sparse_logs_.at(i); }

  // 所有已创建索引的 Add 调用总数。
  int total_add_calls() const {
    int sum = 0;
    for (const auto& log : dense_logs_) {
      sum += log->add_calls();
    }
    for (const auto& log : sparse_logs_) {
      sum += log->add_calls();
    }
    return sum;
  }
  int total_build_calls() const {
    int sum = 0;
    for (const auto& log : dense_logs_) {
      sum += log->build_calls();
    }
    for (const auto& log : sparse_logs_) {
      sum += log->build_calls();
    }
    return sum;
  }

 private:
  std::vector<std::shared_ptr<IndexCallLog>> dense_logs_;
  std::vector<std::shared_ptr<IndexCallLog>> sparse_logs_;
};

}  // namespace test_support
}  // namespace hector

#endif  // HECTOR_TESTS_CORE_COUNTING_INDEX_H_
