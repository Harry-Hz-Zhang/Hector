#ifndef HECTOR_TESTS_INDEX_FAKE_ACCESSORS_H_
#define HECTOR_TESTS_INDEX_FAKE_ACCESSORS_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "hector/common/types.h"
#include "hector/common/vector_accessor.h"

namespace hector {
namespace testing {

// 原子计数器数组：并发用例中多个检索线程同时计数，tsan 下不能报竞争。
class GetCounter {
 public:
  explicit GetCounter(size_t n)
      : n_(n), counts_(std::make_unique<std::atomic<int>[]>(n)) {}

  void Record(RowId id) const {
    counts_[id].fetch_add(1, std::memory_order_relaxed);
  }
  int count(RowId id) const {
    return counts_[id].load(std::memory_order_relaxed);
  }
  int total() const {
    int sum = 0;
    for (size_t i = 0; i < n_; ++i) {
      sum += counts_[i].load(std::memory_order_relaxed);
    }
    return sum;
  }
  void Reset() const {
    for (size_t i = 0; i < n_; ++i) {
      counts_[i].store(0, std::memory_order_relaxed);
    }
  }

 private:
  size_t n_;
  std::unique_ptr<std::atomic<int>[]> counts_;
};

// 内存稠密访问器：行在构造时一次给定，之后不变；记录每个 row id 的 Get 次数。
class FakeDenseAccessor final : public DenseVectorAccessor {
 public:
  FakeDenseAccessor(uint32_t dim, std::vector<std::vector<float>> rows)
      : dim_(dim), rows_(std::move(rows)), gets_(rows_.size()) {}

  uint32_t dim() const override { return dim_; }
  RowId size() const override { return static_cast<RowId>(rows_.size()); }
  std::span<const float> Get(RowId id) const override {
    gets_.Record(id);
    return rows_[id];
  }

  int get_count(RowId id) const { return gets_.count(id); }
  int total_gets() const { return gets_.total(); }
  void ResetCounts() const { gets_.Reset(); }

 private:
  uint32_t dim_;
  std::vector<std::vector<float>> rows_;
  GetCounter gets_;
};

// 内存稀疏访问器：每行 indices 严格升序；记录每个 row id 的 Get 次数。
class FakeSparseAccessor final : public SparseVectorAccessor {
 public:
  explicit FakeSparseAccessor(std::vector<SparseVector> rows)
      : rows_(std::move(rows)), gets_(rows_.size()) {}

  RowId size() const override { return static_cast<RowId>(rows_.size()); }
  SparseVectorView Get(RowId id) const override {
    gets_.Record(id);
    return SparseVectorView{rows_[id].indices, rows_[id].values};
  }

  int get_count(RowId id) const { return gets_.count(id); }
  int total_gets() const { return gets_.total(); }

 private:
  std::vector<SparseVector> rows_;
  GetCounter gets_;
};

}  // namespace testing
}  // namespace hector

#endif  // HECTOR_TESTS_INDEX_FAKE_ACCESSORS_H_
