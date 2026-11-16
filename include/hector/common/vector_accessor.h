#ifndef HECTOR_COMMON_VECTOR_ACCESSOR_H_
#define HECTOR_COMMON_VECTOR_ACCESSOR_H_

#include <cstdint>
#include <span>

#include "hector/common/types.h"

namespace hector {

// 原始稠密向量只存一份，索引通过它读取。
class DenseVectorAccessor {
 public:
  virtual ~DenseVectorAccessor() = default;

  virtual uint32_t dim() const = 0;
  // 写者已写入的行数（可能大于已发布行数）；实现用 atomic acquire
  // 读取，任何线程可调用。 读者判断可见性必须用 RowFilter::row_count()，不得用
  // size()。
  virtual RowId size() const = 0;
  // 前置条件 id < size()；视图在表存活期间永远有效、内容不变。
  virtual std::span<const float> Get(RowId id) const = 0;
};

// 原始稀疏向量只存一份，索引通过它读取。
class SparseVectorAccessor {
 public:
  virtual ~SparseVectorAccessor() = default;

  // 线程语义同 DenseVectorAccessor::size()：atomic acquire，可见性以
  // RowFilter::row_count() 为准。
  virtual RowId size() const = 0;
  // 前置条件与视图有效期同 DenseVectorAccessor::Get()。
  virtual SparseVectorView Get(RowId id) const = 0;
};

}  // namespace hector

#endif  // HECTOR_COMMON_VECTOR_ACCESSOR_H_
