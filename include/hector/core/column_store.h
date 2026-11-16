#ifndef HECTOR_CORE_COLUMN_STORE_H_
#define HECTOR_CORE_COLUMN_STORE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "absl/status/status.h"
#include "hector/common/byte_io.h"
#include "hector/common/types.h"
#include "hector/common/vector_accessor.h"
#include "hector/core/schema.h"
#include "hector/core/value.h"

namespace hector {

// 列存储按块只追加：每块 1024 行；块目录按 ceil(capacity / 1024)
// 个槽位在构造时一次分配，之后不再移动，因此已写入行的地址永远不变。
// Table 以 TableOptions::max_rows 作为各列与墓碑位图的 capacity。
// 只有一个写者（Append、Load）；读者只读取 row id < 已发布行数 的数据，
// 因此不需要锁。size() 与 MemoryBytes() 任何线程都可以调用。
//
// 各列的 Save 编码都以 u32 版本=1 开头；Load 只能在空列上、发布前由写者调用，
// 数据截断、版本或形状不符、行数超过 capacity 时返回 DataLoss，列保持为空。

class DenseColumn final : public DenseVectorAccessor {
 public:
  // capacity 供测试使用较小容量。
  explicit DenseColumn(uint32_t dim, RowId capacity = kMaxRowsPerTable);
  uint32_t dim() const override;
  RowId size() const override;
  std::span<const float> Get(RowId id) const override;

  // 长度不等于 dim 返回 InvalidArgument；已满时返回 ResourceExhausted
  // （正常路径由 PrepareInsert 提前拦截）。失败时列不变。
  absl::Status Append(std::span<const float> v);
  // 已分配的块与块目录的字节数。
  size_t MemoryBytes() const;
  // 编码：u32 版本、u32 dim、u32 行数，随后逐行 dim 个 f32。
  absl::Status Save(ByteWriter& out) const;
  absl::Status Load(ByteReader& in);

 private:
  uint32_t dim_;
  RowId capacity_;
  size_t chunk_slots_;
  std::unique_ptr<std::unique_ptr<float[]>[]> chunks_;
  std::atomic<size_t> allocated_chunks_{0};
  std::atomic<RowId> size_{0};
};

class SparseColumn final : public SparseVectorAccessor {
 public:
  explicit SparseColumn(RowId capacity = kMaxRowsPerTable);
  RowId size() const override;
  SparseVectorView Get(RowId id) const override;

  // indices 与 values 不等长返回 InvalidArgument；已满时返回
  // ResourceExhausted。排序与范围由 Table 校验。失败时列不变。
  absl::Status Append(SparseVector v);
  // 已分配的块、块目录与各行 indices/values 的字节数。
  size_t MemoryBytes() const;
  // 编码：u32 版本、u32 行数，随后每行 u32 非零个数、各 u32 位置、各 f32 值。
  absl::Status Save(ByteWriter& out) const;
  absl::Status Load(ByteReader& in);

 private:
  RowId capacity_;
  size_t chunk_slots_;
  std::unique_ptr<std::unique_ptr<SparseVector[]>[]> chunks_;
  std::atomic<size_t> allocated_chunks_{0};
  std::atomic<size_t> payload_bytes_{0};
  std::atomic<RowId> size_{0};
};

class ScalarColumn {
 public:
  explicit ScalarColumn(ScalarType t, RowId capacity = kMaxRowsPerTable);
  // 值的类型与列类型不符返回 InvalidArgument；已满时返回
  // ResourceExhausted。失败时列不变。
  absl::Status Append(ScalarValue v);
  // 前置条件 id < size()；引用在列存活期间有效、内容不变。
  const ScalarValue& Get(RowId id) const;
  RowId size() const;

  // 已分配的块、块目录与 string 内容的字节数。
  size_t MemoryBytes() const;
  // 编码：u32 版本、u8 类型、u32 行数，随后逐行的值（int64 按 u64、
  // float 按 f32、string 按带长度前缀的字节串）。
  absl::Status Save(ByteWriter& out) const;
  absl::Status Load(ByteReader& in);

 private:
  ScalarType type_;
  RowId capacity_;
  size_t chunk_slots_;
  std::unique_ptr<std::unique_ptr<ScalarValue[]>[]> chunks_;
  std::atomic<size_t> allocated_chunks_{0};
  std::atomic<size_t> payload_bytes_{0};
  std::atomic<RowId> size_{0};
};

}  // namespace hector

#endif  // HECTOR_CORE_COLUMN_STORE_H_
