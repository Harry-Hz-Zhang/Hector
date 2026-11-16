#ifndef HECTOR_COMMON_CONCURRENT_BITMAP_H_
#define HECTOR_COMMON_CONCURRENT_BITMAP_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "hector/common/byte_io.h"

namespace hector {

// 表级墓碑位图：单写者、多读者，读写都不加锁。
// 内部按 65536 位一块懒分配；块目录在构造时一次分配好，之后不再移动。
// 位只会从 0 变为 1（没有 Clear），因此读者看到的每一位与 Count() 都单调不减。
class ConcurrentBitmap {
 public:
  // 容量固定。
  explicit ConcurrentBitmap(size_t capacity);
  ~ConcurrentBitmap();

  ConcurrentBitmap(const ConcurrentBitmap&) = delete;
  ConcurrentBitmap& operator=(const ConcurrentBitmap&) = delete;

  // 仅写者调用；store(release)；i >= capacity 时不做任何事。
  void Set(size_t i);
  // 任意线程；load(acquire)；未分配的块视为全 0。
  bool Test(size_t i) const;
  // 写者维护的计数，load(acquire)。
  size_t Count() const;
  size_t capacity() const;

  // 编码：u64 capacity、u32 已分配块数，随后每块 u32 块号 + 1024 个 u64 字。
  absl::Status Save(ByteWriter& out) const;
  // 只能在发布前、由写者调用；要求当前位图为空且容量与数据一致。
  // 失败时位图不变。
  absl::Status Load(ByteReader& in);

 private:
  static constexpr size_t kBitsPerChunk = 65536;
  static constexpr size_t kWordsPerChunk = kBitsPerChunk / 64;
  using Chunk = std::array<std::atomic<uint64_t>, kWordsPerChunk>;

  size_t capacity_;
  size_t chunk_count_;
  // 块目录：构造时按 chunk_count_ 一次分配，之后不再重新分配。
  std::unique_ptr<std::atomic<Chunk*>[]> chunks_;
  std::atomic<size_t> count_{0};
};

}  // namespace hector

#endif  // HECTOR_COMMON_CONCURRENT_BITMAP_H_
