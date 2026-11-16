#ifndef HECTOR_COMMON_BITMAP_H_
#define HECTOR_COMMON_BITMAP_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "hector/common/byte_io.h"

namespace hector {

// 单次查询内使用的普通位图（例如 Filter 求出的 allow 集合），非线程安全。
class Bitmap {
 public:
  explicit Bitmap(size_t size = 0);

  // 新增位为 0；缩小时截掉的位被丢弃。
  void Resize(size_t size);
  // i >= size() 时不做任何事。
  void Set(size_t i);
  void Clear(size_t i);
  // i >= size() 时返回 false。
  bool Test(size_t i) const;
  size_t size() const;
  size_t Count() const;
  // 升序遍历置位下标。
  void ForEachSet(const std::function<void(size_t)>& fn) const;

  // 编码：u64 size，随后 ceil(size/64) 个 u64 字。
  absl::Status Save(ByteWriter& out) const;
  // 成功时整体替换当前内容；失败时当前内容不变。
  absl::Status Load(ByteReader& in);

 private:
  std::vector<uint64_t> words_;
  size_t size_ = 0;
};

}  // namespace hector

#endif  // HECTOR_COMMON_BITMAP_H_
