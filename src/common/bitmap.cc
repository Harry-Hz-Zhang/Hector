#include "hector/common/bitmap.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/status.h"

namespace hector {
namespace {

constexpr size_t kBitsPerWord = 64;

// 向上取整且不溢出：损坏数据里的超大 size 不会回绕成很小的字数。
size_t WordsFor(size_t bits) {
  return bits / kBitsPerWord + (bits % kBitsPerWord != 0 ? 1 : 0);
}

// 最后一个字中属于位图的位掩码；bits 是 64 的整数倍时为全 1。
uint64_t TailMask(size_t bits) {
  const size_t used = bits % kBitsPerWord;
  return used == 0 ? ~uint64_t{0} : (uint64_t{1} << used) - 1;
}

}  // namespace

Bitmap::Bitmap(size_t size) : words_(WordsFor(size), 0), size_(size) {}

void Bitmap::Resize(size_t size) {
  words_.resize(WordsFor(size), 0);
  size_ = size;
  // 缩小后清掉最后一个字中越界的位，保证之后放大时新位为 0。
  if (!words_.empty()) {
    words_.back() &= TailMask(size_);
  }
}

void Bitmap::Set(size_t i) {
  if (i >= size_) {
    return;
  }
  words_[i / kBitsPerWord] |= uint64_t{1} << (i % kBitsPerWord);
}

void Bitmap::Clear(size_t i) {
  if (i >= size_) {
    return;
  }
  words_[i / kBitsPerWord] &= ~(uint64_t{1} << (i % kBitsPerWord));
}

bool Bitmap::Test(size_t i) const {
  if (i >= size_) {
    return false;
  }
  return ((words_[i / kBitsPerWord] >> (i % kBitsPerWord)) & 1U) != 0;
}

size_t Bitmap::size() const { return size_; }

size_t Bitmap::Count() const {
  size_t count = 0;
  for (const uint64_t word : words_) {
    count += static_cast<size_t>(std::popcount(word));
  }
  return count;
}

void Bitmap::ForEachSet(const std::function<void(size_t)>& fn) const {
  for (size_t w = 0; w < words_.size(); ++w) {
    uint64_t word = words_[w];
    while (word != 0) {
      const auto bit = static_cast<size_t>(std::countr_zero(word));
      fn(w * kBitsPerWord + bit);
      word &= word - 1;
    }
  }
}

absl::Status Bitmap::Save(ByteWriter& out) const {
  HECTOR_RETURN_IF_ERROR(out.WriteU64(size_));
  for (const uint64_t word : words_) {
    HECTOR_RETURN_IF_ERROR(out.WriteU64(word));
  }
  return absl::OkStatus();
}

absl::Status Bitmap::Load(ByteReader& in) {
  HECTOR_ASSIGN_OR_RETURN(const uint64_t size, in.ReadU64());
  // 逐字读入：损坏的 size 只会在数据耗尽时失败，不会预先申请巨大内存。
  std::vector<uint64_t> words;
  const size_t word_count = WordsFor(size);
  for (size_t w = 0; w < word_count; ++w) {
    HECTOR_ASSIGN_OR_RETURN(const uint64_t word, in.ReadU64());
    words.push_back(word);
  }
  if (!words.empty() && (words.back() & ~TailMask(size)) != 0) {
    return absl::DataLossError("bitmap has bits set beyond its size");
  }
  words_ = std::move(words);
  size_ = size;
  return absl::OkStatus();
}

}  // namespace hector
