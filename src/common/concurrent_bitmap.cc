#include "hector/common/concurrent_bitmap.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/status.h"

namespace hector {
namespace {

constexpr size_t kBitsPerWord = 64;

}  // namespace

ConcurrentBitmap::ConcurrentBitmap(size_t capacity)
    : capacity_(capacity),
      chunk_count_(capacity / kBitsPerChunk +
                   (capacity % kBitsPerChunk != 0 ? 1 : 0)),
      // 块目录唯一一次分配；值初始化为 nullptr，之后只写入元素、不再重新分配。
      chunks_(std::make_unique<std::atomic<Chunk*>[]>(chunk_count_)) {}

ConcurrentBitmap::~ConcurrentBitmap() {
  for (size_t c = 0; c < chunk_count_; ++c) {
    delete chunks_[c].load(std::memory_order_relaxed);
  }
}

void ConcurrentBitmap::Set(size_t i) {
  if (i >= capacity_) {
    return;
  }
  std::atomic<Chunk*>& slot = chunks_[i / kBitsPerChunk];
  // 只有写者修改块指针，因此写者自己读可以用 relaxed。
  Chunk* chunk = slot.load(std::memory_order_relaxed);
  if (chunk == nullptr) {
    // std::atomic 在 C++20 中默认值初始化为 0。
    chunk = new Chunk();
    slot.store(chunk, std::memory_order_release);
  }
  const size_t bit = i % kBitsPerChunk;
  std::atomic<uint64_t>& word = (*chunk)[bit / kBitsPerWord];
  const uint64_t mask = uint64_t{1} << (bit % kBitsPerWord);
  const uint64_t old = word.load(std::memory_order_relaxed);
  if ((old & mask) != 0) {
    return;
  }
  word.store(old | mask, std::memory_order_release);
  count_.store(count_.load(std::memory_order_relaxed) + 1,
               std::memory_order_release);
}

bool ConcurrentBitmap::Test(size_t i) const {
  if (i >= capacity_) {
    return false;
  }
  const Chunk* chunk =
      chunks_[i / kBitsPerChunk].load(std::memory_order_acquire);
  if (chunk == nullptr) {
    return false;
  }
  const size_t bit = i % kBitsPerChunk;
  const uint64_t word =
      (*chunk)[bit / kBitsPerWord].load(std::memory_order_acquire);
  return ((word >> (bit % kBitsPerWord)) & 1U) != 0;
}

size_t ConcurrentBitmap::Count() const {
  return count_.load(std::memory_order_acquire);
}

size_t ConcurrentBitmap::capacity() const { return capacity_; }

absl::Status ConcurrentBitmap::Save(ByteWriter& out) const {
  std::vector<size_t> allocated;
  for (size_t c = 0; c < chunk_count_; ++c) {
    if (chunks_[c].load(std::memory_order_acquire) != nullptr) {
      allocated.push_back(c);
    }
  }
  HECTOR_RETURN_IF_ERROR(out.WriteU64(capacity_));
  HECTOR_RETURN_IF_ERROR(out.WriteU32(static_cast<uint32_t>(allocated.size())));
  for (const size_t c : allocated) {
    const Chunk* chunk = chunks_[c].load(std::memory_order_acquire);
    HECTOR_RETURN_IF_ERROR(out.WriteU32(static_cast<uint32_t>(c)));
    for (const std::atomic<uint64_t>& word : *chunk) {
      HECTOR_RETURN_IF_ERROR(
          out.WriteU64(word.load(std::memory_order_acquire)));
    }
  }
  return absl::OkStatus();
}

absl::Status ConcurrentBitmap::Load(ByteReader& in) {
  if (count_.load(std::memory_order_relaxed) != 0) {
    return absl::FailedPreconditionError(
        "ConcurrentBitmap::Load requires an empty bitmap");
  }
  for (size_t c = 0; c < chunk_count_; ++c) {
    if (chunks_[c].load(std::memory_order_relaxed) != nullptr) {
      return absl::FailedPreconditionError(
          "ConcurrentBitmap::Load requires an empty bitmap");
    }
  }

  HECTOR_ASSIGN_OR_RETURN(const uint64_t capacity, in.ReadU64());
  if (capacity != capacity_) {
    return absl::DataLossError("tombstone capacity " +
                               std::to_string(capacity) + " does not match " +
                               std::to_string(capacity_));
  }
  HECTOR_ASSIGN_OR_RETURN(const uint32_t present, in.ReadU32());
  if (present > chunk_count_) {
    return absl::DataLossError("tombstone chunk count " +
                               std::to_string(present) + " exceeds " +
                               std::to_string(chunk_count_));
  }

  // 先完整读入并校验，成功后再安装，失败时位图保持为空。
  using Words = std::array<uint64_t, kWordsPerChunk>;
  std::vector<std::pair<size_t, std::unique_ptr<Words>>> parsed;
  parsed.reserve(present);
  size_t count = 0;
  for (uint32_t k = 0; k < present; ++k) {
    HECTOR_ASSIGN_OR_RETURN(const uint32_t index, in.ReadU32());
    if (index >= chunk_count_ ||
        (!parsed.empty() && index <= parsed.back().first)) {
      return absl::DataLossError("tombstone chunk index " +
                                 std::to_string(index) + " is out of order");
    }
    auto words = std::make_unique<Words>();
    for (uint64_t& word : *words) {
      HECTOR_ASSIGN_OR_RETURN(word, in.ReadU64());
      count += static_cast<size_t>(std::popcount(word));
    }
    parsed.emplace_back(index, std::move(words));
  }

  // 校验最后一块中容量之外的位为 0。
  if (!parsed.empty() && parsed.back().first == chunk_count_ - 1) {
    const Words& last = *parsed.back().second;
    const size_t valid_bits = capacity_ - (chunk_count_ - 1) * kBitsPerChunk;
    for (size_t bit = valid_bits; bit < kBitsPerChunk; ++bit) {
      if (((last[bit / kBitsPerWord] >> (bit % kBitsPerWord)) & 1U) != 0) {
        return absl::DataLossError("tombstone has bits set beyond capacity");
      }
    }
  }

  for (const auto& [index, words] : parsed) {
    auto chunk = std::make_unique<Chunk>();
    for (size_t w = 0; w < kWordsPerChunk; ++w) {
      (*chunk)[w].store((*words)[w], std::memory_order_relaxed);
    }
    chunks_[index].store(chunk.release(), std::memory_order_release);
  }
  count_.store(count, std::memory_order_release);
  return absl::OkStatus();
}

}  // namespace hector
