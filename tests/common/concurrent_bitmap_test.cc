#include "hector/common/concurrent_bitmap.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <random>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "hector/common/byte_io.h"

namespace hector {
namespace {

TEST(ConcurrentBitmapTest, LazyChunksAndCapacity) {
  ConcurrentBitmap bitmap(200000);
  EXPECT_EQ(bitmap.capacity(), 200000U);
  EXPECT_EQ(bitmap.Count(), 0U);
  EXPECT_FALSE(bitmap.Test(150000));  // 块尚未分配，视为全 0

  bitmap.Set(0);
  bitmap.Set(150000);
  bitmap.Set(200000);  // 越界，不做任何事
  EXPECT_TRUE(bitmap.Test(0));
  EXPECT_TRUE(bitmap.Test(150000));
  EXPECT_FALSE(bitmap.Test(70000));
  EXPECT_FALSE(bitmap.Test(200000));
  EXPECT_EQ(bitmap.Count(), 2U);

  bitmap.Set(150000);  // 重复置位不改变计数
  EXPECT_EQ(bitmap.Count(), 2U);

  // 块目录不移动：继续跨越其余所有块置位后，此前置位的仍为 true，计数不回退。
  const size_t count_before = bitmap.Count();
  bitmap.Set(65536);
  bitmap.Set(131071);
  bitmap.Set(199999);
  EXPECT_TRUE(bitmap.Test(0));
  EXPECT_TRUE(bitmap.Test(150000));
  EXPECT_TRUE(bitmap.Test(65536));
  EXPECT_TRUE(bitmap.Test(131071));
  EXPECT_TRUE(bitmap.Test(199999));
  EXPECT_GE(bitmap.Count(), count_before);
  EXPECT_EQ(bitmap.Count(), 5U);
}

TEST(ConcurrentBitmapTest, SaveLoadRoundTrip) {
  ConcurrentBitmap bitmap(200000);
  bitmap.Set(1);
  bitmap.Set(150000);
  bitmap.Set(199999);

  StringByteWriter writer;
  ASSERT_TRUE(bitmap.Save(writer).ok());

  StringByteReader reader(writer.data());
  ConcurrentBitmap loaded(200000);
  ASSERT_TRUE(loaded.Load(reader).ok());
  EXPECT_EQ(reader.remaining(), 0U);
  EXPECT_EQ(loaded.Count(), 3U);
  EXPECT_TRUE(loaded.Test(1));
  EXPECT_TRUE(loaded.Test(150000));
  EXPECT_TRUE(loaded.Test(199999));
  EXPECT_FALSE(loaded.Test(70000));

  // 容量不一致的数据不能加载。
  StringByteReader mismatch_reader(writer.data());
  ConcurrentBitmap other(100);
  EXPECT_FALSE(other.Load(mismatch_reader).ok());
  EXPECT_EQ(other.Count(), 0U);

  // 截断的数据返回错误，且不改变位图。
  const std::string truncated =
      writer.data().substr(0, writer.data().size() / 2);
  StringByteReader truncated_reader(truncated);
  ConcurrentBitmap partial(200000);
  EXPECT_FALSE(partial.Load(truncated_reader).ok());
  EXPECT_EQ(partial.Count(), 0U);
  EXPECT_FALSE(partial.Test(1));
}

TEST(ConcurrentBitmapTest, ConcurrentReadersDuringWrites) {
  constexpr size_t kBits = 100000;
  constexpr int kReaders = 4;
  ConcurrentBitmap bitmap(kBits);
  std::atomic<bool> writer_done{false};
  std::atomic<int> regressions{0};

  std::vector<std::thread> readers;
  readers.reserve(kReaders);
  for (int r = 0; r < kReaders; ++r) {
    readers.emplace_back([&bitmap, &writer_done, &regressions, r] {
      std::mt19937 rng(static_cast<std::mt19937::result_type>(r + 1));
      std::uniform_int_distribution<size_t> pick(0, kBits - 1);
      std::vector<bool> seen(kBits, false);
      size_t last_count = 0;
      // 写者结束后再多读一轮，保证读者至少运行过一段时间。
      bool final_round = false;
      while (!final_round) {
        final_round = writer_done.load(std::memory_order_acquire);
        for (int k = 0; k < 1000; ++k) {
          const size_t i = pick(rng);
          const bool now = bitmap.Test(i);
          if (seen[i] && !now) {
            regressions.fetch_add(1, std::memory_order_relaxed);
          }
          if (now) {
            seen[i] = true;
          }
          const size_t count = bitmap.Count();
          if (count < last_count) {
            regressions.fetch_add(1, std::memory_order_relaxed);
          }
          last_count = count;
        }
      }
    });
  }

  for (size_t i = 0; i < kBits; ++i) {
    bitmap.Set(i);
  }
  writer_done.store(true, std::memory_order_release);
  for (std::thread& t : readers) {
    t.join();
  }

  EXPECT_EQ(regressions.load(), 0);
  EXPECT_EQ(bitmap.Count(), kBits);
  for (size_t i = 0; i < kBits; ++i) {
    ASSERT_TRUE(bitmap.Test(i)) << i;
  }
}

}  // namespace
}  // namespace hector
