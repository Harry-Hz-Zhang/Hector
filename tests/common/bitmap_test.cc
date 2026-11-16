#include "hector/common/bitmap.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "hector/common/byte_io.h"

namespace hector {
namespace {

TEST(BitmapTest, SetClearCountAndOutOfRange) {
  Bitmap bitmap(130);
  bitmap.Set(0);
  bitmap.Set(64);
  bitmap.Set(129);
  bitmap.Clear(64);
  bitmap.Set(500);
  EXPECT_EQ(bitmap.size(), 130U);
  EXPECT_EQ(bitmap.Count(), 2U);
  EXPECT_TRUE(bitmap.Test(0));
  EXPECT_FALSE(bitmap.Test(64));
  EXPECT_TRUE(bitmap.Test(129));
  EXPECT_FALSE(bitmap.Test(200));
  EXPECT_FALSE(bitmap.Test(500));

  bitmap.Resize(300);
  EXPECT_EQ(bitmap.size(), 300U);
  EXPECT_EQ(bitmap.Count(), 2U);
  for (size_t i = 130; i < 300; ++i) {
    EXPECT_FALSE(bitmap.Test(i)) << i;
  }

  // 先缩小再放大：被截掉的位不得「复活」。
  bitmap.Resize(100);
  EXPECT_EQ(bitmap.Count(), 1U);
  bitmap.Resize(300);
  EXPECT_FALSE(bitmap.Test(129));
  EXPECT_EQ(bitmap.Count(), 1U);
}

TEST(BitmapTest, ForEachSetAndSaveLoad) {
  Bitmap bitmap(200);
  bitmap.Set(129);
  bitmap.Set(3);
  bitmap.Set(70);

  std::vector<size_t> visited;
  bitmap.ForEachSet([&visited](size_t i) { visited.push_back(i); });
  EXPECT_EQ(visited, (std::vector<size_t>{3, 70, 129}));

  StringByteWriter writer;
  ASSERT_TRUE(bitmap.Save(writer).ok());
  StringByteReader reader(writer.data());
  Bitmap loaded;
  ASSERT_TRUE(loaded.Load(reader).ok());
  EXPECT_EQ(reader.remaining(), 0U);
  EXPECT_EQ(loaded.size(), bitmap.size());
  EXPECT_EQ(loaded.Count(), bitmap.Count());
  for (size_t i = 0; i < bitmap.size(); ++i) {
    EXPECT_EQ(loaded.Test(i), bitmap.Test(i)) << i;
  }
}

}  // namespace
}  // namespace hector
