#include "hector/common/byte_io.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "absl/status/status.h"

namespace hector {
namespace {

TEST(ByteIoTest, RoundTripAllPrimitives) {
  StringByteWriter writer;
  ASSERT_TRUE(writer.WriteU8(0xAB).ok());
  ASSERT_TRUE(writer.WriteU16(0xBEEF).ok());
  ASSERT_TRUE(writer.WriteU32(0xDEADBEEF).ok());
  ASSERT_TRUE(writer.WriteU64(0x0123456789ABCDEFULL).ok());
  ASSERT_TRUE(writer.WriteF32(-1.5F).ok());
  ASSERT_TRUE(writer
                  .WriteString("h\xC3\xA9"
                               "ctor")
                  .ok());

  StringByteReader reader(writer.data());
  absl::StatusOr<uint8_t> u8 = reader.ReadU8();
  ASSERT_TRUE(u8.ok());
  EXPECT_EQ(*u8, 0xAB);
  absl::StatusOr<uint16_t> u16 = reader.ReadU16();
  ASSERT_TRUE(u16.ok());
  EXPECT_EQ(*u16, 0xBEEF);
  absl::StatusOr<uint32_t> u32 = reader.ReadU32();
  ASSERT_TRUE(u32.ok());
  EXPECT_EQ(*u32, 0xDEADBEEFU);
  absl::StatusOr<uint64_t> u64 = reader.ReadU64();
  ASSERT_TRUE(u64.ok());
  EXPECT_EQ(*u64, 0x0123456789ABCDEFULL);
  absl::StatusOr<float> f32 = reader.ReadF32();
  ASSERT_TRUE(f32.ok());
  EXPECT_EQ(*f32, -1.5F);
  absl::StatusOr<std::string> str = reader.ReadString();
  ASSERT_TRUE(str.ok());
  EXPECT_EQ(*str,
            "h\xC3\xA9"
            "ctor");
  EXPECT_EQ(reader.remaining(), 0U);
}

TEST(ByteIoTest, LittleEndianLayout) {
  StringByteWriter writer;
  ASSERT_TRUE(writer.WriteU32(0x01020304).ok());
  EXPECT_EQ(writer.data(), std::string("\x04\x03\x02\x01", 4));
}

TEST(ByteIoTest, ReadPastEndReturnsOutOfRange) {
  StringByteWriter writer;
  ASSERT_TRUE(writer.WriteU16(0x0102).ok());
  ASSERT_TRUE(writer.WriteU8(0x03).ok());
  StringByteReader reader(writer.data());
  absl::StatusOr<uint32_t> u32 = reader.ReadU32();
  EXPECT_EQ(u32.status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(reader.remaining(), 3U);
  // 读位置未移动：剩余的 3 字节仍可按原样读出。
  absl::StatusOr<uint16_t> u16 = reader.ReadU16();
  ASSERT_TRUE(u16.ok());
  EXPECT_EQ(*u16, 0x0102);

  StringByteWriter short_string;
  ASSERT_TRUE(short_string.WriteU32(100).ok());
  ASSERT_TRUE(short_string.WriteU8('a').ok());
  ASSERT_TRUE(short_string.WriteU8('b').ok());
  StringByteReader string_reader(short_string.data());
  EXPECT_EQ(string_reader.ReadString().status().code(),
            absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace hector
