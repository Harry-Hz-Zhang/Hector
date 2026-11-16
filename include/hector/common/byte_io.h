#ifndef HECTOR_COMMON_BYTE_IO_H_
#define HECTOR_COMMON_BYTE_IO_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace hector {

// 字节写出端。所有多字节整数与浮点数按小端序写出；字符串用 u32 长度前缀。
class ByteWriter {
 public:
  virtual ~ByteWriter() = default;

  virtual absl::Status WriteBytes(std::span<const std::byte> data) = 0;

  absl::Status WriteU8(uint8_t value);
  absl::Status WriteU16(uint16_t value);
  absl::Status WriteU32(uint32_t value);
  absl::Status WriteU64(uint64_t value);
  absl::Status WriteF32(float value);
  // 长度超过 u32 上限时返回 InvalidArgument。
  absl::Status WriteString(std::string_view value);
};

// 字节读入端，与 ByteWriter 的编码对应。
class ByteReader {
 public:
  virtual ~ByteReader() = default;

  // 数据不足返回 OutOfRange，且不移动读位置。
  virtual absl::Status ReadBytes(std::span<std::byte> out) = 0;

  // 基本类型读取失败时不移动读位置。
  absl::StatusOr<uint8_t> ReadU8();
  absl::StatusOr<uint16_t> ReadU16();
  absl::StatusOr<uint32_t> ReadU32();
  absl::StatusOr<uint64_t> ReadU64();
  absl::StatusOr<float> ReadF32();
  // 失败后读位置未定义，调用方应放弃该 reader。
  absl::StatusOr<std::string> ReadString();
};

// 写入内存字符串。
class StringByteWriter final : public ByteWriter {
 public:
  absl::Status WriteBytes(std::span<const std::byte> data) override;
  const std::string& data() const;

 private:
  std::string data_;
};

// 从内存读取。不持有数据：调用方保证 data 在 reader 存活期间有效。
class StringByteReader final : public ByteReader {
 public:
  explicit StringByteReader(std::string_view data);

  absl::Status ReadBytes(std::span<std::byte> out) override;
  size_t remaining() const;

 private:
  std::string_view data_;
  size_t pos_ = 0;
};

}  // namespace hector

#endif  // HECTOR_COMMON_BYTE_IO_H_
