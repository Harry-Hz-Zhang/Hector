#include "hector/common/byte_io.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace hector {
namespace {

// 字符串按块读入，避免损坏的长度前缀一次性申请巨大内存。
constexpr size_t kStringReadChunk = size_t{64} * 1024;

template <typename T>
absl::Status WriteLittleEndian(ByteWriter& writer, T value) {
  std::array<std::byte, sizeof(T)> bytes;
  for (size_t i = 0; i < sizeof(T); ++i) {
    bytes[i] = static_cast<std::byte>(
        static_cast<uint8_t>(static_cast<uint64_t>(value) >> (8 * i)));
  }
  return writer.WriteBytes(bytes);
}

template <typename T>
absl::StatusOr<T> ReadLittleEndian(ByteReader& reader) {
  std::array<std::byte, sizeof(T)> bytes;
  if (absl::Status status = reader.ReadBytes(bytes); !status.ok()) {
    return status;
  }
  uint64_t value = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
  }
  return static_cast<T>(value);
}

}  // namespace

absl::Status ByteWriter::WriteU8(uint8_t value) {
  return WriteLittleEndian(*this, value);
}

absl::Status ByteWriter::WriteU16(uint16_t value) {
  return WriteLittleEndian(*this, value);
}

absl::Status ByteWriter::WriteU32(uint32_t value) {
  return WriteLittleEndian(*this, value);
}

absl::Status ByteWriter::WriteU64(uint64_t value) {
  return WriteLittleEndian(*this, value);
}

absl::Status ByteWriter::WriteF32(float value) {
  return WriteLittleEndian(*this, std::bit_cast<uint32_t>(value));
}

absl::Status ByteWriter::WriteString(std::string_view value) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError("string of " +
                                      std::to_string(value.size()) +
                                      " bytes exceeds the u32 length prefix");
  }
  if (absl::Status status = WriteU32(static_cast<uint32_t>(value.size()));
      !status.ok()) {
    return status;
  }
  return WriteBytes(std::as_bytes(std::span(value.data(), value.size())));
}

absl::StatusOr<uint8_t> ByteReader::ReadU8() {
  return ReadLittleEndian<uint8_t>(*this);
}

absl::StatusOr<uint16_t> ByteReader::ReadU16() {
  return ReadLittleEndian<uint16_t>(*this);
}

absl::StatusOr<uint32_t> ByteReader::ReadU32() {
  return ReadLittleEndian<uint32_t>(*this);
}

absl::StatusOr<uint64_t> ByteReader::ReadU64() {
  return ReadLittleEndian<uint64_t>(*this);
}

absl::StatusOr<float> ByteReader::ReadF32() {
  absl::StatusOr<uint32_t> bits = ReadU32();
  if (!bits.ok()) {
    return bits.status();
  }
  return std::bit_cast<float>(*bits);
}

absl::StatusOr<std::string> ByteReader::ReadString() {
  absl::StatusOr<uint32_t> length = ReadU32();
  if (!length.ok()) {
    return length.status();
  }
  std::string result;
  size_t left = *length;
  while (left > 0) {
    const size_t chunk = std::min(left, kStringReadChunk);
    const size_t offset = result.size();
    result.resize(offset + chunk);
    if (absl::Status status = ReadBytes(
            std::as_writable_bytes(std::span(result.data() + offset, chunk)));
        !status.ok()) {
      return status;
    }
    left -= chunk;
  }
  return result;
}

absl::Status StringByteWriter::WriteBytes(std::span<const std::byte> data) {
  data_.append(reinterpret_cast<const char*>(data.data()), data.size());
  return absl::OkStatus();
}

const std::string& StringByteWriter::data() const { return data_; }

StringByteReader::StringByteReader(std::string_view data) : data_(data) {}

absl::Status StringByteReader::ReadBytes(std::span<std::byte> out) {
  if (out.size() > remaining()) {
    return absl::OutOfRangeError("need " + std::to_string(out.size()) +
                                 " bytes, only " + std::to_string(remaining()) +
                                 " remaining");
  }
  if (!out.empty()) {
    std::memcpy(out.data(), data_.data() + pos_, out.size());
  }
  pos_ += out.size();
  return absl::OkStatus();
}

size_t StringByteReader::remaining() const { return data_.size() - pos_; }

}  // namespace hector
