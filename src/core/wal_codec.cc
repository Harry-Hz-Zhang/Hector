#include "hector/core/wal_codec.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "hector/common/byte_io.h"
#include "hector/common/status.h"
#include "hector/common/types.h"
#include "hector/core/schema.h"
#include "hector/core/value.h"

namespace hector {
namespace {

constexpr uint8_t kTagInt64 = 0;
constexpr uint8_t kTagFloat = 1;
constexpr uint8_t kTagString = 2;
constexpr uint8_t kTagDense = 3;
constexpr uint8_t kTagSparse = 4;

absl::Status DataLoss(std::string_view what) {
  return absl::DataLossError(absl::StrCat("wal payload: ", what));
}

// 读取失败统一转为 DataLoss。
template <typename T>
absl::StatusOr<T> OrDataLoss(absl::StatusOr<T> value) {
  if (!value.ok()) {
    return DataLoss("truncated");
  }
  return value;
}

absl::Status WriteCount(ByteWriter& out, size_t count) {
  if (count > UINT32_MAX) {
    return absl::InvalidArgumentError(
        absl::StrCat("wal payload: count ", count, " exceeds u32"));
  }
  return out.WriteU32(static_cast<uint32_t>(count));
}

// 读一个元素个数；每个元素至少占 min_bytes 字节，超过剩余字节数时视为损坏，
// 避免按损坏的计数分配内存。
absl::StatusOr<uint32_t> ReadCount(StringByteReader& in, size_t min_bytes) {
  HECTOR_ASSIGN_OR_RETURN(const uint32_t count, OrDataLoss(in.ReadU32()));
  if (static_cast<uint64_t>(count) * min_bytes > in.remaining()) {
    return DataLoss(absl::StrCat("count ", count, " exceeds remaining ",
                                 in.remaining(), " bytes"));
  }
  return count;
}

absl::Status Finish(const StringByteReader& in) {
  if (in.remaining() != 0) {
    return DataLoss(absl::StrCat(in.remaining(), " trailing bytes"));
  }
  return absl::OkStatus();
}

absl::Status WriteValue(ByteWriter& out, const Value& value) {
  if (const auto* i = std::get_if<int64_t>(&value)) {
    HECTOR_RETURN_IF_ERROR(out.WriteU8(kTagInt64));
    return out.WriteU64(static_cast<uint64_t>(*i));
  }
  if (const auto* f = std::get_if<float>(&value)) {
    HECTOR_RETURN_IF_ERROR(out.WriteU8(kTagFloat));
    return out.WriteF32(*f);
  }
  if (const auto* s = std::get_if<std::string>(&value)) {
    HECTOR_RETURN_IF_ERROR(out.WriteU8(kTagString));
    return out.WriteString(*s);
  }
  if (const auto* d = std::get_if<std::vector<float>>(&value)) {
    HECTOR_RETURN_IF_ERROR(out.WriteU8(kTagDense));
    HECTOR_RETURN_IF_ERROR(WriteCount(out, d->size()));
    for (float x : *d) {
      HECTOR_RETURN_IF_ERROR(out.WriteF32(x));
    }
    return absl::OkStatus();
  }
  const auto& sp = std::get<SparseVector>(value);
  if (sp.indices.size() != sp.values.size()) {
    return absl::InvalidArgumentError(
        "wal payload: sparse indices and values differ in length");
  }
  HECTOR_RETURN_IF_ERROR(out.WriteU8(kTagSparse));
  HECTOR_RETURN_IF_ERROR(WriteCount(out, sp.indices.size()));
  for (uint32_t index : sp.indices) {
    HECTOR_RETURN_IF_ERROR(out.WriteU32(index));
  }
  for (float x : sp.values) {
    HECTOR_RETURN_IF_ERROR(out.WriteF32(x));
  }
  return absl::OkStatus();
}

absl::StatusOr<Value> ReadValue(StringByteReader& in) {
  HECTOR_ASSIGN_OR_RETURN(const uint8_t tag, OrDataLoss(in.ReadU8()));
  switch (tag) {
    case kTagInt64: {
      HECTOR_ASSIGN_OR_RETURN(const uint64_t bits, OrDataLoss(in.ReadU64()));
      return Value{static_cast<int64_t>(bits)};
    }
    case kTagFloat: {
      HECTOR_ASSIGN_OR_RETURN(const float f, OrDataLoss(in.ReadF32()));
      return Value{f};
    }
    case kTagString: {
      HECTOR_ASSIGN_OR_RETURN(std::string s, OrDataLoss(in.ReadString()));
      return Value{std::move(s)};
    }
    case kTagDense: {
      HECTOR_ASSIGN_OR_RETURN(const uint32_t n, ReadCount(in, sizeof(float)));
      std::vector<float> v;
      v.reserve(n);
      for (uint32_t i = 0; i < n; ++i) {
        HECTOR_ASSIGN_OR_RETURN(const float f, OrDataLoss(in.ReadF32()));
        v.push_back(f);
      }
      return Value{std::move(v)};
    }
    case kTagSparse: {
      HECTOR_ASSIGN_OR_RETURN(const uint32_t n,
                              ReadCount(in, sizeof(uint32_t) + sizeof(float)));
      SparseVector sp;
      sp.indices.reserve(n);
      sp.values.reserve(n);
      for (uint32_t i = 0; i < n; ++i) {
        HECTOR_ASSIGN_OR_RETURN(const uint32_t index, OrDataLoss(in.ReadU32()));
        sp.indices.push_back(index);
      }
      for (uint32_t i = 0; i < n; ++i) {
        HECTOR_ASSIGN_OR_RETURN(const float f, OrDataLoss(in.ReadF32()));
        sp.values.push_back(f);
      }
      return Value{std::move(sp)};
    }
    default:
      return DataLoss(absl::StrCat("unknown value tag ", tag));
  }
}

absl::StatusOr<std::string> EncodeName(std::string_view name) {
  StringByteWriter out;
  HECTOR_RETURN_IF_ERROR(out.WriteString(name));
  return out.data();
}

absl::StatusOr<DatabaseRecord> DecodeName(std::string_view payload) {
  StringByteReader in(payload);
  DatabaseRecord record;
  HECTOR_ASSIGN_OR_RETURN(record.database, OrDataLoss(in.ReadString()));
  HECTOR_RETURN_IF_ERROR(Finish(in));
  return record;
}

absl::StatusOr<std::string> EncodeRows(std::string_view database,
                                       std::string_view table, bool upsert,
                                       std::span<const RowInput> rows) {
  StringByteWriter out;
  HECTOR_RETURN_IF_ERROR(out.WriteString(database));
  HECTOR_RETURN_IF_ERROR(out.WriteString(table));
  HECTOR_RETURN_IF_ERROR(out.WriteU8(upsert ? 1 : 0));
  HECTOR_RETURN_IF_ERROR(WriteCount(out, rows.size()));
  for (const RowInput& row : rows) {
    HECTOR_RETURN_IF_ERROR(out.WriteString(row.primary_key));
    HECTOR_RETURN_IF_ERROR(WriteCount(out, row.fields.size()));
    for (const auto& [name, value] : row.fields) {
      HECTOR_RETURN_IF_ERROR(out.WriteString(name));
      HECTOR_RETURN_IF_ERROR(WriteValue(out, value));
    }
  }
  return out.data();
}

absl::StatusOr<WriteRowsRecord> DecodeRows(std::string_view payload,
                                           bool expect_upsert) {
  StringByteReader in(payload);
  WriteRowsRecord record;
  HECTOR_ASSIGN_OR_RETURN(record.database, OrDataLoss(in.ReadString()));
  HECTOR_ASSIGN_OR_RETURN(record.table, OrDataLoss(in.ReadString()));
  HECTOR_ASSIGN_OR_RETURN(const uint8_t upsert, OrDataLoss(in.ReadU8()));
  if (upsert != (expect_upsert ? 1 : 0)) {
    return DataLoss(absl::StrCat("upsert flag ", upsert, " does not match ",
                                 expect_upsert ? "upsert" : "insert",
                                 " record"));
  }
  record.upsert = expect_upsert;
  // 每行至少有主键长度与字段数两个 u32。
  HECTOR_ASSIGN_OR_RETURN(const uint32_t row_count,
                          ReadCount(in, 2 * sizeof(uint32_t)));
  record.rows.reserve(row_count);
  for (uint32_t r = 0; r < row_count; ++r) {
    RowInput row;
    HECTOR_ASSIGN_OR_RETURN(row.primary_key, OrDataLoss(in.ReadString()));
    // 每个字段至少有列名长度与类别。
    HECTOR_ASSIGN_OR_RETURN(const uint32_t field_count,
                            ReadCount(in, sizeof(uint32_t) + 1));
    row.fields.reserve(field_count);
    for (uint32_t f = 0; f < field_count; ++f) {
      HECTOR_ASSIGN_OR_RETURN(std::string name, OrDataLoss(in.ReadString()));
      HECTOR_ASSIGN_OR_RETURN(Value value, ReadValue(in));
      if (!row.fields.emplace(name, std::move(value)).second) {
        return DataLoss(
            absl::StrCat("rows[", r, "]: duplicate field \"", name, "\""));
      }
    }
    record.rows.push_back(std::move(row));
  }
  HECTOR_RETURN_IF_ERROR(Finish(in));
  return record;
}

}  // namespace

absl::StatusOr<std::string> EncodeCreateDatabase(std::string_view database) {
  return EncodeName(database);
}

absl::StatusOr<DatabaseRecord> DecodeCreateDatabase(std::string_view payload) {
  return DecodeName(payload);
}

absl::StatusOr<std::string> EncodeDropDatabase(std::string_view database) {
  return EncodeName(database);
}

absl::StatusOr<DatabaseRecord> DecodeDropDatabase(std::string_view payload) {
  return DecodeName(payload);
}

absl::StatusOr<std::string> EncodeCreateTable(std::string_view database,
                                              const Schema& schema) {
  StringByteWriter out;
  HECTOR_RETURN_IF_ERROR(out.WriteString(database));
  HECTOR_RETURN_IF_ERROR(schema.Save(out));
  return out.data();
}

absl::StatusOr<CreateTableRecord> DecodeCreateTable(std::string_view payload) {
  StringByteReader in(payload);
  HECTOR_ASSIGN_OR_RETURN(std::string database, OrDataLoss(in.ReadString()));
  // Schema::Load 对截断与不合法的 Schema 已返回 DataLoss。
  HECTOR_ASSIGN_OR_RETURN(Schema schema, Schema::Load(in));
  HECTOR_RETURN_IF_ERROR(Finish(in));
  return CreateTableRecord{.database = std::move(database),
                           .schema = std::move(schema)};
}

absl::StatusOr<std::string> EncodeDropTable(std::string_view database,
                                            std::string_view table) {
  StringByteWriter out;
  HECTOR_RETURN_IF_ERROR(out.WriteString(database));
  HECTOR_RETURN_IF_ERROR(out.WriteString(table));
  return out.data();
}

absl::StatusOr<DropTableRecord> DecodeDropTable(std::string_view payload) {
  StringByteReader in(payload);
  DropTableRecord record;
  HECTOR_ASSIGN_OR_RETURN(record.database, OrDataLoss(in.ReadString()));
  HECTOR_ASSIGN_OR_RETURN(record.table, OrDataLoss(in.ReadString()));
  HECTOR_RETURN_IF_ERROR(Finish(in));
  return record;
}

absl::StatusOr<std::string> EncodeInsert(std::string_view database,
                                         std::string_view table,
                                         std::span<const RowInput> rows) {
  return EncodeRows(database, table, /*upsert=*/false, rows);
}

absl::StatusOr<WriteRowsRecord> DecodeInsert(std::string_view payload) {
  return DecodeRows(payload, /*expect_upsert=*/false);
}

absl::StatusOr<std::string> EncodeUpsert(std::string_view database,
                                         std::string_view table,
                                         std::span<const RowInput> rows) {
  return EncodeRows(database, table, /*upsert=*/true, rows);
}

absl::StatusOr<WriteRowsRecord> DecodeUpsert(std::string_view payload) {
  return DecodeRows(payload, /*expect_upsert=*/true);
}

absl::StatusOr<std::string> EncodeDelete(
    std::string_view database, std::string_view table,
    std::span<const std::string> primary_keys) {
  StringByteWriter out;
  HECTOR_RETURN_IF_ERROR(out.WriteString(database));
  HECTOR_RETURN_IF_ERROR(out.WriteString(table));
  HECTOR_RETURN_IF_ERROR(WriteCount(out, primary_keys.size()));
  for (const std::string& pk : primary_keys) {
    HECTOR_RETURN_IF_ERROR(out.WriteString(pk));
  }
  return out.data();
}

absl::StatusOr<DeleteRecord> DecodeDelete(std::string_view payload) {
  StringByteReader in(payload);
  DeleteRecord record;
  HECTOR_ASSIGN_OR_RETURN(record.database, OrDataLoss(in.ReadString()));
  HECTOR_ASSIGN_OR_RETURN(record.table, OrDataLoss(in.ReadString()));
  HECTOR_ASSIGN_OR_RETURN(const uint32_t count,
                          ReadCount(in, sizeof(uint32_t)));
  record.primary_keys.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    HECTOR_ASSIGN_OR_RETURN(std::string pk, OrDataLoss(in.ReadString()));
    record.primary_keys.push_back(std::move(pk));
  }
  HECTOR_RETURN_IF_ERROR(Finish(in));
  return record;
}

}  // namespace hector
