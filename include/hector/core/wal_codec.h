#ifndef HECTOR_CORE_WAL_CODEC_H_
#define HECTOR_CORE_WAL_CODEC_H_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "hector/core/schema.h"
#include "hector/core/value.h"

namespace hector {

// Engine 写入 WAL 的记录类型（WalRecord::type）。
// NOLINTNEXTLINE(performance-enum-size): 与 WalRecord::type 的 u16 一致
enum class WalRecordType : uint16_t {
  kCreateDatabase = 1,
  kDropDatabase = 2,
  kCreateTable = 3,
  kDropTable = 4,
  kInsert = 5,
  kUpsert = 6,
  kDelete = 7,
};

// 类型 1、2 的内容。
struct DatabaseRecord {
  std::string database;
};

// 类型 3 的内容。
struct CreateTableRecord {
  std::string database;
  Schema schema;
};

// 类型 4 的内容。
struct DropTableRecord {
  std::string database;
  std::string table;
};

// 类型 5、6 的内容：PreparedBatch.rows（归一化与稀疏排序之后的行）与 upsert。
struct WriteRowsRecord {
  std::string database;
  std::string table;
  bool upsert = false;
  std::vector<RowInput> rows;
};

// 类型 7 的内容：Delete 请求中的主键列表。
struct DeleteRecord {
  std::string database;
  std::string table;
  std::vector<std::string> primary_keys;
};

// payload 用 ByteWriter 编码（小端序，字符串带 u32 长度前缀）：
//   1、2：库名。
//   3：库名，再接 Schema::Save 的输出。
//   4：库名、表名。
//   5、6：库名、表名、u8 upsert（Insert 为 0，Upsert 为 1）、u32 行数；
//         每行为主键、u32 字段数，每个字段为列名、u8 类别与值：
//         0 int64（u64）、1 float（f32）、2 string、3 稠密（u32 长度与 f32）、
//         4 稀疏（u32 长度、各位置 u32、各值 f32）。
//   7：库名、表名、u32 主键个数与各主键。
// Encode* 只在字符串超过 u32 长度上限时失败。Decode* 遇到数据截断、多余字节、
// 未知类别、重复字段名或 upsert 标志与类型不符时返回 DataLoss；解码不做业务
// 校验（名字规则、Schema 匹配等由回放时的 Prepare* 负责）。

absl::StatusOr<std::string> EncodeCreateDatabase(std::string_view database);
absl::StatusOr<DatabaseRecord> DecodeCreateDatabase(std::string_view payload);

absl::StatusOr<std::string> EncodeDropDatabase(std::string_view database);
absl::StatusOr<DatabaseRecord> DecodeDropDatabase(std::string_view payload);

absl::StatusOr<std::string> EncodeCreateTable(std::string_view database,
                                              const Schema& schema);
absl::StatusOr<CreateTableRecord> DecodeCreateTable(std::string_view payload);

absl::StatusOr<std::string> EncodeDropTable(std::string_view database,
                                            std::string_view table);
absl::StatusOr<DropTableRecord> DecodeDropTable(std::string_view payload);

absl::StatusOr<std::string> EncodeInsert(std::string_view database,
                                         std::string_view table,
                                         std::span<const RowInput> rows);
absl::StatusOr<WriteRowsRecord> DecodeInsert(std::string_view payload);

absl::StatusOr<std::string> EncodeUpsert(std::string_view database,
                                         std::string_view table,
                                         std::span<const RowInput> rows);
absl::StatusOr<WriteRowsRecord> DecodeUpsert(std::string_view payload);

absl::StatusOr<std::string> EncodeDelete(
    std::string_view database, std::string_view table,
    std::span<const std::string> primary_keys);
absl::StatusOr<DeleteRecord> DecodeDelete(std::string_view payload);

}  // namespace hector

#endif  // HECTOR_CORE_WAL_CODEC_H_
