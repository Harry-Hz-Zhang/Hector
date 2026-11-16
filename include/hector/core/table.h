#ifndef HECTOR_CORE_TABLE_H_
#define HECTOR_CORE_TABLE_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/common/byte_io.h"
#include "hector/common/concurrent_bitmap.h"
#include "hector/common/types.h"
#include "hector/common/vector_accessor.h"
#include "hector/core/column_store.h"
#include "hector/core/schema.h"
#include "hector/core/search.h"
#include "hector/core/value.h"
#include "hector/index/index_factory.h"
#include "hector/index/sparse_index.h"
#include "hector/index/vector_index.h"

namespace hector {

// 测试注入点；默认值包装 MakeVectorIndex / MakeSparseIndex。
struct IndexFactory {
  std::function<absl::StatusOr<std::unique_ptr<VectorIndex>>(
      const DenseIndexConfig&, const DenseVectorAccessor*)>
      dense;
  std::function<absl::StatusOr<std::unique_ptr<SparseIndex>>(
      const SparseIndexConfig&, const SparseVectorAccessor*)>
      sparse;
};

IndexFactory DefaultIndexFactory();

// 测试可设较小容量；必须在 1 到 kMaxRowsPerTable 之间。
struct TableOptions {
  RowId max_rows = kMaxRowsPerTable;
};

// kReplay：照常校验，但跳过归一化与稀疏排序（WAL 中已是归一化后的数据）。
// 因此 kReplay 下稀疏位置必须已经严格升序，否则返回 InvalidArgument。
// NOLINTNEXTLINE(performance-enum-size): 声明与 design.md 4.4 一致
enum class PrepareMode { kOnline, kReplay };

// 已校验、已归一化、已排序的行，以及 Upsert 需要打墓碑的旧 row id；
// rows 与 upsert 编码进 WAL。
struct PreparedBatch {
  // 归一化与稀疏排序之后的行（WAL 只保存这一部分）。
  std::vector<RowInput> rows;
  bool upsert = false;
  // Upsert 覆盖的旧行；由 PrepareInsert 按当前主键映射算出，回放时重新计算，
  // 不写入 WAL。
  std::vector<RowId> tombstoned_row_ids;
};

struct ColumnStatsRow {
  std::string name;
  // 普通列为 monostate。
  std::variant<std::monostate, DenseIndexKind, SparseIndexKind> index;
  uint64_t indexed_rows;
  // 原始数据加索引派生结构的估算字节数。
  uint64_t memory_bytes;
};

struct TableStats {
  uint64_t live_rows;
  uint64_t deleted_rows;
  // 按 Schema 列顺序。
  std::vector<ColumnStatsRow> columns;
};

// 一张表：列存储、每个向量列的索引、主键映射与墓碑位图。
// 写操作（Prepare*、Apply*、Publish）只能由同一时刻的单个写者调用；
// 读操作（Get、Search、Stats）可与写者和其他读者并发。
class Table {
 public:
  // 为每个向量列调用工厂创建索引；工厂返回的错误（如 HNSW、INVERTED、int8
  // 的 Unimplemented）原样返回。
  static absl::StatusOr<std::unique_ptr<Table>> Create(
      Schema schema, const IndexFactory& factory = DefaultIndexFactory(),
      const TableOptions& options = {});

  const Schema& schema() const;

  // 以下三组写操作只能由同一时刻的单个写者调用（由 Engine 的写互斥锁保证）。

  // 不修改状态。行数不在 1 到 100000 或任一行校验失败时返回
  // InvalidArgument（消息含 rows[i]）；Insert 遇到已存在的主键返回
  // AlreadyExists；已写入行数加本批行数超过 max_rows 时返回
  // ResourceExhausted，保证达到上限时不写 WAL。
  absl::StatusOr<PreparedBatch> PrepareInsert(std::vector<RowInput> rows,
                                              bool upsert,
                                              PrepareMode mode) const;
  // 写列存储、调用各索引 Add、登记主键；新行暂不可见。前置条件：batch
  // 由 PrepareInsert 在当前状态上产出（不满足时返回 FailedPrecondition，
  // 状态不变）。索引 Add 中途失败后列存储与索引已部分写入，Table 不再可写，
  // 由 Engine 进入失败态。
  absl::Status ApplyInsert(PreparedBatch batch, Lsn lsn);
  // 主键个数不在 1 到 100000 时返回 InvalidArgument；不存在的主键静默跳过，
  // 重复的主键只计一次。
  absl::StatusOr<std::vector<RowId>> PrepareDelete(
      std::span<const std::string> primary_keys) const;
  // 解除主键映射；墓碑暂不生效。
  absl::Status ApplyDelete(std::span<const RowId> row_ids, Lsn lsn);
  // 按 LSN 顺序让 <= up_to 的已 Apply 改动对读者可见：先发布新行数
  // （release），再设置墓碑。
  void Publish(Lsn up_to);

  // 读操作，可与写者和其他读者并发。

  // 只返回已发布且未删除的行，按 primary_keys 的顺序；其余主键追加到
  // missing（可以为空指针）。output_columns 为空时返回全部列（含向量），
  // 返回的列按 Schema 顺序排列。主键个数不在 1 到 100000 或列不存在时
  // 返回 InvalidArgument。
  absl::StatusOr<std::vector<RowOutput>> Get(
      std::span<const std::string> primary_keys,
      std::span<const std::string> output_columns,
      std::vector<std::string>* missing) const;
  // 先用 acquire 读到已发布行数 n，再按 design.md 第 5 节校验请求（失败返回
  // InvalidArgument，此时不调用任何索引）；Filter 只看 row id < n 的行，
  // 与墓碑一起排除。单路检索向索引要 top_k 条，分数为索引距离的转换
  // （L2 原样，其余取负）；多路检索每路要实际 candidate_limit 条，再用 Fuse
  // 融合。最后取出主键与 output_columns 中的普通列（按 Schema 列顺序）。
  absl::StatusOr<SearchResult> Search(const SearchRequest& request) const;
  TableStats Stats() const;

  // 快照挂载点（c5 实现；c0 返回 Unimplemented）。
  absl::Status Save(ByteWriter& out) const;
  static absl::StatusOr<std::unique_ptr<Table>> Load(
      ByteReader& in, const IndexFactory& factory = DefaultIndexFactory());

 private:
  // Schema 中一列的存储与索引；按列类别只有对应的成员非空。
  struct ColumnSlot {
    std::unique_ptr<DenseColumn> dense;
    std::unique_ptr<VectorIndex> dense_index;
    std::unique_ptr<SparseColumn> sparse;
    std::unique_ptr<SparseIndex> sparse_index;
    std::unique_ptr<ScalarColumn> scalar;
  };

  // 已 Apply、尚未 Publish 的一批改动。
  struct PendingChange {
    Lsn lsn;
    RowId row_count;  // Apply 之后的已写入行数
    std::vector<RowId> tombstones;
  };

  Table(Schema schema, RowId max_rows);

  // 前置条件：row < 已写入行数。
  Value ReadValue(size_t column, RowId row) const;
  // 已写入行数（含未发布行），只由写者调用。
  RowId written_rows() const;

  const Schema schema_;
  const RowId max_rows_;
  std::vector<ColumnSlot> columns_;
  // 第 i 行的主键；与各列一样按单写者多读者方式工作。
  ScalarColumn primary_keys_;
  ConcurrentBitmap tombstones_;
  // 对读者可见的行数：Publish 时 store(release)，读者 load(acquire)。
  std::atomic<RowId> published_rows_{0};

  // 主键到最新 row id 的映射；只在 Apply* 修改时短暂持写锁，读取持读锁。
  mutable std::shared_mutex pk_mu_;
  absl::flat_hash_map<std::string, RowId> pk_to_row_;

  // 只由写者访问。
  std::deque<PendingChange> pending_;
};

}  // namespace hector

#endif  // HECTOR_CORE_TABLE_H_
