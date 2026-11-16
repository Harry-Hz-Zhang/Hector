#include "hector/core/column_store.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
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

constexpr uint32_t kFormatVersion = 1;
// 每块的行数。
constexpr RowId kChunkRows = 1024;

size_t ChunkSlots(RowId capacity) {
  return (static_cast<size_t>(capacity) + kChunkRows - 1) / kChunkRows;
}

absl::Status Full(RowId capacity) {
  return absl::ResourceExhaustedError(
      absl::StrCat("column is full: capacity is ", capacity, " rows"));
}

absl::Status DataLoss(std::string_view column, std::string_view what) {
  return absl::DataLossError(absl::StrCat(column, " data: ", what));
}

// 读取失败统一转为 DataLoss。
template <typename T>
absl::StatusOr<T> OrDataLoss(std::string_view column, absl::StatusOr<T> value) {
  if (!value.ok()) {
    return DataLoss(column, "truncated");
  }
  return value;
}

absl::Status CheckEmpty(RowId size) {
  if (size != 0) {
    return absl::FailedPreconditionError(
        "column Load requires an empty column");
  }
  return absl::OkStatus();
}

// 读出版本与行数并校验；行数不超过 capacity。
absl::StatusOr<RowId> ReadHeaderRows(std::string_view column, ByteReader& in,
                                     RowId capacity) {
  HECTOR_ASSIGN_OR_RETURN(const uint32_t rows,
                          OrDataLoss(column, in.ReadU32()));
  if (rows > capacity) {
    return DataLoss(column, absl::StrCat("row count ", rows,
                                         " exceeds capacity ", capacity));
  }
  return rows;
}

absl::Status ReadVersion(std::string_view column, ByteReader& in) {
  HECTOR_ASSIGN_OR_RETURN(const uint32_t version,
                          OrDataLoss(column, in.ReadU32()));
  if (version != kFormatVersion) {
    return DataLoss(column, absl::StrCat("unsupported version ", version));
  }
  return absl::OkStatus();
}

// ScalarValue 的候选下标与 ScalarType 的对应关系。
size_t AlternativeOf(ScalarType t) {
  switch (t) {
    case ScalarType::kInt64:
      return 0;
    case ScalarType::kFloat:
      return 1;
    case ScalarType::kString:
      return 2;
  }
  return std::variant_npos;
}

}  // namespace

// ---------------------------------------------------------------------------
// DenseColumn

DenseColumn::DenseColumn(uint32_t dim, RowId capacity)
    : dim_(dim),
      capacity_(capacity),
      chunk_slots_(ChunkSlots(capacity)),
      // 块目录唯一一次分配；之后只写入槽位、不再重新分配。
      chunks_(std::make_unique<std::unique_ptr<float[]>[]>(chunk_slots_)) {}

uint32_t DenseColumn::dim() const { return dim_; }

RowId DenseColumn::size() const {
  return size_.load(std::memory_order_acquire);
}

std::span<const float> DenseColumn::Get(RowId id) const {
  const float* chunk = chunks_[id / kChunkRows].get();
  return {chunk + static_cast<size_t>(id % kChunkRows) * dim_, dim_};
}

absl::Status DenseColumn::Append(std::span<const float> v) {
  if (v.size() != dim_) {
    return absl::InvalidArgumentError(
        absl::StrCat("dense column expects ", dim_, " values, got ", v.size()));
  }
  // 只有写者修改 size_，写者自己读可以用 relaxed。
  const RowId n = size_.load(std::memory_order_relaxed);
  if (n >= capacity_) {
    return Full(capacity_);
  }
  std::unique_ptr<float[]>& chunk = chunks_[n / kChunkRows];
  if (chunk == nullptr) {
    chunk = std::make_unique<float[]>(static_cast<size_t>(kChunkRows) * dim_);
    allocated_chunks_.fetch_add(1, std::memory_order_relaxed);
  }
  std::copy(v.begin(), v.end(),
            chunk.get() + static_cast<size_t>(n % kChunkRows) * dim_);
  size_.store(n + 1, std::memory_order_release);
  return absl::OkStatus();
}

size_t DenseColumn::MemoryBytes() const {
  return chunk_slots_ * sizeof(std::unique_ptr<float[]>) +
         allocated_chunks_.load(std::memory_order_relaxed) * kChunkRows * dim_ *
             sizeof(float);
}

absl::Status DenseColumn::Save(ByteWriter& out) const {
  const RowId rows = size();
  HECTOR_RETURN_IF_ERROR(out.WriteU32(kFormatVersion));
  HECTOR_RETURN_IF_ERROR(out.WriteU32(dim_));
  HECTOR_RETURN_IF_ERROR(out.WriteU32(rows));
  for (RowId id = 0; id < rows; ++id) {
    for (float f : Get(id)) {
      HECTOR_RETURN_IF_ERROR(out.WriteF32(f));
    }
  }
  return absl::OkStatus();
}

absl::Status DenseColumn::Load(ByteReader& in) {
  constexpr std::string_view kName = "dense column";
  HECTOR_RETURN_IF_ERROR(CheckEmpty(size()));
  HECTOR_RETURN_IF_ERROR(ReadVersion(kName, in));
  HECTOR_ASSIGN_OR_RETURN(const uint32_t dim, OrDataLoss(kName, in.ReadU32()));
  if (dim != dim_) {
    return DataLoss(kName, absl::StrCat("dim ", dim, " does not match ", dim_));
  }
  HECTOR_ASSIGN_OR_RETURN(const RowId rows,
                          ReadHeaderRows(kName, in, capacity_));
  // 先全部读出再追加，失败时列保持为空。不按 rows 预分配，避免损坏的行数
  // 触发巨大的分配。
  std::vector<float> values;
  for (size_t i = 0; i < static_cast<size_t>(rows) * dim_; ++i) {
    HECTOR_ASSIGN_OR_RETURN(const float f, OrDataLoss(kName, in.ReadF32()));
    values.push_back(f);
  }
  for (RowId id = 0; id < rows; ++id) {
    HECTOR_RETURN_IF_ERROR(Append(std::span<const float>(
        values.data() + static_cast<size_t>(id) * dim_, dim_)));
  }
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// SparseColumn

SparseColumn::SparseColumn(RowId capacity)
    : capacity_(capacity),
      chunk_slots_(ChunkSlots(capacity)),
      chunks_(
          std::make_unique<std::unique_ptr<SparseVector[]>[]>(chunk_slots_)) {}

RowId SparseColumn::size() const {
  return size_.load(std::memory_order_acquire);
}

SparseVectorView SparseColumn::Get(RowId id) const {
  const SparseVector& v = chunks_[id / kChunkRows][id % kChunkRows];
  return SparseVectorView{v.indices, v.values};
}

absl::Status SparseColumn::Append(SparseVector v) {
  if (v.indices.size() != v.values.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("sparse vector has ", v.indices.size(), " indices but ",
                     v.values.size(), " values"));
  }
  const RowId n = size_.load(std::memory_order_relaxed);
  if (n >= capacity_) {
    return Full(capacity_);
  }
  std::unique_ptr<SparseVector[]>& chunk = chunks_[n / kChunkRows];
  if (chunk == nullptr) {
    chunk = std::make_unique<SparseVector[]>(kChunkRows);
    allocated_chunks_.fetch_add(1, std::memory_order_relaxed);
  }
  payload_bytes_.fetch_add(
      v.indices.size() * sizeof(uint32_t) + v.values.size() * sizeof(float),
      std::memory_order_relaxed);
  chunk[n % kChunkRows] = std::move(v);
  size_.store(n + 1, std::memory_order_release);
  return absl::OkStatus();
}

size_t SparseColumn::MemoryBytes() const {
  return chunk_slots_ * sizeof(std::unique_ptr<SparseVector[]>) +
         allocated_chunks_.load(std::memory_order_relaxed) * kChunkRows *
             sizeof(SparseVector) +
         payload_bytes_.load(std::memory_order_relaxed);
}

absl::Status SparseColumn::Save(ByteWriter& out) const {
  const RowId rows = size();
  HECTOR_RETURN_IF_ERROR(out.WriteU32(kFormatVersion));
  HECTOR_RETURN_IF_ERROR(out.WriteU32(rows));
  for (RowId id = 0; id < rows; ++id) {
    const SparseVectorView v = Get(id);
    HECTOR_RETURN_IF_ERROR(
        out.WriteU32(static_cast<uint32_t>(v.indices.size())));
    for (uint32_t index : v.indices) {
      HECTOR_RETURN_IF_ERROR(out.WriteU32(index));
    }
    for (float value : v.values) {
      HECTOR_RETURN_IF_ERROR(out.WriteF32(value));
    }
  }
  return absl::OkStatus();
}

absl::Status SparseColumn::Load(ByteReader& in) {
  constexpr std::string_view kName = "sparse column";
  HECTOR_RETURN_IF_ERROR(CheckEmpty(size()));
  HECTOR_RETURN_IF_ERROR(ReadVersion(kName, in));
  HECTOR_ASSIGN_OR_RETURN(const RowId rows,
                          ReadHeaderRows(kName, in, capacity_));
  std::vector<SparseVector> loaded;
  for (RowId id = 0; id < rows; ++id) {
    HECTOR_ASSIGN_OR_RETURN(const uint32_t nnz,
                            OrDataLoss(kName, in.ReadU32()));
    SparseVector v;
    for (uint32_t i = 0; i < nnz; ++i) {
      HECTOR_ASSIGN_OR_RETURN(const uint32_t index,
                              OrDataLoss(kName, in.ReadU32()));
      v.indices.push_back(index);
    }
    for (uint32_t i = 0; i < nnz; ++i) {
      HECTOR_ASSIGN_OR_RETURN(const float value,
                              OrDataLoss(kName, in.ReadF32()));
      v.values.push_back(value);
    }
    loaded.push_back(std::move(v));
  }
  for (SparseVector& v : loaded) {
    HECTOR_RETURN_IF_ERROR(Append(std::move(v)));
  }
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// ScalarColumn

ScalarColumn::ScalarColumn(ScalarType t, RowId capacity)
    : type_(t),
      capacity_(capacity),
      chunk_slots_(ChunkSlots(capacity)),
      chunks_(
          std::make_unique<std::unique_ptr<ScalarValue[]>[]>(chunk_slots_)) {}

absl::Status ScalarColumn::Append(ScalarValue v) {
  if (v.index() != AlternativeOf(type_)) {
    return absl::InvalidArgumentError(
        "scalar value type does not match the column type");
  }
  const RowId n = size_.load(std::memory_order_relaxed);
  if (n >= capacity_) {
    return Full(capacity_);
  }
  std::unique_ptr<ScalarValue[]>& chunk = chunks_[n / kChunkRows];
  if (chunk == nullptr) {
    chunk = std::make_unique<ScalarValue[]>(kChunkRows);
    allocated_chunks_.fetch_add(1, std::memory_order_relaxed);
  }
  if (const auto* s = std::get_if<std::string>(&v)) {
    payload_bytes_.fetch_add(s->size(), std::memory_order_relaxed);
  }
  chunk[n % kChunkRows] = std::move(v);
  size_.store(n + 1, std::memory_order_release);
  return absl::OkStatus();
}

const ScalarValue& ScalarColumn::Get(RowId id) const {
  return chunks_[id / kChunkRows][id % kChunkRows];
}

RowId ScalarColumn::size() const {
  return size_.load(std::memory_order_acquire);
}

size_t ScalarColumn::MemoryBytes() const {
  return chunk_slots_ * sizeof(std::unique_ptr<ScalarValue[]>) +
         allocated_chunks_.load(std::memory_order_relaxed) * kChunkRows *
             sizeof(ScalarValue) +
         payload_bytes_.load(std::memory_order_relaxed);
}

absl::Status ScalarColumn::Save(ByteWriter& out) const {
  const RowId rows = size();
  HECTOR_RETURN_IF_ERROR(out.WriteU32(kFormatVersion));
  HECTOR_RETURN_IF_ERROR(out.WriteU8(static_cast<uint8_t>(type_)));
  HECTOR_RETURN_IF_ERROR(out.WriteU32(rows));
  for (RowId id = 0; id < rows; ++id) {
    const ScalarValue& v = Get(id);
    if (const auto* i = std::get_if<int64_t>(&v)) {
      HECTOR_RETURN_IF_ERROR(out.WriteU64(std::bit_cast<uint64_t>(*i)));
    } else if (const auto* f = std::get_if<float>(&v)) {
      HECTOR_RETURN_IF_ERROR(out.WriteF32(*f));
    } else {
      HECTOR_RETURN_IF_ERROR(out.WriteString(std::get<std::string>(v)));
    }
  }
  return absl::OkStatus();
}

absl::Status ScalarColumn::Load(ByteReader& in) {
  constexpr std::string_view kName = "scalar column";
  HECTOR_RETURN_IF_ERROR(CheckEmpty(size()));
  HECTOR_RETURN_IF_ERROR(ReadVersion(kName, in));
  HECTOR_ASSIGN_OR_RETURN(const uint8_t type, OrDataLoss(kName, in.ReadU8()));
  if (type != static_cast<uint8_t>(type_)) {
    return DataLoss(kName, absl::StrCat("type ", type, " does not match ",
                                        static_cast<int>(type_)));
  }
  HECTOR_ASSIGN_OR_RETURN(const RowId rows,
                          ReadHeaderRows(kName, in, capacity_));
  std::vector<ScalarValue> loaded;
  for (RowId id = 0; id < rows; ++id) {
    switch (type_) {
      case ScalarType::kInt64: {
        HECTOR_ASSIGN_OR_RETURN(const uint64_t bits,
                                OrDataLoss(kName, in.ReadU64()));
        loaded.emplace_back(std::bit_cast<int64_t>(bits));
        break;
      }
      case ScalarType::kFloat: {
        HECTOR_ASSIGN_OR_RETURN(const float f, OrDataLoss(kName, in.ReadF32()));
        loaded.emplace_back(f);
        break;
      }
      case ScalarType::kString: {
        HECTOR_ASSIGN_OR_RETURN(std::string s,
                                OrDataLoss(kName, in.ReadString()));
        loaded.emplace_back(std::move(s));
        break;
      }
    }
  }
  for (ScalarValue& v : loaded) {
    HECTOR_RETURN_IF_ERROR(Append(std::move(v)));
  }
  return absl::OkStatus();
}

}  // namespace hector
