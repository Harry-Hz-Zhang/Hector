#include "hector/core/database.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "hector/core/table.h"

namespace hector {

Database::Database(std::string name) : name_(std::move(name)) {}

const std::string& Database::name() const { return name_; }

absl::Status Database::AddTable(std::shared_ptr<Table> table) {
  if (table == nullptr) {
    return absl::InvalidArgumentError("table must not be null");
  }
  const std::string& table_name = table->schema().table_name();
  if (tables_.contains(table_name)) {
    return absl::AlreadyExistsError(
        absl::StrCat("table \"", table_name, "\" already exists in database \"",
                     name_, "\""));
  }
  tables_.emplace(table_name, std::move(table));
  return absl::OkStatus();
}

absl::Status Database::RemoveTable(std::string_view name) {
  auto it = tables_.find(name);
  if (it == tables_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "table \"", name, "\" not found in database \"", name_, "\""));
  }
  tables_.erase(it);
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<Table>> Database::FindTable(
    std::string_view name) const {
  auto it = tables_.find(name);
  if (it == tables_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "table \"", name, "\" not found in database \"", name_, "\""));
  }
  return it->second;
}

std::vector<std::string> Database::TableNames() const {
  std::vector<std::string> names;
  names.reserve(tables_.size());
  for (const auto& [name, table] : tables_) {
    names.push_back(name);
  }
  return names;
}

}  // namespace hector
