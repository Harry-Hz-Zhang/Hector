#ifndef HECTOR_CORE_DATABASE_H_
#define HECTOR_CORE_DATABASE_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "hector/core/table.h"

namespace hector {

// 库内表目录，由 Engine 持有。本类不加锁：Engine 的目录读写锁保护所有库的
// 表映射，调用方必须在持有该锁时调用（修改持写锁，查询持读锁）。
class Database {
 public:
  explicit Database(std::string name);

  const std::string& name() const;
  // 按 table->schema().table_name() 登记；重名返回 AlreadyExists。
  absl::Status AddTable(std::shared_ptr<Table> table);
  // 不存在返回 NotFound。已经取到 shared_ptr 的读者不受影响。
  absl::Status RemoveTable(std::string_view name);
  // 不存在返回 NotFound。
  absl::StatusOr<std::shared_ptr<Table>> FindTable(std::string_view name) const;
  // 字典序。
  std::vector<std::string> TableNames() const;

 private:
  std::string name_;
  std::map<std::string, std::shared_ptr<Table>, std::less<>> tables_;
};

}  // namespace hector

#endif  // HECTOR_CORE_DATABASE_H_
