#include "hector/common/status.h"

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace hector {
namespace {

absl::Status IncrementAfter(const absl::Status& first, int& counter) {
  HECTOR_RETURN_IF_ERROR(first);
  ++counter;
  return absl::OkStatus();
}

absl::StatusOr<int> DoubleOf(absl::StatusOr<int> input) {
  HECTOR_ASSIGN_OR_RETURN(const int value, std::move(input));
  return value * 2;
}

absl::StatusOr<int> SumOf(absl::StatusOr<int> a, absl::StatusOr<int> b) {
  int total = 0;
  HECTOR_ASSIGN_OR_RETURN(total, std::move(a));
  HECTOR_ASSIGN_OR_RETURN(const int second, std::move(b));
  return total + second;
}

absl::StatusOr<int> Unwrap(absl::StatusOr<std::unique_ptr<int>> input) {
  HECTOR_ASSIGN_OR_RETURN(std::unique_ptr<int> ptr, std::move(input));
  return *ptr;
}

TEST(StatusMacroTest, ReturnIfErrorShortCircuits) {
  int counter = 0;
  const absl::Status failed =
      IncrementAfter(absl::NotFoundError("missing row"), counter);
  EXPECT_EQ(failed.code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(failed.message(), "missing row");
  EXPECT_EQ(counter, 0);

  EXPECT_TRUE(IncrementAfter(absl::OkStatus(), counter).ok());
  EXPECT_EQ(counter, 1);

  absl::StatusOr<int> doubled = DoubleOf(21);
  ASSERT_TRUE(doubled.ok());
  EXPECT_EQ(*doubled, 42);

  absl::StatusOr<int> not_doubled = DoubleOf(absl::InvalidArgumentError("bad"));
  EXPECT_EQ(not_doubled.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(not_doubled.status().message(), "bad");

  // 同一函数内多次使用，以及给已有变量赋值。
  absl::StatusOr<int> sum = SumOf(2, 3);
  ASSERT_TRUE(sum.ok());
  EXPECT_EQ(*sum, 5);
  EXPECT_EQ(SumOf(2, absl::DataLossError("x")).status().code(),
            absl::StatusCode::kDataLoss);

  // 只能移动的类型。
  absl::StatusOr<int> unwrapped = Unwrap(std::make_unique<int>(7));
  ASSERT_TRUE(unwrapped.ok());
  EXPECT_EQ(*unwrapped, 7);
}

}  // namespace
}  // namespace hector
