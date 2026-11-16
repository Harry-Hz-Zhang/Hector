// 保证 hector_tests 在各层用例加入之前也至少有一个用例可运行。

#include <gtest/gtest.h>

namespace {

TEST(PlaceholderTest, Builds) { EXPECT_TRUE(true); }

}  // namespace
