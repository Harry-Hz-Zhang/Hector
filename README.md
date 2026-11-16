# Hector

## 构建

### 环境准备

本项目在 Apple Silicon 的 macOS 上开发，需要 CMake 3.28 及以上版本、Ninja 与 Apple clang（C++20）。gRPC、protobuf、abseil 使用 Homebrew 安装的版本；clang-format 与 clang-tidy 是必需工具，缺任何一个时配置会直接失败并提示安装命令。

```bash
brew install cmake ninja grpc clang-format llvm
```

Homebrew 的 llvm 是 keg-only，clang-tidy 不在 PATH 中，CMake 会自动到 `/opt/homebrew/opt/llvm/bin` 查找；也可以用 `-DHECTOR_CLANG_TIDY_EXE=<路径>` 显式指定。GoogleTest `v1.17.0`、Google Benchmark `v1.9.4`、spdlog `v1.15.3` 由 FetchContent 在第一次配置时下载，因此第一次配置需要联网。

### 预设

| 预设 | 构建目录 | 说明 |
| --- | --- | --- |
| `debug` | `build/debug` | Debug，开启 ASan 与 UBSan，导出 `compile_commands.json` |
| `release` | `build/release` | Release（`-O3`） |
| `tsan` | `build/tsan` | RelWithDebInfo，开启 TSan，只构建 `hector_tests` 并只运行名字含 `Concurrent` 的用例 |

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

构建完成后，`build/debug/` 下会生成 `hectord`、`hector_tests`、`hector_server_tests` 与 `hector_bench`。

### 质量检查

格式检查与静态检查既是构建目标，也注册成了 CTest 用例：

```bash
cmake --build --preset debug --target hector_format_check   # clang-format --dry-run --Werror
cmake --build --preset debug --target hector_tidy_check     # clang-tidy，基于 build/debug/compile_commands.json
ctest --preset debug -R 'FormatTest|TidyTest' --output-on-failure
```

### CMake 选项

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `HECTOR_ENABLE_ASAN_UBSAN` | OFF（`debug` 预设为 ON） | 给 Hector 目标加 `-fsanitize=address,undefined` |
| `HECTOR_ENABLE_TSAN` | OFF（`tsan` 预设为 ON） | 给 Hector 目标加 `-fsanitize=thread`；与上一项互斥 |
| `HECTOR_CLANG_FORMAT_EXE` | 自动查找 | clang-format 路径 |
| `HECTOR_CLANG_TIDY_EXE` | 自动查找 | clang-tidy 路径 |
| `HECTOR_LAYERING_SELFTEST` | OFF | 仅供分层检查的自测使用，平时不要打开 |

## 运行 hectord

`hectord` 是 Hector 的 gRPC 服务端（协议见 `proto/hector/v1/hector.proto`）。在仓库根目录执行，它会在前台运行，按 Ctrl+C（SIGINT）或发送 SIGTERM 后正常退出，退出码为 0：

```bash
build/debug/hectord --data_dir=./hector_data --listen_address=127.0.0.1:50051
```

启动成功后，stderr 上会打印一行 `hectord listening on 127.0.0.1:50051 data_dir=./hector_data replayed <n> records`。日志全部写到 stderr。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data_dir` | `./hector_data` | 数据目录，不存在则创建；同一目录同时只能被一个 hectord 使用，否则报 `data_dir is locked` 并以退出码 1 退出 |
| `--listen_address` | `127.0.0.1:50051` | 监听地址；端口写 0 时由系统分配，实际端口打印在启动日志中 |
| `--max_message_bytes` | `4194304` | 收发消息的上限（1048576 到 268435456）；超过上限的请求或响应返回 `RESOURCE_EXHAUSTED` |

端口 50051 被占用时，用 `--listen_address` 换一个端口，并在下文的演示命令里使用同一个地址。安装了 grpcurl 时，可以在另一个终端检查服务是否可用，空数据目录下输出 `{}`：

```bash
grpcurl -plaintext -import-path proto -proto hector/v1/hector.proto 127.0.0.1:50051 hector.v1.HectorService/ListDatabases
```

## Python 演示

Python 客户端位于 `clients/python/`，包名 `hector_client`，需要 Python 3.14。下面的命令在仓库根目录执行：先创建虚拟环境并安装依赖（需要联网），再生成 gRPC 存根。生成的代码在 `clients/python/hector_client/_gen/`，与 `.venv/` 一样不入库；proto 变化后重新执行 `gen_proto.sh` 即可。

```bash
python3 -m venv clients/python/.venv
clients/python/.venv/bin/pip install -e 'clients/python[dev]'
clients/python/scripts/gen_proto.sh
```

保持上一节的 hectord 在运行，在另一个终端执行演示脚本。它依次演示建库、建表、插入 20 条玩具文档、稠密检索、稀疏检索、混合检索、带筛选的检索、删除、Get、GetStats、删库，最后打印 `demo finished`；任一步失败时以退出码 1 退出：

```bash
clients/python/.venv/bin/python clients/python/examples/demo.py --address 127.0.0.1:50051
```

运行客户端测试。端到端用例会用临时数据目录和系统分配的端口自行启动 hectord，不依赖上面那个进程；hectord 默认取 `build/debug/hectord`，可以用环境变量 `HECTOR_BIN` 指定其他路径，找不到时端到端用例会被跳过：

```bash
clients/python/.venv/bin/python -m pytest clients/python/tests -v
```

在代码中使用客户端：

```python
from hector_client import HectorClient, hector_pb2
from hector_client.bm25 import BM25Encoder

with HectorClient("127.0.0.1:50051") as client:
    print(client.list_databases())
```

`HectorClient` 为每个 RPC 提供同名的 snake_case 方法（如 `create_table`、`search`、`get_stats`），请求与返回使用 `hector_pb2` 中的消息；服务端返回的错误以 `grpc.RpcError` 原样抛出。`insert` 与 `upsert` 会自动分块：每次请求序列化后不超过 `max_message_bytes - 65536` 字节，且不超过 100000 行；`max_message_bytes` 默认 4 MB，应与 hectord 的 `--max_message_bytes` 一致。单行就超过上限时，在发送任何请求之前抛出 `ValueError`。**分块后的多次请求不是一个原子操作**：中途某一块失败时，之前的块已经写入，不会回滚。`BM25Encoder` 把文本编码成稀疏向量，查询向量与文档向量的内积就是 BM25 分数，`vocab_size` 可作为稀疏列的 `max_dim`。
