"""Hector 向量数据库的 Python 演示客户端。

生成代码位于 `_gen/`（由 `scripts/gen_proto.sh` 生成，不入库），其内部用
`from hector.v1 import hector_pb2` 导入彼此，所以在导入前把 `_gen` 插入 `sys.path` 首位。
"""

import os
import sys

_GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gen")
if _GEN_DIR not in sys.path:
    sys.path.insert(0, _GEN_DIR)

try:
    from hector.v1 import hector_pb2, hector_pb2_grpc  # noqa: E402
except ImportError as exc:  # pragma: no cover - 仅在未生成代码时触发
    raise ImportError(
        "hector_client 的 gRPC 生成代码缺失，请先执行 clients/python/scripts/gen_proto.sh"
    ) from exc

from hector_client.client import HectorClient  # noqa: E402

__all__ = ["HectorClient", "hector_pb2", "hector_pb2_grpc"]
