"""HectorClient：hector.v1.HectorService 的同步封装（design 第 8 节）。

每个 RPC 对应一个 snake_case 方法；gRPC 错误以 `grpc.RpcError` 原样抛出，不做包装。

`insert` 与 `upsert` 会按序列化字节数与行数自动分块：每次请求序列化后不超过
`max_message_bytes - 65536` 字节，且不超过 100000 行。注意：分块后的多次请求不是一个
原子操作。某一块失败时，之前的块已经写入并且不会回滚；调用方需要自行处理部分写入。
"""

import grpc

from hector.v1 import hector_pb2, hector_pb2_grpc

DEFAULT_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
# 每次写入请求为 gRPC 帧与 HTTP/2 头部预留的字节数。
REQUEST_HEADROOM_BYTES = 65536
# 服务端单次写入的行数上限（design 第 5 节）。
MAX_ROWS_PER_REQUEST = 100000


def _varint_size(value):
    size = 1
    while value >= 0x80:
        value >>= 7
        size += 1
    return size


class HectorClient:
    """hector.v1.HectorService 的客户端。可用作上下文管理器，退出时关闭 channel。"""

    def __init__(self, address, max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES):
        if max_message_bytes <= REQUEST_HEADROOM_BYTES:
            raise ValueError(
                f"max_message_bytes must be greater than {REQUEST_HEADROOM_BYTES}, "
                f"got {max_message_bytes}"
            )
        self._max_message_bytes = max_message_bytes
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", max_message_bytes),
                ("grpc.max_receive_message_length", max_message_bytes),
            ],
        )
        self._stub = hector_pb2_grpc.HectorServiceStub(self._channel)

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    # ---- 库 ----

    def create_database(self, name):
        self._stub.CreateDatabase(hector_pb2.CreateDatabaseRequest(name=name))

    def drop_database(self, name):
        self._stub.DropDatabase(hector_pb2.DropDatabaseRequest(name=name))

    def list_databases(self):
        """返回按字典序排列的库名列表。"""
        return list(self._stub.ListDatabases(hector_pb2.ListDatabasesRequest()).names)

    # ---- 表 ----

    def create_table(self, database, schema):
        """`schema` 是 `hector_pb2.TableSchema`。"""
        self._stub.CreateTable(hector_pb2.CreateTableRequest(database=database, schema=schema))

    def drop_table(self, database, table):
        self._stub.DropTable(hector_pb2.DropTableRequest(database=database, table=table))

    def list_tables(self, database):
        return list(self._stub.ListTables(hector_pb2.ListTablesRequest(database=database)).names)

    def describe_table(self, database, table):
        """返回填好默认值的 `hector_pb2.TableSchema`。"""
        request = hector_pb2.DescribeTableRequest(database=database, table=table)
        return self._stub.DescribeTable(request).schema

    # ---- 数据 ----

    def insert(self, database, table, rows):
        """插入 `hector_pb2.Row` 序列，返回插入的行数。自动分块，分块之间不保证原子性。"""
        chunks = self._plan_chunks(hector_pb2.InsertRequest, database, table, rows)
        return sum(
            self._stub.Insert(
                hector_pb2.InsertRequest(database=database, table=table, rows=chunk)
            ).inserted
            for chunk in chunks
        )

    def upsert(self, database, table, rows):
        """写入或覆盖 `hector_pb2.Row` 序列，返回写入的行数。分块规则同 `insert`。"""
        chunks = self._plan_chunks(hector_pb2.UpsertRequest, database, table, rows)
        return sum(
            self._stub.Upsert(
                hector_pb2.UpsertRequest(database=database, table=table, rows=chunk)
            ).upserted
            for chunk in chunks
        )

    def delete(self, database, table, primary_keys):
        """删除主键对应的行，返回实际删除的行数；不存在的主键被静默跳过。"""
        request = hector_pb2.DeleteRequest(
            database=database, table=table, primary_keys=primary_keys
        )
        return self._stub.Delete(request).deleted

    def get(self, database, table, primary_keys, output_columns=None):
        """返回 `hector_pb2.GetResponse`（`rows` 与 `missing_primary_keys`）。"""
        request = hector_pb2.GetRequest(
            database=database,
            table=table,
            primary_keys=primary_keys,
            output_columns=output_columns or [],
        )
        return self._stub.Get(request)

    def search(
        self,
        database,
        table,
        top_k,
        routes,
        filter=None,  # noqa: A002 - 与 proto 字段同名
        fusion=None,
        output_columns=None,
    ):
        """`routes` 是 `hector_pb2.SearchRoute` 序列；返回 `hector_pb2.SearchResponse`。"""
        request = hector_pb2.SearchRequest(
            database=database,
            table=table,
            top_k=top_k,
            routes=routes,
            filter=filter,
            fusion=fusion,
            output_columns=output_columns or [],
        )
        return self._stub.Search(request)

    # ---- 运维 ----

    def create_snapshot(self):
        """返回快照对应的 LSN（c0 的服务端返回 UNIMPLEMENTED）。"""
        return self._stub.CreateSnapshot(hector_pb2.CreateSnapshotRequest()).lsn

    def get_stats(self, database, table):
        """返回 `hector_pb2.GetStatsResponse`。"""
        return self._stub.GetStats(hector_pb2.GetStatsRequest(database=database, table=table))

    # ---- 分块 ----

    def _plan_chunks(self, request_type, database, table, rows):
        """在发送任何请求之前把行全部分好块；任一行单独就超过上限时抛 ValueError。"""
        limit = self._max_message_bytes - REQUEST_HEADROOM_BYTES
        base = request_type(database=database, table=table).ByteSize()
        chunks, chunk, size = [], [], base
        for i, row in enumerate(rows):
            row_bytes = row.ByteSize()
            # repeated Row rows = 3：1 字节 tag + 长度 varint + 行本身。
            cost = 1 + _varint_size(row_bytes) + row_bytes
            if base + cost > limit:
                raise ValueError(
                    f"rows[{i}] (primary_key={row.primary_key!r}) serializes to "
                    f"{base + cost} bytes in a request, exceeding the limit of {limit} bytes "
                    f"(max_message_bytes - {REQUEST_HEADROOM_BYTES})"
                )
            if chunk and (size + cost > limit or len(chunk) >= MAX_ROWS_PER_REQUEST):
                chunks.append(chunk)
                chunk, size = [], base
            chunk.append(row)
            size += cost
        if chunk:
            chunks.append(chunk)
        return chunks
