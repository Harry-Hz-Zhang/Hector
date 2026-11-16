"""insert / upsert 的自动分块（design 第 8 节，python-client spec「自动分块插入」）。"""

import pytest

from hector_client import HectorClient, hector_pb2


class RecordingStub:
    """记录每次 Insert / Upsert 请求的替身存根，不发网络请求。"""

    def __init__(self):
        self.insert_requests = []
        self.upsert_requests = []

    def Insert(self, request):
        copy = hector_pb2.InsertRequest()
        copy.CopyFrom(request)
        self.insert_requests.append(copy)
        return hector_pb2.InsertResponse(inserted=len(request.rows), lsn=len(self.insert_requests))

    def Upsert(self, request):
        copy = hector_pb2.UpsertRequest()
        copy.CopyFrom(request)
        self.upsert_requests.append(copy)
        return hector_pb2.UpsertResponse(upserted=len(request.rows), lsn=len(self.upsert_requests))


def _client(max_message_bytes):
    # channel 是惰性连接的，地址不需要真实存在。
    client = HectorClient("127.0.0.1:1", max_message_bytes=max_message_bytes)
    stub = RecordingStub()
    client._stub = stub
    return client, stub


def _dense_row(pk, dim, value=0.5):
    row = hector_pb2.Row(primary_key=pk)
    row.fields["emb"].dense.values.extend([value] * dim)
    return row


def test_chunking_respects_byte_limit():
    client, stub = _client(1048576)
    rows = [_dense_row(f"pk{i:05d}", 256) for i in range(3000)]  # 每行约 1 KB
    assert 1000 < rows[0].ByteSize() < 1100

    inserted = client.insert("db", "t", rows)

    requests = stub.insert_requests
    assert len(requests) > 1
    assert all(r.ByteSize() <= 1048576 - 65536 for r in requests)
    assert all(r.database == "db" and r.table == "t" for r in requests)
    sent = [row.primary_key for r in requests for row in r.rows]
    assert sent == [row.primary_key for row in rows]
    assert inserted == 3000
    # 贪心装满：除最后一块外，再多放下一行就会超过上限，所以块数没有被无谓地放大。
    for current, following in zip(requests, requests[1:]):
        grown = hector_pb2.InsertRequest()
        grown.CopyFrom(current)
        grown.rows.append(following.rows[0])
        assert grown.ByteSize() > 983040

    # upsert 走同一套分块逻辑。
    upserted = client.upsert("db", "t", rows)
    assert len(stub.upsert_requests) == len(requests)
    assert all(r.ByteSize() <= 983040 for r in stub.upsert_requests)
    assert upserted == 3000


def test_chunking_respects_row_limit():
    client, stub = _client(268435456)
    rows = [_dense_row(str(i), 1) for i in range(100001)]

    inserted = client.insert("db", "t", rows)

    requests = stub.insert_requests
    assert len(requests) >= 2
    assert all(len(r.rows) <= 100000 for r in requests)
    assert sum(len(r.rows) for r in requests) == 100001
    assert inserted == 100001


def test_chunking_rejects_oversized_row():
    client, stub = _client(1048576)
    rows = [_dense_row("small", 4), _dense_row("huge", 512 * 1024)]  # 约 2 MB
    assert rows[1].ByteSize() > 2 * 1000 * 1000

    with pytest.raises(ValueError):
        client.insert("db", "t", rows)
    with pytest.raises(ValueError):
        client.upsert("db", "t", rows)
    assert stub.insert_requests == []
    assert stub.upsert_requests == []


def test_chunking_empty_input():
    client, stub = _client(1048576)

    assert client.insert("db", "t", []) == 0
    assert client.upsert("db", "t", []) == 0
    assert stub.insert_requests == []
    assert stub.upsert_requests == []
