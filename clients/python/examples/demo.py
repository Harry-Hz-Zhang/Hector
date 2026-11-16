"""Hector 演示：建库建表、插入、稠密 / 稀疏 / 混合 / 带筛选检索、删除、Get、GetStats、删库。

用法：python examples/demo.py [--address 127.0.0.1:50051]

每一步打印步骤名与结果；任一步失败时打印原因并以退出码 1 退出（会尽力删除本次创建的演示库）。
稠密向量由固定随机种子生成，不依赖嵌入模型；稀疏向量由 BM25Encoder 编码。
"""

import argparse
import random
import sys

from hector_client import HectorClient, hector_pb2
from hector_client.bm25 import BM25Encoder

DATABASE = "hector_demo"
TABLE = "docs"
DIM = 8
SEED = 42

# (主键, 文本, category, year) 共 20 条玩具文档。
DOCS = [
    ("doc00", "vector database stores dense embeddings", "db", 2021),
    ("doc01", "inverted index speeds up keyword search", "search", 2018),
    ("doc02", "hnsw graph index for approximate nearest neighbor search", "db", 2023),
    ("doc03", "bm25 ranks documents by term frequency", "search", 2016),
    ("doc04", "write ahead log makes the database durable", "db", 2019),
    ("doc05", "reciprocal rank fusion merges ranked lists", "search", 2022),
    ("doc06", "transformers produce sentence embeddings", "ml", 2020),
    ("doc07", "quantization shrinks vector memory usage", "db", 2024),
    ("doc08", "sparse vectors capture exact keyword matches", "search", 2021),
    ("doc09", "cosine similarity compares embedding directions", "ml", 2017),
    ("doc10", "snapshots speed up database recovery", "db", 2022),
    ("doc11", "learning to rank improves search relevance", "ml", 2019),
    ("doc12", "filtered vector search with scalar predicates", "db", 2023),
    ("doc13", "tokenizers split text into terms", "ml", 2015),
    ("doc14", "grpc carries requests between client and server", "infra", 2020),
    ("doc15", "hybrid search combines dense and sparse retrieval", "search", 2024),
    ("doc16", "simd kernels accelerate distance computation", "infra", 2021),
    ("doc17", "column storage keeps vectors contiguous in memory", "db", 2018),
    ("doc18", "recall measures how many true neighbors are found", "ml", 2022),
    ("doc19", "benchmarks report latency and throughput", "infra", 2023),
]


def dense_vectors():
    rng = random.Random(SEED)
    return [[rng.gauss(0.0, 1.0) for _ in range(DIM)] for _ in DOCS]


def build_schema(sparse_max_dim):
    return hector_pb2.TableSchema(
        name=TABLE,
        columns=[
            hector_pb2.ColumnDef(
                name="emb",
                dense=hector_pb2.DenseVectorColumn(dim=DIM, metric=hector_pb2.METRIC_COSINE),
            ),
            hector_pb2.ColumnDef(
                name="text_sparse",
                sparse=hector_pb2.SparseVectorColumn(max_dim=sparse_max_dim),
            ),
            hector_pb2.ColumnDef(
                name="category",
                scalar=hector_pb2.ScalarColumn(type=hector_pb2.SCALAR_TYPE_STRING),
            ),
            hector_pb2.ColumnDef(
                name="year",
                scalar=hector_pb2.ScalarColumn(type=hector_pb2.SCALAR_TYPE_INT64),
            ),
        ],
    )


def build_rows(vectors, sparse_docs):
    rows = []
    for (pk, _, category, year), dense, (indices, values) in zip(DOCS, vectors, sparse_docs):
        row = hector_pb2.Row(primary_key=pk)
        row.fields["emb"].dense.values.extend(dense)
        row.fields["text_sparse"].sparse.indices.extend(indices)
        row.fields["text_sparse"].sparse.values.extend(values)
        row.fields["category"].string_value = category
        row.fields["year"].int64_value = year
        rows.append(row)
    return rows


def dense_route(vector):
    return hector_pb2.SearchRoute(column="emb", dense=hector_pb2.DenseVector(values=vector))


def sparse_route(indices, values):
    return hector_pb2.SearchRoute(
        column="text_sparse", sparse=hector_pb2.SparseVector(indices=indices, values=values)
    )


def show_hits(response):
    kind = hector_pb2.ScoreKind.Name(response.score_kind)
    print(f"  score_kind={kind}")
    for hit in response.hits:
        extras = ""
        if hit.fields:
            extras = " " + " ".join(
                f"{name}={getattr(value, value.WhichOneof('kind'))}"
                for name, value in sorted(hit.fields.items())
            )
        print(f"  {hit.primary_key} score={hit.score:.6f}{extras}")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run(client, state):
    vectors = dense_vectors()
    encoder = BM25Encoder().fit([text for _, text, _, _ in DOCS])
    sparse_docs = encoder.encode_documents([text for _, text, _, _ in DOCS])

    def step(name):
        state["step"] = name
        print(f"== {name}", flush=True)

    step("create database")
    client.create_database(DATABASE)
    state["created"] = True
    print(f"  databases={client.list_databases()}")

    step("create table")
    client.create_table(DATABASE, build_schema(encoder.vocab_size))
    schema = client.describe_table(DATABASE, TABLE)
    print(f"  tables={client.list_tables(DATABASE)} columns={[c.name for c in schema.columns]}")

    step("insert")
    inserted = client.insert(DATABASE, TABLE, build_rows(vectors, sparse_docs))
    print(f"  inserted={inserted}")
    check(inserted == len(DOCS), f"expected {len(DOCS)} inserted rows, got {inserted}")

    step("dense search")
    response = client.search(DATABASE, TABLE, top_k=3, routes=[dense_route(vectors[3])])
    show_hits(response)
    check(response.hits[0].primary_key == "doc03", "dense search should rank doc03 first")

    step("sparse search")
    q_indices, q_values = encoder.encode_query("keyword search")
    response = client.search(DATABASE, TABLE, top_k=3, routes=[sparse_route(q_indices, q_values)])
    show_hits(response)
    check(len(response.hits) > 0, "sparse search returned no hits")

    step("hybrid search")
    q_indices, q_values = encoder.encode_query("vector database")
    response = client.search(
        DATABASE,
        TABLE,
        top_k=5,
        routes=[dense_route(vectors[0]), sparse_route(q_indices, q_values)],
        fusion=hector_pb2.FusionSpec(method=hector_pb2.FUSION_METHOD_RRF, rrf_k=60),
    )
    show_hits(response)
    check(response.score_kind == hector_pb2.SCORE_KIND_RRF, "hybrid search should use RRF")
    check(response.hits[0].primary_key == "doc00", "hybrid search should rank doc00 first")

    step("filtered search")
    flt = hector_pb2.Filter(
        conditions=[
            hector_pb2.Condition(
                column="category",
                op=hector_pb2.COMPARE_OP_EQ,
                values=[hector_pb2.ScalarValue(string_value="db")],
            ),
            hector_pb2.Condition(
                column="year",
                op=hector_pb2.COMPARE_OP_GE,
                values=[hector_pb2.ScalarValue(int64_value=2020)],
            ),
        ]
    )
    response = client.search(
        DATABASE,
        TABLE,
        top_k=10,
        routes=[dense_route(vectors[0])],
        filter=flt,
        output_columns=["category", "year"],
    )
    show_hits(response)
    check(len(response.hits) > 0, "filtered search returned no hits")
    for hit in response.hits:
        category = hit.fields["category"].string_value
        year = hit.fields["year"].int64_value
        check(category == "db" and year >= 2020, f"{hit.primary_key} violates the filter")

    step("delete")
    deleted = client.delete(DATABASE, TABLE, ["doc03", "no_such_doc"])
    print(f"  deleted={deleted}")
    check(deleted == 1, f"expected 1 deleted row, got {deleted}")

    step("get")
    response = client.get(DATABASE, TABLE, ["doc03", "doc04"], output_columns=["category", "year"])
    for row in response.rows:
        print(
            f"  {row.primary_key} category={row.fields['category'].string_value} "
            f"year={row.fields['year'].int64_value}"
        )
    print(f"  missing={list(response.missing_primary_keys)}")
    check([r.primary_key for r in response.rows] == ["doc04"], "get should return doc04 only")
    check(list(response.missing_primary_keys) == ["doc03"], "doc03 should be missing")

    step("get stats")
    stats = client.get_stats(DATABASE, TABLE)
    print(
        f"  live_rows={stats.live_rows} deleted_rows={stats.deleted_rows} "
        f"last_lsn={stats.last_lsn}"
    )
    for column in stats.columns:
        print(
            f"  column {column.name}: indexed_rows={column.indexed_rows} "
            f"memory_bytes={column.memory_bytes}"
        )
    check(stats.live_rows == len(DOCS) - 1, f"expected {len(DOCS) - 1} live rows")

    step("drop database")
    client.drop_database(DATABASE)
    state["created"] = False
    print(f"  databases={client.list_databases()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--address", default="127.0.0.1:50051", help="hectord 地址")
    args = parser.parse_args(argv)

    state = {"created": False, "step": None}
    with HectorClient(args.address) as client:
        try:
            run(client, state)
        except Exception as exc:  # noqa: BLE001 - 演示脚本把任何失败都转成退出码 1
            print(
                f"demo failed at step {state['step']!r}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if state["created"]:
                try:
                    client.drop_database(DATABASE)
                except Exception:  # noqa: BLE001 - 清理是尽力而为
                    pass
            return 1
    print("demo finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
