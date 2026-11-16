"""BM25Encoder 的公式与输出形状（design 第 8 节）。"""

import math

from hector_client.bm25 import BM25Encoder


def _dot(a, b):
    """两个 (indices, values) 稀疏向量的内积。"""
    weights = dict(zip(a[0], a[1]))
    return sum(weights.get(i, 0.0) * v for i, v in zip(b[0], b[1]))


def test_bm25_dot_equals_formula():
    corpus = ["a b", "a c c"]
    encoder = BM25Encoder(k1=1.2, b=0.75)
    encoder.fit(corpus)

    docs = encoder.encode_documents(corpus)
    query = encoder.encode_query("c")
    score = _dot(query, docs[1])

    # 手算：N=2，df(c)=1，tf=2，dl=3，avgdl=(2+3)/2=2.5。
    n, df, tf, dl, avgdl, k1, b = 2, 1, 2, 3, 2.5, 1.2, 0.75
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    assert abs(score - expected) < 1e-9
    # 第 1 篇文档不含 c，内积为 0。
    assert _dot(query, docs[0]) == 0.0


def test_bm25_unknown_terms_and_order():
    corpus = ["Zeta alpha beta", "gamma alpha delta delta", "beta epsilon"]
    encoder = BM25Encoder()
    encoder.fit(corpus)

    assert encoder.encode_query("unknown words only") == ([], [])

    for indices, values in encoder.encode_documents(corpus):
        assert len(indices) == len(values) > 0
        assert all(x < y for x, y in zip(indices, indices[1:]))
        assert all(v > 0 for v in values)
    q_indices, q_values = encoder.encode_query("delta ALPHA zeta delta")
    assert all(x < y for x, y in zip(q_indices, q_indices[1:]))
    assert q_values == [1.0] * 3
