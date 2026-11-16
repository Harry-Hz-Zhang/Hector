"""BM25 稀疏向量编码（design 第 8 节）。

文档向量的每个词权重为 `idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))`，
查询向量对出现的已知词给权重 1，于是两者的内积就是该文档对该查询的 BM25 分数。
idf 取 `ln(1 + (N - df + 0.5) / (df + 0.5))`。分词规则是转小写后按 `\\w+` 切分。
"""

import math
import re
from collections import Counter

_TOKEN = re.compile(r"\w+")


def tokenize(text):
    """转小写后按 `\\w+` 切分。"""
    return _TOKEN.findall(text.lower())


class BM25Encoder:
    """把文本编码为 `(indices, values)` 稀疏向量，indices 严格升序。"""

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self._vocab = {}
        self._idf = []
        self._avgdl = 0.0

    @property
    def vocab_size(self):
        """词表大小；可直接作为稀疏列的 `max_dim`。"""
        return len(self._vocab)

    def fit(self, corpus):
        """由语料建立词表（按词的字典序编号）、每个词的 idf 与平均文档词数。"""
        docs = [tokenize(text) for text in corpus]
        if not docs:
            raise ValueError("corpus must not be empty")
        df = Counter(term for tokens in docs for term in set(tokens))
        n = len(docs)
        terms = sorted(df)
        self._vocab = {term: i for i, term in enumerate(terms)}
        self._idf = [math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5)) for term in terms]
        self._avgdl = sum(len(tokens) for tokens in docs) / n
        return self

    def encode_documents(self, texts):
        """返回每篇文档的 `(indices, values)`；不在词表中的词被忽略。"""
        return [self._encode_document(text) for text in texts]

    def encode_query(self, text):
        """返回查询的 `(indices, values)`：每个出现的已知词权重为 1。"""
        indices = sorted({self._vocab[t] for t in tokenize(text) if t in self._vocab})
        return indices, [1.0] * len(indices)

    def _encode_document(self, text):
        tokens = tokenize(text)
        length_norm = 1 - self.b + self.b * len(tokens) / self._avgdl if self._avgdl > 0 else 1.0
        weights = {}
        for term, tf in Counter(tokens).items():
            index = self._vocab.get(term)
            if index is not None:
                weights[index] = (
                    self._idf[index] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
                )
        indices = sorted(weights)
        return indices, [weights[i] for i in indices]
