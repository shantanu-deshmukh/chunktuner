"""Lexical (BM25) retrieval and rank fusion for the evaluator's retrieval pipeline."""

from typing import Any

import numpy as np

_BM25_MISSING = (
    "BM25 retrieval needs the optional 'bm25' extra: uv add 'chunktuner[bm25]' "
    "or pip install 'chunktuner[bm25]'"
)


def _tokenize(bm25s: Any, texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(
        texts,
        lower=True,
        stopwords="english",
        stemmer=None,
        return_ids=False,
        show_progress=False,
    )


class Bm25Index:
    """BM25 index over ``texts`` (one entry per chunk); IDF is computed over all of them.

    Texts with no tokens (after lowercasing and English stopword removal) score 0 for every
    query, and a query with no tokens scores 0 against every text.
    """

    def __init__(self, texts: list[str]):
        try:
            import bm25s
        except ImportError as exc:
            raise ImportError(_BM25_MISSING) from exc
        self._bm25s = bm25s
        self._n = len(texts)
        corpus_tokens = _tokenize(bm25s, texts) if texts else []
        self._retriever: Any = None
        # bm25s fails on a corpus where no text has any token (empty vocabulary).
        if any(corpus_tokens):
            self._retriever = bm25s.BM25()
            self._retriever.index(corpus_tokens, show_progress=False)

    def __len__(self) -> int:
        return self._n

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of ``query`` against every indexed text (float64, length ``len(texts)``)."""
        if self._retriever is None:
            return np.zeros(self._n, dtype=np.float64)
        query_tokens = _tokenize(self._bm25s, [query])[0]
        # bm25s raises IndexError on an empty query token list.
        if not query_tokens:
            return np.zeros(self._n, dtype=np.float64)
        return np.asarray(self._retriever.get_scores(query_tokens), dtype=np.float64)


def rrf_fuse(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion of rankings over the same candidate set.

    Each ranking is an array of candidate indices, best first. A candidate's fused score is
    ``sum(1 / (k + rank))`` over the rankings (rank starts at 1). Returns candidate indices
    by fused score, best first; ties keep their order in the first ranking.
    """
    if not rankings:
        return np.array([], dtype=np.int64)
    first = np.asarray(rankings[0], dtype=np.int64)
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(np.asarray(ranking, dtype=np.int64).tolist(), start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    position = {idx: pos for pos, idx in enumerate(first.tolist())}
    tail = len(position)
    order = sorted(fused, key=lambda idx: (-fused[idx], position.get(idx, tail), idx))
    return np.array(order, dtype=np.int64)
