"""D17: BM25 index and reciprocal rank fusion (``chunktuner.eval.lexical``)."""

import sys

import numpy as np
import pytest

from chunktuner.eval.lexical import Bm25Index, rrf_fuse
from tests._extras import require_extra


@pytest.fixture
def bm25() -> None:
    require_extra("bm25", "bm25s")


def test_rrf_fuse_known_example() -> None:
    # a=[0,2,1], b=[1,0,2]: 0 -> 1/61+1/62, 1 -> 1/63+1/61, 2 -> 1/62+1/63.
    fused = rrf_fuse([np.array([0, 2, 1]), np.array([1, 0, 2])])
    assert fused.tolist() == [0, 1, 2]


def test_rrf_fuse_ties_follow_first_ranking() -> None:
    # Mirror-image rankings: 3 and 0 tie (1/61+1/64), as do 1 and 2 (1/62+1/63).
    assert rrf_fuse([np.array([3, 1, 2, 0]), np.array([0, 2, 1, 3])]).tolist() == [3, 0, 1, 2]
    assert rrf_fuse([np.array([0, 2, 1, 3]), np.array([3, 1, 2, 0])]).tolist() == [0, 3, 2, 1]


def test_rrf_fuse_single_ranking_is_identity_and_deterministic() -> None:
    r = np.array([4, 0, 3, 1, 2])
    assert rrf_fuse([r]).tolist() == r.tolist()
    a, b = np.array([2, 0, 1]), np.array([0, 1, 2])
    assert rrf_fuse([a, b]).tolist() == rrf_fuse([a, b]).tolist()
    assert rrf_fuse([]).tolist() == []


def test_bm25_scores_keyword_match(bm25: None) -> None:
    idx = Bm25Index(["the banana is yellow", "apple pie recipe", "banana banana bread"])
    s = idx.scores("banana")
    assert s.dtype == np.float64 and s.shape == (3,)
    assert s[1] == 0.0 and s[0] > 0.0 and s[2] > 0.0


def test_bm25_empty_query_and_empty_texts(bm25: None) -> None:
    idx = Bm25Index(["banana bread", "", "the of and"])
    assert idx.scores("the of").tolist() == [0.0, 0.0, 0.0]
    assert idx.scores("").tolist() == [0.0, 0.0, 0.0]
    s = idx.scores("banana")
    assert s[0] > 0.0 and s[1] == 0.0 and s[2] == 0.0


def test_bm25_corpus_without_any_token(bm25: None) -> None:
    idx = Bm25Index(["the", "of and"])
    assert idx.scores("banana").tolist() == [0.0, 0.0]
    assert len(Bm25Index([])) == 0
    assert Bm25Index([]).scores("banana").tolist() == []


def test_bm25_missing_extra_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "bm25s", None)
    with pytest.raises(ImportError, match=r"'bm25' extra"):
        Bm25Index(["x"])
