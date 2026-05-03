"""Embedding cache and wrapped embedder behaviour."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from chunktuner.cache.embedding_cache import EmbeddingCache
from chunktuner.cache.wrapped_embeddings import CachedEmbeddingFunction
from chunktuner.eval.embeddings import DummyEmbeddingFunction


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


def _vec_close(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(
        math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-6) for x, y in zip(a, b, strict=True)
    )


def test_embedding_cache_hit_miss(tmp_db: Path) -> None:
    with EmbeddingCache(tmp_db, "test-model") as cache:
        assert cache.get("hello") is None
        cache.set("hello", [0.1, 0.2, 0.3])
        assert _vec_close(cache.get("hello") or [], [0.1, 0.2, 0.3])


def test_embedding_cache_model_isolation(tmp_db: Path) -> None:
    with EmbeddingCache(tmp_db, "model-a") as a:
        a.set("hello", [1.0])
    with EmbeddingCache(tmp_db, "model-b") as b:
        assert b.get("hello") is None


def test_cached_fn_returns_correct_length(tmp_db: Path) -> None:
    """Mixed hit/miss must return len(texts) vectors (regression for partial cache)."""
    inner = DummyEmbeddingFunction()
    with EmbeddingCache(tmp_db, "dummy") as cache:
        wrapped = CachedEmbeddingFunction(inner, cache)
        texts = ["a", "b", "c"]
        first = wrapped.embed_documents(texts)
        assert len(first) == 3
        second = wrapped.embed_documents(texts)
        assert len(second) == 3
        for a, b in zip(first, second, strict=True):
            assert _vec_close(a, b)
        mixed = wrapped.embed_documents(["a", "new_text", "c"])
        assert len(mixed) == 3


def test_cached_embedding_invariant_not_stripped_under_optimization(tmp_path: Path) -> None:
    """Invariant check raises RuntimeError instead of assert when slots stay None."""
    cache = EmbeddingCache(tmp_path / "e.sqlite", "dummy")
    inner = DummyEmbeddingFunction()
    wrapped = CachedEmbeddingFunction(inner, cache)
    result = wrapped.embed_documents(["hello", "world"])
    assert len(result) == 2


def test_cache_closed_on_exception(tmp_db: Path) -> None:
    try:
        with EmbeddingCache(tmp_db, "model") as cache:
            cache.set("k", [1.0])
            raise RuntimeError("simulated")
    except RuntimeError:
        pass
    with EmbeddingCache(tmp_db, "model") as cache:
        assert _vec_close(cache.get("k") or [], [1.0])
