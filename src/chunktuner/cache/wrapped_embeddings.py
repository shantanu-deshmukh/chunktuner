"""Wrap an ``EmbeddingFunction`` with SQLite caching."""

from __future__ import annotations

from chunktuner.cache.embedding_cache import EmbeddingCache
from chunktuner.models import EmbeddingFunction


class CachedEmbeddingFunction:
    """Delegates to ``inner`` while reading/writing vectors through `EmbeddingCache`."""

    def __init__(self, inner: EmbeddingFunction, cache: EmbeddingCache):
        self._inner = inner
        self._cache = cache
        self.profile_name = inner.profile_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [self._cache.get(t) for t in texts]
        missing_idx = [i for i, v in enumerate(results) if v is None]
        if missing_idx:
            batch = [texts[i] for i in missing_idx]
            fresh = self._inner.embed_documents(batch)
            for i, vec in zip(missing_idx, fresh, strict=True):
                self._cache.set(texts[i], vec)
                results[i] = vec
        if any(v is None for v in results):
            missing = [i for i, v in enumerate(results) if v is None]
            raise RuntimeError(
                "Cache invariant violated: "
                f"{len(missing)} embedding slot(s) still None after fetch "
                f"(indices: {missing[:5]}). This is a bug in CachedEmbeddingFunction."
            )
        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        vec = self._inner.embed_query(text)
        self._cache.set(text, vec)
        return vec
