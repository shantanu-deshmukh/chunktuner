"""Embedding backends — dummy for tests, LiteLLM for live runs."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

_DIM = 256


def _hash_vec(text: str, seed: int = 0) -> list[float]:
    out: list[float] = []
    cur = f"{seed}:{text}".encode()
    while len(out) < _DIM:
        cur = hashlib.sha256(cur).digest()
        for i in range(0, 32, 4):
            out.append(int.from_bytes(cur[i : i + 4], "little") / 2**32)
            if len(out) >= _DIM:
                break
    raw = np.array(out[:_DIM], dtype=np.float64)
    raw = raw / (float(np.linalg.norm(raw)) + 1e-9)
    return raw.tolist()


class DummyEmbeddingFunction:
    """Deterministic pseudo-embeddings for unit tests (no network)."""

    def __init__(self, profile_name: str = "dummy/test"):
        self.profile_name = profile_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vec(t, 0) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _hash_vec(text, 1)


class LiteLLMEmbeddingFunction:
    """LiteLLM-backed embeddings (calls provider APIs)."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        import litellm

        self._litellm = litellm
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.profile_name = model

    def _provider_kwargs(self) -> dict[str, str]:
        kw: dict[str, str] = {}
        if self.api_base:
            kw["api_base"] = self.api_base
        if self.api_key:
            kw["api_key"] = self.api_key
        return kw

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._litellm.embedding(model=self.model, input=texts, **self._provider_kwargs())
        data = sorted(resp["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
    def embed_query(self, text: str) -> list[float]:
        resp = self._litellm.embedding(model=self.model, input=[text], **self._provider_kwargs())
        return list(resp["data"][0]["embedding"])


def embedding_from_workspace(
    model: str | None,
    *,
    prefer_dummy: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
) -> Any:
    if prefer_dummy:
        return DummyEmbeddingFunction()
    if model:
        return LiteLLMEmbeddingFunction(model, api_base=api_base, api_key=api_key)
    return DummyEmbeddingFunction()
