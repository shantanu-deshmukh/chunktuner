"""Reranker backends — dummy for tests, local cross-encoder, and LiteLLM for hosted APIs."""

import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from chunktuner.models import Reranker

DEFAULT_LOCAL_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

_LOCAL_PREFIX = "local:"
_DUMMY_SPEC = "dummy"
_LOCAL_SPEC = "local"
_WORD = re.compile(r"\w+")

_CROSS_ENCODER_MISSING = (
    "Local reranking needs the optional 'late' extra (sentence-transformers): "
    "uv add 'chunktuner[late]' or pip install 'chunktuner[late]'"
)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class DummyReranker:
    """Deterministic lexical-overlap reranker for tests (no network, no dependencies).

    Score = fraction of distinct query words present in the document, plus a tiny bonus for
    total query-word occurrences so documents with equal coverage still separate.
    """

    def __init__(self, profile_name: str = "dummy/rerank", max_input_tokens: int | None = 512):
        self.profile_name = profile_name
        self.max_input_tokens = max_input_tokens

    def score(self, query: str, documents: list[str]) -> list[float]:
        q_words = set(_words(query))
        out: list[float] = []
        for doc in documents:
            d_words = _words(doc)
            if not q_words:
                out.append(0.0)
                continue
            present = q_words & set(d_words)
            occurrences = sum(1 for w in d_words if w in q_words)
            out.append(len(present) / len(q_words) + 1e-6 * occurrences)
        return out


class CrossEncoderReranker:
    """Local sentence-transformers cross-encoder (downloads the model on first use)."""

    def __init__(
        self,
        model: str = DEFAULT_LOCAL_RERANK_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(_CROSS_ENCODER_MISSING) from exc
        self.model = model
        self.batch_size = batch_size
        self._encoder: Any = CrossEncoder(model, device=device)
        self.profile_name = f"{_LOCAL_PREFIX}{model}"
        self.max_input_tokens = self._window(self._encoder)

    @staticmethod
    def _window(encoder: Any) -> int | None:
        # sentence-transformers 6.x: ``max_seq_length`` (``max_length`` deprecated there);
        # 5.x: ``max_length``. Either may be ``None``.
        for attr in ("max_seq_length", "max_length"):
            value = getattr(encoder, attr, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self._encoder.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


class LiteLLMReranker:
    """LiteLLM-backed reranking (calls provider rerank APIs, e.g. ``cohere/rerank-v3.5``)."""

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
        self.max_input_tokens: int | None = None

    def _provider_kwargs(self) -> dict[str, str]:
        kw: dict[str, str] = {}
        if self.api_base:
            kw["api_base"] = self.api_base
        if self.api_key:
            kw["api_key"] = self.api_key
        return kw

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        resp = self._litellm.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=len(documents),
            return_documents=False,
            **self._provider_kwargs(),
        )
        scores = [float("-inf")] * len(documents)
        for item in resp.results or []:
            idx = int(item["index"])
            if 0 <= idx < len(documents):
                scores[idx] = float(item["relevance_score"])
        return scores


def validate_reranker_spec(spec: str) -> str:
    """Check a reranker spec string without importing any backend; returns it stripped.

    Valid specs: ``dummy``, ``local``, ``local:<model id>``, or a LiteLLM rerank model string.
    """
    cleaned = spec.strip()
    if not cleaned:
        raise ValueError("Reranker spec must not be empty or whitespace")
    if cleaned.startswith(_LOCAL_PREFIX) and not cleaned[len(_LOCAL_PREFIX) :].strip():
        raise ValueError("Reranker spec 'local:' needs a model id, e.g. 'local:<model id>'")
    return cleaned


def reranker_from_spec(
    spec: str | None,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
) -> Reranker | None:
    """Build a reranker from a spec string (see `validate_reranker_spec`); ``None``/"" -> None."""
    if spec is None or spec == "":
        return None
    cleaned = validate_reranker_spec(spec)
    if cleaned == _DUMMY_SPEC:
        return DummyReranker()
    if cleaned == _LOCAL_SPEC:
        return CrossEncoderReranker()
    if cleaned.startswith(_LOCAL_PREFIX):
        return CrossEncoderReranker(cleaned[len(_LOCAL_PREFIX) :].strip())
    return LiteLLMReranker(cleaned, api_base=api_base, api_key=api_key)


def reranker_spec(reranker: Reranker | None) -> str | None:
    """Spec string that `reranker_from_spec` turns back into an equivalent reranker."""
    if reranker is None:
        return None
    if isinstance(reranker, DummyReranker):
        return _DUMMY_SPEC
    if isinstance(reranker, CrossEncoderReranker):
        return f"{_LOCAL_PREFIX}{reranker.model}"
    if isinstance(reranker, LiteLLMReranker):
        return reranker.model
    raise ValueError(
        f"Cannot express reranker {type(reranker).__name__!r} as a spec string; "
        "only DummyReranker, CrossEncoderReranker and LiteLLMReranker are supported"
    )
