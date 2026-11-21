"""D17: reranker backends and spec helpers (no network, no model downloads)."""

import sys
import types

import pytest
from tenacity import wait_none

from chunktuner.eval import rerankers
from chunktuner.eval.rerankers import (
    DEFAULT_LOCAL_RERANK_MODEL,
    CrossEncoderReranker,
    DummyReranker,
    LiteLLMReranker,
    reranker_from_spec,
    reranker_spec,
    validate_reranker_spec,
)


def test_dummy_reranker_scores_overlap() -> None:
    rr = DummyReranker()
    assert rr.profile_name == "dummy/rerank" and rr.max_input_tokens == 512
    s = rr.score("zebra stripes", ["no match", "zebra here", "zebra stripes zebra"])
    assert s[0] == 0.0
    assert s[2] > s[1] > s[0]
    assert rr.score("zebra", []) == []
    assert rr.score("", ["anything"]) == [0.0]
    assert rr.score("zebra stripes", ["zebra"]) == rr.score("zebra stripes", ["zebra"])


class _FakeResponse:
    def __init__(self, results: list[dict]) -> None:
        self.results = results


class _FakeLiteLLM(types.ModuleType):
    def __init__(self, fail_times: int = 0) -> None:
        super().__init__("litellm")
        self.fail_times = fail_times
        self.calls: list[dict] = []

    def rerank(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("transient")
        # Out of order, and document 1 is missing.
        return _FakeResponse(
            [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]
        )


def test_litellm_reranker_maps_scores_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    rr = LiteLLMReranker("cohere/rerank-test", api_base="http://x", api_key="k")
    assert rr.profile_name == "cohere/rerank-test" and rr.max_input_tokens is None
    scores = rr.score("q", ["a", "b", "c"])
    assert scores == [0.2, float("-inf"), 0.9]
    call = fake.calls[0]
    assert call["model"] == "cohere/rerank-test" and call["documents"] == ["a", "b", "c"]
    assert call["top_n"] == 3 and call["return_documents"] is False
    assert call["api_base"] == "http://x" and call["api_key"] == "k"
    assert rr.score("q", []) == []
    assert len(fake.calls) == 1


def test_litellm_reranker_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(fail_times=1)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setattr(LiteLLMReranker.score.retry, "wait", wait_none())
    rr = LiteLLMReranker("cohere/rerank-test")
    assert rr.score("q", ["a", "b", "c"]) == [0.2, float("-inf"), 0.9]
    assert len(fake.calls) == 2
    assert "api_base" not in fake.calls[0] and "api_key" not in fake.calls[0]


def _fake_st(monkeypatch: pytest.MonkeyPatch, **window: object) -> list[dict]:
    created: list[dict] = []

    class FakeCrossEncoder:
        def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
            created.append({"model": model_name_or_path, "device": device})
            for k, v in window.items():
                setattr(self, k, v)

        def predict(
            self,
            pairs: list[tuple[str, str]],
            batch_size: int = 32,
            show_progress_bar: bool | None = None,
        ) -> list[float]:
            created[-1]["batch_size"] = batch_size
            return [float(len(d)) for _, d in pairs]

    mod = types.ModuleType("sentence_transformers")
    mod.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return created


def test_cross_encoder_reranker_prefers_max_seq_length(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _fake_st(monkeypatch, max_seq_length=256, max_length=999)
    rr = CrossEncoderReranker(device="cpu", batch_size=8)
    assert created[0] == {"model": DEFAULT_LOCAL_RERANK_MODEL, "device": "cpu"}
    assert rr.profile_name == f"local:{DEFAULT_LOCAL_RERANK_MODEL}"
    assert rr.max_input_tokens == 256
    assert rr.score("q", ["ab", "abcd"]) == [2.0, 4.0]
    assert created[0]["batch_size"] == 8
    assert rr.score("q", []) == []


def test_cross_encoder_reranker_window_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_st(monkeypatch, max_length=512)
    assert CrossEncoderReranker("m").max_input_tokens == 512
    _fake_st(monkeypatch, max_seq_length=None, max_length=128)
    assert CrossEncoderReranker("m").max_input_tokens == 128
    _fake_st(monkeypatch, max_seq_length=None, max_length=None)
    assert CrossEncoderReranker("m").max_input_tokens is None
    _fake_st(monkeypatch)
    assert CrossEncoderReranker("m").max_input_tokens is None


def test_cross_encoder_reranker_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"'late' extra"):
        CrossEncoderReranker()


@pytest.mark.parametrize("spec", ["", "   ", "local:", "local:  "])
def test_validate_reranker_spec_rejects(spec: str) -> None:
    with pytest.raises(ValueError):
        validate_reranker_spec(spec)


def test_validate_reranker_spec_accepts() -> None:
    assert validate_reranker_spec(" dummy ") == "dummy"
    assert validate_reranker_spec("local") == "local"
    assert validate_reranker_spec("local:org/model") == "local:org/model"
    assert validate_reranker_spec("cohere/rerank-v3.5") == "cohere/rerank-v3.5"


def test_reranker_from_spec_and_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_st(monkeypatch, max_seq_length=512)
    monkeypatch.setitem(sys.modules, "litellm", _FakeLiteLLM())
    assert reranker_from_spec(None) is None
    assert reranker_from_spec("") is None
    assert reranker_spec(None) is None

    dummy = reranker_from_spec("dummy")
    assert isinstance(dummy, DummyReranker)
    local = reranker_from_spec("local")
    assert isinstance(local, CrossEncoderReranker) and local.model == DEFAULT_LOCAL_RERANK_MODEL
    custom = reranker_from_spec("local:org/model")
    assert isinstance(custom, CrossEncoderReranker) and custom.model == "org/model"
    hosted = reranker_from_spec("cohere/rerank-v3.5", api_key="k")
    assert isinstance(hosted, LiteLLMReranker) and hosted.api_key == "k"

    assert reranker_spec(dummy) == "dummy"
    assert reranker_spec(local) == f"local:{DEFAULT_LOCAL_RERANK_MODEL}"
    assert reranker_spec(custom) == "local:org/model"
    assert reranker_spec(hosted) == "cohere/rerank-v3.5"
    for r in (dummy, local, custom, hosted):
        again = reranker_from_spec(reranker_spec(r))
        assert type(again) is type(r) and again.profile_name == r.profile_name

    with pytest.raises(ValueError):
        reranker_from_spec("local:")
    with pytest.raises(ValueError):
        reranker_spec(object())  # type: ignore[arg-type]


def test_module_exports_default_model() -> None:
    assert rerankers.DEFAULT_LOCAL_RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L6-v2"
