"""D7 / D20: AutoTuner default strategy selection and per-content-type baseline."""

import pytest

from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.models import ChunkConfig, Document
from chunktuner.tuner.auto_tuner import AutoTuner

PROSE = " ".join(f"Paragraph {i} discusses retrieval and chunking tradeoffs." for i in range(200))
CODE = "\n\n".join(
    f"def func_{i}(x):\n    '''Compute value {i}.'''\n    return x * {i} + 1\n" for i in range(60)
)


def _tuner() -> AutoTuner:
    return AutoTuner(
        build_full_registry(), Evaluator(DummyEmbeddingFunction()), ScoreCalculator("rag_qa")
    )


def _ran(rec) -> set[str]:
    return {r.strategy_name for r in rec.ranked}


@pytest.fixture
def text_docs() -> list[Document]:
    return [Document(id="t1", content=PROSE, content_type="text")]


def test_default_recommend_excludes_late_chunking(text_docs) -> None:
    rec = _tuner().recommend(text_docs, "rag_qa")
    assert "late_chunking" not in _ran(rec)
    assert "fixed_tokens" in _ran(rec)


def _fail_if_called(*_a, **_k):
    raise AssertionError("agentic must not run unless named explicitly")


def test_default_recommend_excludes_agentic_without_api_key(text_docs, monkeypatch) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    tuner = _tuner()
    assert "agentic" in tuner.strategies.names("text")
    monkeypatch.setattr(tuner.strategies.get("agentic"), "chunk", _fail_if_called)
    rec = tuner.recommend(text_docs, "rag_qa")
    assert "agentic" not in _ran(rec)
    assert "fixed_tokens" in _ran(rec)


def test_explicit_agentic_is_included(text_docs, monkeypatch) -> None:
    tuner = _tuner()
    fixed = tuner.strategies.get("fixed_tokens")

    def fake_chunk(doc, config):
        return fixed.chunk(doc, ChunkConfig(name="fixed_tokens", params={"max_tokens": 256}))

    monkeypatch.setattr(tuner.strategies.get("agentic"), "chunk", fake_chunk)
    rec = tuner.recommend(text_docs, "rag_qa", strategies=["agentic"])
    assert "agentic" in _ran(rec)


def test_explicit_late_chunking_is_included(text_docs) -> None:
    rec = _tuner().recommend(text_docs, "rag_qa", strategies=["late_chunking"])
    assert "late_chunking" in _ran(rec)


def test_text_baseline_is_fixed_tokens(text_docs) -> None:
    rec = _tuner().recommend(text_docs, "rag_qa", strategies=["recursive_character"])
    assert rec.baseline is not None
    assert rec.baseline.strategy_name == "fixed_tokens"
    assert rec.baseline.config.params == {"max_tokens": 512, "overlap_tokens": 0}


def test_recommend_on_code_uses_code_window_baseline() -> None:
    docs = [Document(id="c1", content=CODE, content_type="code")]
    rec = _tuner().recommend(docs, "code_assist", strategies=["code_window"])
    assert rec.baseline is not None
    assert rec.baseline.strategy_name == "code_window"
    assert rec.baseline.config.params == {"max_tokens": 512, "overlap_lines": 0}
    assert rec.ranked


def test_recommend_on_code_default_strategies_does_not_raise() -> None:
    docs = [Document(id="c1", content=CODE, content_type="code")]
    tuner = _tuner()
    names = [n for n in tuner.strategies.names("code") if n != "agentic"]
    rec = tuner.recommend(docs, "code_assist", strategies=names)
    assert "code_window" in _ran(rec)
