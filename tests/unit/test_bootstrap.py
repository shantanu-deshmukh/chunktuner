"""``build_full_registry`` must only register strategies whose optional deps are importable."""

import sys

import pytest

from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.models import Document
from chunktuner.tuner.auto_tuner import AutoTuner


@pytest.fixture
def no_semchunk(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ``None`` entry in sys.modules makes ``import semchunk`` raise ImportError.
    monkeypatch.setitem(sys.modules, "semchunk", None)


@pytest.mark.usefixtures("no_semchunk")
def test_semantic_not_registered_without_semchunk() -> None:
    reg = build_full_registry()
    assert "semantic" not in reg.names()
    assert "fixed_tokens" in reg.names()


@pytest.mark.usefixtures("no_semchunk")
def test_recommend_default_strategies_without_semchunk() -> None:
    # The LLM-backed ``agentic`` strategy is explicit-only, so the full registry runs offline.
    tuner = AutoTuner(
        build_full_registry(), Evaluator(DummyEmbeddingFunction()), ScoreCalculator("rag_qa")
    )
    para = "Chunking splits documents into retrievable pieces. " * 20
    doc = Document(id="t1", content=f"{para}\n\n{para}", content_type="text")
    rec = tuner.recommend([doc], "rag_qa", content_type="text")
    assert rec.ranked
    assert all(r.strategy_name != "semantic" for r in rec.ranked)
