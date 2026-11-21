"""``build_full_registry`` must only register strategies whose optional deps are importable."""

import sys

import pytest

from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.chunking.registry import StrategyRegistry
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
    # Drop the LLM-backed strategy (it needs an API key); everything else runs by default.
    reg = StrategyRegistry()
    for strat in build_full_registry().list():
        if strat.name != "agentic":
            reg.register(strat)
    tuner = AutoTuner(reg, Evaluator(DummyEmbeddingFunction()), ScoreCalculator("rag_qa"))
    para = "Chunking splits documents into retrievable pieces. " * 20
    doc = Document(id="t1", content=f"{para}\n\n{para}", content_type="text")
    rec = tuner.recommend([doc], "rag_qa", content_type="text")
    assert rec.ranked
    assert all(r.strategy_name != "semantic" for r in rec.ranked)
