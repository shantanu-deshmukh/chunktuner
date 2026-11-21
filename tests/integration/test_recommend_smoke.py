"""End-to-end recommend with dummy embeddings."""

import pytest

from chunktuner.chunking import default_registry
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.tuner.auto_tuner import AutoTuner


@pytest.fixture
def fixture_docs(tmp_path):
    for i in range(3):
        (tmp_path / f"doc{i}.txt").write_text("The quick brown fox. " * 80)
    return FileIngestor(root=tmp_path).ingest_dir(tmp_path)


def _make_tuner() -> AutoTuner:
    ev = Evaluator(DummyEmbeddingFunction(), top_k=3)
    return AutoTuner(default_registry, ev, ScoreCalculator("rag_qa"))


def test_auto_tuner_recommend_minimal(tmp_path) -> None:
    f = tmp_path / "note.md"
    f.write_text("# Doc\n\n" + ("Some content. " * 80))
    fi = FileIngestor(root=tmp_path)
    docs = fi.ingest_dir(tmp_path)
    assert len(docs) == 1

    ev = Evaluator(DummyEmbeddingFunction(), top_k=3)
    tuner = AutoTuner(default_registry, ev, ScoreCalculator("rag_qa"))
    rec = tuner.recommend(
        docs,
        "rag_qa",
        strategies=["fixed_tokens"],
        param_grid={"fixed_tokens": [{"max_tokens": 128, "overlap_tokens": 0}]},
        baseline=False,
        max_docs=5,
    )
    assert rec.best.strategy_name
    assert rec.ranked


def test_baseline_is_always_fixed_512(fixture_docs) -> None:
    result = _make_tuner().recommend(
        fixture_docs,
        use_case="rag_qa",
        strategies=["fixed_tokens"],
        param_grid={"fixed_tokens": [{"max_tokens": 128, "overlap_tokens": 0}]},
        max_docs=3,
    )
    assert result.baseline is not None
    assert result.baseline.strategy_name == "fixed_tokens"
    assert result.baseline.config.params.get("max_tokens") == 512


def test_ranked_list_is_sorted_descending(fixture_docs) -> None:
    result = _make_tuner().recommend(
        fixture_docs,
        use_case="rag_qa",
        strategies=["fixed_tokens", "recursive_character"],
        param_grid={
            "fixed_tokens": [{"max_tokens": 64, "overlap_tokens": 0}],
            "recursive_character": [{"chunk_size_chars": 120, "chunk_overlap_chars": 0}],
        },
        max_docs=3,
        baseline=False,
    )
    scores = [r.score for r in result.ranked]
    assert scores == sorted(scores, reverse=True)
