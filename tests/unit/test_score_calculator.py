"""ScoreCalculator behaviour across use-case profiles."""

import pytest

from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.models import EvalMetrics


def _metrics(**kwargs: float | dict[int, float]) -> EvalMetrics:
    defaults: dict = dict(
        token_iou=0.5,
        token_precision=0.6,
        token_recall=0.7,
        recall_at_k={1: 0.4, 3: 0.7, 5: 0.85},
        mrr=0.6,
        ndcg_at_k={3: 0.65, 5: 0.75},
        avg_tokens_per_query=400.0,
        duplication_ratio=0.1,
        avg_chunk_length=350.0,
        chunk_length_std=50.0,
        embedding_latency_ms=10.0,
        total_embedding_tokens=1000,
    )
    defaults.update(kwargs)
    return EvalMetrics(**defaults)


@pytest.mark.parametrize("use_case", ["rag_qa", "search", "summarization", "code_assist"])
def test_higher_recall_scores_higher(use_case: str) -> None:
    scorer = ScoreCalculator(use_case=use_case)
    good = scorer.score(_metrics(token_recall=0.9, mrr=0.85))
    bad = scorer.score(_metrics(token_recall=0.3, mrr=0.2))
    assert good > bad


@pytest.mark.parametrize("use_case", ["rag_qa", "search", "summarization", "code_assist"])
def test_high_duplication_penalised(use_case: str) -> None:
    scorer = ScoreCalculator(use_case=use_case)
    clean = scorer.score(_metrics(duplication_ratio=0.05))
    noisy = scorer.score(_metrics(duplication_ratio=0.9))
    assert clean > noisy


def test_custom_weights_requires_positive_sum() -> None:
    with pytest.raises(ValueError, match="at least one positive weight"):
        ScoreCalculator("rag_qa", custom_weights={"token_recall": -1.0, "mrr": 0.0})
