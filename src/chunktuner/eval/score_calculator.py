"""Weighted scalar score from ``EvalMetrics`` for a use case."""

from __future__ import annotations

from chunktuner.config import score_profile_weights
from chunktuner.models import EvalMetrics, RetrievalScope


class ScoreCalculator:
    """Weighted sum of `EvalMetrics` fields for a use-case profile.

    ``retrieval_scope`` selects which ranking feeds the rank-based keys: with ``"corpus"``,
    ``mrr`` reads ``corpus_mrr`` and ``recall_at_1`` reads ``corpus_recall_at_k[1]``
    (missing corpus values count as 0.0). No weight profile references nDCG.
    ``token_recall`` / ``token_iou`` / ``token_precision`` / ``duplication_ratio`` and the
    length metrics are always document-based: they measure what fits in the context budget
    for the query's own document, not ranking against distractors.
    """

    def __init__(
        self,
        use_case: str,
        custom_weights: dict[str, float] | None = None,
        *,
        retrieval_scope: RetrievalScope = "document",
    ):
        self.use_case = use_case
        self.retrieval_scope: RetrievalScope = retrieval_scope
        self.user_custom_weights: dict[str, float] | None = (
            dict(custom_weights) if custom_weights is not None else None
        )
        self.weights = dict(custom_weights) if custom_weights else score_profile_weights(use_case)
        if custom_weights is not None:
            pos = sum(w for w in self.weights.values() if w > 0)
            if pos <= 0:
                raise ValueError(
                    f"custom_weights must contain at least one positive weight; got {self.weights}"
                )

    def score(self, metrics: EvalMetrics) -> float:
        """Return a scalar score from weighted metric contributions (missing keys skipped)."""
        total = 0.0
        for key, w in self.weights.items():
            val = _metric_value(metrics, key, self.retrieval_scope)
            if val is None:
                continue
            total += w * float(val)
        return total


def _corpus_metric_value(m: EvalMetrics, key: str) -> float | None:
    """Corpus-scope value for a rank-based key, or ``None`` if ``key`` is not rank-based."""
    if key == "mrr":
        return m.corpus_mrr if m.corpus_mrr is not None else 0.0
    if key == "recall_at_1":
        return (m.corpus_recall_at_k or {}).get(1, 0.0)
    return None


def _metric_value(
    m: EvalMetrics, key: str, retrieval_scope: RetrievalScope = "document"
) -> float | None:
    if retrieval_scope == "corpus":
        corpus_val = _corpus_metric_value(m, key)
        if corpus_val is not None:
            return corpus_val
    if key == "faithfulness":
        return m.faithfulness if m.faithfulness is not None else 0.0
    if key == "answer_relevancy":
        return m.answer_relevancy if m.answer_relevancy is not None else 0.0
    if key == "recall_at_1":
        return m.recall_at_k.get(1, 0.0)
    if key == "token_recall":
        return m.token_recall
    if key == "token_iou":
        return m.token_iou
    if key == "mrr":
        return m.mrr
    if key == "avg_tokens_per_query":
        return m.avg_tokens_per_query
    if key == "duplication_ratio":
        return m.duplication_ratio
    if key == "avg_chunk_length":
        return m.avg_chunk_length
    if key == "chunk_length_std":
        return m.chunk_length_std
    return getattr(m, key, None)
