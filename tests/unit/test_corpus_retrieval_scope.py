"""Corpus-wide retrieval scope (D6) and trivial dataset marker (D5)."""

import pytest

from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.eval.trivial_dataset import is_trivial, trivial_dataset_for_docs
from chunktuner.models import ChunkConfig, Document, EvalDataset, EvalMetrics, EvalQuery


class KeywordEmbedding:
    """Deterministic 2-d embedding: (count of 'banana', count of 'apple') + small epsilon."""

    profile_name = "keyword-test"

    def _vec(self, text: str) -> list[float]:
        low = text.lower()
        return [low.count("banana") + 0.01, low.count("apple") + 0.01]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _one_chunk_cfg() -> ChunkConfig:
    return ChunkConfig(name="fixed_tokens", params={"max_tokens": 512, "overlap_tokens": 0})


def test_single_doc_corpus_metrics_equal_document_metrics() -> None:
    doc = Document(id="d1", content="The quick brown fox jumps. " * 60, content_type="text")
    ds = trivial_dataset_for_docs([doc], span_chars=80)
    cfg = ChunkConfig(name="fixed_tokens", params={"max_tokens": 40, "overlap_tokens": 0})
    m = Evaluator(DummyEmbeddingFunction()).evaluate(FixedTokenStrategy(), cfg, [doc], ds).metrics
    assert m.corpus_mrr == pytest.approx(m.mrr)
    assert m.corpus_recall_at_k == pytest.approx(m.recall_at_k)
    assert m.corpus_ndcg_at_k == pytest.approx(m.ndcg_at_k)


def test_distractor_doc_lowers_corpus_mrr() -> None:
    target = Document(id="a", content="banana apple apple apple.", content_type="text")
    distractor = Document(id="b", content="banana banana banana.", content_type="text")
    ds = EvalDataset(
        name="x",
        queries=[
            EvalQuery(
                id="q1",
                question="banana",
                document_id="a",
                answer_spans=[(0, len(target.content))],
            )
        ],
        source="user_provided",
    )
    ev = Evaluator(KeywordEmbedding())
    res = ev.evaluate(FixedTokenStrategy(), _one_chunk_cfg(), [target, distractor], ds)
    m = res.metrics
    assert m.mrr == pytest.approx(1.0)
    assert m.recall_at_k[1] == pytest.approx(1.0)
    assert m.corpus_mrr == pytest.approx(0.5)
    assert m.corpus_mrr < m.mrr
    assert m.corpus_recall_at_k == pytest.approx({1: 0.0, 3: 1.0, 5: 1.0})

    # Evaluator-level retrieval_scope changes the default scorer, not the metrics.
    res_c = Evaluator(KeywordEmbedding(), retrieval_scope="corpus").evaluate(
        FixedTokenStrategy(), _one_chunk_cfg(), [target, distractor], ds
    )
    assert res_c.metrics == m.model_copy(
        update={"embedding_latency_ms": res_c.metrics.embedding_latency_ms}
    )
    assert res_c.score < res.score


def _metrics() -> EvalMetrics:
    return EvalMetrics(
        token_recall=0.8,
        token_iou=0.4,
        mrr=1.0,
        recall_at_k={1: 1.0, 3: 1.0, 5: 1.0},
        corpus_mrr=0.25,
        corpus_recall_at_k={1: 0.0, 3: 1.0, 5: 1.0},
    )


def test_score_calculator_corpus_scope_uses_corpus_values() -> None:
    m = _metrics()
    doc_sc = ScoreCalculator("rag_qa")
    corpus_sc = ScoreCalculator("rag_qa", retrieval_scope="corpus")
    w = doc_sc.weights
    # Only the mrr contribution differs (0.30 * (1.0 - 0.25)).
    assert doc_sc.score(m) - corpus_sc.score(m) == pytest.approx(w["mrr"] * (1.0 - 0.25))

    search_doc = ScoreCalculator("search").score(m)
    search_corpus = ScoreCalculator("search", retrieval_scope="corpus").score(m)
    assert search_doc > search_corpus


def test_score_calculator_default_scope_unchanged() -> None:
    m = _metrics()
    expected = ScoreCalculator("rag_qa").score(m.model_copy(update={"corpus_mrr": None}))
    assert ScoreCalculator("rag_qa").score(m) == expected
    assert ScoreCalculator("rag_qa").retrieval_scope == "document"


def test_score_calculator_corpus_scope_missing_values_count_as_zero() -> None:
    m = EvalMetrics(mrr=1.0, recall_at_k={1: 1.0})
    sc = ScoreCalculator("rag_qa", custom_weights={"mrr": 1.0}, retrieval_scope="corpus")
    assert sc.score(m) == 0.0


def test_old_serialized_metrics_still_load() -> None:
    m = EvalMetrics.model_validate({"mrr": 0.5, "recall_at_k": {"1": 1.0}})
    assert m.corpus_mrr is None and m.corpus_recall_at_k is None and m.corpus_ndcg_at_k is None


def test_trivial_dataset_is_marked() -> None:
    doc = Document(id="d1", content="hello world", content_type="text")
    ds = trivial_dataset_for_docs([doc])
    assert ds.source == "trivial"
    assert is_trivial(ds)
    assert not is_trivial(ds.model_copy(update={"source": "user_provided"}))
