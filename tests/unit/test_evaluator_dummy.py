"""Evaluator smoke tests with ``DummyEmbeddingFunction``."""

import uuid

from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.eval.trivial_dataset import trivial_dataset_for_docs
from chunktuner.models import ChunkConfig, Document, EvalDataset, EvalQuery


def test_evaluate_returns_result() -> None:
    doc = Document(
        id=str(uuid.uuid4()),
        content="The quick brown fox jumps over the lazy dog. " * 30,
        content_type="text",
    )
    ds = trivial_dataset_for_docs([doc], span_chars=50)
    ev = Evaluator(DummyEmbeddingFunction(), top_k=3)
    strat = FixedTokenStrategy()
    cfg = ChunkConfig(name="fixed_tokens", params={"max_tokens": 40, "overlap_tokens": 0})
    scorer = ScoreCalculator("rag_qa")
    res = ev.evaluate(strat, cfg, [doc], ds, scorer=scorer)
    assert res.strategy_name == "fixed_tokens"
    assert res.score == res.score  # numeric
    assert res.metrics.token_recall >= 0.0


def test_evaluator_skips_unknown_document_query() -> None:
    doc = Document(id="d1", content="hello world " * 20, content_type="text")
    ds = EvalDataset(
        name="x",
        queries=[
            EvalQuery(
                id="q1",
                question="?",
                document_id="missing",
                answer_spans=[(0, 5)],
            )
        ],
        source="user_provided",
    )
    ev = Evaluator(DummyEmbeddingFunction())
    strat = FixedTokenStrategy()
    cfg = ChunkConfig(name="fixed_tokens", params={"max_tokens": 20, "overlap_tokens": 0})
    res = ev.evaluate(strat, cfg, [doc], ds, scorer=ScoreCalculator("rag_qa"))
    assert res.metrics.token_recall == 0.0
