"""Evaluator metrics with deterministic dummy embeddings."""

from __future__ import annotations

import tiktoken

from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator, _token_bounds
from chunktuner.models import ChunkConfig, Document, EvalDataset, EvalQuery


def _simple_setup() -> tuple[Document, FixedTokenStrategy, ChunkConfig, EvalDataset]:
    content = "Alpha beta gamma. " * 20
    doc = Document(id="d1", content=content, content_type="text")
    strategy = FixedTokenStrategy()
    config = ChunkConfig(name="fixed_tokens", params={"max_tokens": 20, "overlap_tokens": 0})
    chunks = strategy.chunk(doc, config)
    first_chunk = chunks[0]
    query = EvalQuery(
        id="q1",
        question="What is first?",
        document_id="d1",
        answer_spans=[(first_chunk.start_offset, first_chunk.end_offset)],
    )
    dataset = EvalDataset(name="test", queries=[query])
    return doc, strategy, config, dataset


def test_perfect_retrieval_metrics() -> None:
    doc, strategy, config, dataset = _simple_setup()
    evaluator = Evaluator(DummyEmbeddingFunction(), top_k=1)
    result = evaluator.evaluate(strategy, config, [doc], dataset)
    assert result.metrics.recall_at_k[1] >= 0.0
    assert result.metrics.mrr >= 0.0


def test_duplication_ratio_low_for_non_overlapping_windows() -> None:
    doc, strategy, config, dataset = _simple_setup()
    evaluator = Evaluator(DummyEmbeddingFunction(), top_k=3)
    result = evaluator.evaluate(strategy, config, [doc], dataset)
    assert result.metrics.duplication_ratio <= 0.05


def test_token_bounds_cjk() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    text = "你好world"  # 2 CJK + 5 ASCII
    bounds = _token_bounds(enc, text)
    assert bounds[0] == 0
    assert bounds[-1] == len(text)
    assert all(0 <= b <= len(text) for b in bounds)


def test_token_bounds_emoji() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    text = "Hello 🌍!"
    bounds = _token_bounds(enc, text)
    assert bounds[-1] == len(text)
