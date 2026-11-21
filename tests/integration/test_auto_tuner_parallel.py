"""Parallel vs sequential AutoTuner parity."""

from pathlib import Path

import pytest

from chunktuner.cache.embedding_cache import EmbeddingCache
from chunktuner.cache.wrapped_embeddings import CachedEmbeddingFunction
from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.rerankers import DummyReranker
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.tuner.auto_tuner import AutoTuner
from tests._extras import require_extra


@pytest.fixture
def fixture_docs(tmp_path: Path):
    for i in range(3):
        (tmp_path / f"doc{i}.txt").write_text("The quick brown fox. " * 100)
    return FileIngestor(root=tmp_path).ingest_dir(tmp_path)


def test_parallel_equals_sequential(fixture_docs) -> None:
    registry = build_full_registry()
    embedding_fn = DummyEmbeddingFunction()

    def make_tuner() -> AutoTuner:
        return AutoTuner(
            strategies=registry,
            evaluator=Evaluator(embedding_fn, top_k=3),
            scorer=ScoreCalculator(use_case="rag_qa"),
        )

    small_grid = {
        "fixed_tokens": [{"max_tokens": 64, "overlap_tokens": 0}],
        "recursive_character": [{"chunk_size_chars": 200, "chunk_overlap_chars": 0}],
    }

    result_seq = make_tuner().recommend(
        fixture_docs,
        use_case="rag_qa",
        strategies=["fixed_tokens", "recursive_character"],
        param_grid=small_grid,
        max_docs=3,
        parallel=False,
        baseline=False,
    )
    result_par = make_tuner().recommend(
        fixture_docs,
        use_case="rag_qa",
        strategies=["fixed_tokens", "recursive_character"],
        param_grid=small_grid,
        max_docs=3,
        parallel=True,
        max_workers=2,
        baseline=False,
    )

    assert len(result_seq.ranked) == len(result_par.ranked)
    assert result_seq.best.strategy_name == result_par.best.strategy_name
    for r_seq, r_par in zip(
        sorted(result_seq.ranked, key=lambda r: r.strategy_name),
        sorted(result_par.ranked, key=lambda r: r.strategy_name),
        strict=True,
    ):
        assert abs(r_seq.score - r_par.score) < 1e-6, (
            f"Score mismatch for {r_seq.strategy_name}: seq={r_seq.score}, par={r_par.score}"
        )


def test_parallel_warm_cache_matches_sequential_with_sqlite(fixture_docs, tmp_path: Path) -> None:
    """B4-1: optional warm_cache pre-embeds so workers reuse SQLite-backed vectors."""
    registry = build_full_registry()
    db_path = tmp_path / "embed_cache.sqlite"
    small_grid = {"fixed_tokens": [{"max_tokens": 64, "overlap_tokens": 0}]}
    rec_kw = {
        "use_case": "rag_qa",
        "strategies": ["fixed_tokens"],
        "param_grid": small_grid,
        "max_docs": 3,
        "baseline": False,
    }

    with EmbeddingCache(db_path, "dummy/test") as cache:
        embed_fn = CachedEmbeddingFunction(DummyEmbeddingFunction(), cache)
        tuner_seq = AutoTuner(
            strategies=registry,
            evaluator=Evaluator(embed_fn, top_k=3),
            scorer=ScoreCalculator(use_case="rag_qa"),
        )
        result_seq = tuner_seq.recommend(fixture_docs, **rec_kw, parallel=False)

    with EmbeddingCache(db_path, "dummy/test") as cache:
        embed_fn = CachedEmbeddingFunction(DummyEmbeddingFunction(), cache)
        tuner_par = AutoTuner(
            strategies=registry,
            evaluator=Evaluator(embed_fn, top_k=3),
            scorer=ScoreCalculator(use_case="rag_qa"),
        )
        result_par = tuner_par.recommend(
            fixture_docs,
            **rec_kw,
            parallel=True,
            max_workers=2,
            warm_cache=True,
        )

    assert result_seq.best.strategy_name == result_par.best.strategy_name
    assert abs(result_seq.best.score - result_par.best.score) < 1e-6


def test_parallel_equals_sequential_corpus_scope(tmp_path: Path) -> None:
    """retrieval_scope reaches the worker's Evaluator and ScoreCalculator (mp_worker path)."""
    for i, topic in enumerate(["apples", "rivers", "stars"]):
        (tmp_path / f"d{i}.txt").write_text(f"Sentence about {topic} number {i}. " * 60)
    docs = FileIngestor(root=tmp_path).ingest_dir(tmp_path)
    registry = build_full_registry()

    def make_tuner() -> AutoTuner:
        return AutoTuner(
            strategies=registry,
            evaluator=Evaluator(DummyEmbeddingFunction(), retrieval_scope="corpus"),
            scorer=ScoreCalculator("search", retrieval_scope="corpus"),
        )

    grid = {"fixed_tokens": [{"max_tokens": 32, "overlap_tokens": 0}]}
    kw = {"strategies": ["fixed_tokens"], "param_grid": grid, "baseline": False}
    seq = make_tuner().recommend(docs, "search", **kw)
    par = make_tuner().recommend(docs, "search", parallel=True, max_workers=2, **kw)
    assert [r.score for r in seq.ranked] == pytest.approx([r.score for r in par.ranked])
    corp = ScoreCalculator("search", retrieval_scope="corpus")
    for r in par.ranked:
        assert r.score == pytest.approx(corp.score(r.metrics))


class _InlinePool:
    """Stand-in for ProcessPoolExecutor: runs tasks in-process and records them."""

    tasks: list[dict] = []

    def __init__(self, max_workers: int | None = None) -> None:
        pass

    def __enter__(self) -> "_InlinePool":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def submit(self, fn, task):  # type: ignore[no-untyped-def]
        from concurrent.futures import Future

        _InlinePool.tasks.append(task)
        fut: Future = Future()
        fut.set_result(fn(task))
        return fut


@pytest.mark.parametrize("encoding", [None, "o200k_base"])
def test_parallel_tasks_carry_encoding(fixture_docs, monkeypatch, encoding) -> None:
    import chunktuner.tuner.auto_tuner as auto_tuner_mod

    monkeypatch.setattr(auto_tuner_mod, "ProcessPoolExecutor", _InlinePool)
    _InlinePool.tasks = []
    kw = {} if encoding is None else {"encoding_name": encoding}
    tuner = AutoTuner(
        build_full_registry(),
        Evaluator(DummyEmbeddingFunction(), top_k=3),
        ScoreCalculator(use_case="rag_qa"),
        **kw,
    )
    rec = tuner.recommend(
        fixture_docs,
        use_case="rag_qa",
        strategies=["fixed_tokens"],
        param_grid={"fixed_tokens": [{"max_tokens": 64, "overlap_tokens": 0}]},
        parallel=True,
        baseline=False,
    )
    assert rec.ranked
    assert _InlinePool.tasks
    assert {t["encoding"] for t in _InlinePool.tasks} == {encoding or "cl100k_base"}


def _topic_docs(tmp_path: Path, text: str = "Sentence about {topic} number {i}. ") -> list:
    for i, topic in enumerate(["apples", "rivers", "stars"]):
        (tmp_path / f"d{i}.txt").write_text(text.format(topic=topic, i=i) * 60)
    return FileIngestor(root=tmp_path).ingest_dir(tmp_path)


def _assert_parallel_matches_sequential(docs, make_evaluator, monkeypatch=None) -> list[dict]:
    """Run a tiny grid sequentially and via the worker; metrics and scores must match."""
    registry = build_full_registry()

    def make_tuner() -> AutoTuner:
        return AutoTuner(registry, make_evaluator(), ScoreCalculator("rag_qa"))

    grid = {
        "fixed_tokens": [{"max_tokens": 32, "overlap_tokens": 0}],
        "recursive_character": [{"chunk_size_chars": 150, "chunk_overlap_chars": 0}],
    }
    kw = {"strategies": list(grid), "param_grid": grid, "baseline": False}
    seq = make_tuner().recommend(docs, "rag_qa", **kw)
    _InlinePool.tasks = []
    if monkeypatch is not None:
        import chunktuner.tuner.auto_tuner as auto_tuner_mod

        monkeypatch.setattr(auto_tuner_mod, "ProcessPoolExecutor", _InlinePool)
    par = make_tuner().recommend(docs, "rag_qa", parallel=True, max_workers=2, **kw)
    by_name_seq = {r.strategy_name: r for r in seq.ranked}
    by_name_par = {r.strategy_name: r for r in par.ranked}
    assert by_name_seq.keys() == by_name_par.keys()
    for name, r_seq in by_name_seq.items():
        r_par = by_name_par[name]
        assert r_seq.score == pytest.approx(r_par.score), name
        m_seq, m_par = (
            {k: v for k, v in r.metrics.model_dump().items() if "latency" not in k}
            for r in (r_seq, r_par)
        )
        assert m_seq == m_par, name
    return list(_InlinePool.tasks)


def test_parallel_equals_sequential_bm25(tmp_path: Path) -> None:
    """D17: retrieval_mode="bm25" reaches the worker's Evaluator."""
    require_extra("bm25", "bm25s")
    docs = _topic_docs(tmp_path)
    _assert_parallel_matches_sequential(
        docs, lambda: Evaluator(DummyEmbeddingFunction(), top_k=3, retrieval_mode="bm25")
    )


def test_parallel_equals_sequential_hybrid_rerank(tmp_path: Path) -> None:
    """D17: retrieval_mode="hybrid" plus a DummyReranker survive the worker round-trip."""
    require_extra("bm25", "bm25s")
    docs = _topic_docs(tmp_path)
    _assert_parallel_matches_sequential(
        docs,
        lambda: Evaluator(
            DummyEmbeddingFunction(),
            top_k=3,
            retrieval_mode="hybrid",
            reranker=DummyReranker(),
            rerank_candidates=4,
        ),
    )


def test_parallel_equals_sequential_o200k_evaluator(tmp_path: Path) -> None:
    """D27: the worker Evaluator counts tokens with the parent's encoding, not cl100k_base."""
    docs = _topic_docs(tmp_path, "Die Quantenchromodynamik über {topic} beschreibt Gluonen {i}. ")
    _assert_parallel_matches_sequential(
        docs,
        lambda: Evaluator(
            DummyEmbeddingFunction(), top_k=3, encoding_name="o200k_base", context_budget_tokens=300
        ),
    )


def test_parallel_tasks_carry_evaluator_fields(tmp_path: Path, monkeypatch) -> None:
    """Task dicts serialise retrieval mode, reranker, encoding, budget and batch size."""
    docs = _topic_docs(tmp_path)
    tasks = _assert_parallel_matches_sequential(
        docs,
        lambda: Evaluator(
            DummyEmbeddingFunction(),
            top_k=3,
            encoding_name="o200k_base",
            context_budget_tokens=300,
            batch_size=7,
            reranker=DummyReranker(),
            rerank_candidates=5,
        ),
        monkeypatch,
    )
    assert tasks
    for t in tasks:
        assert t["retrieval_mode"] == "dense"
        assert t["reranker_spec"] == "dummy"
        assert "reranker_api_key" not in t
        assert t["rerank_candidates"] == 5
        assert t["encoding_name"] == "o200k_base"
        assert t["encoding"] == "cl100k_base"
        assert t["context_budget_tokens"] == 300
        assert t["batch_size"] == 7


def test_parallel_task_defaults(fixture_docs, monkeypatch) -> None:
    """Default Evaluator: worker fields match the Evaluator defaults (no behaviour change)."""
    import chunktuner.tuner.auto_tuner as auto_tuner_mod

    monkeypatch.setattr(auto_tuner_mod, "ProcessPoolExecutor", _InlinePool)
    _InlinePool.tasks = []
    AutoTuner(
        build_full_registry(), Evaluator(DummyEmbeddingFunction()), ScoreCalculator("rag_qa")
    ).recommend(
        fixture_docs,
        "rag_qa",
        strategies=["fixed_tokens"],
        param_grid={"fixed_tokens": [{"max_tokens": 64, "overlap_tokens": 0}]},
        parallel=True,
        baseline=False,
    )
    assert _InlinePool.tasks
    for t in _InlinePool.tasks:
        assert t["retrieval_mode"] == "dense"
        assert t["reranker_spec"] is None
        assert t["rerank_candidates"] == 20
        assert t["encoding_name"] == "cl100k_base"
        assert t["context_budget_tokens"] == 2000
        assert t["batch_size"] == 64


def test_rerank_task_fields_litellm_round_trip() -> None:
    from chunktuner.eval.rerankers import LiteLLMReranker, reranker_from_spec
    from chunktuner.tuner.auto_tuner import _rerank_task_fields

    assert _rerank_task_fields(None) == {"reranker_spec": None}
    fields = _rerank_task_fields(
        LiteLLMReranker("cohere/rerank-v3.5", api_base="http://x", api_key="k")
    )
    assert fields == {
        "reranker_spec": "cohere/rerank-v3.5",
        "reranker_api_base": "http://x",
        "reranker_api_key": "k",
    }
    rebuilt = reranker_from_spec(
        fields["reranker_spec"],
        api_base=fields["reranker_api_base"],
        api_key=fields["reranker_api_key"],
    )
    assert isinstance(rebuilt, LiteLLMReranker)
    assert (rebuilt.model, rebuilt.api_base, rebuilt.api_key) == (
        "cohere/rerank-v3.5",
        "http://x",
        "k",
    )
