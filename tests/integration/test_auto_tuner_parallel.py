"""Parallel vs sequential AutoTuner parity."""

from pathlib import Path

import pytest

from chunktuner.cache.embedding_cache import EmbeddingCache
from chunktuner.cache.wrapped_embeddings import CachedEmbeddingFunction
from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.tuner.auto_tuner import AutoTuner


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
