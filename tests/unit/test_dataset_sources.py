"""Dataset resolution helpers, LLM-dataset cost line, exact embedding cache, mp scope."""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunktuner.cache.embedding_cache import EmbeddingCache
from chunktuner.cache.wrapped_embeddings import CachedEmbeddingFunction, maybe_cache_embeddings
from chunktuner.eval.cost_estimator import CostEstimator, estimate_dataset_generation
from chunktuner.eval.dataset_sources import (
    align_user_dataset,
    generated_dataset_cache_path,
    load_or_generate_dataset,
)
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.models import Document, EvalDataset, EvalQuery
from chunktuner.tuner.mp_worker import mp_evaluate_task


def _docs(prefix: str = "id") -> list[Document]:
    return [
        Document(
            id=f"{prefix}{i}",
            content=f"Document {i} talks about topic {i}. " * 30,
            content_type="text",
            path=f"/corpus/sub/f{i}.txt",
        )
        for i in range(3)
    ]


def _q(doc_id: str) -> EvalQuery:
    return EvalQuery(id=f"q-{doc_id}", question="?", document_id=doc_id, answer_spans=[(0, 10)])


def test_align_user_dataset_matches_paths_and_names() -> None:
    docs = _docs()
    ds = EvalDataset(
        name="u",
        queries=[_q("sub/f0.txt"), _q("f1.txt"), _q("/corpus/sub/f2.txt"), _q("missing.txt")],
        source="user_provided",
    )
    out, warnings = align_user_dataset(ds, docs, Path("/corpus"))
    assert [q.document_id for q in out.queries] == ["id0", "id1", "id2"]
    assert any("Skipped 1" in w for w in warnings)


def test_align_user_dataset_no_match_raises() -> None:
    ds = EvalDataset(name="u", queries=[_q("x")], source="user_provided")
    with pytest.raises(ValueError, match="reference an ingested document"):
        align_user_dataset(ds, _docs(), None)


def test_cache_path_depends_on_content_and_settings(tmp_path: Path) -> None:
    a = generated_dataset_cache_path(tmp_path, _docs("a"), "m1")
    assert a == generated_dataset_cache_path(tmp_path, _docs("b"), "m1")  # ids don't matter
    assert a != generated_dataset_cache_path(tmp_path, _docs(), "m2")
    assert a != generated_dataset_cache_path(tmp_path, _docs(), "m1", questions_per_doc=2)
    assert a != generated_dataset_cache_path(tmp_path, _docs(), "m1", max_queries=10)
    assert a != generated_dataset_cache_path(tmp_path, _docs()[:2], "m1")


def test_generated_dataset_remapped_to_new_doc_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    calls: list[int] = []

    def fake_completion(*, messages: list[dict], **_: object) -> SimpleNamespace:
        calls.append(1)
        doc = messages[0]["content"].split("DOCUMENT:\n", 1)[1]
        body = json.dumps(
            {"queries": [{"question": "q", "start": 0, "end": 30, "reference_answer": doc[:30]}]}
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    first_docs = _docs("run1-")
    order = [d.id for d in first_docs]
    ds1, path, hit1 = load_or_generate_dataset(first_docs, llm_model="m", cache_dir=tmp_path)
    assert not hit1 and path.is_file() and len(calls) == 3
    assert [d.id for d in first_docs] == order  # caller's list not shuffled

    second_docs = _docs("run2-")
    ds2, path2, hit2 = load_or_generate_dataset(
        second_docs,
        llm_model="m",
        cache_dir=tmp_path,
        before_generate=lambda: pytest.fail("must not regenerate on cache hit"),
    )
    assert hit2 and path2 == path and len(calls) == 3
    assert {q.document_id for q in ds2.queries} == {d.id for d in second_docs}
    by_old = {q.id: q for q in ds1.queries}
    for q in ds2.queries:
        assert q.answer_spans == by_old[q.id].answer_spans


def test_generation_with_no_queries_raises_and_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    def fake_completion(**_: object) -> SimpleNamespace:
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    with pytest.raises(ValueError, match="no valid queries"):
        load_or_generate_dataset(_docs(), llm_model="m", cache_dir=tmp_path)
    assert not (tmp_path / "datasets").exists()


def test_cost_estimator_llm_fields_only_with_llm_model() -> None:
    docs = _docs()
    legacy = CostEstimator().estimate(docs, ["fixed_tokens"], {}, "emb")
    assert legacy.llm_model is None and legacy.llm_dataset_tokens is None
    est = CostEstimator().estimate(docs, ["fixed_tokens"], {}, "emb", llm_model="some/llm")
    tokens, cost = estimate_dataset_generation(docs, "some/llm")
    assert est.llm_model == "some/llm"
    assert est.llm_dataset_tokens == tokens > 0
    assert est.llm_cost_usd == cost > 0
    off = CostEstimator().estimate(
        docs, ["fixed_tokens"], {}, "emb", generate_dataset=False, llm_model="some/llm"
    )
    assert off.llm_dataset_tokens is None and off.llm_cost_usd == 0.0


def test_embedding_cache_round_trips_exactly(tmp_path: Path) -> None:
    vec = [0.1234567890123, -2.0000000001, 1e-12]
    with EmbeddingCache(tmp_path / "e.sqlite", "m") as c:
        c.set("t", vec)
        assert c.get("t") == vec


def test_embedding_cache_reads_legacy_float32_rows(tmp_path: Path) -> None:
    db = tmp_path / "e.sqlite"
    with EmbeddingCache(db, "m") as c:
        key = c._key("old")
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO embeddings (k, dim, vec) VALUES (?, ?, ?)",
        (key, 2, struct.pack("2f", 0.5, 0.25)),
    )
    conn.commit()
    conn.close()
    with EmbeddingCache(db, "m") as c:
        assert c.get("old") == [0.5, 0.25]


def test_maybe_cache_embeddings_skips_dummy_and_disabled(tmp_path: Path) -> None:
    dummy = DummyEmbeddingFunction()
    assert maybe_cache_embeddings(dummy, tmp_path) is dummy

    class Real:
        profile_name = "real/model"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    real = Real()
    assert maybe_cache_embeddings(real, tmp_path, enabled=False) is real
    wrapped = maybe_cache_embeddings(real, tmp_path)
    assert isinstance(wrapped, CachedEmbeddingFunction)
    assert wrapped.profile_name == "real/model"


def test_mp_worker_honours_score_retrieval_scope() -> None:
    docs = _docs()
    task = {
        "strategy_name": "fixed_tokens",
        "config": {"name": "fixed_tokens", "params": {"max_tokens": 32, "overlap_tokens": 0}},
        "docs": [d.model_dump(mode="json") for d in docs],
        "dataset": EvalDataset(
            name="d", queries=[_q(d.id) for d in docs], source="trivial"
        ).model_dump(mode="json"),
        "use_case": "search",
    }
    from chunktuner.eval.score_calculator import ScoreCalculator
    from chunktuner.models import EvalResult

    doc_res = EvalResult.model_validate(mp_evaluate_task(task))
    corp_res = EvalResult.model_validate(
        mp_evaluate_task({**task, "score_retrieval_scope": "corpus", "retrieval_scope": "corpus"})
    )
    corp_sc = ScoreCalculator("search", retrieval_scope="corpus")
    assert corp_res.score == pytest.approx(corp_sc.score(corp_res.metrics))
    assert doc_res.score == pytest.approx(ScoreCalculator("search").score(doc_res.metrics))
