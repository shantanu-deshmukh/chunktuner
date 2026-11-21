"""MCP service impls: dataset_path, generate_dataset, retrieval_scope, content_type, warnings."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunktuner.mcp import service
from chunktuner.mcp.service import (
    evaluate_chunking_impl,
    preview_chunks_impl,
    recommend_config_impl,
)


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "base"
    corpus = base / "corpus"
    corpus.mkdir(parents=True)
    for i, topic in enumerate(["apples", "rivers", "stars"]):
        (corpus / f"doc{i}.txt").write_text(
            " ".join(f"Sentence {j} about {topic}." for j in range(60)), encoding="utf-8"
        )
    monkeypatch.setenv("CHUNK_TUNER_BASE_DIR", str(base))
    monkeypatch.setattr(service, "mcp_cache_dir", lambda: tmp_path / "cache")
    return corpus


def _write_dataset(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "queries": [
                    {"question": "apples?", "document_id": "doc0.txt", "answer_spans": [[0, 50]]}
                ]
            }
        )
    )
    return path


def test_trivial_dataset_warning_in_response(corpus: Path) -> None:
    out = evaluate_chunking_impl(str(corpus), "rag_qa", strategies=["fixed_tokens"])
    assert out["dataset_summary"] == {"queries": 3, "source": "trivial"}
    assert len(out["warnings"]) == 1 and "trivial" in out["warnings"][0]
    rec = recommend_config_impl(str(corpus), "rag_qa", strategies=["fixed_tokens"])
    assert "trivial" in rec["warnings"][0]
    assert rec["best"]


def test_dataset_path_used(corpus: Path) -> None:
    ds = _write_dataset(corpus.parent / "ds.json")
    out = evaluate_chunking_impl(
        str(corpus), "rag_qa", strategies=["fixed_tokens"], dataset_path=str(ds)
    )
    assert out["dataset_summary"] == {"queries": 1, "source": "user_provided"}
    assert not any("trivial" in w for w in out["warnings"])


def test_dataset_path_outside_base_rejected(corpus: Path, tmp_path: Path) -> None:
    outside = _write_dataset(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="CHUNK_TUNER_BASE_DIR"):
        evaluate_chunking_impl(str(corpus), "rag_qa", dataset_path=str(outside))
    with pytest.raises(ValueError, match="CHUNK_TUNER_BASE_DIR"):
        evaluate_chunking_impl(str(corpus), "rag_qa", dataset_path=str(outside), dry_run=True)
    with pytest.raises(ValueError, match="CHUNK_TUNER_BASE_DIR"):
        recommend_config_impl(str(corpus), "rag_qa", dataset_path=str(outside))


def test_dataset_args_mutually_exclusive(corpus: Path) -> None:
    ds = _write_dataset(corpus.parent / "ds.json")
    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluate_chunking_impl(str(corpus), "rag_qa", dataset_path=str(ds), generate_dataset=True)


def test_retrieval_scope_validated(corpus: Path) -> None:
    with pytest.raises(ValueError, match="retrieval_scope"):
        evaluate_chunking_impl(str(corpus), "rag_qa", retrieval_scope="global")


def test_content_type_filters_docs_and_strategies(corpus: Path) -> None:
    (corpus / "mod.py").write_text("def a():\n    return 1\n\n\ndef b():\n    return 2\n")
    # code_window supports only code; markdown_semantic does not support code.
    out = evaluate_chunking_impl(
        str(corpus),
        "rag_qa",
        content_type="code",
        strategies=["markdown_semantic", "code_window"],
    )
    names = {r["strategy_name"] for r in out["results"]}
    assert names == {"code_window"}
    assert out["dataset_summary"]["queries"] == 1  # only the .py doc
    with pytest.raises(ValueError, match="content_type"):
        evaluate_chunking_impl(
            str(corpus), "rag_qa", content_type="code", strategies=["markdown_semantic"]
        )
    with pytest.raises(ValueError, match="No documents with content_type"):
        evaluate_chunking_impl(
            str(corpus), "rag_qa", content_type="html", strategies=["fixed_tokens"]
        )


def test_dry_run_generate_dataset_reports_llm_tokens(corpus: Path) -> None:
    plain = evaluate_chunking_impl(str(corpus), "rag_qa", dry_run=True)
    assert "llm_dataset_tokens" not in plain
    est = evaluate_chunking_impl(str(corpus), "rag_qa", dry_run=True, generate_dataset=True)
    assert est["llm_dataset_tokens"] > 0
    assert est["llm_model"]


def test_dry_run_bm25_has_no_embedding_cost(corpus: Path) -> None:
    from tests._extras import require_extra

    require_extra("bm25", "bm25s")
    dense = evaluate_chunking_impl(str(corpus), "rag_qa", dry_run=True)
    assert dense["total_tokens"] > 0
    est = evaluate_chunking_impl(str(corpus), "rag_qa", dry_run=True, retrieval_mode="bm25")
    assert est["total_tokens"] == 0
    assert est["embedding_cost_usd"] == 0.0
    assert est["strategy_configs"] == dense["strategy_configs"]


def test_generate_dataset_cached(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    calls: list[str] = []

    def fake_completion(*, model: str, messages: list[dict], **_: object) -> SimpleNamespace:
        calls.append(model)
        doc = messages[0]["content"].split("DOCUMENT:\n", 1)[1]
        content = json.dumps(
            {"queries": [{"question": "q?", "start": 0, "end": 40, "reference_answer": doc[:40]}]}
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    out = evaluate_chunking_impl(
        str(corpus), "rag_qa", strategies=["fixed_tokens"], generate_dataset=True
    )
    assert out["dataset_summary"] == {"queries": 3, "source": "llm_generated"}
    n = len(calls)
    assert n == 3
    rec = recommend_config_impl(
        str(corpus), "rag_qa", strategies=["fixed_tokens"], generate_dataset=True
    )
    assert len(calls) == n
    assert rec["best"]


def test_preview_content_type(corpus: Path) -> None:
    code = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    out = preview_chunks_impl(code, "code_window", {}, content_type="code")
    assert out and out[0]["start_offset"] == 0
    # default stays markdown
    md = preview_chunks_impl("# T\n\nbody", "fixed_tokens", {"max_tokens": 16})
    assert md
    with pytest.raises(ValueError):
        preview_chunks_impl("x", "fixed_tokens", {}, content_type="not-a-type")


def test_retrieval_mode_validated(corpus: Path) -> None:
    with pytest.raises(ValueError, match="retrieval_mode"):
        evaluate_chunking_impl(str(corpus), "rag_qa", retrieval_mode="sparse")
    with pytest.raises(ValueError, match="retrieval_mode"):
        evaluate_chunking_impl(str(corpus), "rag_qa", retrieval_mode="sparse", dry_run=True)
    with pytest.raises(ValueError, match="retrieval_mode"):
        recommend_config_impl(str(corpus), "rag_qa", retrieval_mode="sparse")


@pytest.mark.parametrize("spec", ["", "   ", "local:", "local:  "])
def test_reranker_spec_validated(corpus: Path, spec: str) -> None:
    with pytest.raises(ValueError, match="[Rr]eranker spec"):
        evaluate_chunking_impl(str(corpus), "rag_qa", reranker=spec)
    with pytest.raises(ValueError, match="[Rr]eranker spec"):
        evaluate_chunking_impl(str(corpus), "rag_qa", reranker=spec, dry_run=True)
    with pytest.raises(ValueError, match="[Rr]eranker spec"):
        recommend_config_impl(str(corpus), "rag_qa", reranker=spec)


def test_missing_bm25_extra_is_value_error(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "bm25s", None)  # makes ``import bm25s`` raise ImportError
    for mode in ("bm25", "hybrid"):
        with pytest.raises(ValueError, match="'bm25' extra"):
            evaluate_chunking_impl(str(corpus), "rag_qa", retrieval_mode=mode)
        with pytest.raises(ValueError, match="'bm25' extra"):
            recommend_config_impl(str(corpus), "rag_qa", retrieval_mode=mode)


def test_missing_reranker_extra_is_value_error(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ValueError, match="'late' extra"):
        evaluate_chunking_impl(
            str(corpus), "rag_qa", strategies=["fixed_tokens"], reranker="local:some/model"
        )


def test_dense_dummy_reranker_metrics(corpus: Path) -> None:
    out = evaluate_chunking_impl(
        str(corpus), "rag_qa", strategies=["fixed_tokens"], reranker="dummy"
    )
    for r in out["results"]:
        assert r["metrics"]["retrieval_mode"] == "dense"
        assert r["metrics"]["reranker"] == "dummy/rerank"
    plain = evaluate_chunking_impl(str(corpus), "rag_qa", strategies=["fixed_tokens"])
    assert all(r["metrics"]["reranker"] is None for r in plain["results"])


def test_recommend_hybrid_with_dummy_reranker(corpus: Path) -> None:
    from tests._extras import require_extra

    require_extra("bm25", "bm25s")
    rec = recommend_config_impl(
        str(corpus),
        "rag_qa",
        strategies=["fixed_tokens"],
        retrieval_mode="hybrid",
        reranker="dummy",
    )
    assert rec["best"]
    metrics = rec["best"]["metrics"]
    assert metrics["retrieval_mode"] == "hybrid"
    assert metrics["reranker"] == "dummy/rerank"
