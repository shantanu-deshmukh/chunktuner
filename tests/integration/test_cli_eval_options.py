"""CLI eval options: --dataset, --generate-dataset, --retrieval-scope, --no-cache.

No real API calls: litellm.completion / litellm.embedding are monkeypatched.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from chunktuner.cli import app
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.models import EvalMetrics

try:
    # Click < 8.2 mixes stderr into stdout unless told otherwise.
    runner = CliRunner(mix_stderr=False)
except TypeError:
    # Click >= 8.2 removed `mix_stderr`; stdout and stderr are always separate.
    runner = CliRunner()

_TRIVIAL_WARNING = "trivial eval dataset"


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return ``(corpus_dir, config_path, cache_dir)`` with three distinct docs."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    topics = ["apples and orchards", "rivers and boats", "stars and telescopes"]
    for i, topic in enumerate(topics):
        body = " ".join(f"Sentence {j} about {topic} number {i}." for j in range(80))
        (corpus / f"doc{i}.txt").write_text(body, encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cfg = tmp_path / ".autochunk.yaml"
    cfg.write_text(yaml.safe_dump({"cache_dir": str(cache_dir), "llm_model": "fake/llm"}))
    return corpus, cfg, cache_dir


def _fake_completion_factory(calls: list[str]):
    def fake_completion(*, model: str, messages: list[dict], **_: object) -> SimpleNamespace:
        calls.append(model)
        prompt = messages[0]["content"]
        doc = prompt.split("DOCUMENT:\n", 1)[1]
        rows = [
            {
                "question": f"What does part {k} say?",
                "start": k * 40,
                "end": k * 40 + 60,
                "reference_answer": doc[k * 40 : k * 40 + 60],
            }
            for k in range(3)
        ]
        content = json.dumps({"queries": rows})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return fake_completion


def _fake_embedding_factory(calls: list[int]):
    dummy = DummyEmbeddingFunction()

    def fake_embedding(*, model: str, input: list[str], **_: object) -> dict:
        calls.append(len(input))
        vecs = dummy.embed_documents(input)
        return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]}

    return fake_embedding


def _scores(stdout: str) -> list[float]:
    return [r["score"] for r in json.loads(stdout)]


def test_dataset_file_used_and_no_trivial_warning(workspace, tmp_path: Path) -> None:
    corpus, cfg, _ = workspace
    ds_file = tmp_path / "ds.yaml"
    ds_file.write_text(
        yaml.safe_dump(
            {
                "name": "mine",
                "queries": [
                    {
                        "question": "Which topic?",
                        "document_id": "doc1.txt",
                        "answer_spans": [[0, 80]],
                    },
                    {
                        "question": "Stars?",
                        "document_id": str(corpus / "doc2.txt"),
                        "answer_spans": [[100, 200]],
                    },
                ],
            }
        )
    )
    res = runner.invoke(
        app,
        [
            "evaluate",
            str(corpus),
            "--strategies",
            "fixed_tokens",
            "--config",
            str(cfg),
            "--dataset",
            str(ds_file),
            "--output-format",
            "json",
        ],
    )
    assert res.exit_code == 0, res.stderr
    assert _TRIVIAL_WARNING not in res.stderr
    assert "only 2 usable queries" in res.stderr
    assert len(json.loads(res.stdout)) == 9


def test_dataset_file_unmatched_documents_errors(workspace, tmp_path: Path) -> None:
    corpus, cfg, _ = workspace
    ds_file = tmp_path / "ds.json"
    ds_file.write_text(
        json.dumps({"queries": [{"question": "q", "document_id": "nope.txt", "answer_spans": []}]})
    )
    res = runner.invoke(
        app,
        ["evaluate", str(corpus), "--config", str(cfg), "--dataset", str(ds_file)],
    )
    assert res.exit_code == 2
    assert "reference an ingested document" in res.stderr


def test_default_prints_trivial_warning_on_stderr(workspace) -> None:
    corpus, cfg, _ = workspace
    res = runner.invoke(
        app,
        [
            "evaluate",
            str(corpus),
            "--strategies",
            "fixed_tokens",
            "--config",
            str(cfg),
            "--output-format",
            "json",
        ],
    )
    assert res.exit_code == 0
    assert _TRIVIAL_WARNING in res.stderr
    json.loads(res.stdout)  # stdout stays pure JSON


@pytest.mark.parametrize("cmd", ["evaluate", "compare", "recommend"])
def test_dataset_flags_mutually_exclusive(workspace, tmp_path: Path, cmd: str) -> None:
    corpus, cfg, _ = workspace
    ds_file = tmp_path / "ds.json"
    ds_file.write_text(json.dumps({"queries": []}))
    res = runner.invoke(
        app,
        [
            cmd,
            str(corpus),
            "--strategies",
            "fixed_tokens",
            "--config",
            str(cfg),
            "--dataset",
            str(ds_file),
            "--generate-dataset",
            "--yes",
        ],
    )
    assert res.exit_code == 2
    assert "mutually exclusive" in res.stderr


def test_generate_dataset_cached_and_reused(workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    corpus, cfg, cache_dir = workspace
    calls: list[str] = []
    monkeypatch.setattr(litellm, "completion", _fake_completion_factory(calls))
    args = [
        "recommend",
        str(corpus),
        "--strategies",
        "fixed_tokens",
        "--config",
        str(cfg),
        "--generate-dataset",
        "--yes",
        "--no-baseline",
        "--output-format",
        "json",
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.stderr
    assert calls and set(calls) == {"fake/llm"}
    assert "Cached new generated dataset (9 queries)" in first.stderr
    assert _TRIVIAL_WARNING not in first.stderr
    cached = list((cache_dir / "datasets").glob("generated_*.json"))
    assert len(cached) == 1

    n_calls = len(calls)
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.stderr
    assert len(calls) == n_calls, "cache hit must not call the LLM again"
    assert "Reusing cached generated dataset" in second.stderr


def test_generate_dataset_prompts_without_yes(workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    corpus, cfg, cache_dir = workspace
    calls: list[str] = []
    monkeypatch.setattr(litellm, "completion", _fake_completion_factory(calls))
    res = runner.invoke(
        app,
        [
            "evaluate",
            str(corpus),
            "--strategies",
            "fixed_tokens",
            "--config",
            str(cfg),
            "--generate-dataset",
        ],
        input="n\n",
    )
    assert res.exit_code != 0
    assert "fake/llm" in res.stdout + res.stderr
    assert calls == []
    assert not (cache_dir / "datasets").exists()


def test_estimate_generate_dataset_adds_llm_fields(workspace) -> None:
    corpus, cfg, _ = workspace
    plain = runner.invoke(app, ["estimate", str(corpus), "--output-format", "json"])
    assert plain.exit_code == 0
    assert "llm_dataset_tokens" not in json.loads(plain.stdout)
    res = runner.invoke(
        app,
        [
            "estimate",
            str(corpus),
            "--config",
            str(cfg),
            "--generate-dataset",
            "--output-format",
            "json",
        ],
    )
    assert res.exit_code == 0
    est = json.loads(res.stdout)
    assert est["llm_model"] == "fake/llm"
    assert est["llm_dataset_tokens"] > 0
    assert est["llm_cost_usd"] > 0


def test_retrieval_scope_corpus_scores_from_corpus_metrics(workspace) -> None:
    corpus, cfg, _ = workspace
    base = [
        "evaluate",
        str(corpus),
        "--strategies",
        "fixed_tokens",
        "--config",
        str(cfg),
        "--output-format",
        "json",
    ]
    doc_res = runner.invoke(app, base)
    corp_res = runner.invoke(app, [*base, "--retrieval-scope", "corpus"])
    assert doc_res.exit_code == 0 and corp_res.exit_code == 0
    doc_rows = json.loads(doc_res.stdout)
    corp_rows = json.loads(corp_res.stdout)
    doc_sc = ScoreCalculator("rag_qa")
    corp_sc = ScoreCalculator("rag_qa", retrieval_scope="corpus")
    for d_row, c_row in zip(doc_rows, corp_rows, strict=True):
        m = EvalMetrics.model_validate(c_row["metrics"])
        assert c_row["score"] == pytest.approx(corp_sc.score(m))
        assert d_row["score"] == pytest.approx(doc_sc.score(m))
    # Three docs → corpus-wide ranking differs from per-document ranking somewhere.
    assert _scores(doc_res.stdout) != _scores(corp_res.stdout)


def test_retrieval_scope_rejects_unknown(workspace) -> None:
    corpus, cfg, _ = workspace
    res = runner.invoke(
        app, ["evaluate", str(corpus), "--config", str(cfg), "--retrieval-scope", "global"]
    )
    assert res.exit_code == 2


def _strip_latency(stdout: str) -> str:
    return re.sub(r'"embedding_latency_ms": [0-9.e-]+', "", stdout)


def test_embedding_cache_default_on_and_no_cache(
    workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm

    corpus, cfg, cache_dir = workspace
    calls: list[int] = []
    monkeypatch.setattr(litellm, "embedding", _fake_embedding_factory(calls))
    args = [
        "evaluate",
        str(corpus),
        "--strategies",
        "fixed_tokens",
        "--config",
        str(cfg),
        "--embedding-model",
        "fake/embed",
        "--yes",
        "--output-format",
        "json",
    ]
    nocache = runner.invoke(app, [*args, "--no-cache"])
    assert nocache.exit_code == 0, nocache.stderr
    assert calls
    assert not (cache_dir / "chunktuner_cache.sqlite").exists()

    calls.clear()
    cold = runner.invoke(app, args)
    assert cold.exit_code == 0, cold.stderr
    assert calls
    assert (cache_dir / "chunktuner_cache.sqlite").is_file()

    calls.clear()
    warm = runner.invoke(app, args)
    assert warm.exit_code == 0, warm.stderr
    assert calls == [], "warm cache must not call the embedding API"

    # Caching must not change any result.
    assert _strip_latency(nocache.stdout) == _strip_latency(cold.stdout)
    assert _strip_latency(cold.stdout) == _strip_latency(warm.stdout)


def test_dummy_embeddings_never_create_cache(workspace) -> None:
    corpus, cfg, cache_dir = workspace
    res = runner.invoke(
        app, ["compare", str(corpus), "--strategies", "fixed_tokens", "--config", str(cfg)]
    )
    assert res.exit_code == 0
    assert not (cache_dir / "chunktuner_cache.sqlite").exists()
