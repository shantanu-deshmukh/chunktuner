"""`--retrieval-mode` / `--reranker` CLI flags and their `.autochunk.yaml` keys (D17).

Precedence: explicit CLI flag > key written in the yaml > dense / no reranker.
Dummy embeddings and the dummy reranker only; no API calls, no model downloads.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from chunktuner.cli import app
from chunktuner.config import WorkspaceConfig
from chunktuner.eval.evaluator import Evaluator
from tests._extras import require_extra

try:
    # Click < 8.2 mixes stderr into stdout unless told otherwise.
    runner = CliRunner(mix_stderr=False)
except TypeError:
    # Click >= 8.2 removed `mix_stderr`; stdout and stderr are always separate.
    runner = CliRunner()

EVAL_CMDS = ["evaluate", "compare", "recommend"]
ALL_CMDS = [*EVAL_CMDS, "estimate"]


def _write_docs(corpus: Path, n: int, sentences: int = 60) -> None:
    corpus.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        body = " ".join(f"Sentence {j} about topic {i}." for j in range(sentences))
        (corpus / f"doc{i:02d}.txt").write_text(body, encoding="utf-8")


def _cfg(tmp_path: Path, **keys: object) -> Path:
    cfg = tmp_path / ".autochunk.yaml"
    keys.setdefault("cache_dir", str(tmp_path / "cache"))
    cfg.write_text(yaml.safe_dump(keys), encoding="utf-8")
    return cfg


def _run(*args: str, input: str | None = None):  # type: ignore[no-untyped-def]
    argv = list(args) if args[0] == "estimate" else [*args, "--no-cache"]
    return runner.invoke(app, argv, input=input)


@pytest.fixture
def ev_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record ``retrieval_mode`` and reranker profile of every `Evaluator` built."""
    calls: list[dict] = []
    real = Evaluator.__init__

    def wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        real(self, *args, **kwargs)
        calls.append(
            {
                "mode": self.retrieval_mode,
                "reranker": self.reranker.profile_name if self.reranker else None,
            }
        )

    monkeypatch.setattr(Evaluator, "__init__", wrapped)
    return calls


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    c = tmp_path / "c"
    _write_docs(c, 2)
    return c


# ---------------------------------------------------------------------------
# Flag validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd", ALL_CMDS)
def test_retrieval_mode_rejects_unknown(tmp_path: Path, corpus: Path, cmd: str) -> None:
    res = _run(cmd, str(corpus), "--config", str(_cfg(tmp_path)), "--retrieval-mode", "sparse")
    assert res.exit_code == 2
    assert "--retrieval-mode" in res.stderr


@pytest.mark.parametrize("cmd", ALL_CMDS)
@pytest.mark.parametrize("spec", ["local:", "  "])
def test_reranker_rejects_bad_spec(tmp_path: Path, corpus: Path, cmd: str, spec: str) -> None:
    res = _run(cmd, str(corpus), "--config", str(_cfg(tmp_path)), "--reranker", spec)
    assert res.exit_code == 2
    assert "--reranker" in res.stderr


# ---------------------------------------------------------------------------
# Precedence: flag > yaml > default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd", EVAL_CMDS)
def test_retrieval_default_is_dense_without_reranker(
    tmp_path: Path, corpus: Path, ev_spy: list[dict], cmd: str
) -> None:
    cfg = _cfg(tmp_path)
    res = _run(cmd, str(corpus), "--config", str(cfg), "--strategies", "fixed_tokens")
    assert res.exit_code == 0, res.output
    assert ev_spy == [{"mode": "dense", "reranker": None}]
    assert "reranker's input window" not in res.stderr


@pytest.mark.parametrize("cmd", EVAL_CMDS)
def test_retrieval_yaml_keys_honoured(
    tmp_path: Path, corpus: Path, ev_spy: list[dict], cmd: str
) -> None:
    require_extra("bm25", "bm25s")
    cfg = _cfg(tmp_path, retrieval_mode="hybrid", reranker="dummy")
    res = _run(cmd, str(corpus), "--config", str(cfg), "--strategies", "fixed_tokens")
    assert res.exit_code == 0, res.output
    assert ev_spy == [{"mode": "hybrid", "reranker": "dummy/rerank"}]


@pytest.mark.parametrize("cmd", EVAL_CMDS)
def test_retrieval_flags_beat_yaml(
    tmp_path: Path, corpus: Path, ev_spy: list[dict], cmd: str
) -> None:
    # The yaml reranker would download a model if it were used; the flag wins first.
    cfg = _cfg(tmp_path, retrieval_mode="bm25", reranker="local:some/model")
    res = _run(
        cmd,
        str(corpus),
        "--config",
        str(cfg),
        "--strategies",
        "fixed_tokens",
        "--retrieval-mode",
        "dense",
        "--reranker",
        "dummy",
    )
    assert res.exit_code == 0, res.output
    assert ev_spy == [{"mode": "dense", "reranker": "dummy/rerank"}]


def test_invalid_yaml_reranker_is_clean_error(tmp_path: Path, corpus: Path) -> None:
    cfg = _cfg(tmp_path, reranker="local:")
    res = _run("evaluate", str(corpus), "--config", str(cfg))
    assert res.exit_code == 2
    assert "Error: invalid" in res.stderr and "reranker" in res.stderr
    assert isinstance(res.exception, SystemExit)


def test_invalid_yaml_retrieval_mode_is_clean_error(tmp_path: Path, corpus: Path) -> None:
    cfg = _cfg(tmp_path, retrieval_mode="sparse")
    res = _run("estimate", str(corpus), "--config", str(cfg))
    assert res.exit_code == 2
    assert "retrieval_mode" in res.stderr
    assert isinstance(res.exception, SystemExit)


def test_workspace_config_validates_reranker() -> None:
    assert WorkspaceConfig().retrieval_mode == "dense"
    assert WorkspaceConfig().reranker is None
    assert WorkspaceConfig(reranker=" dummy ").reranker == "dummy"
    with pytest.raises(ValidationError):
        WorkspaceConfig(reranker="local:")
    with pytest.raises(ValidationError):
        WorkspaceConfig(retrieval_mode="sparse")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End to end, warnings and errors
# ---------------------------------------------------------------------------


def test_evaluate_bm25_dummy_reranker_json(tmp_path: Path, corpus: Path) -> None:
    require_extra("bm25", "bm25s")
    res = _run(
        "evaluate",
        str(corpus),
        "--config",
        str(_cfg(tmp_path)),
        "--strategies",
        "fixed_tokens",
        "--retrieval-mode",
        "bm25",
        "--reranker",
        "dummy",
        "--output-format",
        "json",
    )
    assert res.exit_code == 0, res.output
    rows = json.loads(res.stdout)
    assert rows
    for row in rows:
        assert row["metrics"]["retrieval_mode"] == "bm25"
        assert row["metrics"]["reranker"] == "dummy/rerank"
        assert row["metrics"]["reranker_overflow_ratio"] is not None


@pytest.mark.parametrize("cmd", ["evaluate", "recommend"])
def test_reranker_overflow_warning_on_stderr(tmp_path: Path, cmd: str) -> None:
    # ~1500-token docs: fixed_tokens configs with max_tokens=1024 exceed the 512 window.
    corpus = tmp_path / "long"
    _write_docs(corpus, 2, sentences=250)
    res = _run(
        cmd,
        str(corpus),
        "--config",
        str(_cfg(tmp_path)),
        "--strategies",
        "fixed_tokens",
        "--reranker",
        "dummy",
        "--output-format",
        "json",
    )
    assert res.exit_code == 0, res.output
    json.loads(res.stdout)  # the warning stays off stdout
    assert res.stderr.count("reranker's input window") == 1
    assert "512 tokens" in res.stderr


def test_reranker_overflow_warning_compare(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # compare runs only each strategy's first (small) config; force an overflow ratio.
    monkeypatch.setattr(Evaluator, "_overflow_ratio", lambda self, tok_lens: 0.25)
    res = _run(
        "compare",
        str(corpus),
        "--config",
        str(_cfg(tmp_path)),
        "--strategies",
        "fixed_tokens,recursive_character",
        "--reranker",
        "dummy",
    )
    assert res.exit_code == 0, res.output
    assert res.stderr.count("reranker's input window") == 1
    assert "25%" in res.stderr


def test_no_overflow_warning_when_chunks_fit(tmp_path: Path, corpus: Path) -> None:
    # 2 short docs (~420 tokens each): no chunk exceeds the 512-token window.
    res = _run(
        "evaluate",
        str(corpus),
        "--config",
        str(_cfg(tmp_path)),
        "--strategies",
        "fixed_tokens",
        "--reranker",
        "dummy",
    )
    assert res.exit_code == 0, res.output
    assert "reranker's input window" not in res.stderr


def test_missing_bm25_extra_is_clean_error(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "bm25s", None)  # import bm25s -> ImportError
    res = _run("evaluate", str(corpus), "--config", str(_cfg(tmp_path)), "--retrieval-mode", "bm25")
    assert res.exit_code == 1
    assert "bm25" in res.stderr and "extra" in res.stderr
    assert isinstance(res.exception, SystemExit)


def test_litellm_reranker_needs_confirmation(
    tmp_path: Path, corpus: Path, ev_spy: list[dict]
) -> None:
    res = _run(
        "evaluate",
        str(corpus),
        "--config",
        str(_cfg(tmp_path)),
        "--reranker",
        "fake/rerank-model",
        input="n\n",
    )
    assert res.exit_code == 1
    assert ev_spy == []  # aborted before any evaluation


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------

_ESTIMATE_PREFIXES = (
    "Total embedding tokens (est.):",
    "Embedding cost (USD est.):",
    "LLM dataset cost (USD est.):",
    "Wall time (min est.):",
    "Strategy-config combinations:",
)


def _lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.strip()]


def test_estimate_default_output_unchanged(tmp_path: Path, corpus: Path) -> None:
    cfg = str(_cfg(tmp_path))
    res = _run("estimate", str(corpus), "--config", cfg)
    assert res.exit_code == 0, res.output
    lines = _lines(res.stdout)
    assert len(lines) == len(_ESTIMATE_PREFIXES)
    assert all(line.startswith(p) for line, p in zip(lines, _ESTIMATE_PREFIXES, strict=True))
    # dense / hybrid / dummy reranker do not change the output.
    for extra in (["--retrieval-mode", "dense"], ["--retrieval-mode", "hybrid"]):
        assert _run("estimate", str(corpus), "--config", cfg, *extra).stdout == res.stdout
    assert _run("estimate", str(corpus), "--config", cfg, "--reranker", "dummy").stdout == (
        res.stdout
    )


def test_estimate_bm25_has_no_embedding_cost(tmp_path: Path, corpus: Path) -> None:
    cfg = str(_cfg(tmp_path))
    res = _run("estimate", str(corpus), "--config", cfg, "--retrieval-mode", "bm25")
    assert res.exit_code == 0, res.output
    assert "Embedding cost (USD est.): 0.0" in res.stdout
    assert "Total embedding tokens (est.): 0" in res.stdout
    assert "no embedding calls" in res.stdout
    js = _run(
        "estimate",
        str(corpus),
        "--config",
        cfg,
        "--retrieval-mode",
        "bm25",
        "--output-format",
        "json",
    )
    est = json.loads(js.stdout)
    assert est["embedding_cost_usd"] == 0.0 and est["total_tokens"] == 0


def test_estimate_bm25_from_yaml(tmp_path: Path, corpus: Path) -> None:
    res = _run("estimate", str(corpus), "--config", str(_cfg(tmp_path, retrieval_mode="bm25")))
    assert res.exit_code == 0, res.output
    assert "Embedding cost (USD est.): 0.0" in res.stdout


def test_estimate_litellm_reranker_line(tmp_path: Path, corpus: Path) -> None:
    cfg = str(_cfg(tmp_path))
    res = _run("estimate", str(corpus), "--config", cfg, "--reranker", "cohere/rerank-v3.5")
    assert res.exit_code == 0, res.output
    rerank = [line for line in res.stdout.splitlines() if line.startswith("Rerank requests")]
    assert len(rerank) == 1
    assert "18 configs" in rerank[0] and "up to 40 documents" in rerank[0]
    assert "no price" in rerank[0]
    js = _run(
        "estimate",
        str(corpus),
        "--config",
        cfg,
        "--reranker",
        "cohere/rerank-v3.5",
        "--output-format",
        "json",
    )
    json.loads(js.stdout)  # note goes to stderr in json mode
    assert "Rerank requests" in js.stderr
