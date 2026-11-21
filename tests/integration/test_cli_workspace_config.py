"""`.autochunk.yaml` keys wired into the CLI (D23) and mixed-corpus filtering (D25).

Precedence: explicit CLI flag > key written in the yaml > the command's own default.
Dummy embeddings only; no API calls.
"""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from chunktuner.cli import app
from chunktuner.eval.evaluator import Evaluator

try:
    # Click < 8.2 mixes stderr into stdout unless told otherwise.
    runner = CliRunner(mix_stderr=False)
except TypeError:
    # Click >= 8.2 removed `mix_stderr`; stdout and stderr are always separate.
    runner = CliRunner()


def _write_docs(corpus: Path, n: int) -> None:
    corpus.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        body = " ".join(f"Sentence {j} about topic {i}." for j in range(60))
        (corpus / f"doc{i:02d}.txt").write_text(body, encoding="utf-8")


def _cfg(tmp_path: Path, **keys: object) -> Path:
    cfg = tmp_path / ".autochunk.yaml"
    cfg.write_text(yaml.safe_dump(keys), encoding="utf-8")
    return cfg


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Record (doc count, use_case, encoding) of every `Evaluator.evaluate` call."""
    calls: list[dict] = []
    real = Evaluator.evaluate

    def wrapped(self, strategy, config, docs, dataset, *, scorer=None):  # type: ignore[no-untyped-def]
        calls.append(
            {
                "docs": len(docs),
                "use_case": scorer.use_case if scorer else None,
                "encoding": self._enc.name,
                "types": sorted({d.content_type for d in docs}),
            }
        )
        return real(self, strategy, config, docs, dataset, scorer=scorer)

    monkeypatch.setattr(Evaluator, "evaluate", wrapped)
    return calls


def _run(*args: str):  # type: ignore[no-untyped-def]
    return runner.invoke(app, [*args, "--no-cache"] if args[0] != "estimate" else list(args))


# ---------------------------------------------------------------------------
# use_case / max_docs precedence
# ---------------------------------------------------------------------------


def test_use_case_absent_key_uses_default(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 2)
    cfg = _cfg(tmp_path, cache_dir=str(tmp_path / "cache"))
    res = _run("recommend", str(corpus), "--config", str(cfg), "--output-format", "json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["use_case"] == "rag_qa"


def test_use_case_yaml_beats_default(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 2)
    cfg = _cfg(tmp_path, use_case="search")
    res = _run("recommend", str(corpus), "--config", str(cfg), "--output-format", "json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["use_case"] == "search"


def test_use_case_flag_beats_yaml(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 2)
    cfg = _cfg(tmp_path, use_case="search")
    res = _run(
        "recommend",
        str(corpus),
        "--config",
        str(cfg),
        "--use-case",
        "summarization",
        "--output-format",
        "json",
    )
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["use_case"] == "summarization"


@pytest.mark.parametrize("cmd", ["evaluate", "compare"])
def test_use_case_yaml_in_evaluate_and_compare(tmp_path: Path, spy: list[dict], cmd: str) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 2)
    cfg = _cfg(tmp_path, use_case="search")
    res = _run(cmd, str(corpus), "--config", str(cfg), "--strategies", "fixed_tokens")
    assert res.exit_code == 0, res.output
    assert spy and {c["use_case"] for c in spy} == {"search"}


@pytest.mark.parametrize(
    ("cmd", "default"),
    [("evaluate", 20), ("compare", 15), ("recommend", 20)],
)
def test_max_docs_precedence(tmp_path: Path, spy: list[dict], cmd: str, default: int) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 22)
    base = ["--strategies", "fixed_tokens"]

    # absent key -> the command's own default (not WorkspaceConfig's 100)
    cfg = _cfg(tmp_path, top_k=5)
    assert _run(cmd, str(corpus), "--config", str(cfg), *base).exit_code == 0
    assert {c["docs"] for c in spy} == {default}

    # yaml beats default
    spy.clear()
    cfg = _cfg(tmp_path, max_docs=3)
    assert _run(cmd, str(corpus), "--config", str(cfg), *base).exit_code == 0
    assert {c["docs"] for c in spy} == {3}

    # flag beats yaml
    spy.clear()
    res = _run(cmd, str(corpus), "--config", str(cfg), *base, "--max-docs", "5")
    assert res.exit_code == 0, res.output
    assert {c["docs"] for c in spy} == {5}


def test_estimate_max_docs_from_yaml(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 6)
    out_default = _run("estimate", str(corpus), "--output-format", "json")
    cfg = _cfg(tmp_path, max_docs=2)
    out_yaml = _run("estimate", str(corpus), "--config", str(cfg), "--output-format", "json")
    assert out_default.exit_code == 0 and out_yaml.exit_code == 0
    assert (
        json.loads(out_yaml.stdout)["total_tokens"] < json.loads(out_default.stdout)["total_tokens"]
    )


# ---------------------------------------------------------------------------
# strip_patterns / tokenizer_encoding / log_level
# ---------------------------------------------------------------------------


def test_strip_patterns_applied_at_ingest(tmp_path: Path) -> None:
    doc = tmp_path / "a.md"
    doc.write_text("# Title\nSECRET-FOOTER line\nBody text here.\n", encoding="utf-8")
    cfg = _cfg(tmp_path, strip_patterns=[r"^SECRET-FOOTER.*$"])
    args = ["preview", str(doc), "-s", "fixed_tokens", "--output-format", "json"]
    plain = runner.invoke(app, args)
    stripped = runner.invoke(app, [*args, "--config", str(cfg)])
    assert plain.exit_code == 0 and stripped.exit_code == 0
    assert "SECRET-FOOTER" in plain.stdout
    assert "SECRET-FOOTER" not in stripped.stdout
    assert "Body text here." in stripped.stdout


def test_strip_patterns_offsets_invariant(tmp_path: Path) -> None:
    from chunktuner.chunking import default_registry
    from chunktuner.ingestion.file_ingestor import FileIngestor
    from chunktuner.models import ChunkConfig

    (tmp_path / "a.txt").write_text(
        "\n".join(f"line {i} DROPME tail" if i % 3 == 0 else f"keep {i}" for i in range(200)),
        encoding="utf-8",
    )
    (doc,) = FileIngestor(root=tmp_path, strip_patterns=[r"DROPME"]).ingest_dir(tmp_path)
    assert "DROPME" not in doc.content
    chunks = default_registry.get("fixed_tokens").chunk(
        doc, ChunkConfig(name="fixed_tokens", params={"max_tokens": 64, "overlap_tokens": 8})
    )
    assert chunks
    for c in chunks:
        assert doc.content[c.start_offset : c.end_offset] == c.text


def test_invalid_strip_pattern_exits_2(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 1)
    cfg = _cfg(tmp_path, strip_patterns=["(unclosed"])
    res = _run("evaluate", str(corpus), "--config", str(cfg))
    assert res.exit_code == 2
    assert "strip_patterns" in res.stderr


def test_tokenizer_encoding_used(tmp_path: Path, spy: list[dict]) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 2)
    assert _run("evaluate", str(corpus), "--strategies", "fixed_tokens").exit_code == 0
    assert {c["encoding"] for c in spy} == {"cl100k_base"}
    spy.clear()
    cfg = _cfg(tmp_path, tokenizer_encoding="o200k_base")
    res = _run("evaluate", str(corpus), "--config", str(cfg), "--strategies", "fixed_tokens")
    assert res.exit_code == 0, res.output
    assert {c["encoding"] for c in spy} == {"o200k_base"}


def test_tokenizer_encoding_preview_and_analyze(tmp_path: Path) -> None:
    doc = tmp_path / "a.txt"
    # ASCII French: o200k_base needs fewer tokens than cl100k_base.
    doc.write_text("Bonjour le monde, comment allez-vous aujourd hui? " * 20, encoding="utf-8")
    cfg = _cfg(tmp_path, tokenizer_encoding="o200k_base")

    def preview_tokens(*extra: str) -> int:
        res = runner.invoke(
            app,
            ["preview", str(doc), "-s", "fixed_tokens", "--output-format", "json", *extra],
        )
        assert res.exit_code == 0, res.output
        return sum(r["tokens"] for r in json.loads(res.stdout))

    assert preview_tokens("--config", str(cfg)) < preview_tokens()

    def analyze_tokens(*extra: str) -> int:
        res = runner.invoke(app, ["analyze", str(doc), "--output-format", "json", *extra])
        assert res.exit_code == 0, res.output
        return json.loads(res.stdout)["token_count"]

    assert analyze_tokens("--config", str(cfg)) < analyze_tokens()


def test_unknown_tokenizer_encoding_exits_2(tmp_path: Path) -> None:
    doc = tmp_path / "a.txt"
    doc.write_text("hello", encoding="utf-8")
    cfg = _cfg(tmp_path, tokenizer_encoding="not_an_encoding")
    res = runner.invoke(app, ["analyze", str(doc), "--config", str(cfg)])
    assert res.exit_code == 2
    assert "tokenizer_encoding" in res.stderr


def test_log_level_sets_chunktuner_logger(tmp_path: Path) -> None:
    import logging

    corpus = tmp_path / "c"
    _write_docs(corpus, 1)
    logger = logging.getLogger("chunktuner")
    before = logger.level
    try:
        cfg = _cfg(tmp_path, log_level="debug")
        res = _run("estimate", str(corpus), "--config", str(cfg))
        assert res.exit_code == 0, res.output
        assert logger.level == logging.DEBUG
        bad = _cfg(tmp_path, log_level="LOUD")
        res = _run("estimate", str(corpus), "--config", str(bad))
        assert res.exit_code == 2
        assert "log_level" in res.stderr
    finally:
        logger.setLevel(before)


def test_log_level_absent_leaves_logging_alone(tmp_path: Path) -> None:
    import logging

    corpus = tmp_path / "c"
    _write_docs(corpus, 1)
    logger = logging.getLogger("chunktuner")
    before = logger.level
    cfg = _cfg(tmp_path, top_k=5)
    assert _run("estimate", str(corpus), "--config", str(cfg)).exit_code == 0
    assert logger.level == before


# ---------------------------------------------------------------------------
# max_tokens_per_run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd", ["evaluate", "compare", "recommend"])
def test_max_tokens_per_run_stops_without_yes(tmp_path: Path, spy: list[dict], cmd: str) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 3)
    cfg = _cfg(tmp_path, max_tokens_per_run=10)
    res = _run(cmd, str(corpus), "--config", str(cfg), "--strategies", "fixed_tokens")
    assert res.exit_code == 1
    assert "max_tokens_per_run=10" in res.stderr
    assert spy == []


@pytest.mark.parametrize("cmd", ["evaluate", "compare", "recommend"])
def test_max_tokens_per_run_continues_with_yes(tmp_path: Path, spy: list[dict], cmd: str) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 3)
    cfg = _cfg(tmp_path, max_tokens_per_run=10)
    res = _run(cmd, str(corpus), "--config", str(cfg), "--strategies", "fixed_tokens", "--yes")
    assert res.exit_code == 0, res.output
    assert "max_tokens_per_run=10" in res.stderr
    assert spy


def test_max_tokens_per_run_under_cap_is_silent(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    _write_docs(corpus, 1)
    cfg = _cfg(tmp_path, max_tokens_per_run=10_000_000)
    res = _run("evaluate", str(corpus), "--config", str(cfg), "--strategies", "fixed_tokens")
    assert res.exit_code == 0, res.output
    assert "max_tokens_per_run" not in res.stderr


# ---------------------------------------------------------------------------
# D25: mixed corpora
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed(tmp_path: Path) -> Path:
    corpus = tmp_path / "mixed"
    corpus.mkdir()
    (corpus / "guide.md").write_text(
        "# Guide\n\n" + " ".join(f"Point {i} about setup." for i in range(80)), encoding="utf-8"
    )
    (corpus / "tool.py").write_text(
        "\n\n".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(20)), encoding="utf-8"
    )
    return corpus


@pytest.mark.parametrize("cmd", ["evaluate", "compare", "recommend"])
def test_mixed_folder_skips_unsupported_with_warning(
    mixed: Path, spy: list[dict], cmd: str
) -> None:
    res = _run(cmd, str(mixed), "--strategies", "fixed_tokens,recursive_character")
    assert res.exit_code == 0, res.output
    assert "Skipped 1 code file(s) not supported by fixed_tokens, recursive_character" in (
        res.stderr
    )
    assert "--strategies code_ast,code_window" in res.stderr
    assert spy and all(c["types"] == ["markdown"] for c in spy)


def test_mixed_folder_recommend_json_content_type(mixed: Path) -> None:
    res = _run("recommend", str(mixed), "--output-format", "json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["content_type"] == "markdown"


def test_mixed_folder_code_strategies_keep_code_only(mixed: Path, spy: list[dict]) -> None:
    res = _run("recommend", str(mixed), "--strategies", "code_window", "--output-format", "json")
    assert res.exit_code == 0, res.output
    assert "Skipped 1 markdown file(s)" in res.stderr
    assert json.loads(res.stdout)["content_type"] == "code"
    assert all(c["types"] == ["code"] for c in spy)


@pytest.mark.parametrize("cmd", ["evaluate", "compare", "recommend"])
def test_only_code_default_strategies_exit_1_with_hint(tmp_path: Path, cmd: str) -> None:
    corpus = tmp_path / "code"
    corpus.mkdir()
    (corpus / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    res = _run(cmd, str(corpus), "--strategies", "fixed_tokens,recursive_character")
    assert res.exit_code == 1
    assert "no documents" in res.stderr
    assert "--strategies code_ast,code_window" in res.stderr


def test_supported_mixed_types_still_evaluated_together(tmp_path: Path, spy: list[dict]) -> None:
    corpus = tmp_path / "mt"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\n" + "alpha beta " * 100, encoding="utf-8")
    (corpus / "b.txt").write_text("gamma delta " * 100, encoding="utf-8")
    res = _run("recommend", str(corpus), "--strategies", "fixed_tokens", "--output-format", "json")
    assert res.exit_code == 0, res.output
    assert "Skipped" not in res.stderr
    assert spy and all(c["types"] == ["markdown", "text"] for c in spy)


def test_unknown_strategy_exits_2(mixed: Path) -> None:
    res = _run("evaluate", str(mixed), "--strategies", "nope")
    assert res.exit_code == 2
    assert "Unknown strategy 'nope'" in res.stderr
