"""D25: MCP service skips docs the selected strategies cannot chunk (dummy embeddings)."""

from pathlib import Path

import pytest

from chunktuner.mcp.service import evaluate_chunking_impl, recommend_config_impl


@pytest.fixture
def mixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    corpus = tmp_path / "mixed"
    corpus.mkdir()
    (corpus / "guide.md").write_text(
        "# Guide\n\n" + " ".join(f"Point {i} about setup." for i in range(80)), encoding="utf-8"
    )
    (corpus / "tool.py").write_text(
        "\n\n".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(20)), encoding="utf-8"
    )
    monkeypatch.setenv("CHUNK_TUNER_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("CHUNKTUNER_CACHE_DIR", str(tmp_path / "cache"))
    return corpus


_SKIP = "Skipped 1 code file(s) not supported by fixed_tokens, recursive_character"
_HINT = 'strategies=["code_ast", "code_window"]'


def test_recommend_mixed_skips_code_with_warning(mixed: Path) -> None:
    out = recommend_config_impl(str(mixed), "rag_qa", use_cache=False)
    assert out["content_type"] == "markdown"
    assert any(_SKIP in w and _HINT in w for w in out["warnings"])


def test_evaluate_mixed_skips_code_with_warning(mixed: Path) -> None:
    out = evaluate_chunking_impl(str(mixed), "rag_qa", use_cache=False)
    assert out["results"]
    assert any(_SKIP in w and _HINT in w for w in out["warnings"])


def test_evaluate_dry_run_mixed_reports_skip(mixed: Path) -> None:
    out = evaluate_chunking_impl(str(mixed), "rag_qa", dry_run=True)
    assert "total_tokens" in out
    assert any(_SKIP in w for w in out["warnings"])


def test_only_code_default_strategies_errors(mixed: Path) -> None:
    (mixed / "guide.md").unlink()
    with pytest.raises(ValueError, match="No documents under path are supported"):
        recommend_config_impl(str(mixed), "rag_qa", use_cache=False)
    with pytest.raises(ValueError, match="code_ast"):
        evaluate_chunking_impl(str(mixed), "rag_qa", use_cache=False)


def test_recommend_content_type_code_runs_code_baseline(mixed: Path) -> None:
    # content_type=code with the default (text) strategies: only the code baseline runs.
    out = recommend_config_impl(str(mixed), "rag_qa", content_type="code", use_cache=False)
    assert out["content_type"] == "code"
    assert out["best"]["strategy_name"] == "code_window"


def test_single_type_corpus_has_no_skip_warning(mixed: Path) -> None:
    (mixed / "tool.py").unlink()
    out = recommend_config_impl(str(mixed), "rag_qa", use_cache=False)
    assert not any("Skipped" in w for w in out["warnings"])
    dry = evaluate_chunking_impl(str(mixed), "rag_qa", dry_run=True)
    assert "warnings" not in dry


def test_dry_run_only_code_default_strategies_errors(mixed: Path) -> None:
    (mixed / "guide.md").unlink()
    with pytest.raises(ValueError, match="code_ast"):
        evaluate_chunking_impl(str(mixed), "rag_qa", dry_run=True)


def test_dry_run_max_docs_applied_after_filtering(mixed: Path) -> None:
    (mixed / "a_tool.py").write_text("def g():\n    return 2\n", encoding="utf-8")
    out = evaluate_chunking_impl(str(mixed), "rag_qa", dry_run=True, max_docs=1)
    only_md = evaluate_chunking_impl(str(mixed / "guide.md"), "rag_qa", dry_run=True)
    assert {k: v for k, v in out.items() if k != "warnings"} == only_md
