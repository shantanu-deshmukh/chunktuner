"""CLI smoke tests via Typer CliRunner."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from chunktuner.cli import app

runner = CliRunner()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for i in range(3):
        (tmp_path / f"doc{i}.txt").write_text("Sample text for testing. " * 50)
    return tmp_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_analyze_exits_zero(corpus: Path) -> None:
    result = runner.invoke(app, ["analyze", str(corpus)])
    assert result.exit_code == 0
    assert "token_count" in result.output.lower() or "content_type" in result.output.lower()


def test_estimate_exits_zero(corpus: Path) -> None:
    result = runner.invoke(app, ["estimate", str(corpus)])
    assert result.exit_code == 0
    assert "token" in result.output.lower()


def test_preview_exits_zero(corpus: Path) -> None:
    first_file = next(corpus.iterdir())
    result = runner.invoke(app, ["preview", str(first_file), "--strategy", "fixed_tokens"])
    assert result.exit_code == 0


def test_empty_dir_exits_nonzero(empty_dir: Path) -> None:
    result = runner.invoke(app, ["evaluate", str(empty_dir)])
    assert result.exit_code != 0
    assert "no supported documents" in result.output.lower()


def test_recommend_json_output(corpus: Path) -> None:
    result = runner.invoke(
        app,
        ["recommend", str(corpus), "--output-format", "json", "--no-baseline"],
    )
    assert result.exit_code == 0


def test_cache_stats_no_db(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["cache", "stats", "--config", str(tmp_path / "missing.yaml")],
    )
    assert result.exit_code == 0
    assert "No cache database" in result.output


def test_cache_clear_idempotent(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["cache", "clear", "--config", str(tmp_path / "missing.yaml")],
    )
    assert result.exit_code == 0
    assert "cleared" in result.output.lower()


def test_recommend_rejects_unknown_output_format(corpus: Path) -> None:
    result = runner.invoke(
        app,
        ["recommend", str(corpus), "--output-format", "bogus", "--no-baseline"],
    )
    assert result.exit_code != 0
    assert "Unknown output format" in result.output or "bogus" in result.output
