"""CLI smoke tests via Typer CliRunner."""

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


def test_root_no_subcommand_shows_help_exit_2() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage:" in result.output


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


def test_compare_exits_zero(corpus: Path) -> None:
    result = runner.invoke(
        app,
        ["compare", str(corpus), "--strategies", "fixed_tokens,recursive_character"],
    )
    assert result.exit_code == 0


def test_compare_invalid_strategy(corpus: Path) -> None:
    result = runner.invoke(
        app,
        ["compare", str(corpus), "--strategies", "nonexistent_strategy"],
    )
    assert result.exit_code != 0


def test_estimate_yaml_output(corpus: Path) -> None:
    result = runner.invoke(app, ["estimate", str(corpus), "--output-format", "yaml"])
    assert result.exit_code == 0
    assert "total_tokens" in result.output


def test_estimate_rejects_unknown_output_format(corpus: Path) -> None:
    result = runner.invoke(app, ["estimate", str(corpus), "--output-format", "xml"])
    assert result.exit_code != 0


def _cache_workspace(tmp_path: Path) -> tuple[Path, Path]:
    from chunktuner.cache.chunk_cache import ChunkCache
    from chunktuner.cache.embedding_cache import EmbeddingCache, default_embedding_db_path

    cache_dir = tmp_path / "cache"
    cfg = tmp_path / ".autochunk.yaml"
    cfg.write_text(f"cache_dir: {cache_dir}\n")
    db = default_embedding_db_path(cache_dir)
    with EmbeddingCache(db, "model-a") as a:
        a.set("x", [1.0])
        a.set("y", [2.0])
    with EmbeddingCache(db, "model-b") as b:
        b.set("x", [3.0])
    ChunkCache(db).close()
    return cfg, db


def test_cache_stats_model_filter(tmp_path: Path) -> None:
    cfg, _ = _cache_workspace(tmp_path)
    all_rows = runner.invoke(app, ["cache", "stats", "--config", str(cfg)])
    assert all_rows.exit_code == 0, all_rows.output
    assert "Embeddings: {'rows': 3," in all_rows.output
    only_a = runner.invoke(app, ["cache", "stats", "--config", str(cfg), "--model", "model-a"])
    assert only_a.exit_code == 0, only_a.output
    assert "Embeddings: {'rows': 2," in only_a.output


def test_cache_clear_model_filter(tmp_path: Path) -> None:
    from chunktuner.cache.embedding_cache import EmbeddingCache

    cfg, db = _cache_workspace(tmp_path)
    result = runner.invoke(app, ["cache", "clear", "--config", str(cfg), "--model", "model-a"])
    assert result.exit_code == 0, result.output
    assert "Caches cleared." in result.output
    with EmbeddingCache(db, "model-b") as emb:
        assert emb.stats()["rows"] == 1
        assert emb.get("x") == [3.0]
    result = runner.invoke(app, ["cache", "clear", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    with EmbeddingCache(db, "model-b") as emb:
        assert emb.stats()["rows"] == 0
