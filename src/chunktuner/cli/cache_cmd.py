"""``chunk-tune cache`` — stats and clear."""

from pathlib import Path

import typer

from chunktuner.cache.chunk_cache import ChunkCache
from chunktuner.cache.embedding_cache import EmbeddingCache, default_embedding_db_path
from chunktuner.cli.common import load_workspace_path
from chunktuner.config import (
    DEFAULT_EMBEDDING_MODEL,
    load_workspace_config,
    workspace_cache_dir,
)

MODEL_FILTER_HELP = (
    "Only embeddings cached for this model (default: all rows; chunk rows untouched when set). "
    "Rows written before 0.3 have no model and are only counted/cleared without --model."
)


def register(app: typer.Typer) -> None:
    cache_app = typer.Typer(help="SQLite embedding + chunk caches")

    @cache_app.command("stats")
    def cache_stats(
        config: Path | None = typer.Option(None, "--config"),
        model: str | None = typer.Option(
            None,
            "--model",
            help=MODEL_FILTER_HELP,
        ),
    ) -> None:
        ws = load_workspace_config(load_workspace_path(config))
        effective_model = model or ws.embedding_model or DEFAULT_EMBEDDING_MODEL
        cache_dir = workspace_cache_dir(ws)
        db = default_embedding_db_path(cache_dir)
        if not db.is_file():
            typer.echo("No cache database on disk yet.")
            raise typer.Exit(0)
        with EmbeddingCache(db, effective_model) as emb, ChunkCache(db) as ch:
            typer.echo(f"Database: {db}")
            typer.echo(f"Embeddings: {emb.stats(model)}")
            typer.echo(f"Chunks: {ch.stats()}")

    @cache_app.command("clear")
    def cache_clear(
        config: Path | None = typer.Option(None, "--config"),
        model: str | None = typer.Option(
            None,
            "--model",
            help=MODEL_FILTER_HELP,
        ),
    ) -> None:
        ws = load_workspace_config(load_workspace_path(config))
        effective_model = model or ws.embedding_model or DEFAULT_EMBEDDING_MODEL
        cache_dir = workspace_cache_dir(ws)
        db = default_embedding_db_path(cache_dir)
        if db.is_file():
            with EmbeddingCache(db, effective_model) as emb, ChunkCache(db) as ch:
                if model is None:
                    emb.clear()
                    ch.clear()
                else:
                    # Chunk rows don't depend on the embedding model; leave them.
                    emb.clear(model)
        typer.echo("Caches cleared.")

    app.add_typer(cache_app, name="cache")
