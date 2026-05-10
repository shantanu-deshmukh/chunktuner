"""``chunk-tune init`` — bootstrap workspace config."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from chunktuner.config import default_init_yaml


def register(app: typer.Typer) -> None:
    @app.command("init")
    def init_cmd(
        model: str | None = typer.Option(
            None,
            "--model",
            "-m",
            help=(
                "Embedding model id (LiteLLM). "
                "Examples: text-embedding-3-small (OpenAI), "
                "gemini/gemini-embedding-001 (Google), "
                "openai/<id> for local servers. "
                "Omit to keep dummy embeddings (no API calls)."
            ),
        ),
    ) -> None:
        """Create ``.autochunk.yaml`` in the current directory.

        Embedding model defaults to null (DummyEmbeddingFunction, no API calls).
        Set --model to enable real embeddings via LiteLLM for any provider.
        """
        path = Path.cwd() / ".autochunk.yaml"
        if path.exists():
            typer.echo(
                f"Refusing to overwrite existing {path}. "
                "Delete the file manually and re-run, or edit it directly.",
                err=True,
            )
            raise typer.Exit(code=1)
        data = default_init_yaml()
        if model is not None:
            data["embedding_model"] = model
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        typer.echo(f"Wrote {path}")
