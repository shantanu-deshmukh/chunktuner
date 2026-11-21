"""``chunk-tune preview`` — inspect chunks for one strategy."""

import json
from pathlib import Path

import typer

from chunktuner.cli.common import CONFIG_HELP, file_ingestor, load_workspace, registry_for
from chunktuner.models import ChunkConfig, Document


def register(app: typer.Typer) -> None:
    @app.command("preview")
    def preview_cmd(
        target: str = typer.Argument(..., help="File path or inline text"),
        strategy: str = typer.Option(..., "--strategy", "-s", help="Strategy name"),
        params_json: str = typer.Option(
            "{}",
            "--params",
            help='JSON params object (e.g. {"max_tokens":256})',
        ),
        output_format: str = typer.Option("table", "--output-format"),
        config: Path | None = typer.Option(None, "--config", help=CONFIG_HELP),
    ) -> None:
        """Chunk a single document and print chunk boundaries (no embeddings)."""
        ws = load_workspace(config)
        p = Path(target)
        if p.is_file():
            fi = file_ingestor(p.resolve().parent, ws)
            doc = fi.ingest_path(p)[0]
        else:
            doc = Document(id="preview-inline", content=target, content_type="text")

        try:
            params = json.loads(params_json)
        except json.JSONDecodeError as e:
            typer.echo(f"Invalid --params JSON: {e}", err=True)
            raise typer.Exit(1) from e

        strat = registry_for(ws).get(strategy)
        cfg = ChunkConfig(name=strategy, params=params)
        chunks = strat.chunk(doc, cfg)
        rows = [
            {
                "id": c.id,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
                "tokens": c.tokens,
                "preview": (c.text[:120] + "...") if len(c.text) > 120 else c.text,
            }
            for c in chunks
        ]
        if output_format == "json":
            typer.echo(json.dumps(rows, indent=2))
        elif output_format == "yaml":
            typer.echo(__import__("yaml").safe_dump(rows, sort_keys=False))
        else:
            for r in rows:
                typer.echo(
                    f"[{r['start_offset']}:{r['end_offset']}] tok={r['tokens']} {r['preview']!r}"
                )
