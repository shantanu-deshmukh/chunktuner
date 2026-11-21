"""``chunk-tune analyze`` — structural scan without API calls."""

import re
from pathlib import Path

import tiktoken
import typer

from chunktuner.cli.common import (
    CONFIG_HELP,
    load_workspace,
    tokenizer_encoding,
    validate_output_format,
)
from chunktuner.ingestion.content_type import detect_content_type

_ANALYZE_DIR_MAX_CHARS = 50_000
_ANALYZE_FILE_MAX_CHARS = 200_000


def register(app: typer.Typer) -> None:
    @app.command("analyze")
    def analyze_cmd(
        path: Path = typer.Argument(..., exists=True, help="File or directory"),
        content_type: str | None = typer.Option(
            None,
            "--content-type",
            help="Override detected content type",
        ),
        output_format: str = typer.Option(
            "table",
            "--output-format",
            help="table | json | yaml",
        ),
        config: Path | None = typer.Option(None, "--config", help=CONFIG_HELP),
    ) -> None:
        """Quick structural scan (no embeddings)."""
        validate_output_format(output_format)
        ws = load_workspace(config)
        if path.is_dir():
            typer.echo("Directory analyze: summarizing first matching .md/.txt file", err=True)
            sample = next(
                (p for p in sorted(path.rglob("*")) if p.suffix.lower() in {".md", ".txt"}),
                None,
            )
            if sample is None:
                typer.echo("No .md or .txt files found", err=True)
                raise typer.Exit(1)
            text = sample.read_text(encoding="utf-8", errors="replace")[:_ANALYZE_DIR_MAX_CHARS]
            target = sample
        else:
            text = path.read_text(encoding="utf-8", errors="replace")[:_ANALYZE_FILE_MAX_CHARS]
            target = path

        ct = content_type or detect_content_type(target, text)
        enc = tiktoken.get_encoding(tokenizer_encoding(ws))
        ntok = len(enc.encode(text))
        headers = len(re.findall(r"(?m)^#{1,3}\s+\S", text))
        code_blocks = len(re.findall(r"(?m)^```", text)) // 2
        paras = [p for p in text.split("\n\n") if p.strip()]
        avg_para = sum(len(p) for p in paras) / max(1, len(paras))

        hint = "recursive_character (~1600 chars)"
        if headers / max(1, len(paras)) > 0.4:
            hint = "markdown_semantic (when available)"
        if code_blocks > 3:
            hint = "code_ast (when available)"

        payload = {
            "path": str(path),
            "sample_path": str(target) if path.is_dir() else str(path),
            "content_type": ct,
            "token_count": ntok,
            "header_hits": headers,
            "fenced_code_blocks": code_blocks,
            "avg_paragraph_chars": round(avg_para, 1),
            "heuristic_starting_strategy": hint,
        }
        if output_format == "json":
            typer.echo(__import__("json").dumps(payload, indent=2))
        elif output_format == "yaml":
            typer.echo(__import__("yaml").safe_dump(payload, sort_keys=False))
        else:
            for k, v in payload.items():
                typer.echo(f"{k}: {v}")
