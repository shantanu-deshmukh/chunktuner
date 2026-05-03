"""``chunk-tune compare`` — side-by-side strategy comparison."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from chunktuner.chunking import default_registry
from chunktuner.cli.common import load_workspace_path
from chunktuner.config import load_workspace_config
from chunktuner.eval.embeddings import DummyEmbeddingFunction, LiteLLMEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.eval.trivial_dataset import trivial_dataset_for_docs
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.models import ChunkConfig, UseCase

console = Console()


def register(app: typer.Typer) -> None:
    @app.command("compare")
    def compare_cmd(
        path: Path = typer.Argument(..., exists=True),
        strategies: str = typer.Option(..., "--strategies", help="Comma-separated names"),
        use_case: str = typer.Option("rag_qa", "--use-case"),
        max_docs: int = typer.Option(15, "--max-docs"),
        top_k: int = typer.Option(5, "--top-k"),
        config: Path | None = typer.Option(None, "--config"),
        report: Path | None = typer.Option(None, "--report", help="Write Markdown report path"),
        embedding_model: str | None = typer.Option(None, "--embedding-model"),
        yes: bool = typer.Option(False, "--yes"),
    ) -> None:
        """Compare a small set of strategies on the same corpus."""
        if embedding_model and not yes:
            typer.confirm(
                "Embedding model set — this will call external APIs. Continue?",
                default=False,
                abort=True,
            )
        embed = (
            LiteLLMEmbeddingFunction(embedding_model)
            if embedding_model
            else DummyEmbeddingFunction()
        )
        ws = load_workspace_config(load_workspace_path(config))
        root = path.resolve().parent if path.is_file() else path.resolve()
        fi = FileIngestor(root=root)
        docs = fi.ingest_path(path) if path.is_file() else fi.ingest_dir(path)
        docs = docs[:max_docs]
        ds = trivial_dataset_for_docs(docs)
        names = [s.strip() for s in strategies.split(",") if s.strip()]
        if not names:
            typer.echo("Provide at least one strategy in --strategies", err=True)
            raise typer.Exit(2)
        scorer = ScoreCalculator(cast(UseCase, use_case))
        ev = Evaluator(embed, top_k=top_k or ws.top_k)
        rows = []
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            transient=True,
        ) as progress:
            tid = progress.add_task("Comparing...", total=len(names))
            for n in names:
                progress.update(tid, description=f"Comparing {n}...")
                strat = default_registry.get(n)
                cfg = ChunkConfig(name=n, params=strat.default_param_grid()[0])
                res = ev.evaluate(strat, cfg, docs, ds, scorer=scorer)
                rows.append(
                    {
                        "strategy": n,
                        "score": round(res.score, 4),
                        "token_recall": round(res.metrics.token_recall, 4),
                        "mrr": round(res.metrics.mrr, 4),
                        "params": str(cfg.params),
                    }
                )
                progress.advance(tid)
        table = Table(title="Strategy comparison")
        for key in rows[0].keys():
            table.add_column(key)
        for r in rows:
            table.add_row(*[str(r[k]) for k in rows[0].keys()])
        console.print(table)
        if report:
            lines = [
                "# Chunking comparison\n",
                f"Path: `{path}`\n\n",
                "| " + " | ".join(rows[0].keys()) + " |\n",
            ]
            lines.append("| " + " | ".join("---" for _ in rows[0]) + " |\n")
            for r in rows:
                lines.append("| " + " | ".join(str(r[k]) for k in rows[0].keys()) + " |\n")
            report.write_text("".join(lines), encoding="utf-8")
            typer.echo(f"Wrote {report}")
