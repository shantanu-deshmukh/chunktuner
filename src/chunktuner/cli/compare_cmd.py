"""``chunk-tune compare`` — side-by-side strategy comparison."""

from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from chunktuner.cache.wrapped_embeddings import maybe_cache_embeddings
from chunktuner.chunking import default_registry
from chunktuner.cli.common import (
    DATASET_HELP,
    GENERATE_DATASET_HELP,
    NO_CACHE_HELP,
    TOP_K_HELP,
    check_dataset_flags,
    load_workspace_path,
    resolve_eval_dataset,
    retrieval_scope_option,
)
from chunktuner.config import (
    load_workspace_config,
    resolve_provider_config,
    workspace_cache_dir,
)
from chunktuner.eval.embeddings import DummyEmbeddingFunction, LiteLLMEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.models import ChunkConfig, RetrievalScope, UseCase

console = Console()


def register(app: typer.Typer) -> None:
    @app.command("compare")
    def compare_cmd(
        path: Path = typer.Argument(..., exists=True),
        strategies: str = typer.Option(..., "--strategies", help="Comma-separated names"),
        use_case: str = typer.Option("rag_qa", "--use-case"),
        max_docs: int = typer.Option(15, "--max-docs"),
        top_k: int = typer.Option(5, "--top-k", help=TOP_K_HELP),
        config: Path | None = typer.Option(None, "--config"),
        report: Path | None = typer.Option(None, "--report", help="Write Markdown report path"),
        embedding_model: str | None = typer.Option(None, "--embedding-model"),
        api_base: str | None = typer.Option(None, "--api-base"),
        api_key: str | None = typer.Option(None, "--api-key"),
        llm_model: str | None = typer.Option(None, "--llm-model"),
        yes: bool = typer.Option(False, "--yes"),
        dataset: Path | None = typer.Option(
            None, "--dataset", exists=True, dir_okay=False, help=DATASET_HELP
        ),
        generate_dataset: bool = typer.Option(
            False, "--generate-dataset", help=GENERATE_DATASET_HELP
        ),
        retrieval_scope: str = retrieval_scope_option(),
        no_cache: bool = typer.Option(False, "--no-cache", help=NO_CACHE_HELP),
    ) -> None:
        """Compare a small set of strategies on the same corpus."""
        check_dataset_flags(dataset, generate_dataset)
        ws = load_workspace_config(load_workspace_path(config))
        cache_dir = workspace_cache_dir(ws)
        resolved_api_base, resolved_api_key = resolve_provider_config(ws)
        resolved_api_base = api_base or resolved_api_base
        resolved_api_key = api_key or resolved_api_key
        resolved_embedding_model = (
            embedding_model if embedding_model is not None else ws.embedding_model
        )
        resolved_llm_model = llm_model if llm_model is not None else ws.llm_model

        if resolved_embedding_model and not yes:
            typer.confirm(
                "Embedding model set — this will call external APIs. Continue?",
                default=False,
                abort=True,
            )
        embed = (
            LiteLLMEmbeddingFunction(
                resolved_embedding_model,
                api_base=resolved_api_base,
                api_key=resolved_api_key,
            )
            if resolved_embedding_model
            else DummyEmbeddingFunction()
        )
        embed = maybe_cache_embeddings(embed, cache_dir, enabled=not no_cache)
        root = path.resolve().parent if path.is_file() else path.resolve()
        fi = FileIngestor(root=root)
        docs = fi.ingest_path(path) if path.is_file() else fi.ingest_dir(path)
        docs = docs[:max_docs]
        ds = resolve_eval_dataset(
            docs,
            root=root,
            dataset=dataset,
            generate_dataset=generate_dataset,
            llm_model=resolved_llm_model,
            api_base=resolved_api_base,
            api_key=resolved_api_key,
            cache_dir=cache_dir,
            yes=yes,
        )
        names = [s.strip() for s in strategies.split(",") if s.strip()]
        if not names:
            typer.echo("Provide at least one strategy in --strategies", err=True)
            raise typer.Exit(2)
        scorer = ScoreCalculator(
            cast(UseCase, use_case), retrieval_scope=cast(RetrievalScope, retrieval_scope)
        )
        ev = Evaluator(
            embed,
            top_k=top_k or ws.top_k,
            llm_answer_model=resolved_llm_model,
            llm_api_base=resolved_api_base,
            llm_api_key=resolved_api_key,
            retrieval_scope=cast(RetrievalScope, retrieval_scope),
        )
        rows = []
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            transient=True,
        ) as progress:
            tid = progress.add_task("Comparing...", total=len(names))
            for n in names:
                progress.update(tid, description=f"Comparing {n}...")
                try:
                    strat = default_registry.get(n)
                except KeyError:
                    available = sorted(default_registry.names())
                    typer.secho(
                        f"Unknown strategy {n!r}. Available: {available}",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    raise typer.Exit(2) from None
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
