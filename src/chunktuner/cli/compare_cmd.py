"""``chunk-tune compare`` — side-by-side strategy comparison."""

from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from chunktuner.cache.wrapped_embeddings import maybe_cache_embeddings
from chunktuner.cli.common import (
    CONFIG_HELP,
    DATASET_HELP,
    DEFAULT_USE_CASE,
    GENERATE_DATASET_HELP,
    NO_CACHE_HELP,
    TOP_K_HELP,
    USE_CASE_HELP,
    check_dataset_flags,
    check_strategy_names,
    check_token_budget,
    ingest_corpus,
    load_workspace,
    max_docs_help,
    registry_for,
    resolve_eval_dataset,
    retrieval_scope_option,
    select_supported_docs,
    tokenizer_encoding,
    ws_setting,
)
from chunktuner.config import (
    resolve_provider_config,
    workspace_cache_dir,
)
from chunktuner.eval.embeddings import DummyEmbeddingFunction, LiteLLMEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.models import ChunkConfig, RetrievalScope, UseCase

console = Console()


def register(app: typer.Typer) -> None:
    @app.command("compare")
    def compare_cmd(
        path: Path = typer.Argument(..., exists=True),
        strategies: str = typer.Option(..., "--strategies", help="Comma-separated names"),
        use_case: str | None = typer.Option(None, "--use-case", help=USE_CASE_HELP),
        max_docs: int | None = typer.Option(None, "--max-docs", help=max_docs_help(15)),
        top_k: int = typer.Option(5, "--top-k", help=TOP_K_HELP),
        config: Path | None = typer.Option(None, "--config", help=CONFIG_HELP),
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
        ws = load_workspace(config)
        use_case = ws_setting(ws, "use_case", use_case, DEFAULT_USE_CASE)
        max_docs = ws_setting(ws, "max_docs", max_docs, 15)
        registry = registry_for(ws)
        names = [s.strip() for s in strategies.split(",") if s.strip()]
        if not names:
            typer.echo("Provide at least one strategy in --strategies", err=True)
            raise typer.Exit(2)
        check_strategy_names(names, registry)
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
        docs, root = ingest_corpus(path, ws)
        docs = select_supported_docs(docs, names, registry, path)[:max_docs]
        # compare runs only the first config of each strategy's default grid.
        check_token_budget(
            ws,
            docs,
            names,
            {n: registry.get(n).default_param_grid()[:1] for n in names},
            embedding_model=resolved_embedding_model,
            yes=yes,
        )
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
            encoding_name=tokenizer_encoding(ws),
        )
        scorer = ScoreCalculator(
            cast(UseCase, use_case), retrieval_scope=cast(RetrievalScope, retrieval_scope)
        )
        ev = Evaluator(
            embed,
            top_k=top_k or ws.top_k,
            encoding_name=tokenizer_encoding(ws),
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
                strat = registry.get(n)
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
