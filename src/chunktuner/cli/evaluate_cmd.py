"""``chunk-tune evaluate`` — run metrics for each strategy/config."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

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


def register(app: typer.Typer) -> None:
    @app.command("evaluate")
    def evaluate_cmd(
        path: Path = typer.Argument(..., exists=True),
        strategies: str = typer.Option(
            "fixed_tokens,recursive_character",
            "--strategies",
        ),
        use_case: str = typer.Option("rag_qa", "--use-case"),
        max_docs: int = typer.Option(20, "--max-docs"),
        top_k: int = typer.Option(5, "--top-k", help=TOP_K_HELP),
        config: Path | None = typer.Option(None, "--config"),
        output_format: str = typer.Option("table", "--output-format"),
        embedding_model: str | None = typer.Option(
            None,
            "--embedding-model",
            help="If set, calls LiteLLM (paid). Omit to use dummy embeddings.",
        ),
        api_base: str | None = typer.Option(
            None,
            "--api-base",
            help="Custom OpenAI-compatible API base URL (LM Studio, Ollama, Azure, etc.).",
        ),
        api_key: str | None = typer.Option(
            None,
            "--api-key",
            help="API key for the embedding provider (overrides env / workspace).",
        ),
        llm_model: str | None = typer.Option(
            None,
            "--llm-model",
            help="LLM for generation-related paths; defaults to llm_model in .autochunk.yaml.",
        ),
        yes: bool = typer.Option(
            False, "--yes", help="Skip confirmation for paid embedding / LLM calls"
        ),
        dataset: Path | None = typer.Option(
            None, "--dataset", exists=True, dir_okay=False, help=DATASET_HELP
        ),
        generate_dataset: bool = typer.Option(
            False, "--generate-dataset", help=GENERATE_DATASET_HELP
        ),
        retrieval_scope: str = retrieval_scope_option(),
        no_cache: bool = typer.Option(False, "--no-cache", help=NO_CACHE_HELP),
    ) -> None:
        """Evaluate chunking strategies (dummy embeddings by default)."""
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
        embed_fn = (
            LiteLLMEmbeddingFunction(
                resolved_embedding_model,
                api_base=resolved_api_base,
                api_key=resolved_api_key,
            )
            if resolved_embedding_model
            else DummyEmbeddingFunction()
        )
        embed_fn = maybe_cache_embeddings(embed_fn, cache_dir, enabled=not no_cache)
        root = path.resolve().parent if path.is_file() else path.resolve()
        fi = FileIngestor(root=root)
        docs = fi.ingest_path(path) if path.is_file() else fi.ingest_dir(path)
        if not docs:
            exts = ", ".join(sorted(FileIngestor.SUPPORTED_EXTENSIONS))
            typer.secho(
                f"No supported documents found in {path}.\nSupported extensions: {exts}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
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
        scorer = ScoreCalculator(
            cast(UseCase, use_case), retrieval_scope=cast(RetrievalScope, retrieval_scope)
        )
        ev = Evaluator(
            embed_fn,
            top_k=top_k or ws.top_k,
            llm_answer_model=resolved_llm_model,
            llm_api_base=resolved_api_base,
            llm_api_key=resolved_api_key,
            retrieval_scope=cast(RetrievalScope, retrieval_scope),
        )
        results = []
        total_jobs = sum(len(default_registry.get(n).default_param_grid()) for n in names)
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            transient=True,
        ) as progress:
            task_id = progress.add_task("Evaluating...", total=total_jobs)
            for n in names:
                strat = default_registry.get(n)
                for params in strat.default_param_grid():
                    progress.update(task_id, description=f"Evaluating {n} {params!r}...")
                    cfg = ChunkConfig(name=n, params=dict(params))
                    results.append(ev.evaluate(strat, cfg, docs, ds, scorer=scorer))
                    progress.advance(task_id)
        if output_format == "json":
            typer.echo(__import__("json").dumps([r.model_dump() for r in results], indent=2))
        elif output_format == "yaml":
            typer.echo(
                __import__("yaml").safe_dump(
                    [r.model_dump() for r in results],
                    sort_keys=False,
                )
            )
        else:
            for r in results:
                typer.echo(
                    f"{r.strategy_name} {r.config.params} score={r.score:.4f} "
                    f"recall={r.metrics.token_recall:.3f} iou={r.metrics.token_iou:.3f}"
                )
