"""``chunk-tune recommend`` — full grid + best config."""

from pathlib import Path
from typing import cast

import typer

from chunktuner.cache.wrapped_embeddings import maybe_cache_embeddings
from chunktuner.cli.common import (
    CONFIG_HELP,
    DATASET_HELP,
    DEFAULT_USE_CASE,
    GENERATE_DATASET_HELP,
    NO_CACHE_HELP,
    TOP_K_HELP,
    USE_CASE_HELP,
    baseline_unsupported,
    check_dataset_flags,
    check_strategy_names,
    check_token_budget,
    emit_output,
    ingest_corpus,
    load_workspace,
    max_docs_help,
    registry_for,
    resolve_eval_dataset,
    retrieval_scope_option,
    select_supported_docs,
    tokenizer_encoding,
    warn,
    ws_setting,
)
from chunktuner.config import (
    resolve_provider_config,
    workspace_cache_dir,
)
from chunktuner.eval.embeddings import DummyEmbeddingFunction, LiteLLMEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.models import RetrievalScope, UseCase
from chunktuner.tuner.auto_tuner import AutoTuner


def register(app: typer.Typer) -> None:
    @app.command("recommend")
    def recommend_cmd(
        path: Path = typer.Argument(..., exists=True),
        strategies: str = typer.Option(
            "fixed_tokens,recursive_character",
            "--strategies",
        ),
        use_case: str | None = typer.Option(None, "--use-case", help=USE_CASE_HELP),
        max_docs: int | None = typer.Option(None, "--max-docs", help=max_docs_help(20)),
        top_k: int = typer.Option(5, "--top-k", help=TOP_K_HELP),
        config: Path | None = typer.Option(None, "--config", help=CONFIG_HELP),
        output_format: str = typer.Option("table", "--output-format"),
        embedding_model: str | None = typer.Option(None, "--embedding-model"),
        api_base: str | None = typer.Option(
            None,
            "--api-base",
            help=(
                "Custom base URL for an OpenAI-compatible API. Overrides api_base in "
                ".autochunk.yaml and CHUNKTUNER_API_BASE. Examples: http://localhost:1234/v1 "
                "(LM Studio), http://localhost:11434/v1 (Ollama)."
            ),
        ),
        api_key: str | None = typer.Option(
            None,
            "--api-key",
            help=(
                "API key for the provider. Overrides api_key in .autochunk.yaml and "
                "CHUNKTUNER_API_KEY. For LM Studio any non-empty string is accepted."
            ),
        ),
        llm_model: str | None = typer.Option(
            None,
            "--llm-model",
            help=(
                "LLM for agentic strategy and generation metrics. Defaults to llm_model in "
                ".autochunk.yaml. Examples: claude-3-haiku-20240307, gemini/gemini-1.5-flash, "
                "openai/llama-3.2-3b-instruct (with --api-base for local servers)."
            ),
        ),
        yes: bool = typer.Option(False, "--yes"),
        no_baseline: bool = typer.Option(False, "--no-baseline"),
        dataset: Path | None = typer.Option(
            None, "--dataset", exists=True, dir_okay=False, help=DATASET_HELP
        ),
        generate_dataset: bool = typer.Option(
            False, "--generate-dataset", help=GENERATE_DATASET_HELP
        ),
        retrieval_scope: str = retrieval_scope_option(),
        no_cache: bool = typer.Option(False, "--no-cache", help=NO_CACHE_HELP),
    ) -> None:
        """Run tuner and print the best chunking configuration."""
        check_dataset_flags(dataset, generate_dataset)
        ws = load_workspace(config)
        use_case = ws_setting(ws, "use_case", use_case, DEFAULT_USE_CASE)
        max_docs = ws_setting(ws, "max_docs", max_docs, 20)
        registry = registry_for(ws)
        names = [s.strip() for s in strategies.split(",") if s.strip()]
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
        docs, root = ingest_corpus(path, ws)
        if not docs:
            exts = ", ".join(sorted(FileIngestor.SUPPORTED_EXTENSIONS))
            typer.secho(
                f"No supported documents found in {path}.\nSupported extensions: {exts}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        docs = select_supported_docs(docs, names, registry, path)[:max_docs]
        check_token_budget(
            ws,
            docs,
            names,
            {n: registry.get(n).default_param_grid() for n in names},
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
            embed_fn,
            top_k=top_k or ws.top_k,
            encoding_name=tokenizer_encoding(ws),
            llm_answer_model=resolved_llm_model,
            llm_api_base=resolved_api_base,
            llm_api_key=resolved_api_key,
            retrieval_scope=cast(RetrievalScope, retrieval_scope),
        )
        tuner = AutoTuner(registry, ev, scorer, encoding_name=tokenizer_encoding(ws))
        # Every remaining doc is supported by every strategy in ``names``; the tuner's
        # content type (strategy selection, baseline, Recommendation.content_type) is the
        # first remaining doc's, as before. Mixed supported types (e.g. markdown + text)
        # are evaluated together.
        content_type = docs[0].content_type
        baseline = not no_baseline
        if baseline:
            bl_msg = baseline_unsupported(tuner, content_type, docs)
            if bl_msg:
                warn(bl_msg)
                baseline = False
        rec = tuner.recommend(
            docs,
            cast(UseCase, use_case),
            content_type=content_type,
            strategies=names,
            max_docs=max_docs,
            baseline=baseline,
            dataset=ds,
        )
        emit_output(rec, output_format)
