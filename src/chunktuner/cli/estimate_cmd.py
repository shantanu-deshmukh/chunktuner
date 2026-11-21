"""``chunk-tune estimate`` — dry-run cost estimate."""

from pathlib import Path

import typer

from chunktuner.cli.common import (
    CONFIG_HELP,
    USE_CASE_HELP,
    check_strategy_names,
    ingest_corpus,
    load_workspace,
    max_docs_help,
    registry_for,
    select_supported_docs,
    tokenizer_encoding,
    validate_output_format,
    ws_setting,
)
from chunktuner.config import DEFAULT_EMBEDDING_MODEL
from chunktuner.eval.cost_estimator import CostEstimator


def register(app: typer.Typer) -> None:
    @app.command("estimate")
    def estimate_cmd(
        path: Path = typer.Argument(..., exists=True),
        strategies: str = typer.Option(
            "fixed_tokens,recursive_character",
            "--strategies",
            help="Comma-separated strategy names",
        ),
        use_case: str | None = typer.Option(
            None, "--use-case", help=f"{USE_CASE_HELP} (does not affect the estimate)"
        ),
        max_docs: int | None = typer.Option(None, "--max-docs", help=max_docs_help(100)),
        config: Path | None = typer.Option(None, "--config", help=CONFIG_HELP),
        output_format: str = typer.Option("table", "--output-format"),
        generate_dataset: bool = typer.Option(
            False,
            "--generate-dataset",
            help=(
                "Estimate LLM eval-dataset generation (as run by --generate-dataset on "
                "evaluate/compare/recommend) with the configured LLM model."
            ),
        ),
        llm_model: str | None = typer.Option(
            None, "--llm-model", help="LLM for dataset generation; default: workspace llm_model."
        ),
    ) -> None:
        """Estimate tokens, cost, and wall time (no API calls).

        Like evaluate, skips docs whose content type a selected strategy does not support.
        """
        validate_output_format(output_format)
        ws = load_workspace(config)
        max_docs = ws_setting(ws, "max_docs", max_docs, 100)
        registry = registry_for(ws)
        names = [s.strip() for s in strategies.split(",") if s.strip()]
        check_strategy_names(names, registry)
        docs, _root = ingest_corpus(path, ws)
        # Same doc selection as evaluate/compare/recommend: skip unsupported content types
        # (warning on stderr), exit 1 if none remain, then cap at max_docs.
        docs = select_supported_docs(docs, names, registry, path)[:max_docs]
        param_grid: dict[str, list[dict]] = {}
        for n in names:
            param_grid[n] = registry.get(n).default_param_grid()
        est = CostEstimator(encoding_name=tokenizer_encoding(ws)).estimate(
            docs,
            names,
            param_grid,
            ws.embedding_model or DEFAULT_EMBEDDING_MODEL,
            generate_dataset=True,
            llm_model=(llm_model or ws.llm_model) if generate_dataset else None,
        )
        if output_format == "json":
            typer.echo(est.model_dump_json(indent=2, exclude_none=True))
        elif output_format == "yaml":
            typer.echo(
                __import__("yaml").safe_dump(est.model_dump(exclude_none=True), sort_keys=False)
            )
        else:
            typer.echo(f"Total embedding tokens (est.): {est.total_tokens}")
            typer.echo(f"Embedding cost (USD est.): {est.embedding_cost_usd}")
            typer.echo(f"LLM dataset cost (USD est.): {est.llm_cost_usd}")
            if est.llm_dataset_tokens is not None:
                typer.echo(f"LLM dataset tokens (est., {est.llm_model}): {est.llm_dataset_tokens}")
            typer.echo(f"Wall time (min est.): {est.estimated_wall_time_min}")
            typer.echo(f"Strategy-config combinations: {est.strategy_configs}")
