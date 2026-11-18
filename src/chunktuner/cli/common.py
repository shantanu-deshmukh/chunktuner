"""Shared CLI helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

console = Console()

_VALID_FORMATS = frozenset({"json", "yaml", "table"})


def validate_output_format(fmt: str) -> None:
    if fmt not in _VALID_FORMATS:
        raise typer.BadParameter(
            f"Unknown output format {fmt!r}. Valid choices: {sorted(_VALID_FORMATS)}",
            param_hint="'--output-format'",
        )


def load_workspace_path(config: Path | None) -> Path | None:
    if config is not None:
        return config
    p = Path.cwd() / ".autochunk.yaml"
    return p if p.is_file() else None


def emit_output(data: Any, fmt: str) -> None:
    validate_output_format(fmt)
    if isinstance(data, BaseModel):
        payload = data.model_dump(mode="json")
    elif isinstance(data, list) and data and isinstance(data[0], BaseModel):
        payload = [x.model_dump(mode="json") for x in data]
    else:
        payload = data

    if fmt == "json":
        console.print(json.dumps(payload, indent=2, default=str))
    elif fmt == "yaml":
        console.print(yaml.safe_dump(payload, sort_keys=False))
    else:
        _emit_table(payload)


def _emit_table(payload: Any) -> None:
    if isinstance(payload, dict) and "best" in payload:
        rec = payload
        console.print("[bold]Best configuration[/bold]")
        console.print_json(json.dumps(rec.get("best"), indent=2))
        return
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        keys = list(payload[0].keys())
        table = Table(show_header=True)
        for k in keys:
            table.add_column(k)
        for row in payload[:50]:
            table.add_row(*[str(row.get(k, "")) for k in keys])
        console.print(table)
        return
    console.print(str(payload))


# ---------------------------------------------------------------------------
# Shared evaluate / compare / recommend options and helpers
# ---------------------------------------------------------------------------

TOP_K_HELP = (
    "Kept for compatibility; does NOT affect metrics or scores. Retrieval depth is the "
    "effective k = 2000-token context budget / avg chunk tokens."
)
DATASET_HELP = (
    "JSON/YAML eval dataset ('queries': question, document_id, answer_spans, "
    "reference_answer). document_id may be the file path relative to PATH, an absolute "
    "path, or the file name. Mutually exclusive with --generate-dataset."
)
GENERATE_DATASET_HELP = (
    "Generate eval questions with the configured LLM (--llm-model / llm_model; paid unless "
    "local). Run `chunk-tune estimate --generate-dataset` first. Prompts unless --yes; the "
    "result is cached under cache_dir and reused on re-runs."
)
RETRIEVAL_SCOPE_HELP = (
    "Which ranking feeds score's rank metrics (mrr / recall@1): 'document' ranks only the "
    "gold document's chunks; 'corpus' ranks all chunks of all documents."
)
NO_CACHE_HELP = (
    "Disable the SQLite embedding cache under cache_dir (dummy embeddings are never cached)."
)


_RETRIEVAL_SCOPES = ("document", "corpus")


def _validate_retrieval_scope(value: str) -> str:
    if value not in _RETRIEVAL_SCOPES:
        raise typer.BadParameter(f"must be one of: {', '.join(_RETRIEVAL_SCOPES)}")
    return value


def retrieval_scope_option() -> Any:
    # A callback is used instead of `click_type=` so this works on every typer version
    # allowed by pyproject (typer 0.29 removed the `click_type` keyword).
    return typer.Option(
        "document",
        "--retrieval-scope",
        callback=_validate_retrieval_scope,
        help=f"{RETRIEVAL_SCOPE_HELP} [document|corpus]",
    )


def warn(msg: str) -> None:
    """Print a warning on stderr (keeps stdout clean for JSON/YAML output)."""
    typer.secho(f"Warning: {msg}", fg=typer.colors.YELLOW, err=True)


def check_dataset_flags(dataset: Path | None, generate_dataset: bool) -> None:
    if dataset is not None and generate_dataset:
        raise typer.BadParameter(
            "--dataset and --generate-dataset are mutually exclusive",
            param_hint="'--dataset' / '--generate-dataset'",
        )


def resolve_eval_dataset(
    docs: list,
    *,
    root: Path,
    dataset: Path | None,
    generate_dataset: bool,
    llm_model: str,
    api_base: str | None,
    api_key: str | None,
    cache_dir: Path,
    yes: bool,
) -> Any:
    """Return the `EvalDataset` for a CLI run (user file, cached LLM generation, or trivial)."""
    from chunktuner.eval.cost_estimator import estimate_dataset_generation
    from chunktuner.eval.dataset_sources import (
        DEFAULT_MAX_QUERIES,
        DEFAULT_QUESTIONS_PER_DOC,
        load_or_generate_dataset,
        load_user_dataset,
        trivial_dataset_warning,
    )
    from chunktuner.eval.trivial_dataset import trivial_dataset_for_docs

    check_dataset_flags(dataset, generate_dataset)
    if dataset is not None:
        try:
            ds, warnings = load_user_dataset(dataset, docs, root)
        except (ValueError, OSError) as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from None
        for w in warnings:
            warn(w)
        return ds
    if generate_dataset:

        def _confirm() -> None:
            if yes:
                return
            tokens, cost = estimate_dataset_generation(
                docs,
                llm_model,
                questions_per_doc=DEFAULT_QUESTIONS_PER_DOC,
                max_queries=DEFAULT_MAX_QUERIES,
            )
            typer.confirm(
                f"Generating an eval dataset calls LLM {llm_model!r} "
                f"(~{tokens} tokens, ~${cost:.4f} est.). Continue?",
                default=False,
                abort=True,
            )

        try:
            ds, path, hit = load_or_generate_dataset(
                docs,
                llm_model=llm_model,
                cache_dir=cache_dir,
                llm_api_base=api_base,
                llm_api_key=api_key,
                before_generate=_confirm,
            )
        except ValueError as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None
        verb = "Reusing cached" if hit else "Cached new"
        typer.echo(f"{verb} generated dataset ({len(ds.queries)} queries): {path}", err=True)
        if len(ds.queries) < 20:
            warn(f"Only {len(ds.queries)} eval queries (20+ recommended); metrics will be noisy.")
        return ds
    warn(trivial_dataset_warning("--dataset FILE", "--generate-dataset"))
    return trivial_dataset_for_docs(docs)
