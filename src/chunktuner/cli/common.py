"""Shared CLI helpers."""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table

from chunktuner.chunking.registry import StrategyRegistry
from chunktuner.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOKENIZER_ENCODING,
    WorkspaceConfig,
    load_workspace_config,
)
from chunktuner.models import Document

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


RETRIEVAL_MODE_HELP = (
    "First-stage retrieval (default: retrieval_mode in .autochunk.yaml, else dense): 'dense' "
    "ranks by embeddings; 'bm25' ranks lexically and makes no embedding calls; 'hybrid' fuses "
    "both. bm25 and hybrid need the 'bm25' extra."
)
RERANKER_HELP = (
    "Optional second-stage reranker (default: reranker in .autochunk.yaml, else none): "
    "'dummy' (free, lexical, for tests); 'local' or 'local:<hf-model-id>' (cross-encoder, "
    "needs the 'late' extra, downloads a Hugging Face model); anything else is a LiteLLM "
    "rerank model string, e.g. cohere/rerank-english-v3.0 (calls a paid API; its key comes "
    "from the provider's env vars, not --api-base / --api-key)."
)

_RETRIEVAL_MODES = ("dense", "bm25", "hybrid")
DEFAULT_RETRIEVAL_MODE = "dense"


def _validate_retrieval_mode(value: str | None) -> str | None:
    if value is not None and value not in _RETRIEVAL_MODES:
        raise typer.BadParameter(f"must be one of: {', '.join(_RETRIEVAL_MODES)}")
    return value


def retrieval_mode_option() -> Any:
    return typer.Option(
        None,
        "--retrieval-mode",
        callback=_validate_retrieval_mode,
        help=f"{RETRIEVAL_MODE_HELP} [dense|bm25|hybrid]",
    )


def _validate_reranker(value: str | None) -> str | None:
    if value is None:
        return None
    from chunktuner.eval.rerankers import validate_reranker_spec

    try:
        return validate_reranker_spec(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def reranker_option() -> Any:
    return typer.Option(None, "--reranker", callback=_validate_reranker, help=RERANKER_HELP)


def is_paid_reranker_spec(spec: str | None) -> bool:
    """True for a LiteLLM rerank spec (not ``dummy`` / ``local`` / ``local:<id>``)."""
    return (
        spec is not None and spec != "dummy" and spec != "local" and not spec.startswith("local:")
    )


def resolve_retrieval(
    ws: WorkspaceConfig, retrieval_mode: str | None, reranker: str | None
) -> tuple[str, str | None]:
    """``(retrieval_mode, reranker spec)``: flag > yaml key > dense / no reranker."""
    mode = ws_setting(ws, "retrieval_mode", retrieval_mode, DEFAULT_RETRIEVAL_MODE)
    spec = ws_setting(ws, "reranker", reranker, None)
    return mode, spec


def build_retrieval(mode: str, spec: str | None, *, yes: bool) -> Any:
    """Check the bm25 extra for ``mode`` and build the reranker for ``spec``.

    Exits 1 with the install hint when an optional extra is missing. A LiteLLM rerank spec
    asks for confirmation unless ``yes``; it gets no api_base / api_key (the embedding
    provider's may belong to another provider), so LiteLLM reads the provider env vars.
    """
    from chunktuner.eval.lexical import Bm25Index
    from chunktuner.eval.rerankers import reranker_from_spec

    try:
        if mode != "dense":
            Bm25Index([])
        if is_paid_reranker_spec(spec) and not yes:
            typer.confirm(
                f"Reranker {spec!r} calls an external rerank API (paid). Continue?",
                default=False,
                abort=True,
            )
        return reranker_from_spec(spec)
    except ImportError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None


def warn_reranker_overflow(results: list[Any], reranker: Any) -> None:
    """Warn once if any result has chunks longer than the reranker's input window."""
    ratios = [
        r.metrics.reranker_overflow_ratio
        for r in results
        if r is not None and r.metrics.reranker_overflow_ratio
    ]
    if not ratios or reranker is None:
        return
    worst = max(ratios)
    scope = "of chunks" if len(results) == 1 else "of chunks (worst config)"
    warn(
        f"{worst:.0%} {scope} exceed the reranker's input window "
        f"({reranker.max_input_tokens} tokens, estimated with tiktoken); they are truncated "
        "by the reranker."
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
    encoding_name: str = DEFAULT_TOKENIZER_ENCODING,
) -> Any:
    """Return the `EvalDataset` for a CLI run (user file, cached LLM generation, or trivial).

    ``encoding_name`` only affects the token count shown in the generation confirm prompt.
    """
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
                encoding_name=encoding_name,
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


# ---------------------------------------------------------------------------
# Workspace (.autochunk.yaml) settings
# ---------------------------------------------------------------------------

DEFAULT_USE_CASE = "rag_qa"
USE_CASE_HELP = "Scoring profile (default: use_case in .autochunk.yaml, else rag_qa)"
CONFIG_HELP = "Path to .autochunk.yaml (default: ./.autochunk.yaml if present)"


def max_docs_help(default: int) -> str:
    return f"Max documents to use (default: max_docs in .autochunk.yaml, else {default})"


def ws_setting(ws: WorkspaceConfig, field: str, cli_value: Any, default: Any) -> Any:
    """Resolve a setting: explicit CLI flag > key written in the yaml > command default.

    A key counts as written only if it is present in the loaded yaml (``model_fields_set``),
    so the `WorkspaceConfig` field defaults never override a command's own default.
    """
    if cli_value is not None:
        return cli_value
    if field in ws.model_fields_set:
        return getattr(ws, field)
    return default


def configure_logging(ws: WorkspaceConfig) -> None:
    """Apply ``log_level`` from the yaml to the ``chunktuner`` loggers (only when written)."""
    if "log_level" not in ws.model_fields_set:
        return
    level = logging.getLevelName(str(ws.log_level).upper())
    if not isinstance(level, int):
        typer.secho(
            f"Error: invalid log_level {ws.log_level!r} in .autochunk.yaml "
            "(use DEBUG, INFO, WARNING, ERROR or CRITICAL)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    root = logging.getLogger()
    if not root.handlers:
        # Handler on stderr; third-party loggers keep the root's WARNING level.
        logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("chunktuner").setLevel(level)


def load_workspace(config: Path | None) -> WorkspaceConfig:
    """Load ``.autochunk.yaml`` (``--config`` or cwd) and apply its ``log_level``."""
    path = load_workspace_path(config)
    try:
        ws = load_workspace_config(path)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'config'}: {err['msg']}"
            for err in exc.errors()
        )
        typer.secho(
            f"Error: invalid {path or '.autochunk.yaml'}: {problems}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2) from None
    configure_logging(ws)
    return ws


def tokenizer_encoding(ws: WorkspaceConfig) -> str:
    """``tokenizer_encoding`` from the yaml (validated), else ``cl100k_base``."""
    enc = ws_setting(ws, "tokenizer_encoding", None, DEFAULT_TOKENIZER_ENCODING)
    if enc != DEFAULT_TOKENIZER_ENCODING:
        import tiktoken

        try:
            tiktoken.get_encoding(enc)
        except (ValueError, KeyError):
            typer.secho(
                f"Error: unknown tokenizer_encoding {enc!r} in .autochunk.yaml "
                "(any tiktoken encoding, e.g. cl100k_base, o200k_base)",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2) from None
    return enc


def registry_for(ws: WorkspaceConfig) -> StrategyRegistry:
    """Strategy registry using the workspace tokenizer (shared default registry otherwise)."""
    enc = tokenizer_encoding(ws)
    if enc == DEFAULT_TOKENIZER_ENCODING:
        from chunktuner.chunking import default_registry

        return default_registry
    from chunktuner.chunking.bootstrap import build_full_registry

    return build_full_registry(enc)


def file_ingestor(root: Path, ws: WorkspaceConfig) -> Any:
    """`FileIngestor` for ``root`` applying the yaml ``strip_patterns`` (validated)."""
    from chunktuner.ingestion.file_ingestor import FileIngestor

    patterns = list(ws.strip_patterns or [])
    for pat in patterns:
        try:
            re.compile(pat)
        except re.error as exc:
            typer.secho(
                f"Error: invalid strip_patterns regex {pat!r} in .autochunk.yaml: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2) from None
    return FileIngestor(root=root, strip_patterns=patterns)


def ingest_corpus(path: Path, ws: WorkspaceConfig) -> tuple[list[Document], Path]:
    """Ingest ``path`` (file or directory) with workspace settings; returns ``(docs, root)``."""
    root = path.resolve().parent if path.is_file() else path.resolve()
    fi = file_ingestor(root, ws)
    docs = fi.ingest_path(path) if path.is_file() else fi.ingest_dir(path)
    return docs, root


def check_strategy_names(names: list[str], registry: StrategyRegistry) -> None:
    """Exit 2 on an unknown strategy name (same message as ``compare``)."""
    for n in names:
        try:
            registry.get(n)
        except KeyError:
            available = sorted(registry.names())
            typer.secho(
                f"Unknown strategy {n!r}. Available: {available}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(2) from None


# Strategies never suggested for skipped docs (explicit-only fallbacks; see AutoTuner).
_NO_SUGGEST = frozenset({"late_chunking"})


def cli_strategies_hint(names: list[str]) -> str:
    return "--strategies " + ",".join(names)


def filter_supported_docs(
    docs: list[Document],
    names: list[str],
    registry: StrategyRegistry,
    *,
    hint: Callable[[list[str]], str] = cli_strategies_hint,
) -> tuple[list[Document], list[str]]:
    """Keep docs whose content_type every strategy in ``names`` supports.

    Returns ``(kept, messages)`` with one message per skipped content type, suggesting the
    registered strategies that do support it. Order of ``docs`` is preserved. Shared by the
    CLI (``evaluate`` / ``compare`` / ``recommend``) and the MCP service.
    """
    supported: set[str] | None = None
    for n in names:
        types = set(registry.get(n).supported_content_types)
        supported = types if supported is None else supported & types
    if supported is None:
        return list(docs), []
    kept = [d for d in docs if d.content_type in supported]
    skipped: dict[str, int] = {}
    for d in docs:
        if d.content_type not in supported:
            skipped[d.content_type] = skipped.get(d.content_type, 0) + 1
    messages = []
    for ct, count in skipped.items():
        msg = f"Skipped {count} {ct} file(s) not supported by {', '.join(names)}"
        alternatives = [s for s in registry.names(ct) if s not in _NO_SUGGEST]
        if alternatives:
            msg += f"; run with {hint(alternatives)} for {ct}."
        else:
            msg += f"; no registered strategy supports {ct}."
        messages.append(msg)
    return kept, messages


def select_supported_docs(
    docs: list[Document], names: list[str], registry: StrategyRegistry, path: Path
) -> list[Document]:
    """CLI wrapper of `filter_supported_docs`: warn on skips, exit 1 if nothing remains."""
    if not docs:
        from chunktuner.ingestion.file_ingestor import FileIngestor

        exts = ", ".join(sorted(FileIngestor.SUPPORTED_EXTENSIONS))
        typer.secho(
            f"No supported documents found in {path}.\nSupported extensions: {exts}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    kept, messages = filter_supported_docs(docs, names, registry)
    if not kept:
        typer.secho(
            f"Error: no documents in {path} are supported by {', '.join(names)}.",
            fg=typer.colors.RED,
            err=True,
        )
        for m in messages:
            typer.echo(m, err=True)
        raise typer.Exit(code=1)
    for m in messages:
        warn(m)
    return kept


def check_token_budget(
    ws: WorkspaceConfig,
    docs: list[Document],
    names: list[str],
    param_grid: dict[str, list[dict]],
    *,
    embedding_model: str | None,
    yes: bool,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> None:
    """Enforce ``max_tokens_per_run`` (only when written in the yaml; bm25 embeds nothing).

    Uses the same `CostEstimator` total (and workspace tokenizer) as ``chunk-tune estimate``.
    Over the cap: exit 1, or with ``--yes`` warn and continue.
    """
    if "max_tokens_per_run" not in ws.model_fields_set or retrieval_mode == "bm25":
        return
    from chunktuner.eval.cost_estimator import CostEstimator

    est = CostEstimator(encoding_name=tokenizer_encoding(ws)).estimate(
        docs, names, param_grid, embedding_model or DEFAULT_EMBEDDING_MODEL
    )
    cap = ws.max_tokens_per_run
    if est.total_tokens <= cap:
        return
    msg = (
        f"estimated {est.total_tokens} embedding tokens exceeds max_tokens_per_run={cap} "
        "in .autochunk.yaml (see `chunk-tune estimate`)."
    )
    if yes:
        warn(f"{msg} Continuing because --yes was given.")
        return
    typer.secho(
        f"Error: {msg} Reduce --max-docs / --strategies, raise the cap, or pass --yes.",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)


def baseline_unsupported(tuner: Any, content_type: str, docs: list[Document]) -> str | None:
    """Message if the tuner's baseline for ``content_type`` cannot chunk every doc type.

    Only relevant for mixed (all supported) content types, e.g. markdown + pdf with
    ``structural_semantic``: the ``fixed_tokens`` baseline would fail on the pdf docs.
    """
    types = {d.content_type for d in docs}
    if len(types) <= 1:
        return None
    cfg = tuner._baseline_config(content_type)
    if cfg is None:
        return None
    missing = sorted(types - set(tuner.strategies.get(cfg.name).supported_content_types))
    if not missing:
        return None
    return (
        f"Baseline {cfg.name} skipped: it does not support content_type "
        f"{', '.join(missing)} present in the corpus."
    )
