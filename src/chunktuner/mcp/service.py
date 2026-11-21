"""MCP / HTTP shared logic (path validation for corpus ``path`` arguments)."""

import json
import os
from pathlib import Path
from typing import Any, cast

from chunktuner.api.security import require_under_base
from chunktuner.cache.wrapped_embeddings import maybe_cache_embeddings
from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.chunking.registry import StrategyRegistry
from chunktuner.cli.common import baseline_unsupported, filter_supported_docs
from chunktuner.config import (
    DEFAULT_LLM_MODEL,
    WorkspaceConfig,
    resolve_provider_config,
    workspace_cache_dir,
)
from chunktuner.eval.cost_estimator import CostEstimator
from chunktuner.eval.dataset_sources import (
    load_or_generate_dataset,
    load_user_dataset,
    trivial_dataset_warning,
)
from chunktuner.eval.embeddings import DummyEmbeddingFunction, LiteLLMEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.rerankers import reranker_from_spec, validate_reranker_spec
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.eval.trivial_dataset import trivial_dataset_for_docs
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.models import (
    ChunkConfig,
    Document,
    EmbeddingFunction,
    EvalDataset,
    Reranker,
    RetrievalMode,
    RetrievalScope,
    UseCase,
)
from chunktuner.tuner.auto_tuner import AutoTuner

DEFAULT_MAX_PREVIEW_CHARS = 500_000

_ws_defaults = WorkspaceConfig()
_MCP_API_BASE, _MCP_API_KEY = resolve_provider_config(_ws_defaults)
_MCP_LLM_MODEL = os.environ.get("CHUNKTUNER_LLM_MODEL") or DEFAULT_LLM_MODEL


def max_preview_chars() -> int:
    raw = os.environ.get("CHUNKTUNER_MAX_PREVIEW_CHARS")
    if raw is None or not str(raw).strip():
        return DEFAULT_MAX_PREVIEW_CHARS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_PREVIEW_CHARS


def _validate_strategies(names: list[str]) -> None:
    reg = build_full_registry()
    available = set(reg.names())
    invalid = [n for n in names if n not in available]
    if invalid:
        raise ValueError(f"Unknown strategy name(s): {invalid}. Available: {sorted(available)}")


def list_strategies_impl(content_type: str | None = None) -> list[dict]:
    reg = build_full_registry()
    out: list[dict] = []
    for s in reg.list(content_type):
        out.append(
            {
                "name": s.name,
                "description": s.description,
                "supported_content_types": s.supported_content_types,
                "supported_params": s.param_schema(),
            }
        )
    return out


def mcp_cache_dir() -> Path:
    """Embedding + generated-dataset cache dir (workspace default, as ``chunk-tune cache``)."""
    return workspace_cache_dir(_ws_defaults)


def _check_retrieval_scope(retrieval_scope: str) -> None:
    if retrieval_scope not in ("document", "corpus"):
        raise ValueError(f"retrieval_scope must be 'document' or 'corpus', got {retrieval_scope!r}")


def _check_retrieval_mode(retrieval_mode: str) -> None:
    """Validate ``retrieval_mode``; ``bm25``/``hybrid`` also need the optional ``bm25`` extra."""
    if retrieval_mode not in ("dense", "bm25", "hybrid"):
        raise ValueError(
            f"retrieval_mode must be 'dense', 'bm25' or 'hybrid', got {retrieval_mode!r}"
        )
    if retrieval_mode != "dense":
        try:
            import bm25s  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                f"retrieval_mode={retrieval_mode!r} needs the optional 'bm25' extra: "
                "uv add 'chunktuner[bm25]' or pip install 'chunktuner[bm25]'"
            ) from exc


def _check_reranker(reranker: str | None) -> str | None:
    """Validate a reranker spec string without building it (``None`` = no reranker)."""
    if reranker is None:
        return None
    return validate_reranker_spec(reranker)


def _build_reranker(reranker: str | None) -> Reranker | None:
    """Build the reranker; a missing optional extra becomes a ``ValueError`` (tool error)."""
    try:
        return reranker_from_spec(reranker)
    except ImportError as exc:
        raise ValueError(str(exc)) from exc


def _check_dataset_args(dataset_path: str | None, generate_dataset: bool) -> Path | None:
    """Validate dataset args up front; ``dataset_path`` must be under CHUNK_TUNER_BASE_DIR."""
    if dataset_path is not None and generate_dataset:
        raise ValueError("dataset_path and generate_dataset are mutually exclusive")
    if dataset_path is None:
        return None
    dp = require_under_base(dataset_path)
    if not dp.is_file():
        raise ValueError("dataset_path does not exist or is not a file")
    return dp


def _embedding_fn(embedding_model: str | None, use_cache: bool) -> EmbeddingFunction:
    embed: EmbeddingFunction = (
        LiteLLMEmbeddingFunction(
            embedding_model,
            api_base=_MCP_API_BASE,
            api_key=_MCP_API_KEY,
        )
        if embedding_model
        else DummyEmbeddingFunction()
    )
    return maybe_cache_embeddings(embed, mcp_cache_dir(), enabled=use_cache)


def _mcp_strategies_hint(names: list[str]) -> str:
    return f"strategies={json.dumps(names)}"


def _supported_docs(
    docs: list[Document], names: list[str], reg: StrategyRegistry
) -> tuple[list[Document], list[str]]:
    """Keep docs every strategy in ``names`` supports; skip messages go to ``warnings``."""
    kept, messages = filter_supported_docs(docs, names, reg, hint=_mcp_strategies_hint)
    if not kept:
        detail = " ".join(messages)
        raise ValueError(
            f"No documents under path are supported by {', '.join(names)}. {detail}".strip()
        )
    return kept, messages


def _ingest(
    p: Path,
    max_docs: int,
    content_type: str | None = None,
    *,
    names: list[str] | None = None,
    reg: StrategyRegistry | None = None,
) -> tuple[list[Document], Path, list[str]]:
    """Ingest ``p``; with ``content_type``, keep only matching docs (error if none match).

    With ``names``/``reg``, also drop docs some strategy in ``names`` cannot chunk (before
    the ``max_docs`` cap); the third item holds the skip messages.
    """
    root = p.parent if p.is_file() else p
    fi = FileIngestor(root=root)
    docs = fi.ingest_path(p) if p.is_file() else fi.ingest_dir(p)
    if content_type is not None:
        found = sorted({d.content_type for d in docs})
        docs = [d for d in docs if d.content_type == content_type]
        if not docs:
            raise ValueError(
                f"No documents with content_type={content_type!r} under path (found: {found})"
            )
    skipped: list[str] = []
    if names and reg is not None:
        docs, skipped = _supported_docs(docs, names, reg)
    return docs[:max_docs], root, skipped


def _resolve_dataset(
    docs: list[Document],
    root: Path,
    dataset_file: Path | None,
    generate_dataset: bool,
) -> tuple[EvalDataset, list[str]]:
    if dataset_file is not None:
        return load_user_dataset(dataset_file, docs, root)
    if generate_dataset:
        ds, _path, _hit = load_or_generate_dataset(
            docs,
            llm_model=_MCP_LLM_MODEL,
            cache_dir=mcp_cache_dir(),
            llm_api_base=_MCP_API_BASE,
            llm_api_key=_MCP_API_KEY,
        )
        warnings = []
        if len(ds.queries) < 20:
            warnings.append(
                f"Only {len(ds.queries)} eval queries (20+ recommended); metrics will be noisy."
            )
        return ds, warnings
    return trivial_dataset_for_docs(docs), [
        trivial_dataset_warning("dataset_path", "generate_dataset=true")
    ]


def preview_chunks_impl(
    text: str,
    strategy_name: str,
    config: dict[str, Any],
    content_type: str = "markdown",
) -> list[dict]:
    lim = max_preview_chars()
    if len(text) > lim:
        raise ValueError(
            f"preview_chunks text length {len(text)} exceeds limit of {lim} chars. "
            "Trim your input or use evaluate_chunking with a file path."
        )
    reg = build_full_registry()
    strat = reg.get(strategy_name)
    doc = Document(id="preview", content=text, content_type=cast(Any, content_type))
    cfg = ChunkConfig(name=strategy_name, params=config)
    chunks = strat.chunk(doc, cfg)
    return [
        {
            "id": c.id,
            "text": c.text,
            "start_offset": c.start_offset,
            "end_offset": c.end_offset,
            "tokens": c.tokens,
        }
        for c in chunks
    ]


def evaluate_chunking_impl(
    path: str,
    use_case: str,
    *,
    content_type: str | None = None,
    strategies: list[str] | None = None,
    max_docs: int = 20,
    top_k: int = 5,
    dry_run: bool = False,
    embedding_model: str | None = None,
    dataset_path: str | None = None,
    generate_dataset: bool = False,
    retrieval_scope: str = "document",
    retrieval_mode: str = "dense",
    reranker: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Evaluate strategies' default grids (or estimate cost with ``dry_run``).

    ``content_type`` keeps only documents of that type and the strategies supporting it
    (`recommend_config` uses it the same way to select strategies). ``top_k`` does not
    affect metrics (effective k = 2000-token budget / avg chunk tokens). With
    ``generate_dataset`` there is no confirmation prompt: call with ``dry_run=True`` first
    to see the LLM cost.
    """
    p = require_under_base(path)
    if not p.exists():
        raise ValueError("path does not exist")
    dataset_file = _check_dataset_args(dataset_path, generate_dataset)
    _check_retrieval_scope(retrieval_scope)
    _check_retrieval_mode(retrieval_mode)
    reranker_spec = _check_reranker(reranker)
    if strategies is not None:
        _validate_strategies(strategies)
    reg = build_full_registry()
    names = strategies or ["fixed_tokens", "recursive_character"]
    if content_type is not None:
        allowed = set(reg.names(content_type))
        names = [n for n in names if n in allowed]
        if not names:
            raise ValueError(
                f"None of the requested strategies support content_type={content_type!r}. "
                f"Strategies for it: {sorted(allowed)}"
            )
    if dry_run:
        docs, _root, skipped = _ingest(p, max_docs, content_type, names=names, reg=reg)
        grid: dict[str, list[dict]] = {n: reg.get(n).default_param_grid() for n in names}
        est = CostEstimator().estimate(
            docs,
            names,
            grid,
            embedding_model or "text-embedding-3-small",
            llm_model=_MCP_LLM_MODEL if generate_dataset else None,
        )
        if retrieval_mode == "bm25":
            # BM25 ranks lexically: no chunk or query is embedded (same as the CLI estimate).
            est = est.model_copy(
                update={
                    "total_tokens": 0,
                    "embedding_cost_usd": 0.0,
                    "estimated_wall_time_min": 0.0,
                }
            )
        out = est.model_dump(exclude_none=True)
        if skipped:
            out["warnings"] = skipped
        return out
    embed = _embedding_fn(embedding_model, use_cache)
    docs, root, skipped = _ingest(p, max_docs, content_type, names=names, reg=reg)
    ds, warnings = _resolve_dataset(docs, root, dataset_file, generate_dataset)
    warnings = skipped + warnings
    scope = cast(RetrievalScope, retrieval_scope)
    ev = Evaluator(
        embed,
        top_k=top_k,
        llm_answer_model=_MCP_LLM_MODEL,
        llm_api_base=_MCP_API_BASE,
        llm_api_key=_MCP_API_KEY,
        retrieval_scope=scope,
        retrieval_mode=cast(RetrievalMode, retrieval_mode),
        reranker=_build_reranker(reranker_spec),
    )
    scorer = ScoreCalculator(cast(UseCase, use_case), retrieval_scope=scope)
    results = []
    for n in names:
        strat = reg.get(n)
        for params in strat.default_param_grid():
            cfg = ChunkConfig(name=n, params=dict(params))
            results.append(ev.evaluate(strat, cfg, docs, ds, scorer=scorer))
    return {
        "dataset_summary": {"queries": len(ds.queries), "source": ds.source},
        "results": [r.model_dump() for r in results],
        "warnings": warnings,
    }


def recommend_config_impl(
    path: str,
    use_case: str,
    *,
    content_type: str | None = None,
    strategies: list[str] | None = None,
    max_docs: int = 20,
    top_k: int = 5,
    embedding_model: str | None = None,
    dataset_path: str | None = None,
    generate_dataset: bool = False,
    retrieval_scope: str = "document",
    retrieval_mode: str = "dense",
    reranker: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Run the tuner; ``generate_dataset`` has no prompt — estimate via evaluate dry_run first."""
    p = require_under_base(path)
    if not p.exists():
        raise ValueError("path does not exist")
    dataset_file = _check_dataset_args(dataset_path, generate_dataset)
    _check_retrieval_scope(retrieval_scope)
    _check_retrieval_mode(retrieval_mode)
    reranker_spec = _check_reranker(reranker)
    if strategies is not None:
        _validate_strategies(strategies)
    embed = _embedding_fn(embedding_model, use_cache)
    reg = build_full_registry()
    uc = cast(UseCase, use_case)
    scope = cast(RetrievalScope, retrieval_scope)
    tuner = AutoTuner(
        reg,
        Evaluator(
            embed,
            top_k=top_k,
            llm_answer_model=_MCP_LLM_MODEL,
            llm_api_base=_MCP_API_BASE,
            llm_api_key=_MCP_API_KEY,
            retrieval_scope=scope,
            retrieval_mode=cast(RetrievalMode, retrieval_mode),
            reranker=_build_reranker(reranker_spec),
        ),
        ScoreCalculator(uc, retrieval_scope=scope),
    )
    strat_names = strategies or ["fixed_tokens", "recursive_character"]
    # Strategies that actually run: with ``content_type`` the tuner keeps only those supporting
    # it, and runs just the baseline when none does.
    run_names = strat_names
    if content_type is not None:
        run_names = [n for n in strat_names if n in set(reg.names(content_type))]
        if not run_names:
            bl_cfg = tuner._baseline_config(content_type)
            run_names = [bl_cfg.name] if bl_cfg is not None else []
    docs, root, skipped = _ingest(p, max_docs, names=run_names, reg=reg)
    if not docs:
        raise ValueError("No documents to tune on")
    ds, warnings = _resolve_dataset(docs, root, dataset_file, generate_dataset)
    warnings = skipped + warnings
    ct = content_type or docs[0].content_type
    bl_msg = baseline_unsupported(tuner, ct, docs)
    if bl_msg:
        warnings.append(bl_msg)
    rec = tuner.recommend(
        docs,
        uc,
        strategies=strat_names,
        max_docs=max_docs,
        baseline=bl_msg is None,
        content_type=content_type,
        dataset=ds,
    )
    out = rec.model_dump()
    out["warnings"] = warnings
    return out
