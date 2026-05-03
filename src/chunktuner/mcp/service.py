"""MCP / HTTP shared logic (path validation for corpus ``path`` arguments)."""

from __future__ import annotations

import os
from typing import Any, cast

from chunktuner.api.security import require_under_base
from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.eval.cost_estimator import CostEstimator
from chunktuner.eval.embeddings import DummyEmbeddingFunction, LiteLLMEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.eval.trivial_dataset import trivial_dataset_for_docs
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.models import ChunkConfig, Document, UseCase
from chunktuner.tuner.auto_tuner import AutoTuner

DEFAULT_MAX_PREVIEW_CHARS = 500_000


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
        raise ValueError(
            f"Unknown strategy name(s): {invalid}. Available: {sorted(available)}"
        )


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


def preview_chunks_impl(text: str, strategy_name: str, config: dict[str, Any]) -> list[dict]:
    lim = max_preview_chars()
    if len(text) > lim:
        raise ValueError(
            f"preview_chunks text length {len(text)} exceeds limit of {lim} chars. "
            "Trim your input or use evaluate_chunking with a file path."
        )
    reg = build_full_registry()
    strat = reg.get(strategy_name)
    doc = Document(id="preview", content=text, content_type="markdown")
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
) -> dict:
    p = require_under_base(path)
    if not p.exists():
        raise ValueError("path does not exist")
    if strategies is not None:
        _validate_strategies(strategies)
    names = strategies or ["fixed_tokens", "recursive_character"]
    if dry_run:
        fi = FileIngestor(root=p.parent if p.is_file() else p)
        docs = fi.ingest_path(p) if p.is_file() else fi.ingest_dir(p)
        docs = docs[:max_docs]
        grid: dict[str, list[dict]] = {
            n: build_full_registry().get(n).default_param_grid() for n in names
        }
        est = CostEstimator().estimate(
            docs, names, grid, embedding_model or "text-embedding-3-small"
        )
        return est.model_dump()
    embed = (
        LiteLLMEmbeddingFunction(embedding_model) if embedding_model else DummyEmbeddingFunction()
    )
    fi = FileIngestor(root=p.parent if p.is_file() else p)
    docs = fi.ingest_path(p) if p.is_file() else fi.ingest_dir(p)
    docs = docs[:max_docs]
    ds = trivial_dataset_for_docs(docs)
    reg = build_full_registry()
    ev = Evaluator(embed, top_k=top_k)
    scorer = ScoreCalculator(cast(UseCase, use_case))
    results = []
    for n in names:
        strat = reg.get(n)
        for params in strat.default_param_grid():
            cfg = ChunkConfig(name=n, params=dict(params))
            results.append(ev.evaluate(strat, cfg, docs, ds, scorer=scorer))
    return {
        "dataset_summary": {"queries": len(ds.queries)},
        "results": [r.model_dump() for r in results],
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
) -> dict:
    p = require_under_base(path)
    if not p.exists():
        raise ValueError("path does not exist")
    if strategies is not None:
        _validate_strategies(strategies)
    embed = (
        LiteLLMEmbeddingFunction(embedding_model) if embedding_model else DummyEmbeddingFunction()
    )
    fi = FileIngestor(root=p.parent if p.is_file() else p)
    docs = fi.ingest_path(p) if p.is_file() else fi.ingest_dir(p)
    docs = docs[:max_docs]
    uc = cast(UseCase, use_case)
    tuner = AutoTuner(
        build_full_registry(),
        Evaluator(embed, top_k=top_k),
        ScoreCalculator(uc),
    )
    strat_names = strategies or ["fixed_tokens", "recursive_character"]
    rec = tuner.recommend(
        docs,
        uc,
        strategies=strat_names,
        max_docs=max_docs,
        baseline=True,
        content_type=content_type,
    )
    return rec.model_dump()
