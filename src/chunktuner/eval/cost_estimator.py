"""Dry-run cost and runtime estimates (no API calls)."""

from __future__ import annotations

import logging

from chunktuner.config import THROUGHPUT_API_TOKENS_PER_MIN
from chunktuner.models import CostEstimate, Document

logger = logging.getLogger(__name__)


class CostEstimator:
    """Heuristic token counts and USD estimates without calling embedding or LLM APIs."""

    def estimate(
        self,
        docs: list[Document],
        strategies: list[str],
        param_grid: dict[str, list[dict]],
        embedding_model: str,
        *,
        generate_dataset: bool = True,
        avg_chunks_per_doc: float = 3.0,
        llm_tokens_per_query: int = 400,
    ) -> CostEstimate:
        """Heuristic token counts × strategy configs; USD from litellm when available."""
        doc_tokens = _sum_doc_tokens(docs)
        n_configs = sum(len(param_grid.get(s, [{}])) for s in strategies)
        if n_configs == 0:
            n_configs = len(strategies)

        est_chunks_total = len(docs) * avg_chunks_per_doc * n_configs
        # Heuristic: avg chunk ≈ 64 tokens (covers fixed_tokens/recursive defaults 256–1024 ÷ ~6
        # chunks), plus 32 tokens per query embedding per doc per config (2 queries/doc × 16
        # tok/question avg). These are conservative over-estimates; the max() floor below guards
        # under-estimation.
        embed_tokens = int(est_chunks_total * 64) + int(len(docs) * 32 * n_configs)
        embed_tokens = max(embed_tokens, doc_tokens * n_configs // 4)

        embed_cost = _embedding_cost_usd(embedding_model, embed_tokens)
        llm_cost = 0.0
        if generate_dataset:
            q = max(1, len(docs) * 2)
            llm_cost = _llm_cost_usd(q * llm_tokens_per_query)

        wall = embed_tokens / THROUGHPUT_API_TOKENS_PER_MIN

        return CostEstimate(
            total_tokens=embed_tokens,
            embedding_cost_usd=round(embed_cost, 6),
            llm_cost_usd=round(llm_cost, 6),
            estimated_wall_time_min=round(wall, 3),
            strategy_configs=n_configs,
        )


def _sum_doc_tokens(docs: list[Document]) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(d.content)) for d in docs)
    except Exception:
        logger.warning(
            "tiktoken unavailable; using character-count / 4 token estimate "
            "(highly inaccurate for CJK or dense code)."
        )
        return sum(max(1, len(d.content) // 4) for d in docs)


def _embedding_cost_usd(model: str, tokens: int) -> float:
    try:
        from litellm import model_cost

        info = model_cost.get(model, {})
        p1k = float(info.get("input_cost_per_token", 0) or 0) * 1000
        if p1k > 0:
            return (tokens / 1000.0) * (p1k / 1000.0)
    except Exception:
        pass
    return (tokens / 1_000_000.0) * 0.02


def _llm_cost_usd(tokens: int) -> float:
    return (tokens / 1_000_000.0) * 0.15
