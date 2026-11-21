"""Dry-run cost and runtime estimates (no API calls)."""

import logging

from chunktuner.config import THROUGHPUT_API_TOKENS_PER_MIN
from chunktuner.models import CostEstimate, Document

logger = logging.getLogger(__name__)


class CostEstimator:
    """Heuristic token counts and USD estimates without calling embedding or LLM APIs.

    ``encoding_name`` is the tiktoken encoding used to count document tokens (the workspace
    ``tokenizer_encoding``; default ``cl100k_base``).
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name

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
        llm_model: str | None = None,
        questions_per_doc: int = 3,
        max_queries: int = 50,
    ) -> CostEstimate:
        """Heuristic token counts × strategy configs; USD from litellm when available.

        When ``generate_dataset`` is True and ``llm_model`` is given, the LLM line is computed
        by `estimate_dataset_generation` (same sampling as ``DatasetBuilder.build_from_docs``)
        and ``llm_model`` / ``llm_dataset_tokens`` are filled in; otherwise the legacy
        per-query heuristic is used and those fields stay ``None``.
        """
        doc_tokens = _sum_doc_tokens(docs, self.encoding_name)
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
        llm_dataset_tokens: int | None = None
        if generate_dataset and llm_model:
            llm_dataset_tokens, llm_cost = estimate_dataset_generation(
                docs,
                llm_model,
                questions_per_doc=questions_per_doc,
                max_queries=max_queries,
                encoding_name=self.encoding_name,
            )
        elif generate_dataset:
            q = max(1, len(docs) * 2)
            llm_cost = _llm_cost_usd(q * llm_tokens_per_query)

        wall = embed_tokens / THROUGHPUT_API_TOKENS_PER_MIN

        return CostEstimate(
            total_tokens=embed_tokens,
            embedding_cost_usd=round(embed_cost, 6),
            llm_cost_usd=round(llm_cost, 6),
            estimated_wall_time_min=round(wall, 3),
            strategy_configs=n_configs,
            llm_model=llm_model if llm_dataset_tokens is not None else None,
            llm_dataset_tokens=llm_dataset_tokens,
        )


# Mirrors ``DatasetBuilder.build_from_docs``: prompt = instructions + first 6000 chars of a doc.
_DATASET_PROMPT_OVERHEAD_TOKENS = 120
_DATASET_SNIPPET_CHARS = 6000
_DATASET_OUTPUT_TOKENS_PER_QUESTION = 100


def estimate_dataset_generation(
    docs: list[Document],
    llm_model: str,
    *,
    questions_per_doc: int = 3,
    max_queries: int = 50,
    encoding_name: str = "cl100k_base",
) -> tuple[int, float]:
    """Return ``(llm_tokens, usd)`` for LLM eval-dataset generation (no API calls).

    Uses the same doc budget as ``DatasetBuilder.build_from_docs``
    (``max_queries // questions_per_doc`` docs, one completion each). Pricing comes from
    litellm's model table when known, else a flat $0.15 / 1M tokens.
    """
    if not docs:
        return 0, 0.0
    budget = max(1, max_queries // max(1, questions_per_doc))
    n_docs = min(len(docs), budget)
    # Builder samples randomly; use the largest docs for a conservative upper bound.
    snippets = sorted((d.content[:_DATASET_SNIPPET_CHARS] for d in docs), key=len, reverse=True)
    in_tokens = sum(_count_tokens(s, encoding_name) for s in snippets[:n_docs])
    in_tokens += n_docs * _DATASET_PROMPT_OVERHEAD_TOKENS
    out_tokens = n_docs * questions_per_doc * _DATASET_OUTPUT_TOKENS_PER_QUESTION
    return in_tokens + out_tokens, round(_llm_model_cost_usd(llm_model, in_tokens, out_tokens), 6)


def _count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding(encoding_name).encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _llm_model_cost_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    try:
        from litellm import model_cost

        info = model_cost.get(model, {})
        p_in = float(info.get("input_cost_per_token", 0) or 0)
        p_out = float(info.get("output_cost_per_token", 0) or 0)
        if p_in > 0 or p_out > 0:
            return in_tokens * p_in + out_tokens * p_out
    except Exception:
        pass
    return _llm_cost_usd(in_tokens + out_tokens)


def _sum_doc_tokens(docs: list[Document], encoding_name: str = "cl100k_base") -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding(encoding_name)
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
        per_token = float(info.get("input_cost_per_token", 0) or 0)
        if per_token > 0:
            return tokens * per_token
    except Exception:
        pass
    return (tokens / 1_000_000.0) * 0.02


def _llm_cost_usd(tokens: int) -> float:
    return (tokens / 1_000_000.0) * 0.15
