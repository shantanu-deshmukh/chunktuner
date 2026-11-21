"""FastMCP tool registrations (stdio-safe logging to stderr only)."""

import logging
import time
from threading import BoundedSemaphore
from typing import Annotated, Literal

from pydantic import Field

from chunktuner.mcp.service import (
    evaluate_chunking_impl,
    list_strategies_impl,
    preview_chunks_impl,
    recommend_config_impl,
)

log = logging.getLogger("chunktuner.mcp.tools")

TopK = Annotated[
    int,
    Field(
        description=(
            "Kept for compatibility; does NOT affect metrics or scores. Retrieval depth is the "
            "effective k = 2000-token context budget / avg chunk tokens."
        )
    ),
]
DatasetPath = Annotated[
    str | None,
    Field(
        description=(
            "Optional JSON/YAML eval dataset file (must be under CHUNK_TUNER_BASE_DIR). "
            "document_id may be the file path relative to `path`, an absolute path, or the "
            "file name. Mutually exclusive with generate_dataset."
        )
    ),
]
GenerateDataset = Annotated[
    bool,
    Field(
        description=(
            "Generate eval questions with the server's LLM (CHUNKTUNER_LLM_MODEL; may be paid). "
            "No confirmation prompt: call evaluate_chunking with dry_run=true and "
            "generate_dataset=true first to see llm_dataset_tokens / llm_cost_usd. The dataset "
            "is cached and reused for identical documents. Default: a trivial placeholder "
            "dataset (see `warnings` in the response)."
        )
    ),
]
RetrievalScopeArg = Annotated[
    Literal["document", "corpus"],
    Field(
        description=(
            "Ranking used for score's rank metrics: 'document' (gold document's chunks only) "
            "or 'corpus' (all chunks of all documents)."
        )
    ),
]
RetrievalModeArg = Annotated[
    Literal["dense", "bm25", "hybrid"],
    Field(
        description=(
            "First-stage retrieval for the rank metrics: 'dense' (embedding similarity), "
            "'bm25' (lexical BM25) or 'hybrid' (dense + BM25 fused with reciprocal rank "
            "fusion). 'bm25' and 'hybrid' need the optional 'bm25' extra."
        )
    ),
]
RerankerArg = Annotated[
    str | None,
    Field(
        description=(
            "Optional second-stage reranker spec: 'dummy' (deterministic lexical overlap, no "
            "cost), 'local' or 'local:<hf-model-id>' (runs a local CrossEncoder and downloads "
            "the model; needs the optional 'late' extra), or any other string as a LiteLLM "
            "rerank model (e.g. 'cohere/rerank-v3.5'; uses the provider's env vars and may "
            "cost money). Default: no reranker."
        )
    ),
]
UseCache = Annotated[
    bool,
    Field(description="Use the SQLite embedding cache for real embedding models."),
]
_eval_sem = BoundedSemaphore(5)


def _log_tool(name: str, start: float, ok: bool) -> None:
    ms = int((time.perf_counter() - start) * 1000)
    log.info("mcp_tool name=%s ok=%s latency_ms=%s", name, ok, ms)


def register_tools(mcp: object) -> None:
    from mcp.server.fastmcp import FastMCP

    if not isinstance(mcp, FastMCP):
        raise TypeError("Expected FastMCP instance")

    @mcp.tool()
    def list_strategies(content_type: str | None = None) -> list[dict]:
        """List registered chunking strategies, optionally filtered by content type."""
        t0 = time.perf_counter()
        try:
            out = list_strategies_impl(content_type)
            _log_tool("list_strategies", t0, True)
            return out
        except Exception:
            _log_tool("list_strategies", t0, False)
            raise

    @mcp.tool()
    def preview_chunks(
        text: str,
        strategy_name: str,
        config: dict | None = None,
        content_type: Annotated[
            str, Field(description="Content type of `text` (e.g. markdown, text, code).")
        ] = "markdown",
    ) -> list[dict]:
        """Chunk inline text with one strategy + params (no embeddings)."""
        t0 = time.perf_counter()
        try:
            out = preview_chunks_impl(text, strategy_name, config or {}, content_type)
            _log_tool("preview_chunks", t0, True)
            return out
        except Exception:
            _log_tool("preview_chunks", t0, False)
            raise

    @mcp.tool()
    def evaluate_chunking(
        path: str,
        use_case: str = "rag_qa",
        content_type: str | None = None,
        strategies: list[str] | None = None,
        max_docs: int = 20,
        top_k: TopK = 5,
        dry_run: bool = False,
        embedding_model: str | None = None,
        dataset_path: DatasetPath = None,
        generate_dataset: GenerateDataset = False,
        retrieval_scope: RetrievalScopeArg = "document",
        retrieval_mode: RetrievalModeArg = "dense",
        reranker: RerankerArg = None,
        use_cache: UseCache = True,
    ) -> dict:
        """Dry-run cost estimate or full evaluation (``DummyEmbeddingFunction`` if no model).

        ``content_type`` restricts strategies to those supporting it. Documents some selected
        strategy cannot chunk (e.g. code with text strategies) are skipped. The response includes
        a ``warnings`` list (e.g. skipped documents, or the trivial placeholder dataset is used).
        """
        t0 = time.perf_counter()
        with _eval_sem:
            try:
                out = evaluate_chunking_impl(
                    path,
                    use_case,
                    content_type=content_type,
                    strategies=strategies,
                    max_docs=max_docs,
                    top_k=top_k,
                    dry_run=dry_run,
                    embedding_model=embedding_model,
                    dataset_path=dataset_path,
                    generate_dataset=generate_dataset,
                    retrieval_scope=retrieval_scope,
                    retrieval_mode=retrieval_mode,
                    reranker=reranker,
                    use_cache=use_cache,
                )
                _log_tool("evaluate_chunking", t0, True)
                return out
            except Exception:
                _log_tool("evaluate_chunking", t0, False)
                raise

    @mcp.tool()
    def recommend_config(
        path: str,
        use_case: str = "rag_qa",
        content_type: str | None = None,
        strategies: list[str] | None = None,
        max_docs: int = 20,
        top_k: TopK = 5,
        embedding_model: str | None = None,
        dataset_path: DatasetPath = None,
        generate_dataset: GenerateDataset = False,
        retrieval_scope: RetrievalScopeArg = "document",
        retrieval_mode: RetrievalModeArg = "dense",
        reranker: RerankerArg = None,
        use_cache: UseCache = True,
    ) -> dict:
        """Run tuner and return ranked ``Recommendation`` (uses dummy embeddings by default).

        Documents some selected strategy cannot chunk (e.g. code with text strategies) are
        skipped. The response adds a ``warnings`` list (e.g. skipped documents, or the trivial
        placeholder dataset is used).
        """
        t0 = time.perf_counter()
        with _eval_sem:
            try:
                out = recommend_config_impl(
                    path,
                    use_case,
                    content_type=content_type,
                    strategies=strategies,
                    max_docs=max_docs,
                    top_k=top_k,
                    embedding_model=embedding_model,
                    dataset_path=dataset_path,
                    generate_dataset=generate_dataset,
                    retrieval_scope=retrieval_scope,
                    retrieval_mode=retrieval_mode,
                    reranker=reranker,
                    use_cache=use_cache,
                )
                _log_tool("recommend_config", t0, True)
                return out
            except Exception:
                _log_tool("recommend_config", t0, False)
                raise
