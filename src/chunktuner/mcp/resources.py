"""Read-only MCP resources (documentation URIs)."""

_OVERVIEW = """# chunktuner

Auto chunking tuner for RAG pipelines. Ingests text-heavy corpora, benchmarks chunking
strategies (fixed tokens, recursive character, semantic, PDF/code strategies when installed),
scores retrieval-style metrics, and recommends a configuration.

Use `estimate` / dry-run before paid embedding runs. All corpus paths must stay under
`CHUNK_TUNER_BASE_DIR` when using MCP tools.
"""

_STRATEGY_GUIDE = """# Strategy selection (heuristic)

- **Plain prose / docs**: start with `recursive_character` or `fixed_tokens`.
- **Markdown with headings**: `markdown_semantic` when the semantic extra is installed.
- **PDF / Office exports**: `pdf_structural` or `structural_semantic` after docling ingestion.
- **Repositories**: `code_ast` / `code_window` with the code extra installed.

Always validate chunk offsets (`doc.content[start:end] == chunk.text`) in tests for new strategies.
"""

_METRICS_GUIDE = """# Eval metrics (short)

- **token_recall / token_iou**: overlap between retrieved chunk tokens and gold answer spans.
- **recall@k / MRR / NDCG@k**: retrieval quality vs ranked chunk list per query.
- **duplication_ratio**: retrieved-set token overlap across chunks (wasted context).
- **faithfulness / answer_relevancy**: optional RAGAS generation metrics when enabled.

Composite **score** comes from `ScoreCalculator` use-case weights (`rag_qa`, `search`, etc.).
"""


def register_resources(mcp: object) -> None:
    from mcp.server.fastmcp import FastMCP

    if not isinstance(mcp, FastMCP):
        raise TypeError("Expected FastMCP instance")

    @mcp.resource("doc://overview")
    def doc_overview() -> str:
        return _OVERVIEW

    @mcp.resource("doc://strategy_guidelines")
    def doc_strategy_guidelines() -> str:
        return _STRATEGY_GUIDE

    @mcp.resource("doc://metrics_guide")
    def doc_metrics_guide() -> str:
        return _METRICS_GUIDE
