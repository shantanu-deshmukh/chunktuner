# FAQ

## Cost and API keys

### Does chunktuner send my documents to any server?

Only if **you** configure a cloud embedding model (e.g. via `LiteLLMEmbeddingFunction`, or CLI/workspace `embedding_model` resolved to a non-empty LiteLLM id, which calls your provider). With `DummyEmbeddingFunction` (no resolved embedding model from CLI or YAML) or a local model you configure in code, scoring runs locally. The `estimate` CLI command performs a structural/token estimate and does **not** call embedding APIs.

### How much does a real run cost?

Run `chunk-tune estimate ./my_docs` first — it reports token counts and a cost estimate (see `chunktuner.eval.cost_estimator`). Actual spend depends on your provider and model.

### Does the CLI cache embeddings automatically?

The default `chunk-tune recommend` / `evaluate` paths construct `LiteLLMEmbeddingFunction` or `DummyEmbeddingFunction` directly (see `src/chunktuner/cli/recommend_cmd.py` and `evaluate_cmd.py`) — they do **not** wrap `CachedEmbeddingFunction` for you. The library provides `EmbeddingCache` and `CachedEmbeddingFunction` (`chunktuner.cache`) for applications that want SQLite-backed reuse; see [Python API](python_api.md). The `chunk-tune cache` command inspects or clears the on-disk DB used when you opt into caching in your own code.

## Comparison

### How is chunktuner different from "just using LangChain's splitter"?

Frameworks ship splitters; chunktuner **benchmarks** registered strategies on your corpus using retrieval metrics and a tunable score profile (`ScoreCalculator`). See [integrations — LangChain](integrations/langchain.md).

### Does it replace RAGAS?

No. Optional RAGAS-backed fields (`faithfulness`, `answer_relevancy`) are integrated via `RagasBridge` when dependencies and `enable_generation_metrics` are available (`chunktuner.eval.ragas_bridge`, `Evaluator`). RAGAS does not replace multi-strategy search; chunktuner ranks `(strategy, params)` combinations.

## Strategies

### Which strategy should I try first?

Use `chunk-tune analyze ./my_docs`: it prints a **`heuristic_starting_strategy`** field based on the sampled file (see `src/chunktuner/cli/analyze_cmd.py`). In general: `recursive_character` is a solid text/markdown baseline; `markdown_semantic` needs the `semantic` extra (`semchunk`); `code_ast` needs the `code` extra (tree-sitter). For **ingesting** PDF/DOCX/PPTX files, `FileIngestor` uses Docling when the `docling` extra is installed (`src/chunktuner/ingestion/file_ingestor.py`).

### What is the baseline to beat?

When baseline mode is enabled (default), `AutoTuner.recommend` evaluates `fixed_tokens` with `max_tokens=512` and `overlap_tokens=0` first (`src/chunktuner/tuner/auto_tuner.py`). CLI `recommend` exposes `--no-baseline` to skip that run.

## MCP

### Is the MCP server safe to give filesystem access?

Tools that take a `path` argument resolve it under `CHUNK_TUNER_BASE_DIR` via `chunktuner.api.security.require_under_base` (see `src/chunktuner/mcp/service.py`). Paths outside the base directory are rejected.

### Which MCP tools exist?

`list_strategies`, `preview_chunks`, `evaluate_chunking`, and `recommend_config` are registered in `src/chunktuner/mcp/tools.py`. Any host that supports stdio MCP can run `chunk-tune-mcp`.

## Troubleshooting

See [Troubleshooting](troubleshooting.md).
