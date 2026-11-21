# Troubleshooting

## `No module named …` for optional strategies or MCP

Install the matching optional extra (see `pyproject.toml` `[project.optional-dependencies]`):

| Import / error signal | Install |
|----------------------|---------|
| `semchunk` (semantic / markdown strategies) | `uv add "chunktuner[semantic]"` or `pip install 'chunktuner[semantic]'` |
| `docling` (PDF/DOCX/PPTX ingestion in `FileIngestor`) | `uv add "chunktuner[docling]"` |
| `ragas` / `datasets` when enabling RAGAS metrics | `uv add "chunktuner[ragas]"` |
| `tree_sitter` / grammar packages for `code_ast` | `uv add "chunktuner[code]"` |
| `mcp` when running `chunk-tune-mcp` | `uv add "chunktuner[mcp]"` or `uv sync --extra mcp` |
| `bm25s` when using `--retrieval-mode bm25` or `hybrid` | `uv add "chunktuner[bm25]"` |
| `sentence-transformers` when using `--reranker local` or `local:<hf-model-id>` | `uv add "chunktuner[late]"` |

## Authentication errors (embeddings)

LiteLLM reads provider keys from standard environment variables (see [LiteLLM providers](https://docs.litellm.ai/docs/providers)). Examples: `OPENAI_API_KEY` for OpenAI. Pass `--embedding-model` on the CLI only when you intend to call a remote model, and confirm with `--yes` where the command requires it (`recommend`, `evaluate`, `compare`).

## MCP server not visible in your host

1. Run `uvx --from "chunktuner[mcp]" chunk-tune-mcp` (or `uv run chunk-tune-mcp` from a dev checkout with the `mcp` extra) and confirm it stays running without tracebacks.
2. Set `CHUNK_TUNER_BASE_DIR` to an **absolute** path to an existing directory that contains (or is a parent of) the files you pass to tools.
3. Restart the MCP host after editing its config.
4. MCP logging is configured on **stderr** in `chunktuner.mcp.server.run` at `INFO` by default — check the host’s server logs, not stdout (stdout is reserved for JSON-RPC).

## Path errors in MCP (`path outside base dir` or similar)

`require_under_base` rejects paths that do not resolve under `CHUNK_TUNER_BASE_DIR`. Use paths inside that tree, and set the base to a **directory** that is an ancestor of your corpus files.

## Low retrieval metric scores

- Try different `max_tokens` / overlap settings on `fixed_tokens`, or other strategies (`chunk-tune analyze` suggests a starting heuristic).
- Inspect `avg_chunk_length` and duplication in the evaluation output — pathological splits can hurt recall.

## Offset invariant errors (`chunk.text` does not match `doc.content`)

Strategies must satisfy `doc.content[chunk.start_offset:chunk.end_offset] == chunk.text`. If you see a violation, file an issue with the strategy name, parameters, and document type. Offset tests live in `tests/unit/test_chunking_offsets.py`.
