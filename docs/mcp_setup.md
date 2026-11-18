# chunktuner MCP setup (Python / FastMCP)

The MCP server is **pure Python**: it imports the `chunktuner` library directly (no Node.js, no separate HTTP hop for MCP).

## Install

```bash
uv sync --extra mcp
# or
pip install 'chunktuner[mcp]'
```

## Claude Desktop (`.mcp.json`)

```json
{
  "mcpServers": {
    "chunktuner": {
      "command": "uvx",
      "args": ["--from", "chunktuner[mcp]", "chunk-tune-mcp"],
      "env": {
        "CHUNK_TUNER_BASE_DIR": "/absolute/path/to/your/corpus"
      }
    }
  }
}
```

- **`CHUNK_TUNER_BASE_DIR`**: every `path` argument to tools must resolve under this directory (security boundary).
- **`CHUNKTUNER_API_BASE`** / **`CHUNKTUNER_API_KEY`**: optional OpenAI-compatible endpoint and key for LiteLLM when tools pass an `embedding_model` (LM Studio, Ollama, etc.). Do not put secrets in MCP tool arguments.
- **`CHUNKTUNER_LLM_MODEL`**: LiteLLM model id passed to the evaluator as `llm_answer_model` for MCP `evaluate_chunking` / `recommend_config` (dataset generation, agentic, generation metrics when enabled, etc.); defaults to `gpt-4o-mini` when unset (see `chunktuner.config.DEFAULT_LLM_MODEL`). It applies regardless of whether an `embedding_model` is passed to the tool.
- **`CHUNKTUNER_CACHE_DIR`**: used by the library helper `chunktuner.config.default_cache_dir`. MCP tools cache embeddings and generated datasets in the default workspace `cache_dir` (`~/.cache/chunktuner`), which is the same location `chunk-tune cache stats|clear` uses when no workspace config is given.
- **Entry point**: `chunk-tune-mcp` → `chunktuner.mcp.server:run` (stdio JSON-RPC on stdout; **never** `print()` in MCP code).

## Cursor

Cursor’s MCP configuration uses the same **`command` / `args` / `env`** shape as above: register a server named (for example) `chunktuner` with `uvx`, args `["--from", "chunktuner[mcp]", "chunk-tune-mcp"]`, and set `CHUNK_TUNER_BASE_DIR` to an **absolute** corpus root. Where to paste this JSON depends on your Cursor version (global MCP settings vs project config); see the [Cursor MCP documentation](https://docs.cursor.com/context/model-context-protocol) for the current UI.

## Data handling

- **Embeddings:** When MCP tools are called with an `embedding_model`, `LiteLLMEmbeddingFunction` invokes your configured provider via LiteLLM (same as the CLI). With no model, tools use `DummyEmbeddingFunction` — no external embedding calls (`src/chunktuner/mcp/service.py`).
- **Document content:** Ingestion reads files under the validated `path` inside `CHUNK_TUNER_BASE_DIR`. The MCP server does not send document text to a chunktuner-operated cloud; stdout is reserved for the MCP JSON-RPC stream (`src/chunktuner/mcp/server.py`).
- **Logging:** Tool timing is logged at INFO to **stderr** in `chunktuner.mcp.tools` (no document body in those messages).
- **SQLite cache:** When a real `embedding_model` is used, `evaluate_chunking` / `recommend_config` cache embedding vectors in `~/.cache/chunktuner/chunktuner_cache.sqlite`. Pass `use_cache: false` to turn this off. Dummy embeddings are never cached. LLM-generated datasets (`generate_dataset: true`) are stored as JSON under `~/.cache/chunktuner/datasets/`. Use `chunk-tune cache stats|clear` to inspect or clear the embedding cache. See also [FAQ — caching](faq.md).

## Local dev (editable)

```bash
cd /path/to/chunktuner
uv run --extra mcp chunk-tune-mcp
```

## Optional HTTP API

The FastAPI app under `chunktuner.api` is separate from MCP. Start it with:

```bash
uv run uvicorn chunktuner.api.app:create_app --factory --host 127.0.0.1 --port 8765
```

Use this when you want REST clients; MCP hosts talk to `chunk-tune-mcp` only.

## Tools

| Tool | Purpose |
|------|---------|
| `list_strategies` | Strategies + param schemas |
| `preview_chunks` | Chunk inline text (no embeddings). Optional `content_type` (default `markdown`) |
| `evaluate_chunking` | Cost dry-run or full eval (dummy embeddings unless you pass a model + keys) |
| `recommend_config` | Full tuner + ranking |

### Optional fields on `evaluate_chunking` and `recommend_config`

| Field | Default | Description |
|-------|---------|-------------|
| `content_type` | none | `evaluate_chunking` keeps only documents of this type and the strategies that support it. `recommend_config` uses it to select strategies |
| `top_k` | `5` | No effect on metrics. Retrieval depth is the effective k = 2000-token context budget / avg chunk tokens |
| `dataset_path` | none | JSON/YAML eval dataset file. It must resolve under `CHUNK_TUNER_BASE_DIR` or the call is rejected. `document_id` may be the file path relative to `path`, an absolute path, or the file name |
| `generate_dataset` | `false` | Generate questions with `CHUNKTUNER_LLM_MODEL`. MCP has no confirmation prompt, so call `evaluate_chunking` with `dry_run: true, generate_dataset: true` first and check `llm_dataset_tokens` / `llm_cost_usd`. The result is cached and reused. Mutually exclusive with `dataset_path` |
| `retrieval_scope` | `document` | `document` or `corpus`: which ranking feeds the score's rank metrics |
| `use_cache` | `true` | SQLite embedding cache for real embedding models |

Responses (other than dry runs) include a `warnings` list. With neither `dataset_path` nor `generate_dataset`, a placeholder dataset is used (one query per document, whose answer is the first 200 characters), and `warnings` says so. `evaluate_chunking` also reports `dataset_summary.source` (`trivial`, `user_provided` or `llm_generated`).

## Resources & prompts

- Resources: `doc://overview`, `doc://strategy_guidelines`, `doc://metrics_guide`
- Prompts: `explain_chunking_results`, `design_eval_questions`

## See also

- [LangChain integration](integrations/langchain.md)
- [LlamaIndex integration](integrations/llamaindex.md)
- [Haystack integration](integrations/haystack.md)
- [LiteLLM embedding providers](https://docs.litellm.ai/docs/providers)
- [RAGAS documentation](https://docs.ragas.io/)
- [Docling documentation](https://ds4sd.github.io/docling/)
