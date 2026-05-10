# Configuration

## `.autochunk.yaml`

Created in the **current directory** by `chunk-tune init`. Merged at runtime with defaults when keys are missing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | int | `1` | Schema version |
| `provider` | string | `openai` | Provider label (stored for tooling) |
| `embedding_model` | string or omitted | *(none)* | If non-empty, CLI `recommend` / `evaluate` / `compare` use LiteLLM for that model when you confirm (or pass `--yes`); resolution order is **`--embedding-model` CLI flag, else this field**. `chunk-tune init` pre-fills this with `text-embedding-3-small` — remove or set `null` for dummy embeddings without a CLI override |
| `llm_model` | string | `gpt-4o-mini` | Default LLM id for LiteLLM (agentic, generation metrics, dataset builders) |
| `api_base` | string or null | `null` | Optional OpenAI-compatible base URL (LM Studio, Ollama, Azure, vLLM) |
| `api_key` | string or null | `null` | Optional explicit API key; prefer `CHUNKTUNER_API_KEY` in environments where YAML should stay secret-free |
| `use_case` | string | `rag_qa` | Default scoring profile |
| `max_docs` | int | `100` | Default cap on documents per run |
| `max_tokens_per_run` | int | `250_000` | Budget guard for runs that honor workspace config |
| `top_k` | int | `5` | Default retrieval depth |
| `cache_dir` | string | `~/.cache/chunktuner` | SQLite and cache root |
| `log_level` | string | `INFO` | Intended logging level for apps reading this file |
| `strip_patterns` | list of string | `[]` | Optional regex patterns stripped at ingest |
| `tokenizer_encoding` | string | `cl100k_base` | tiktoken encoding name (workspace model) |

CLI commands accept `--config /path/to/.autochunk.yaml` to point elsewhere.

---

## Environment variables

| Variable | Effect |
|----------|--------|
| `CHUNKTUNER_CACHE_DIR` | Overrides default cache directory (`~/.cache/chunktuner`) when resolving paths; also used by `default_cache_dir()` |
| `CHUNK_TUNER_BASE_DIR` | **MCP / API security**: corpus root; every tool path must resolve under this directory |
| `CHUNKTUNER_API_BASE` | Optional API base URL for LiteLLM (CLI fallback when `api_base` is unset in YAML; used by MCP server) |
| `CHUNKTUNER_API_KEY` | Optional API key for LiteLLM (CLI fallback when `api_key` is unset in YAML; used by MCP server) |
| `CHUNKTUNER_LLM_MODEL` | Default LLM model id for the MCP server when tools use an embedding model |
| `CHUNKTUNER_SKIP_OFFSET_VALIDATION` | Set to `1` to skip chunk offset validation (debugging only) |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / other provider keys | Passed through **LiteLLM** for real embeddings and LLM calls |
| `LITELLM_LOG` | LiteLLM logging verbosity |

Logging for the library itself uses the standard `logging` module; configure handlers in your app. The `log_level` field in YAML is for conventions and future tooling.

---

## Workspace init flow

1. Run `chunk-tune init` in the directory where you want `.autochunk.yaml`.
2. Edit `embedding_model`, `cache_dir`, or `use_case` as needed.
3. Run `chunk-tune estimate` before paid `evaluate` / `recommend` whenever a non-empty embedding model is resolved (CLI `--embedding-model` or workspace `embedding_model`).

---

## Provider examples (LiteLLM)

The CLI wires **`LiteLLMEmbeddingFunction`** when `--embedding-model` is set. Examples:

```bash
export OPENAI_API_KEY=sk-...
chunk-tune recommend ./docs --embedding-model text-embedding-3-small --yes
```

For **Ollama** (local), use a LiteLLM-recognized model string such as `ollama/nomic-embed-text` if your LiteLLM build supports it, and ensure the Ollama server is reachable.

For **Cohere**, set `COHERE_API_KEY` and pass the appropriate `cohere/...` model id per LiteLLM docs.

Exact model ids change with providers; verify against [LiteLLM model list](https://docs.litellm.ai/docs/providers).

---

## Related docs

- [MCP setup](mcp_setup.md) — `CHUNK_TUNER_BASE_DIR` in `mcp.json`
- [CLI reference](cli_reference.md)
