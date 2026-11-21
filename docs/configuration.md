# Configuration

## `.autochunk.yaml`

Created in the **current directory** by `chunk-tune init`. Merged at runtime with defaults when keys are missing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | int | `1` | Schema version |
| `embedding_model` | string or omitted | *(none)* | If non-empty, CLI `recommend` / `evaluate` / `compare` use LiteLLM for that model when you confirm (or pass `--yes`); resolution order is **`--embedding-model` CLI flag, else this field**. `chunk-tune init` writes `null` (dummy embeddings) unless you pass `-m/--model` |
| `llm_model` | string | `gpt-4o-mini` | Default LLM id for LiteLLM (agentic, generation metrics, dataset builders) |
| `api_base` | string or null | `null` | Optional OpenAI-compatible base URL (LM Studio, Ollama, Azure, vLLM) |
| `api_key` | string or null | `null` | Optional explicit API key; prefer `CHUNKTUNER_API_KEY` in environments where YAML should stay secret-free |
| `use_case` | string | `rag_qa` | Default scoring profile for `evaluate` / `compare` / `recommend` when `--use-case` is not passed |
| `max_docs` | int | `100` | Document cap for `evaluate` / `compare` / `recommend` / `estimate` when `--max-docs` is not passed; if the key is absent each command keeps its own default (`evaluate` / `recommend` 20, `compare` 15, `estimate` 100) |
| `max_tokens_per_run` | int | `250_000` | If set, `evaluate` / `compare` / `recommend` compute the `chunk-tune estimate` token total; over the cap they exit 1 unless `--yes` (then warn and continue) |
| `top_k` | int | `5` | Kept for compatibility; no effect on metrics (effective k comes from the 2000-token context budget) |
| `cache_dir` | string | `~/.cache/chunktuner` | SQLite and cache root |
| `log_level` | string | `INFO` | Level for `chunktuner` loggers in CLI commands (stderr); not used by the MCP server |
| `strip_patterns` | list of string | `[]` | Regex patterns removed from every document at ingest (`evaluate` / `compare` / `recommend` / `estimate` / `preview`); chunk offsets refer to the stripped text |
| `tokenizer_encoding` | string | `cl100k_base` | tiktoken encoding for chunking, evaluation, and `analyze` / `preview` token counts (`estimate` still counts with `cl100k_base`) |

`analyze`, `preview`, `estimate`, `evaluate`, `recommend`, `compare` and `cache` accept `--config /path/to/.autochunk.yaml` to point elsewhere.

Precedence: CLI flag > key present in `.autochunk.yaml` > command default. `chunk-tune init` writes `use_case`, `max_docs: 100`, `max_tokens_per_run: 250000` and `log_level: INFO`, so in an init-created workspace those values apply.

---

## Environment variables

| Variable | Effect |
|----------|--------|
| `CHUNKTUNER_CACHE_DIR` | Used by `default_cache_dir()`; applies only when `cache_dir` is empty in the workspace config |
| `CHUNK_TUNER_BASE_DIR` | **MCP / API security**: corpus root; every tool path must resolve under this directory |
| `CHUNKTUNER_API_BASE` | Optional API base URL for LiteLLM (CLI fallback when `api_base` is unset in YAML; used by MCP server) |
| `CHUNKTUNER_API_KEY` | Optional API key for LiteLLM (CLI fallback when `api_key` is unset in YAML; used by MCP server) |
| `CHUNKTUNER_LLM_MODEL` | Default LLM model id for the MCP server (dataset generation and generation metrics), whether or not tools pass an embedding model |
| `CHUNKTUNER_SKIP_OFFSET_VALIDATION` | Set to `1` to skip chunk offset validation (debugging only) |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / other provider keys | Passed through **LiteLLM** for real embeddings and LLM calls |
| `LITELLM_LOG` | LiteLLM logging verbosity |

Logging for the library itself uses the standard `logging` module; configure handlers in your app. In CLI commands, the `log_level` field in YAML sets the level of the `chunktuner` loggers (output on stderr); the MCP server ignores it.

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
