# Provider configuration

chunktuner routes LLM and embedding calls through [LiteLLM](https://docs.litellm.ai/), so you can use OpenAI, Anthropic, Google Gemini, Cohere, Together, Azure OpenAI, or local OpenAI-compatible servers (LM Studio, Ollama, vLLM) by choosing the right **model id** and **credentials**.

## Configuration surfaces

| Surface | How to set base URL and API key |
|--------|----------------------------------|
| **Python** | `LiteLLMEmbeddingFunction(model, api_base=..., api_key=...)`, `Evaluator(..., llm_api_base=..., llm_api_key=...)`, `DatasetBuilder(..., llm_api_base=..., llm_api_key=...)`, `AgenticStrategy(api_base=..., api_key=...)` |
| **CLI** | `--api-base`, `--api-key`, `--llm-model` on `chunk-tune recommend`, `evaluate`, and `compare`; or `api_base` / `api_key` / `embedding_model` / `llm_model` in `.autochunk.yaml` |
| **MCP server** | `CHUNKTUNER_API_BASE`, `CHUNKTUNER_API_KEY`, `CHUNKTUNER_LLM_MODEL` (do not pass secrets in MCP tool arguments) |

Priority for base URL and key: CLI flags override workspace YAML, which overrides `CHUNKTUNER_*` environment variables. Provider-specific env vars (for example `OPENAI_API_KEY`, `GEMINI_API_KEY`) are still read by LiteLLM when you use that provider’s models.

## Workspace defaults

In `.autochunk.yaml`, `embedding_model` defaults to `null` — `chunk-tune init` does **not** write an OpenAI model there any more. With `embedding_model: null`, all `evaluate` / `recommend` / `compare` runs use **dummy embeddings** (no API calls, no cost) until you either pass `--embedding-model` on the CLI or set the field in the YAML. Set `llm_model` to the LiteLLM model id used for agentic chunking and generation-style metrics when enabled.

Optional fields:

- `api_base` — custom OpenAI-compatible endpoint (LM Studio, Ollama, Azure host, etc.).
- `api_key` — optional explicit key; prefer `CHUNKTUNER_API_KEY` or provider env vars in CI.

## Quick examples

**OpenAI** — set `OPENAI_API_KEY`, then e.g. `--embedding-model text-embedding-3-small` and `--llm-model gpt-4o-mini`.

**Anthropic / Claude** — set `ANTHROPIC_API_KEY`. Claude has no embeddings endpoint, so pair it with Gemini or a local embedding model:

```bash
chunk-tune recommend ./docs \
  --embedding-model gemini/gemini-embedding-001 \
  --llm-model claude-3-haiku-20240307
```

**Google Gemini** — set `GEMINI_API_KEY`, then e.g. `--embedding-model gemini/gemini-embedding-001` and `--llm-model gemini/gemini-2.0-flash`.

**LM Studio** — start the local server, then:

```bash
export CHUNKTUNER_API_BASE=http://localhost:1234/v1
export CHUNKTUNER_API_KEY=lm-studio
export CHUNKTUNER_LLM_MODEL=openai/llama-3.2-3b-instruct
uv run --extra mcp chunk-tune-mcp
```

Or with the CLI:

```bash
chunk-tune recommend ./docs \
  --embedding-model openai/nomic-embed-text-v1.5 \
  --api-base http://localhost:1234/v1 \
  --api-key lm-studio \
  --llm-model openai/llama-3.2-3b-instruct \
  --yes
```

## MCP note

The MCP server does not accept `api_key` in tool JSON. Configure `CHUNKTUNER_API_BASE`, `CHUNKTUNER_API_KEY`, and optionally `CHUNKTUNER_LLM_MODEL` before launching `chunk-tune-mcp`.

## RAGAS

Faithfulness and answer-relevancy still go through `RagasBridge` (LangChain stack). For non-OpenAI setups, configure the underlying LangChain/OpenAI env vars separately if required; see the evaluator docstring in `eval/evaluator.py` for the known limitation.
