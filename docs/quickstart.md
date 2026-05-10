# Quickstart

## Prerequisites

- Python **3.10+**
- [uv](https://docs.astral.sh/uv/) or pip

Optional: API keys for your embedding provider (e.g. `OPENAI_API_KEY`) when you run real evaluations instead of dummy embeddings.

---

## Install

| Goal | Command |
|------|---------|
| Global CLI | `uv tool install chunktuner` |
| Project dependency | `uv add chunktuner` |
| One-off CLI (no install) | `uvx --from chunktuner chunk-tune --help` |

### Optional extras

| Extra | Purpose |
|-------|---------|
| `chunktuner[docling]` | PDF, DOCX, PPTX ingestion |
| `chunktuner[ragas]` | Faithfulness / answer relevancy metrics |
| `chunktuner[semantic]` | Semantic chunking strategies |
| `chunktuner[code]` | Tree-sitter code strategies |
| `chunktuner[mcp]` | MCP server (`chunk-tune-mcp`) |
| `chunktuner[all]` | All of the above |

```bash
uv add "chunktuner[semantic,ragas]"
```

---

## First run (CLI)

Typical flow: create workspace config, estimate cost, then recommend.

```bash
chunk-tune init

chunk-tune estimate ./my_docs --use-case rag_qa

chunk-tune recommend ./my_docs --use-case rag_qa
```

`estimate` is dry-run (no paid API calls). `init` creates `.autochunk.yaml` with `embedding_model: null` — so `recommend` / `evaluate` / `compare` all use **dummy embeddings** by default (free, no API key needed). To enable real embeddings, pass `--embedding-model <model-id>` on the CLI or set `embedding_model` in `.autochunk.yaml`. LiteLLM runs only when a model is resolved; you are prompted unless you pass `--yes`. Any LiteLLM-supported provider works: OpenAI, Anthropic, Google Gemini, Cohere, Ollama, LM Studio, and more — see [Provider configuration](providers.md).

---

## Minimal Python example

```python
from pathlib import Path

from chunktuner import (
    AutoTuner,
    DummyEmbeddingFunction,
    Evaluator,
    FileIngestor,
    ScoreCalculator,
    default_registry,
)

# 1) Load documents from a directory (respects supported extensions).
docs = FileIngestor().ingest_dir(Path("./my_docs"))

# 2) Embeddings: dummy is free; swap for LiteLLMEmbeddingFunction for real runs.
embedding_fn = DummyEmbeddingFunction()

# 3) Evaluator + scorer for your use case (rag_qa, search, summarization, code_assist).
evaluator = Evaluator(embedding_fn)
scorer = ScoreCalculator(use_case="rag_qa")

# 4) Grid search over registered strategies.
tuner = AutoTuner(
    strategies=default_registry,
    evaluator=evaluator,
    scorer=scorer,
)
result = tuner.recommend(docs, use_case="rag_qa")
print(result.best.config)
```

Use `ingest_path` when you have a single file, or `ingest_dir` for a tree. See [Python API](python_api.md) for `URLIngestor`, `RepoIngestor`, caching, and parallel tuning.

---

## What’s next

- [Configuration](configuration.md) — `.autochunk.yaml` and environment variables
- [Strategy guide](strategy_guide.md)
- [Metrics reference](metrics.md) and [metrics glossary](metrics_glossary.md)
- [MCP setup](mcp_setup.md)

## See also

- [LangChain integration](integrations/langchain.md)
- [LlamaIndex integration](integrations/llamaindex.md)
- [Haystack integration](integrations/haystack.md)
- [LiteLLM embedding providers](https://docs.litellm.ai/docs/providers)
- [RAGAS documentation](https://docs.ragas.io/)
- [Docling documentation](https://ds4sd.github.io/docling/)
