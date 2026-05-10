# chunktuner

[![PyPI version](https://img.shields.io/pypi/v/chunktuner.svg)](https://pypi.org/project/chunktuner/)
[![Python versions](https://img.shields.io/pypi/pyversions/chunktuner.svg)](https://pypi.org/project/chunktuner/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/shantanu-deshmukh/chunktuner/actions/workflows/ci.yml/badge.svg)](https://github.com/shantanu-deshmukh/chunktuner/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://shantanu-deshmukh.github.io/chunktuner/)

Auto chunking tuner and MCP server for RAG pipelines.

Give it your documents. It tries multiple chunking strategies, measures which one lets an AI answer questions most accurately, and tells you the winner.

![chunktuner project flow: documents through strategies, evaluation, to a recommended configuration](https://raw.githubusercontent.com/shantanu-deshmukh/chunktuner/main/docs/assets/project-flow.svg)

---

## What it does

When building a RAG pipeline, how you split documents into chunks directly impacts retrieval quality. `chunktuner` automates the process of finding the optimal chunking strategy for your specific corpus, embedding model, and use case.

It benchmarks strategies like fixed-token windows, recursive character splitting, semantic splitting, PDF structural chunking, and AST-based code chunking — then scores each one against real retrieval metrics (token recall, MRR, NDCG) and optional generation metrics (RAGAS faithfulness, answer relevancy).

---

## Interfaces

- **Python library** — programmatic integration into your pipeline
- **CLI** (`chunk-tune`) — human-driven tuning from the terminal
- **MCP server** — use directly from Claude Desktop or any MCP host

---

## Quickstart

```bash
# Install
uv tool install chunktuner

# Initialize workspace
chunk-tune init --provider openai

# See cost estimate before running anything
chunk-tune estimate ./my_docs --use-case rag_qa

# Get a recommendation
chunk-tune recommend ./my_docs --use-case rag_qa
```

**Python API:**

```python
from pathlib import Path
from chunktuner import FileIngestor, LiteLLMEmbeddingFunction, AutoTuner
from chunktuner import default_registry, Evaluator, ScoreCalculator

docs = FileIngestor().ingest_dir(Path("./my_docs"))
embedding_fn = LiteLLMEmbeddingFunction("text-embedding-3-small")
tuner = AutoTuner(
    strategies=default_registry,
    evaluator=Evaluator(embedding_fn),
    scorer=ScoreCalculator(use_case="rag_qa"),
)
result = tuner.recommend(docs, use_case="rag_qa")
print(result.best.config)
```

---

## Supported strategies

| Strategy              | Best for                              |
| --------------------- | ------------------------------------- |
| `fixed_tokens`        | Baseline; uniform token windows       |
| `recursive_character` | General prose and documentation       |
| `semantic`            | Theme-heavy articles                  |
| `markdown_semantic`   | Structured Markdown docs              |
| `pdf_structural`      | PDFs with layout regions and tables   |
| `structural_semantic` | PDF/DOCX with mixed layout and text   |
| `late_chunking`       | Long docs with dense cross-references |
| `agentic`             | High-value narrative documents        |
| `code_ast`            | Code repos (Python, JavaScript)       |
| `code_window`         | Code baseline (sliding window)        |

---

## MCP server (Claude Desktop)

Python **FastMCP** (`chunk-tune-mcp`, stdio). No Node.js build. See `docs/mcp_setup.md`.

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "chunktuner": {
      "command": "uvx",
      "args": ["--from", "chunktuner[mcp]", "chunk-tune-mcp"],
      "env": {
        "CHUNK_TUNER_BASE_DIR": "/path/to/your/corpus"
      }
    }
  }
}
```

Tools available: `list_strategies`, `preview_chunks`, `evaluate_chunking`, `recommend_config`.

---

## CLI reference

```
chunk-tune init       Bootstrap workspace config
chunk-tune analyze    Quick structural scan (no API cost)
chunk-tune estimate   Dry-run cost/token estimate
chunk-tune evaluate   Full evaluation across strategies
chunk-tune recommend  Evaluation + best config recommendation
chunk-tune compare    Side-by-side comparison of specific strategies
chunk-tune preview    Inspect how a strategy splits a document
chunk-tune cache      Manage embedding and chunk cache
```

---

## Installation options

```bash
uv add chunktuner                    # library
uv tool install chunktuner           # global CLI
uvx --from chunktuner chunk-tune …   # ephemeral CLI (no install)

# With optional extras
uv add "chunktuner[docling]"         # PDF/DOCX support
uv add "chunktuner[ragas]"           # generation metrics
uv add "chunktuner[semantic]"        # semantic chunking
uv add "chunktuner[code]"            # AST code chunking
uv add "chunktuner[all]"             # everything
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 👨🏻‍💻 Author

[Shantanu Deshmukh](https://shantanudeshmukh.com)

Full stack developer with experience in building E2E AI applications.

[Linkedin](https://www.linkedin.com/in/shantanud/)
/ [Twitter](https://twitter.com/askshantanu) / [AngelList](https://angel.co/u/dshantanu)
