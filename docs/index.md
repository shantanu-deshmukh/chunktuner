# chunktuner

[![PyPI version](https://img.shields.io/pypi/v/chunktuner.svg)](https://pypi.org/project/chunktuner/)
[![Python versions](https://img.shields.io/pypi/pyversions/chunktuner.svg)](https://pypi.org/project/chunktuner/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/shantanu-deshmukh/chunktuner/blob/main/LICENSE)
[![CI](https://github.com/shantanu-deshmukh/chunktuner/actions/workflows/ci.yml/badge.svg)](https://github.com/shantanu-deshmukh/chunktuner/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://shantanu-deshmukh.github.io/chunktuner/)

**Give it your documents.** It tries multiple chunking strategies, measures which setup supports retrieval best, and recommends a configuration for your corpus and use case.

---

## What it does

Chunking choices directly affect RAG quality. **chunktuner** benchmarks strategies (fixed windows, recursive splits, semantic splits, PDF structure, code AST, and more), scores them with retrieval metrics (token recall, MRR, NDCG) and optional generation metrics (RAGAS), then surfaces a winner.

```text
your docs → try multiple strategies → measure each → recommend the best config
```

---

## Three ways to use it

| Interface | Best for |
|-----------|----------|
| **Python library** | Embedding pipelines, custom grids, CI |
| **CLI** (`chunk-tune`) | Interactive tuning from the terminal |
| **MCP server** | Claude Desktop and other MCP hosts |

---

## Install

### uv (tool)

```bash
uv tool install chunktuner
```

### pip

```bash
pip install chunktuner
```

### Library only

```bash
uv add chunktuner
```

---

## Where to go next

- [Quickstart](quickstart.md) — install, first commands, minimal Python example
- [Strategy guide](strategy_guide.md) — choosing a strategy
- [CLI reference](cli_reference.md) — every `chunk-tune` command
- [Python API](python_api.md) — library patterns
- [API reference](api/index.md) — auto-generated module docs
- [MCP setup](mcp_setup.md) — Claude Desktop and `CHUNK_TUNER_BASE_DIR`
