# Getting started for implementers

## 1. Dev environment

```bash
git clone https://github.com/shantanu-deshmukh/chunktuner.git
cd chunktuner
uv sync --all-extras --dev
uv run pytest
uv run ruff check src/ tests/
```

Copy `.env.example` to `.env` for local API keys (never commit `.env`). See `AuthorReadme.md` for maintainer-focused notes.

## 2. Project structure

| Area | Role |
|------|------|
| `src/chunktuner/models.py` | **Single source of truth** for Pydantic models and protocols |
| `src/chunktuner/chunking/` | Strategies + `bootstrap.build_full_registry()` |
| `src/chunktuner/eval/` | Evaluator, datasets, score calculator, RAGAS bridge, cost estimator |
| `src/chunktuner/tuner/` | `AutoTuner`, parallel worker |
| `src/chunktuner/cache/` | SQLite embedding + chunk cache, wrapped embedder |
| `src/chunktuner/ingestion/` | `FileIngestor`, preprocessing, content-type detection |
| `src/chunktuner/cli/` | Typer commands |
| `src/chunktuner/mcp/` | FastMCP server (stdio) |
| `tests/unit/`, `tests/integration/` | Pytest suites |

## 3. Adding a new strategy (checklist)

1. Add `<strategy>.py` implementing `ChunkingStrategy` (`name`, `supported_content_types`, `chunk`, `param_schema`, `default_param_grid`).
2. Register in `chunking/bootstrap.py` (try/import optional deps if needed).
3. Call `validate_chunk_offsets(doc, chunks)` before returning from `chunk()` (or rely on a validated inner strategy only if offsets are unchanged).
4. Add `tests/unit/test_chunking_offsets.py` coverage (or dedicated test) for UTF-8 edge cases where relevant.
5. Document in `docs/strategy_guide.md`.

## 4. Evaluation flow (text)

```
Documents → strategy.chunk(doc, config) → chunks
         → Evaluator: offset sample check → embed chunks & questions
         → per-query cosine ranking → token recall / MRR / NDCG / …
         → optional LLM answers → RAGAS (faithfulness, answer relevancy)
         → ScoreCalculator → scalar score
```

`AutoTuner` runs this over a grid of `(strategy, params)` and ranks `EvalResult.score`.

## 5. Running tests by subsystem

| Subsystem | Command |
|-----------|---------|
| Chunking | `uv run pytest tests/unit/test_chunking_offsets.py` |
| Evaluator | `uv run pytest tests/unit/test_evaluator_dummy.py tests/unit/test_evaluator_metrics.py` |
| Tuner | `uv run pytest tests/integration/test_recommend_smoke.py tests/integration/test_auto_tuner_parallel.py` |
| CLI | `uv run pytest tests/integration/test_cli.py` |
| MCP | `uv run pytest tests/integration/test_mcp_*.py` |

## 6. Local MCP server

```bash
uv sync --extra mcp
uv run chunk-tune-mcp
```

Configure `CHUNK_TUNER_BASE_DIR` so all tool paths stay under your corpus root. See `docs/mcp_setup.md` for Claude Desktop `mcp.json` examples.
