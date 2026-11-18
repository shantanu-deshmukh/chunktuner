# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.1] - 2026-11-18

### Fixed

- `chunk-tune` crashed on start with typer >= 0.29 (`Option() got an unexpected keyword
  argument 'click_type'`), which is what a fresh install of 0.2.0 resolves to. The
  `--retrieval-scope` option now validates its value with a callback that works on every
  supported typer version.

## [0.2.0] - 2026-11-18

### Breaking

- Python 3.10 is no longer supported (end of life October 2026). Requires Python >= 3.11;
  tested on 3.11, 3.12, 3.13 and 3.14.
- `late_chunking` is no longer included when the tuner runs all strategies by default. It is
  a fixed-token fallback (true late chunking is not implemented) and runs only when named
  explicitly.

### Added

- `--dataset PATH` and `--generate-dataset` on `evaluate`, `compare` and `recommend` (and the
  matching optional MCP/API fields `dataset_path`, `generate_dataset`). Generated datasets
  show up in `estimate --generate-dataset`, ask for confirmation unless `--yes` is given, and
  are cached.
- Corpus-wide retrieval metrics (`corpus_recall_at_k`, `corpus_mrr`, `corpus_ndcg_at_k`)
  reported next to the existing per-document metrics, and `--retrieval-scope
  [document|corpus]` (MCP: `retrieval_scope`) to choose which set drives the score. The
  default stays `document`.
- Embedding cache is now used by `evaluate`, `compare`, `recommend` and the MCP evaluation
  tools when a real embedding model is configured; disable with `--no-cache` (MCP:
  `use_cache=false`).
- MCP responses include a `warnings` list; `preview_chunks` accepts `content_type`.
- `semantic` strategy: new `overlap_ratio` parameter.

### Changed

- Using the built-in placeholder eval dataset now prints a warning, and its source is
  reported as `trivial` (previously mislabelled `user_provided`).
- `semantic` strategy description now states it is token-budget recursive splitting via
  semchunk and does not use embeddings.
- README and docs describe what the CLI and MCP actually support (local files only; URL/repo
  ingestion and RAGAS metrics are available from the Python API only).
- Dependency bounds: `litellm>=1.83,<2` (excludes the compromised 1.82.7/1.82.8 releases),
  `mcp[cli]>=1.27,<2`, `ragas>=0.3.2,<0.4`; the `ragas` extra now declares `datasets`.

### Deprecated

- `semantic` parameter `similarity_threshold`; use `overlap_ratio` (same meaning).

### Fixed

- `recommend` on code files crashed because the fixed-token baseline does not support code;
  code now uses a `code_window` (512 tokens, no overlap) baseline.
- MCP `evaluate_chunking` ignored its `content_type` argument.
- RAGAS extra failed to import with recent `pyarrow` because of an old `datasets` version.
- Embedding cache stored vectors as float32, so warm-cache runs could differ slightly from
  uncached runs; vectors are now stored at full precision (old entries still load).

## [0.1.2] - 2026-05-11

## [0.1.0] - 2026-05-02

### Added

- Ten chunking strategies: `fixed_tokens`, `recursive_character`, `semantic`,
  `markdown_semantic`, `pdf_structural`, `structural_semantic`, `late_chunking`,
  `agentic`, `code_ast`, `code_window`
- Retrieval evaluation: token recall, IoU, precision, MRR, NDCG@k,
  duplication ratio, average chunk length
- Optional generation metrics via RAGAS: faithfulness, answer relevancy
- LLM-generated and user-provided evaluation datasets
- SQLite-backed embedding and chunk caches
- AutoTuner with optional parallel grid search (`ProcessPoolExecutor`)
- CLI: `init`, `analyze`, `estimate`, `evaluate`, `recommend`, `compare`, `preview`, `cache`
- Python MCP server (FastMCP, stdio): `list_strategies`, `preview_chunks`,
  `evaluate_chunking`, `recommend_config`
- Optional extras: `docling` (PDF/DOCX), `ragas`, `semantic`, `code`, `mcp`, `all`
