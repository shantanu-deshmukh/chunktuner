# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `paragraph_group` strategy: groups consecutive blank-line paragraphs
  (`paragraphs_per_chunk`, `overlap_paragraphs`) under a `max_tokens` cap; a paragraph over
  the cap is split into token windows. Default grid: 2/3/4 paragraphs × 0/1 overlap.
- `semantic` strategy: `min_tokens` now merges chunks smaller than that into a neighbour
  (default 0: off).
- `analyze` and `preview` accept `--config`.

### Changed

- `AutoTuner.recommend()` without an explicit strategy list now also tries `paragraph_group`
  on text, Markdown and HTML, so its recommendation can change. The CLI and MCP defaults
  (`fixed_tokens`, `recursive_character`) are unchanged.
- `.autochunk.yaml` keys `use_case`, `max_docs`, `max_tokens_per_run`, `strip_patterns`,
  `tokenizer_encoding` and `log_level` are now applied (they were stored but ignored). A CLI
  flag still wins; a key missing from the file leaves the command default. Because
  `chunk-tune init` writes `max_docs: 100` and `max_tokens_per_run: 250000`, workspaces
  created by `init` now evaluate up to 100 documents (was 20, or 15 for `compare`) and stop
  when the estimated tokens exceed 250k unless `--yes` is given.
- `code_ast` now honours `merge_small` (set in its default grid): Python files are covered
  completely, including imports and module-level code, and small consecutive units are
  merged up to `max_tokens`. Tuner results for Python code change accordingly; configs
  without `merge_small` keep one chunk per top-level function/class.
- `evaluate`, `compare`, `recommend` and the MCP evaluation tools skip documents whose
  content type a selected strategy does not support, with a warning that names strategies
  that do (MCP: in `warnings`), instead of crashing on mixed folders.
- `cache stats --model` and `cache clear --model` now act only on that model's embeddings
  (the cache records the model for new rows; rows from earlier versions are only included
  without `--model`).

### Fixed

- `estimate` under-reported embedding cost 1000× for models priced in litellm's model table
  (for example, 0.0 instead of 0.0004 USD for the test fixtures).
- On installs without the `semantic` extra, `semantic` was still registered, so
  `AutoTuner.recommend()` without an explicit strategy list failed; it is now registered only
  when `semchunk` is importable.
- `fixed_tokens` (and `late_chunking`, `structural_semantic` and `paragraph_group`, which
  use it) crashed with "Token decode length mismatch" on text where a token splits a
  multibyte character, such as Japanese or emoji.
- `structural_semantic` crashed on DOCX and PPTX documents.
- `evaluate` and `recommend` crashed on unknown strategy names; they now exit with an error
  message. `compare` on a folder with no supported documents now says so.
- `cache stats` under-reported the cache size (vectors are stored as float64).

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
