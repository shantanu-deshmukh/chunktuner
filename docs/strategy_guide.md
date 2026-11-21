# Chunking strategy guide

## `fixed_tokens`

- **Best for:** Plain text, Markdown, HTML when you want a predictable baseline and token-aligned windows.
- **How it works:** Encodes the document with tiktoken, slides a window of `max_tokens` with `overlap_tokens` step, maps token spans back to character offsets.
- **Key parameters:** `max_tokens`, `overlap_tokens` — larger windows mean fewer, richer chunks; overlap preserves boundary context.
- **Performance:** Fast, deterministic, no extra dependencies; embedding cost scales with chunk count.
- **When NOT to use:** When structure (headings, AST, pages) should drive splits rather than fixed windows.

## `recursive_character`

- **Best for:** General prose and docs where hierarchical separators (paragraphs, lines, sentences) should drive breaks.
- **How it works:** Greedy split up to `chunk_size_chars`, preferring the latest separator inside the window; applies `chunk_overlap_chars` between windows.
- **Key parameters:** `chunk_size_chars`, `chunk_overlap_chars`, optional `separators` list.
- **Performance:** Fast, no API calls; overlap must stay **below** chunk size (validated).
- **When NOT to use:** Highly structured tabular PDFs or code where AST-aware strategies are better.

## `paragraph_group`

- **Best for:** Prose, reports and Markdown where blank-line paragraphs are the natural unit of meaning. It was the top-performing strategy in [Shaukat et al. (arXiv 2603.06976)](https://arxiv.org/abs/2603.06976).
- **How it works:** Splits on blank lines into paragraphs (outer whitespace trimmed, exact offsets), then emits windows of `paragraphs_per_chunk` paragraphs that overlap by `overlap_paragraphs`. A window that exceeds `max_tokens` is shortened to the paragraphs that fit; a single paragraph over the cap is split into token windows.
- **Key parameters:** `paragraphs_per_chunk` (default 3), `overlap_paragraphs` (default 0, must be below `paragraphs_per_chunk`), `max_tokens` (default 512). The default grid is 2/3/4 paragraphs × 0/1 overlap; with 1 paragraph of overlap, groups of 2 or 3 repeat a third or more of their text, which the negative `duplication_ratio` score weight penalizes.
- **Performance:** Fast, no API calls or extra dependencies.
- **When NOT to use:** Text without blank-line paragraph breaks (it becomes token windows), or code.

## `semantic` (extra: `chunktuner[semantic]`)

- **Best for:** Long articles and mixed prose where token windows should follow `semchunk` splits.
- **How it works:** Token-budget recursive splitting via `semchunk` with tiktoken sizing; does not use embeddings. Offsets must match the source string or an error is raised.
- **Key parameters:** `max_tokens`, `overlap_tokens`, or `overlap_ratio` (overlap as a fraction of `max_tokens`; `overlap_tokens` wins). `similarity_threshold` is a deprecated alias for `overlap_ratio`. `min_tokens` merges chunks with fewer tokens into the following chunk (the last into the previous); 0 (default) disables it.
- **Performance:** CPU-bound tokenization + semchunk; moderate cost tier.
- **When NOT to use:** Without the optional dependency; very short docs where recursion is simpler.

## `markdown_semantic` (extra: `semantic`)

- **Best for:** Markdown with clear heading hierarchy.
- **How it works:** Splits on headings into sections, runs `semchunk` per section with local offsets mapped back to the full document.
- **Key parameters:** `max_tokens`, `overlap_tokens`.
- **Performance:** Similar to `semantic` per section.
- **When NOT to use:** Non-markdown or flat text without headings (use `semantic`).

## `pdf_structural`

- **Best for:** PDF-flavoured markdown, exports with `## Page N` markers, or coarse markdown headings.
- **How it works:** Finds page/section markers or major headings, then slices long regions into `max_region_chars` windows.
- **Key parameters:** `max_region_chars`.
- **Performance:** Fast string passes; no ML.
- **When NOT to use:** When you need sentence-level semantics inside a section (`structural_semantic` + token windows).

## `structural_semantic`

- **Best for:** PDF/DOCX/PPTX markdown with large regions where you still want token-sized retrieval chunks.
- **How it works:** Uses `pdf_structural` regions (DOCX/PPTX are treated as markdown for region detection), then `fixed_tokens` inside each region with absolute offsets stitched back to the parent document.
- **Key parameters:** `max_tokens`, `overlap_tokens`, `max_region_chars`.
- **Performance:** Two-phase; more chunks than `pdf_structural` alone.
- **When NOT to use:** Tiny documents where a single region is enough.

## `late_chunking`

- **Best for:** Nothing beyond `fixed_tokens` — this is a fixed-token fallback. True late chunking is not implemented (it is an embedding-side technique). Excluded from default tuner runs; only evaluated when named explicitly.
- **How it works:** Delegates to `fixed_tokens` with `chunk_size_tokens` / `overlap_tokens` mapping.
- **Key parameters:** `chunk_size_tokens`, `overlap_tokens`.
- **Performance:** Same as fixed tokens.
- **When NOT to use:** When you specifically need document-level pooling from a compatible embedding model.

## `agentic`

- **Best for:** High-value narrative docs where an LLM can propose span boundaries (API cost). Excluded from default tuner runs; only evaluated when named explicitly.
- **How it works:** Prompts an LLM for JSON offsets; clamps to a 50k character prefix with a warning if truncated; optional fuzzy rejection when the model also supplies `text` / `chunk_text`.
- **Key parameters:** `model`, `max_propositions`.
- **Performance:** Expensive (LLM per document); falls back to `recursive_character` if no valid chunks.
- **When NOT to use:** Batch jobs without API budget; non-UTF-8 binary content.

## `code_ast` (extra: `chunktuner[code]`)

- **Best for:** Python source when tree-sitter is available — with `merge_small` (the default tuner grid) the whole file, including imports and module-level code, with consecutive small units merged up to `max_tokens`; without it, one chunk per top-level function/class when under `max_tokens`.
- **How it works:** Parses UTF-8 bytes with tree-sitter, converts byte offsets to **character** indices via UTF-8 decoding boundaries, splits oversized nodes with `code_window`. With `merge_small`, top-level definitions and the module-level code between them are merged greedily while the span stays within `max_tokens`.
- **Key parameters:** `max_tokens`, `merge_small` (default false: one chunk per top-level function/class; the default grid sets it to true).
- **Performance:** Fast parse; optional heavy dependency.
- **When NOT to use:** Non-Python languages (until more grammars are registered) or without the `code` extra.

## `code_window`

- **Best for:** Any code as a robust baseline (line-batched windows under a token cap).
- **How it works:** Accumulates whole lines until `max_tokens` would be exceeded, optional `overlap_lines` between windows.
- **Key parameters:** `max_tokens`, `overlap_lines`.
- **Performance:** Fast, deterministic.
- **When NOT to use:** When semantic structure (functions/classes) must be preserved (`code_ast`).
