# Python API

Use the **library** when you need custom ingestion, param grids, caching, or embedding wiring inside an application or test. Use the **CLI** for ad-hoc exploration; use **MCP** when an agent host should call tools.

---

## Ingestion

### `FileIngestor`

Load supported files from disk. Use a `root` when you want to restrict paths to a subtree.

```python
from pathlib import Path
from chunktuner import FileIngestor

fi = FileIngestor(root=Path("/safe/corpus"))
docs = fi.ingest_dir(Path("/safe/corpus/docs"))
single = fi.ingest_path(Path("/safe/corpus/readme.md"))
```

`ingest_dir` walks with `glob="**/*"` by default and skips unknown extensions. Binary formats (PDF, DOCX, …) require optional dependencies (`chunktuner[docling]`).

### `URLIngestor`

Fetch a single HTTP(S) URL into one `Document` (`ingest_url`).

```python
from chunktuner import URLIngestor

doc = URLIngestor().ingest_url("https://example.com/page.html")
```

### `RepoIngestor`

Walk a checkout for code and markdown-like files (`ingest_repo`).

```python
from pathlib import Path
from chunktuner import RepoIngestor

docs = RepoIngestor().ingest_repo(Path("./my-repo"))
```

---

## Strategies and registries

- **`default_registry`** — strategies that imported successfully on this install (optional deps may be missing).
- **`build_full_registry(encoding="cl100k_base")`** — rebuild with a specific tiktoken encoding.

```python
from chunktuner import default_registry, build_full_registry

reg = build_full_registry("cl100k_base")
strategy = reg.get("fixed_tokens")
chunks = strategy.chunk(doc, cfg)
```

Custom implementations follow the `ChunkingStrategy` protocol in `chunktuner.models`; register them with `StrategyRegistry.register`.

---

## Evaluation

### `Evaluator`

Chunks documents with a strategy, embeds chunk and query texts, and computes `EvalMetrics`. Pass a `ScoreCalculator` when calling `evaluate` to fill `EvalResult.score`.

```python
from chunktuner import DummyEmbeddingFunction, Evaluator, ScoreCalculator
from chunktuner import trivial_dataset_for_docs

ev = Evaluator(DummyEmbeddingFunction(), top_k=5)
scorer = ScoreCalculator(use_case="rag_qa")
ds = trivial_dataset_for_docs(docs)
result = ev.evaluate(strategy, cfg, docs, ds, scorer=scorer)
```

Enable generation metrics and supply a `RagasBridge` on the evaluator when using RAGAS (see below).

`retrieval_mode` picks the first-stage ranking: `"dense"` (default, embedding cosine similarity), `"bm25"` (lexical BM25 over all corpus chunks with corpus-wide IDF; makes no embedding calls) or `"hybrid"` (dense and BM25 fused with reciprocal rank fusion, k = 60). `bm25` and `hybrid` need `chunktuner[bm25]`; BM25 lowercases, drops English stopwords, does not stem, and its default tokenizer drops 1-character words, so it is approximate for code and CJK text. An optional `reranker` (any object following the `Reranker` protocol in `chunktuner.models`) reorders the top `rerank_candidates` (default 20) first-stage chunks of each ranking with one `score` call per query; effective k is unchanged and token metrics use the reranked order. The defaults (dense, no reranker) leave scores unchanged.

```python
from chunktuner.eval.rerankers import reranker_from_spec

reranker = reranker_from_spec("local")  # or "dummy", "local:<hf-model-id>", a LiteLLM rerank model
ev = Evaluator(DummyEmbeddingFunction(), retrieval_mode="hybrid", reranker=reranker)
```

`reranker_from_spec` builds `DummyReranker` for `"dummy"`, `CrossEncoderReranker` for `"local"` (`cross-encoder/ms-marco-MiniLM-L6-v2`) or `"local:<hf-model-id>"` (needs `chunktuner[late]`; downloads the model on first use), and `LiteLLMReranker` for any other string (calls the provider's rerank API, which may cost money). `EvalMetrics.reranker_overflow_ratio` reports the share of chunks longer than the reranker's input window.

### `ScoreCalculator`

Wraps `EvalMetrics` with use-case weights from `chunktuner.config.score_profile_weights` unless you pass `custom_weights`.

---

## `AutoTuner`

Grid search over `(strategy, params)` jobs, then rank by score.

```python
from chunktuner import AutoTuner, default_registry, Evaluator, ScoreCalculator, DummyEmbeddingFunction

tuner = AutoTuner(
    strategies=default_registry,
    evaluator=Evaluator(DummyEmbeddingFunction()),
    scorer=ScoreCalculator(use_case="rag_qa"),
)
rec = tuner.recommend(
    docs,
    "rag_qa",
    strategies=["fixed_tokens", "recursive_character"],
    param_grid={"fixed_tokens": [{"max_tokens": 256, "overlap_tokens": 0}]},
    max_docs=50,
    parallel=True,
    max_workers=4,
)
```

With `strategies=None`, every strategy compatible with the content type runs except the explicit-only `late_chunking` and `agentic`. Pass `encoding_name=` (default `cl100k_base`) to match a non-default `tokenizer_encoding` when `parallel=True`. Parallel workers rebuild the evaluator with its `encoding_name`, `context_budget_tokens`, `batch_size`, `retrieval_mode`, `reranker` and `rerank_candidates`; only the built-in rerankers (`DummyReranker`, `CrossEncoderReranker`, `LiteLLMReranker`) can be sent to them.

`parallel=True` uses a process pool; embedding payloads must be picklable (see the `AutoTuner` / parallel evaluation notes in the [API reference](api/tuner/auto_tuner.md)).

---

## Caching

`CachedEmbeddingFunction` delegates to an inner `EmbeddingFunction` and persists vectors through `EmbeddingCache`. Chunk text reuse can go through `ChunkCache` for advanced scenarios.

```python
from pathlib import Path
from chunktuner import CachedEmbeddingFunction
from chunktuner.cache.embedding_cache import EmbeddingCache, default_embedding_db_path
from chunktuner.eval.embeddings import LiteLLMEmbeddingFunction

cache_dir = Path.home() / ".cache" / "chunktuner"
db = default_embedding_db_path(cache_dir)
inner = LiteLLMEmbeddingFunction("text-embedding-3-small")
with EmbeddingCache(db, inner.model) as emb_cache:
    embed_fn = CachedEmbeddingFunction(inner, emb_cache)
    vecs = embed_fn.embed_documents(["hello", "world"])
```

---

## RAGAS (`RagasBridge`)

Install `chunktuner[ragas]` and compatible stack. Construct `RagasBridge(llm_client=...)` and pass it to `Evaluator(..., ragas_bridge=bridge, enable_generation_metrics=True)` so faithfulness / answer relevancy can populate when the evaluator’s generation path runs.

`RagasBridge.compute` accepts parallel lists of questions, retrieved contexts, answers, and ground truths; it returns averaged scores or `None` entries when RAGAS is unavailable.

---

## Datasets

- **`trivial_dataset_for_docs`** — one synthetic query per document (no LLM), for smoke tests and defaults.
- **`DatasetBuilder`** — `build_from_user_file`, `build_from_docs`, and code-oriented helpers for richer `EvalDataset` instances.

See [Metrics](metrics.md) for field meanings on `EvalMetrics` / `EvalResult`.

---

## See also

- [API reference](api/index.md) — mkdocstrings module pages
- [Configuration](configuration.md)
- [Contributing](contributing.md) — changes upstream (strategies, tests, docs workflow)
