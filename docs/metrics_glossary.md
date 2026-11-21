# Metrics glossary

Metrics are always relative to an embedding model and corpus — treat absolute numbers as **relative comparators** between strategies in the same run, not as certificates of production quality.

For how scores are combined per use case, see [Metrics reference](metrics.md). Field definitions below match `EvalMetrics` in `chunktuner.models`.

## Retrieval metrics

### `token_recall`

- **Definition:** Share of gold answer tokens (from `answer_spans` in each `EvalQuery`) that appear in the union of retrieved chunk token sets (within the effective context window).
- **Range:** 0–1
- **Direction:** Higher is better
- **Example:** `0.85` means about 85% of answer tokens were covered by at least one retrieved chunk.
- **Use cases:** Strong weight in `rag_qa`, `summarization`, `code_assist` (see `score_profile_weights` in `chunktuner.config`).

### `token_precision`

- **Definition:** Share of retrieved tokens (union of top chunks) that overlap gold answer tokens.
- **Range:** 0–1
- **Direction:** Higher is better
- **Use cases:** Informative for `rag_qa`; often paired with recall.

### `token_iou`

- **Definition:** Token-level IoU between gold and retrieved unions: `|gold ∩ retrieved| / |gold ∪ retrieved|`.
- **Range:** 0–1
- **Direction:** Higher is better
- **Intuition:** Penalizes both missing answer text and noisy extra text.
- **Use cases:** `rag_qa` profile weights IoU alongside recall.

### `recall_at_k` (keys 1, 3, 5)

- **Definition:** For each k, whether any of the top-k retrieved chunks hits a gold token (binary per query), then averaged across queries. Stored on `EvalMetrics.recall_at_k` as integer keys; `ScoreCalculator` maps the `search` profile’s `recall_at_1` weight to `recall_at_k[1]`.
- **Range:** 0–1 per k
- **Direction:** Higher is better
- **Use cases:** `search` weights recall at 1; the other profiles do not weight `recall_at_k`.

### `mrr` (mean reciprocal rank)

- **Definition:** Average of `1 / rank` of the first relevant chunk (0 if none), where relevance means overlap with gold answer tokens.
- **Range:** 0–1
- **Direction:** Higher is better
- **Intuition:** Rewards putting the first “hit” chunk earlier in the ranking.
- **Use cases:** `rag_qa`, `search`, `code_assist`.

### `ndcg_at_k` (keys 1, 3, 5)

- **Definition:** Normalized discounted cumulative gain using binary relevance (chunk hits gold tokens or not), computed per k on `EvalMetrics.ndcg_at_k`.
- **Range:** 0–1 per k
- **Direction:** Higher is better
- **Intuition:** Penalizes relevant chunks that appear late in the ranked list.
- **Use cases:** Ranking quality for retrieval-heavy profiles.

### `corpus_recall_at_k` / `corpus_mrr` / `corpus_ndcg_at_k` (keys 1, 3, 5)

- **Definition:** The same as `recall_at_k`, `mrr` and `ndcg_at_k`, but each query is ranked against every chunk in the corpus instead of only the chunks of its own document. Only chunks of the query's own document can be relevant, so chunks from other documents act as distractors.
- **Range:** 0–1 (per k for the dict fields); `None` on results produced before these fields existed
- **Direction:** Higher is better
- **Use cases:** Always computed. With `retrieval_scope="corpus"` (`--retrieval-scope corpus`), `ScoreCalculator` reads `mrr` from `corpus_mrr` and `recall_at_1` from `corpus_recall_at_k[1]` (missing values count as 0.0). No weight profile references nDCG.

### `duplication_ratio`

- **Definition:** Among token positions covered by retrieved chunks, the fraction of tokens that appear in more than one chunk (within the selected top set). Implemented in `chunktuner.eval.evaluator`.
- **Range:** 0–1
- **Direction:** Lower is better (less redundant context).
- **Use cases:** Penalized in default score profiles via negative weights.

### `avg_chunk_length`

- **Definition:** Mean tiktoken length of all chunks produced for the corpus in this run.
- **Range:** roughly 0–∞ tokens
- **Direction:** Neither strictly higher nor lower; profile-dependent.
- **Use cases:** Exposed to `ScoreCalculator` if you add a custom weight; **default** score profiles in `chunktuner.config.score_profile_weights` do not include this field — compare raw values across strategies instead.

### `chunk_length_std`

- **Definition:** Standard deviation of per-chunk token lengths.
- **Range:** 0–∞
- **Direction:** Lower is often better for uniform pipelines; **default** `code_assist` weights do not include this field (use custom weights if you want it in the composite score).
- **Intuition:** High variance can mean a mix of tiny and oversized chunks.

### `avg_tokens_per_query`

- **Definition:** Average total tokens in retrieved chunks per query (after effective-k truncation).
- **Range:** 0–∞
- **Direction:** Lower is often better for cost/latency; **default** score profiles do not weight this field (you can supply `custom_weights` on `ScoreCalculator` to use it).
- **Intuition:** Measures how “fat” the retrieval window is per query.

### `embedding_latency_ms` / `total_embedding_tokens`

- **Definition:** Wall time for batched embeddings in this evaluation; total tokens embedded (chunks + questions). Both are 0 with `retrieval_mode="bm25"`, which makes no embedding calls.
- **Range:** ms and token count ≥ 0
- **Direction:** Lower latency and predictable token counts help operational planning; not all score profiles weight these directly.

### `retrieval_mode` / `reranker`

- **Definition:** The retrieval pipeline that produced the ranking metrics. `retrieval_mode` is `dense` (embedding cosine similarity), `bm25` (lexical BM25 over all corpus chunks, no embedding calls) or `hybrid` (dense and BM25 rankings fused with reciprocal rank fusion). `reranker` is the reranker's `profile_name` (for example `dummy/rerank`, `local:cross-encoder/ms-marco-MiniLM-L6-v2` or the LiteLLM model string), or `None` without a reranker.
- **Range:** strings; `None` on results produced before these fields existed
- **Direction:** Not a score; labels to compare runs with the same pipeline
- **Intuition:** The reranker reorders the top `rerank_candidates` (default 20) first-stage chunks per query; the rank and token metrics use the reranked order.

### `reranker_overflow_ratio`

- **Definition:** Fraction of chunks whose token length (evaluator tiktoken encoding) exceeds the reranker's input window (`max_input_tokens`); those chunks are truncated by the reranker.
- **Range:** 0–1; `None` without a reranker or when the window is unknown (LiteLLM rerankers)
- **Direction:** Lower is better
- **Use cases:** Not weighted by any score profile. The CLI prints a warning on stderr when it is above 0.

## Generation metrics (optional, RAGAS)

### `faithfulness` / `answer_relevancy`

- **Definition:** When `Evaluator(enable_generation_metrics=True)` and RAGAS dependencies are available, `RagasBridge.compute` fills these from RAGAS `faithfulness` and `answer_relevancy` over generated answers vs retrieved contexts (see `chunktuner.eval.ragas_bridge`).
- **Range:** typically 0–1 when present
- **Direction:** Higher is better; **`None`** if RAGAS or the LLM path is unavailable
- **Use cases:** `rag_qa` default weights include faithfulness; a `None` value scores as 0.0.

When generation metrics are disabled or RAGAS fails, these fields stay `None` and `ScoreCalculator` scores them as 0.0 where weights apply.
