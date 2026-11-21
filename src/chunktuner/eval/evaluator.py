"""Retrieval-style evaluation over chunking strategies."""

import logging
import random
import time
from collections import Counter, defaultdict

import numpy as np
import tiktoken

from chunktuner.config import DEFAULT_LLM_MODEL
from chunktuner.eval.effective_k import compute_effective_k
from chunktuner.eval.lexical import Bm25Index, rrf_fuse
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.models import (
    Chunk,
    ChunkConfig,
    ChunkingStrategy,
    Document,
    EmbeddingFunction,
    EvalDataset,
    EvalMetrics,
    EvalResult,
    Reranker,
    RetrievalMode,
    RetrievalScope,
)

logger = logging.getLogger(__name__)

_EPS: float = 1e-9  # zero-division guard for cosine similarity normalization


def _token_bounds(encoding: tiktoken.Encoding, text: str) -> list[int]:
    """Map token positions to char offsets using byte-level alignment.

    Builds a byte-offset→char-offset table for the full text, then looks up
    each token's byte width via decode_single_token_bytes(). This is correct
    even when token boundaries fall mid-codepoint.
    """
    ids = encoding.encode(text)
    if not ids:
        return [0]
    text_utf8 = text.encode("utf-8")
    # byte_to_char[b] = index of the str character owning UTF-8 byte b; last = len(text)
    byte_to_char: list[int] = [0] * (len(text_utf8) + 1)
    byte_idx = 0
    for char_idx, ch in enumerate(text):
        ch_len = len(ch.encode("utf-8"))
        for k in range(ch_len):
            byte_to_char[byte_idx + k] = char_idx
        byte_idx += ch_len
    byte_to_char[byte_idx] = len(text)  # sentinel

    byte_offset = 0
    bounds: list[int] = [0]
    for tid in ids:
        byte_offset += len(encoding.decode_single_token_bytes(tid))
        bounds.append(byte_to_char[min(byte_offset, len(text_utf8))])
    return bounds


def _token_indices_for_span(bounds: list[int], a: int, b: int) -> set[int]:
    n = len(bounds) - 1
    out: set[int] = set()
    for i in range(n):
        if bounds[i] < b and bounds[i + 1] > a:
            out.add(i)
    return out


def _token_indices_for_chunk(bounds: list[int], chunk: Chunk) -> set[int]:
    return _token_indices_for_span(bounds, chunk.start_offset, chunk.end_offset)


def _duplication_ratio(chunks: list[Chunk], bounds: list[int]) -> float:
    if len(chunks) < 2:
        return 0.0
    freq: Counter[int] = Counter()
    for c in chunks:
        for t in _token_indices_for_chunk(bounds, c):
            freq[t] += 1
    multi = sum(1 for _, c in freq.items() if c > 1)
    return multi / max(1, len(freq))


def _ndcg_at_k(rels: list[float], k: int) -> float:
    rels = rels[:k]
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


_RANK_KS: tuple[int, ...] = (1, 3, 5)

_RETRIEVAL_MODES: tuple[str, ...] = ("dense", "bm25", "hybrid")


def _ranking_metrics(rels: list[float]) -> tuple[dict[int, float], float, dict[int, float]]:
    """Per-query hit@k, reciprocal rank, and nDCG@k from a ranked binary relevance list.

    Shared by document-scope and corpus-scope retrieval so both use identical formulas.
    """
    hits = {kk: 1.0 if any(r > 0 for r in rels[:kk]) else 0.0 for kk in _RANK_KS}
    ndcg = {kk: _ndcg_at_k(rels, kk) for kk in _RANK_KS}
    first_rel = next((i + 1 for i, r in enumerate(rels) if r > 0), None)
    rr = 1.0 / first_rel if first_rel else 0.0
    return hits, rr, ndcg


class Evaluator:
    """Chunks documents, embeds text, runs retrieval metrics (and optional RAGAS).

    Args:
        embedding_fn: Embeds chunk texts and queries.
        top_k: Stored for API compatibility only; it does **not** affect any metric. The
            number of chunks retrieved per query for token-level metrics is the effective k
            from `compute_effective_k`: ``context_budget_tokens / avg chunk tokens``.
        enable_generation_metrics: Compute RAGAS metrics when RAGAS and a judge key exist.
        llm_client: Optional client forwarded to `RagasBridge`.
        context_budget_tokens: Token budget used to derive the effective k.
        retrieval_scope: Which ranking feeds the score's ``mrr`` / ``recall_at_1`` when the
            evaluator builds its default `ScoreCalculator`. Both document-scope and
            corpus-scope ranking metrics are always computed and stored on `EvalMetrics`.
        retrieval_mode: First-stage ranking for both scopes. ``dense`` (default) ranks by
            embedding cosine similarity; ``bm25`` ranks by BM25 over all corpus chunks
            (corpus-wide IDF; needs the ``bm25`` extra) and makes no embedding calls;
            ``hybrid`` fuses the dense and BM25 rankings with reciprocal rank fusion.
        reranker: Optional second stage. Per query it scores the top ``rerank_candidates``
            chunks of each scope's first-stage ranking (one ``score`` call on their union)
            and reorders those by reranker score; the rest keep first-stage order. Token
            metrics and generation contexts use the reranked document-scope order.
        rerank_candidates: How many first-stage chunks per scope the reranker reorders.
    """

    @staticmethod
    def _ragas_available() -> bool:
        try:
            import ragas  # noqa: F401
        except ImportError:
            return False
        from chunktuner.eval.ragas_bridge import RagasBridge

        return RagasBridge.is_configured()

    def __init__(
        self,
        embedding_fn: EmbeddingFunction,
        top_k: int = 5,
        enable_generation_metrics: bool = False,
        llm_client: object | None = None,
        *,
        encoding_name: str = "cl100k_base",
        context_budget_tokens: int = 2000,
        batch_size: int = 64,
        llm_answer_model: str | None = None,
        llm_api_base: str | None = None,
        llm_api_key: str | None = None,
        ragas_bridge: object | None = None,
        retrieval_scope: RetrievalScope = "document",
        retrieval_mode: RetrievalMode = "dense",
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
    ):
        if retrieval_mode not in _RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of {', '.join(_RETRIEVAL_MODES)}; "
                f"got {retrieval_mode!r}"
            )
        if rerank_candidates < 1:
            raise ValueError(f"rerank_candidates must be >= 1, got {rerank_candidates}")
        self.embedding_fn = embedding_fn
        self.top_k = top_k
        wants_generation = enable_generation_metrics
        self.enable_generation_metrics = wants_generation and self._ragas_available()
        if wants_generation and not self.enable_generation_metrics:
            try:
                import ragas  # noqa: F401
            except ImportError:
                logger.info(
                    "Generation metrics disabled: RAGAS is not installed "
                    "(e.g. uv add 'chunktuner[ragas]')."
                )
            else:
                logger.info(
                    "Generation metrics disabled: no API key found for RAGAS "
                    "(set OPENAI_API_KEY, GEMINI_API_KEY, CHUNKTUNER_API_KEY, or another "
                    "provider key RAGAS uses). Local OpenAI-compatible servers without a "
                    "judge LLM key are skipped — use retrieval metrics only."
                )
        self.llm_client = llm_client
        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)
        self.context_budget_tokens = context_budget_tokens
        self.batch_size = batch_size
        self.llm_answer_model = llm_answer_model or DEFAULT_LLM_MODEL
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self.ragas_bridge = ragas_bridge
        self.retrieval_scope: RetrievalScope = retrieval_scope
        self.retrieval_mode: RetrievalMode = retrieval_mode
        self.reranker: Reranker | None = reranker
        self.rerank_candidates = rerank_candidates

    def _llm_provider_kwargs(self) -> dict[str, str]:
        kw: dict[str, str] = {}
        if self.llm_api_base:
            kw["api_base"] = self.llm_api_base
        if self.llm_api_key:
            kw["api_key"] = self.llm_api_key
        return kw

    def evaluate(
        self,
        strategy: ChunkingStrategy,
        config: ChunkConfig,
        docs: list[Document],
        dataset: EvalDataset,
        *,
        scorer: ScoreCalculator | None = None,
    ) -> EvalResult:
        """Evaluate one strategy configuration on a document set and dataset.

        Chunks each document, validates offsets, embeds chunks and dataset queries,
        computes retrieval metrics per query, optionally generation metrics, and
        assigns ``score`` when ``scorer`` is provided.

        Args:
            strategy: Registered chunking implementation.
            config: Strategy name and parameters.
            docs: Corpus (ids must match ``dataset`` references).
            dataset: Queries and gold spans for scoring.
            scorer: If set, used to populate `EvalResult.score` (its own ``retrieval_scope``
                applies). Otherwise a ``rag_qa`` scorer using ``self.retrieval_scope``.

        Returns:
            `EvalResult` with `EvalMetrics` and composite score.
        """
        docs_by_id = {d.id: d for d in docs}
        all_chunks: list[Chunk] = []
        for d in docs:
            all_chunks.extend(strategy.chunk(d, config))

        self._validate_offsets_sample(all_chunks, docs_by_id)

        use_dense = self.retrieval_mode != "bm25"
        use_bm25 = self.retrieval_mode != "dense"
        latency_ms = 0.0
        q_vecs: dict[str, np.ndarray] = {}
        chunk_vecs: np.ndarray | None = None
        if use_dense:
            t0 = time.perf_counter()
            chunk_vecs = self._embed_batched([c.text for c in all_chunks])
            q_texts = [q.question for q in dataset.queries]
            q_vecs_arr = self._embed_batched(q_texts)
            q_vecs = {q.id: q_vecs_arr[i] for i, q in enumerate(dataset.queries)}
            latency_ms = (time.perf_counter() - t0) * 1000
        # One index over every corpus chunk so IDF is corpus-wide for both scopes.
        bm25_index = Bm25Index([c.text for c in all_chunks]) if use_bm25 and all_chunks else None

        by_doc: dict[str, list[tuple[int, Chunk]]] = defaultdict(list)
        for idx, ch in enumerate(all_chunks):
            by_doc[ch.document_id].append((idx, ch))

        tok_lens = [len(self._enc.encode(c.text)) for c in all_chunks]
        avg_chunk_len = float(np.mean(tok_lens)) if all_chunks else 0.0
        chunk_std = float(np.std(tok_lens)) if len(all_chunks) > 1 else 0.0

        recalls_at: dict[int, list[float]] = {k: [] for k in _RANK_KS}
        mrrs: list[float] = []
        ndcgs: dict[int, list[float]] = {k: [] for k in _RANK_KS}
        c_recalls_at: dict[int, list[float]] = {k: [] for k in _RANK_KS}
        c_mrrs: list[float] = []
        c_ndcgs: dict[int, list[float]] = {k: [] for k in _RANK_KS}
        # Normalized vectors of every chunk, reused for corpus-scope ranking (no extra embeds).
        all_matn = (
            chunk_vecs / (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + _EPS)
            if all_chunks and chunk_vecs is not None
            else None
        )
        precs: list[float] = []
        recalls: list[float] = []
        ious: list[float] = []
        dups: list[float] = []
        toks_per_q: list[float] = []
        topk_by_q: dict[str, list[Chunk]] = {}

        for q in dataset.queries:
            doc = docs_by_id.get(q.document_id)
            if doc is None:
                logger.warning(
                    "Query %r references document_id=%r which is not in the ingested corpus. "
                    "Skipping. Check that document IDs in the eval dataset match ingested doc IDs.",
                    q.id,
                    q.document_id,
                )
                continue
            bounds = _token_bounds(self._enc, doc.content)
            gold: set[int] = set()
            for a, b in q.answer_spans:
                gold |= _token_indices_for_span(bounds, a, b)
            if not gold:
                continue

            pairs = by_doc.get(q.document_id, [])
            if not pairs:
                continue
            idxs = [p[0] for p in pairs]
            # ``order``: positions into ``pairs`` best first; ``c_order``: corpus chunk indices.
            order, c_order = self._first_stage_orders(
                q.id, q.question, idxs, chunk_vecs, all_matn, q_vecs, bm25_index
            )
            if self.reranker is not None:
                order, c_order = self._rerank(q.question, idxs, order, c_order, all_chunks)
            ranked_chunks = [pairs[int(j)][1] for j in order]

            eff_k = compute_effective_k(avg_chunk_len, self.context_budget_tokens)
            k_use = min(max(eff_k, 1), len(ranked_chunks))

            rels = [
                1.0 if gold & _token_indices_for_chunk(bounds, ranked_chunks[j]) else 0.0
                for j in range(len(ranked_chunks))
            ]

            hits, rr, ndcg_q = _ranking_metrics(rels)
            for kk in _RANK_KS:
                recalls_at[kk].append(hits[kk])
                ndcgs[kk].append(ndcg_q[kk])
            mrrs.append(rr)

            # Corpus scope: rank against every chunk; only own-doc chunks can be relevant.
            rel_by_idx = {
                idx: (1.0 if gold & _token_indices_for_chunk(bounds, ch) else 0.0)
                for idx, ch in pairs
            }
            c_rels = [rel_by_idx.get(int(j), 0.0) for j in c_order]
            c_hits, c_rr, c_ndcg_q = _ranking_metrics(c_rels)
            for kk in _RANK_KS:
                c_recalls_at[kk].append(c_hits[kk])
                c_ndcgs[kk].append(c_ndcg_q[kk])
            c_mrrs.append(c_rr)

            topk = ranked_chunks[:k_use]
            ret_union: set[int] = set()
            for c in topk:
                ret_union |= _token_indices_for_chunk(bounds, c)
            inter = gold & ret_union
            precs.append(len(inter) / max(1, len(ret_union)))
            recalls.append(len(inter) / max(1, len(gold)))
            union = gold | ret_union
            ious.append(len(inter) / max(1, len(union)))
            dups.append(_duplication_ratio(topk, bounds))
            toks_per_q.append(float(sum(len(self._enc.encode(c.text)) for c in topk)))
            topk_by_q[q.id] = topk

        def mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        recall_at_k_dict = {k: mean(v) for k, v in recalls_at.items()}
        ndcg_at_k_dict = {k: mean(v) for k, v in ndcgs.items()}

        metrics = EvalMetrics(
            token_iou=mean(ious),
            token_precision=mean(precs),
            token_recall=mean(recalls),
            recall_at_k=recall_at_k_dict,
            mrr=mean(mrrs),
            ndcg_at_k=ndcg_at_k_dict,
            corpus_recall_at_k={k: mean(v) for k, v in c_recalls_at.items()},
            corpus_mrr=mean(c_mrrs),
            corpus_ndcg_at_k={k: mean(v) for k, v in c_ndcgs.items()},
            avg_tokens_per_query=mean(toks_per_q),
            duplication_ratio=mean(dups),
            avg_chunk_length=avg_chunk_len,
            chunk_length_std=chunk_std,
            embedding_latency_ms=latency_ms,
            total_embedding_tokens=(
                sum(len(self._enc.encode(c.text)) for c in all_chunks)
                + sum(len(self._enc.encode(q.question)) for q in dataset.queries)
                if use_dense
                else 0
            ),
            retrieval_mode=self.retrieval_mode,
            reranker=self.reranker.profile_name if self.reranker is not None else None,
            reranker_overflow_ratio=self._overflow_ratio(tok_lens),
        )

        if self.enable_generation_metrics and topk_by_q:
            metrics = self._apply_generation_metrics(
                metrics,
                docs_by_id,
                dataset,
                topk_by_q,
            )

        sc = scorer or ScoreCalculator("rag_qa", retrieval_scope=self.retrieval_scope)
        score = sc.score(metrics)
        return EvalResult(
            strategy_name=strategy.name,
            config=config,
            embedding_profile=self.embedding_fn.profile_name,
            metrics=metrics,
            score=score,
        )

    def _first_stage_orders(
        self,
        query_id: str,
        query_text: str,
        idxs: list[int],
        chunk_vecs: np.ndarray | None,
        all_matn: np.ndarray | None,
        q_vecs: dict[str, np.ndarray],
        bm25_index: Bm25Index | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Document-scope order (positions into ``idxs``) and corpus-scope order (chunk ids)."""
        dense_order = dense_c_order = None
        if self.retrieval_mode != "bm25":
            assert chunk_vecs is not None and all_matn is not None
            mat = np.stack([chunk_vecs[i] for i in idxs])
            qv = q_vecs[query_id]
            qn = qv / (np.linalg.norm(qv) + _EPS)
            matn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + _EPS)
            sims = matn @ qn
            # Keep these exact argsort calls: the dense default must stay bit-identical.
            dense_order = np.argsort(-sims)
            dense_c_order = np.argsort(-(all_matn @ qn))
            if self.retrieval_mode == "dense":
                return dense_order, dense_c_order
        assert bm25_index is not None
        bm25_all = bm25_index.scores(query_text)
        bm25_order = np.argsort(-bm25_all[np.asarray(idxs, dtype=np.int64)], kind="stable")
        bm25_c_order = np.argsort(-bm25_all, kind="stable")
        if self.retrieval_mode == "bm25":
            return bm25_order, bm25_c_order
        assert dense_order is not None and dense_c_order is not None
        return rrf_fuse([dense_order, bm25_order]), rrf_fuse([dense_c_order, bm25_c_order])

    def _rerank(
        self,
        query_text: str,
        idxs: list[int],
        order: np.ndarray,
        c_order: np.ndarray,
        all_chunks: list[Chunk],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reorder each scope's top ``rerank_candidates`` by reranker score (one call)."""
        assert self.reranker is not None
        n = self.rerank_candidates
        doc_head = [int(j) for j in order[:n]]
        c_head = [int(j) for j in c_order[:n]]
        # Union of candidate chunk ids across both scopes, deduplicated in stable order.
        cand = list(dict.fromkeys([idxs[j] for j in doc_head] + c_head))
        scores = self.reranker.score(query_text, [all_chunks[i].text for i in cand])
        if len(scores) != len(cand):
            raise ValueError(
                f"Reranker {self.reranker.profile_name!r} returned {len(scores)} scores "
                f"for {len(cand)} documents"
            )
        by_chunk = dict(zip(cand, (float(s) for s in scores), strict=True))
        # sorted() is stable: equal reranker scores keep their first-stage order.
        new_doc_head = sorted(doc_head, key=lambda j: -by_chunk[idxs[j]])
        new_c_head = sorted(c_head, key=lambda i: -by_chunk[i])
        new_order = np.concatenate([np.asarray(new_doc_head, dtype=np.int64), order[n:]])
        new_c_order = np.concatenate([np.asarray(new_c_head, dtype=np.int64), c_order[n:]])
        return new_order, new_c_order

    def _overflow_ratio(self, tok_lens: list[int]) -> float | None:
        """Fraction of chunks longer than the reranker's input window (``None`` if unknown)."""
        if self.reranker is None or not self.reranker.max_input_tokens:
            return None
        if not tok_lens:
            return 0.0
        limit = self.reranker.max_input_tokens
        return sum(1 for t in tok_lens if t > limit) / len(tok_lens)

    def _apply_generation_metrics(
        self,
        metrics: EvalMetrics,
        docs_by_id: dict[str, Document],
        dataset: EvalDataset,
        topk_by_q: dict[str, list[Chunk]],
    ) -> EvalMetrics:
        import litellm

        from chunktuner.eval.ragas_bridge import RagasBridge

        bridge = self.ragas_bridge or RagasBridge(self.llm_client)
        qs: list[str] = []
        ctxs: list[list[str]] = []
        ans: list[str] = []
        gts: list[str] = []
        for q in dataset.queries:
            if q.id not in topk_by_q:
                continue
            doc = docs_by_id.get(q.document_id)
            if doc is None:
                continue
            ctx = [c.text for c in topk_by_q[q.id]]
            con = "\n\n".join(ctx)
            try:
                resp = litellm.completion(
                    model=self.llm_answer_model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Answer using ONLY the context. Be concise.\n\n"
                                f"Context:\n{con}\n\nQuestion: {q.question}"
                            ),
                        }
                    ],
                    max_tokens=256,
                    temperature=0.0,
                    **self._llm_provider_kwargs(),
                )
                answer = (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                logger.warning(
                    "Generation metric LLM call failed: %s. "
                    "Query %r excluded from generation metrics.",
                    exc,
                    q.id,
                )
                continue

            if not answer:
                continue

            ref = q.reference_answer
            if not ref and q.answer_spans:
                a0, b0 = q.answer_spans[0]
                ref = doc.content[a0:b0]
            qs.append(q.question)
            ctxs.append(ctx)
            ans.append(answer)
            gts.append(ref or "")

        if not qs:
            return metrics

        scores = bridge.compute(qs, ctxs, ans, gts)
        updates: dict[str, float | None] = {}
        if scores.get("faithfulness") is not None:
            updates["faithfulness"] = scores["faithfulness"]
        if scores.get("answer_relevancy") is not None:
            updates["answer_relevancy"] = scores["answer_relevancy"]
        return metrics.model_copy(update=updates) if updates else metrics

    def _embed_batched(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self.embedding_fn.embed_documents(batch))
        return np.array(out, dtype=np.float64)

    def _validate_offsets_sample(
        self,
        chunks: list[Chunk],
        docs_by_id: dict[str, Document],
    ) -> None:
        if not chunks:
            return
        n = len(chunks)
        if n <= 500:
            sample = chunks
        else:
            k = max(50, min(200, n // 10))
            sample = random.Random(42).sample(chunks, k)
        for c in sample:
            d = docs_by_id[c.document_id]
            n = len(d.content)
            if not (0 <= c.start_offset < c.end_offset <= n):
                raise ValueError(
                    f"Chunk {c.id!r} offsets [{c.start_offset}:{c.end_offset}] are out of bounds "
                    f"for doc {d.id!r} (length {n})"
                )
            got = d.content[c.start_offset : c.end_offset]
            if got != c.text:
                raise ValueError(
                    f"Offset invariant failed for chunk {c.id!r}: "
                    f"content[{c.start_offset}:{c.end_offset}]={got[:60]!r} "
                    f"!= chunk.text={c.text[:60]!r}"
                )
