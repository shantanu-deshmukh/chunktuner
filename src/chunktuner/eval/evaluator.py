"""Retrieval-style evaluation over chunking strategies."""

from __future__ import annotations

import logging
import random
import time
from collections import Counter, defaultdict

import numpy as np
import tiktoken

from chunktuner.eval.effective_k import compute_effective_k
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


class Evaluator:
    """Chunks documents, embeds text, runs retrieval metrics (and optional RAGAS)."""

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
        ragas_bridge: object | None = None,
    ):
        self.embedding_fn = embedding_fn
        self.top_k = top_k
        self.enable_generation_metrics = enable_generation_metrics
        self.llm_client = llm_client
        self._enc = tiktoken.get_encoding(encoding_name)
        self.context_budget_tokens = context_budget_tokens
        self.batch_size = batch_size
        self.llm_answer_model = llm_answer_model or "gpt-4o-mini"
        self.ragas_bridge = ragas_bridge

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
            scorer: If set, used to populate `EvalResult.score`.

        Returns:
            `EvalResult` with `EvalMetrics` and composite score.
        """
        docs_by_id = {d.id: d for d in docs}
        all_chunks: list[Chunk] = []
        for d in docs:
            all_chunks.extend(strategy.chunk(d, config))

        self._validate_offsets_sample(all_chunks, docs_by_id)

        t0 = time.perf_counter()
        chunk_vecs = self._embed_batched([c.text for c in all_chunks])
        q_vecs: dict[str, np.ndarray] = {}
        for q in dataset.queries:
            v = np.array(self.embedding_fn.embed_query(q.question), dtype=np.float64)
            q_vecs[q.id] = v
        latency_ms = (time.perf_counter() - t0) * 1000

        by_doc: dict[str, list[tuple[int, Chunk]]] = defaultdict(list)
        for idx, ch in enumerate(all_chunks):
            by_doc[ch.document_id].append((idx, ch))

        tok_lens = [len(self._enc.encode(c.text)) for c in all_chunks]
        avg_chunk_len = float(np.mean(tok_lens)) if all_chunks else 0.0
        chunk_std = float(np.std(tok_lens)) if len(all_chunks) > 1 else 0.0

        recalls_at: dict[int, list[float]] = {1: [], 3: [], 5: []}
        mrrs: list[float] = []
        ndcgs: dict[int, list[float]] = {1: [], 3: [], 5: []}
        precs: list[float] = []
        recalls: list[float] = []
        ious: list[float] = []
        dups: list[float] = []
        toks_per_q: list[float] = []
        topk_by_q: dict[str, list[Chunk]] = {}

        for q in dataset.queries:
            doc = docs_by_id.get(q.document_id)
            if doc is None:
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
            mat = np.stack([chunk_vecs[i] for i in idxs])
            qv = q_vecs[q.id]
            qn = qv / (np.linalg.norm(qv) + _EPS)
            matn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + _EPS)
            sims = matn @ qn
            order = np.argsort(-sims)
            ranked_chunks = [pairs[int(j)][1] for j in order]

            eff_k = compute_effective_k(avg_chunk_len, self.context_budget_tokens)
            k_use = min(max(eff_k, 1), len(ranked_chunks))

            rels = [
                1.0 if gold & _token_indices_for_chunk(bounds, ranked_chunks[j]) else 0.0
                for j in range(len(ranked_chunks))
            ]

            for kk in (1, 3, 5):
                top = ranked_chunks[: min(kk, len(ranked_chunks))]
                hit = any(gold & _token_indices_for_chunk(bounds, c) for c in top)
                recalls_at[kk].append(1.0 if hit else 0.0)
                ndcgs[kk].append(_ndcg_at_k(rels, kk))

            first_rel = next((i + 1 for i, r in enumerate(rels) if r > 0), None)
            mrrs.append(1.0 / first_rel if first_rel else 0.0)

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
            avg_tokens_per_query=mean(toks_per_q),
            duplication_ratio=mean(dups),
            avg_chunk_length=avg_chunk_len,
            chunk_length_std=chunk_std,
            embedding_latency_ms=latency_ms,
            total_embedding_tokens=sum(len(self._enc.encode(c.text)) for c in all_chunks)
            + sum(len(self._enc.encode(q.question)) for q in dataset.queries),
        )

        if self.enable_generation_metrics and topk_by_q:
            metrics = self._apply_generation_metrics(
                metrics,
                docs_by_id,
                dataset,
                topk_by_q,
            )

        sc = scorer or ScoreCalculator("rag_qa")
        score = sc.score(metrics)
        return EvalResult(
            strategy_name=strategy.name,
            config=config,
            embedding_profile=self.embedding_fn.profile_name,
            metrics=metrics,
            score=score,
        )

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
                )
                answer = (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                logger.warning(
                    "Generation metric LLM call failed: %s. Skipping for this query.", exc
                )
                answer = ""
            ref = q.reference_answer
            if not ref and q.answer_spans:
                a0, b0 = q.answer_spans[0]
                ref = doc.content[a0:b0]
            qs.append(q.question)
            ctxs.append(ctx)
            ans.append(answer)
            gts.append(ref or "")
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
            got = d.content[c.start_offset : c.end_offset]
            if got != c.text:
                raise ValueError(
                    f"Offset invariant failed for chunk {c.id!r}: "
                    f"content[{c.start_offset}:{c.end_offset}]={got[:60]!r} "
                    f"!= chunk.text={c.text[:60]!r}"
                )
