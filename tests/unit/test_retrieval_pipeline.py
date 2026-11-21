"""D17: evaluator retrieval pipeline — BM25 / hybrid first stage and reranking."""

import numpy as np
import pytest
import tiktoken

from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.rerankers import DummyReranker
from chunktuner.models import Chunk, ChunkConfig, Document, EvalDataset, EvalQuery
from tests._extras import require_extra

CFG = ChunkConfig(name="paragraphs", params={})


@pytest.fixture
def bm25() -> None:
    require_extra("bm25", "bm25s")


class ParagraphStrategy:
    """One chunk per ``\\n\\n``-separated paragraph (separators excluded)."""

    name = "paragraphs"
    supported_content_types = ["text"]
    description = "test double"

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        out: list[Chunk] = []
        start = 0
        for i, part in enumerate(doc.content.split("\n\n")):
            end = start + len(part)
            out.append(
                Chunk.from_document(doc, id=f"{doc.id}:{i}", start_offset=start, end_offset=end)
            )
            start = end + 2
        return out

    def param_schema(self) -> dict:
        return {}

    def default_param_grid(self) -> list[dict]:
        return [{}]


class FillerEmbedding:
    """Dense double that prefers chunks with many 'filler' words, ignoring the query text.

    Chunks embed as ``(filler count, 1)``; known query strings embed as ``(1, 0)`` (the
    evaluator embeds queries through ``embed_documents`` too).
    """

    profile_name = "filler-test"
    QUERIES = frozenset({"zebra", "the of and"})

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] if t in self.QUERIES else [float(t.count("filler")), 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0]


class SpyReranker(DummyReranker):
    def __init__(self, max_input_tokens: int | None = 512) -> None:
        super().__init__(max_input_tokens=max_input_tokens)
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return super().score(query, documents)


def _doc(doc_id: str, paragraphs: list[str]) -> Document:
    return Document(id=doc_id, content="\n\n".join(paragraphs), content_type="text")


def _span(doc: Document, para: int) -> tuple[int, int]:
    parts = doc.content.split("\n\n")
    start = sum(len(p) + 2 for p in parts[:para])
    return (start, start + len(parts[para]))


def _ds(doc: Document, para: int, question: str = "zebra", n: int = 1) -> EvalDataset:
    return EvalDataset(
        name="t",
        queries=[
            EvalQuery(
                id=f"q{i}", question=question, document_id=doc.id, answer_spans=[_span(doc, para)]
            )
            for i in range(n)
        ],
        source="user_provided",
    )


# Dense (FillerEmbedding) document order: P0, P2, P1. BM25 order for "zebra": P1, P0, P2.
TARGET = _doc("a", ["filler filler text here", "zebra stripes are unique", "filler words again"])
DISTRACTOR = _doc("b", ["filler filler filler"])


def _run(mode: str, docs: list[Document] | None = None, **kw: object):
    emb = FillerEmbedding()
    ev = Evaluator(emb, retrieval_mode=mode, **kw)  # type: ignore[arg-type]
    res = ev.evaluate(ParagraphStrategy(), CFG, docs or [TARGET, DISTRACTOR], _ds(TARGET, 1))
    return res, emb


def test_dense_ranks_keyword_chunk_last() -> None:
    res, emb = _run("dense")
    assert res.metrics.mrr == pytest.approx(1 / 3)
    assert res.metrics.corpus_mrr == pytest.approx(1 / 4)
    assert res.metrics.retrieval_mode == "dense"
    assert res.metrics.reranker is None and res.metrics.reranker_overflow_ratio is None
    assert emb.calls > 0


def test_bm25_ranks_keyword_chunk_first_without_embedding(bm25: None) -> None:
    res, emb = _run("bm25")
    assert res.metrics.mrr == pytest.approx(1.0)
    assert res.metrics.corpus_mrr == pytest.approx(1.0)
    assert res.metrics.retrieval_mode == "bm25"
    assert emb.calls == 0
    assert res.metrics.total_embedding_tokens == 0
    assert res.metrics.embedding_latency_ms == 0.0


def test_bm25_query_without_tokens_keeps_chunk_order(bm25: None) -> None:
    emb = FillerEmbedding()
    ev = Evaluator(emb, retrieval_mode="bm25")
    # Stopwords only: every chunk scores 0, the stable order is chunk order (gold P1 -> rank 2).
    res = ev.evaluate(ParagraphStrategy(), CFG, [TARGET], _ds(TARGET, 1, question="the of and"))
    assert res.metrics.mrr == pytest.approx(0.5)
    assert emb.calls == 0


def test_bm25_single_chunk_document(bm25: None) -> None:
    doc = _doc("s", ["only zebra here"])
    res = Evaluator(FillerEmbedding(), retrieval_mode="bm25").evaluate(
        ParagraphStrategy(), CFG, [doc, DISTRACTOR], _ds(doc, 0)
    )
    assert res.metrics.mrr == pytest.approx(1.0)
    assert res.metrics.corpus_mrr == pytest.approx(1.0)


def test_hybrid_fuses_dense_and_bm25(bm25: None) -> None:
    # Doc scope RRF of [P0, P2, P1] and [P1, P0, P2] -> [P0, P1, P2]: gold P1 at rank 2.
    res, emb = _run("hybrid")
    assert res.metrics.mrr == pytest.approx(0.5)
    assert res.metrics.retrieval_mode == "hybrid"
    assert emb.calls > 0
    again, _ = _run("hybrid")
    assert again.metrics.model_dump(exclude={"embedding_latency_ms"}) == res.metrics.model_dump(
        exclude={"embedding_latency_ms"}
    )


def test_invalid_retrieval_mode_and_candidates() -> None:
    with pytest.raises(ValueError, match="retrieval_mode"):
        Evaluator(FillerEmbedding(), retrieval_mode="sparse")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rerank_candidates"):
        Evaluator(FillerEmbedding(), rerank_candidates=0)


# Dense order P0..P4 (filler counts 4, 3, 2, 1, 0); DummyReranker("zebra") favours P3 and P4.
LADDER = _doc(
    "l",
    [
        "filler filler filler filler",
        "filler filler filler",
        "filler filler",
        "filler zebra",
        "zebra zebra",
    ],
)


@pytest.mark.parametrize(
    ("candidates", "gold", "expected_mrr"),
    [
        (2, 3, 1 / 4),  # reranker only sees P0, P1 (both score 0): order unchanged
        (4, 3, 1.0),  # P3 promoted to the top of the reranked head
        (4, 4, 1 / 5),  # P4 is outside the head: tail keeps first-stage order
        (5, 4, 1.0),
    ],
)
def test_reranker_reorders_only_top_candidates(
    candidates: int, gold: int, expected_mrr: float
) -> None:
    rr = SpyReranker()
    ev = Evaluator(FillerEmbedding(), reranker=rr, rerank_candidates=candidates)
    res = ev.evaluate(ParagraphStrategy(), CFG, [LADDER], _ds(LADDER, gold))
    assert res.metrics.mrr == pytest.approx(expected_mrr)
    assert res.metrics.corpus_mrr == pytest.approx(expected_mrr)
    assert res.metrics.reranker == "dummy/rerank"
    assert len(rr.calls) == 1
    assert len(rr.calls[0][1]) == candidates


def test_rerank_order_exact() -> None:
    ev = Evaluator(FillerEmbedding(), reranker=DummyReranker(), rerank_candidates=4)
    chunks = ParagraphStrategy().chunk(LADDER, CFG)
    order, c_order = ev._rerank(
        "zebra", list(range(5)), np.arange(5), np.array([4, 3, 2, 1, 0]), chunks
    )
    assert order.tolist() == [3, 0, 1, 2, 4]
    # Corpus head [4, 3, 2, 1]: P4 (two zebras) > P3 > ties keep first-stage order; tail P0.
    assert c_order.tolist() == [4, 3, 2, 1, 0]


@pytest.mark.parametrize("scope", ["document", "corpus"])
def test_reranker_called_once_per_query_on_candidate_union(scope: str) -> None:
    rr = SpyReranker()
    ev = Evaluator(FillerEmbedding(), reranker=rr, rerank_candidates=2, retrieval_scope=scope)  # type: ignore[arg-type]
    ds = _ds(LADDER, 3, n=3)
    ev.evaluate(ParagraphStrategy(), CFG, [LADDER, DISTRACTOR], ds)
    assert len(rr.calls) == 3
    # Doc head: P0, P1. Corpus head: P0 plus P1 or the distractor (3 fillers, tied with P1);
    # the candidate union is deduplicated.
    for _, docs in rr.calls:
        assert len(docs) == len(set(docs))
        assert 2 <= len(docs) <= 4
        assert LADDER.content.split("\n\n")[0] in docs


def test_reranker_overflow_ratio() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    paras = LADDER.content.split("\n\n")
    limit = 3
    expected = sum(1 for p in paras if len(enc.encode(p)) > limit) / len(paras)
    assert 0.0 < expected < 1.0
    res = Evaluator(FillerEmbedding(), reranker=DummyReranker(max_input_tokens=limit)).evaluate(
        ParagraphStrategy(), CFG, [LADDER], _ds(LADDER, 3)
    )
    assert res.metrics.reranker_overflow_ratio == pytest.approx(expected)
    unknown = Evaluator(FillerEmbedding(), reranker=DummyReranker(max_input_tokens=None)).evaluate(
        ParagraphStrategy(), CFG, [LADDER], _ds(LADDER, 3)
    )
    assert unknown.metrics.reranker_overflow_ratio is None


def test_bm25_with_reranker_and_corpus(bm25: None) -> None:
    rr = SpyReranker()
    emb = FillerEmbedding()
    ev = Evaluator(emb, retrieval_mode="bm25", reranker=rr, rerank_candidates=2)
    res = ev.evaluate(ParagraphStrategy(), CFG, [LADDER, DISTRACTOR], _ds(LADDER, 4))
    assert emb.calls == 0
    assert len(rr.calls) == 1
    assert res.metrics.mrr == pytest.approx(1.0)
    assert res.metrics.corpus_mrr == pytest.approx(1.0)
