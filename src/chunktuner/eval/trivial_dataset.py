"""Minimal eval datasets without LLM calls (Phase 1)."""

from chunktuner.models import Document, EvalDataset, EvalQuery


def trivial_dataset_for_docs(
    docs: list[Document],
    *,
    name: str = "trivial",
    span_chars: int = 200,
) -> EvalDataset:
    """One query per doc; gold span covers the first ``span_chars`` characters."""
    queries: list[EvalQuery] = []
    for d in docs:
        end = min(span_chars, len(d.content)) if d.content else 1
        if end == 0:
            end = 1
        queries.append(
            EvalQuery(
                id=f"q_{d.id}",
                question="What information appears at the start of this document?",
                document_id=d.id,
                answer_spans=[(0, end)],
            )
        )
    return EvalDataset(name=name, queries=queries, source="trivial")


def is_trivial(ds: EvalDataset) -> bool:
    """True if ``ds`` is a placeholder dataset (``source == "trivial"``); callers should warn."""
    return ds.source == "trivial"
