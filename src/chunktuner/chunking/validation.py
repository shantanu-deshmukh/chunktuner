"""Chunk offset invariants against document content."""

from __future__ import annotations

import os

from chunktuner.models import Chunk, Document

_SKIP = os.environ.get("CHUNKTUNER_SKIP_OFFSET_VALIDATION") == "1"


def validate_content_type(strategy_name: str, supported: list[str], doc_content_type: str) -> None:
    """Raise ValueError if doc_content_type is not in supported list."""
    if doc_content_type not in supported:
        raise ValueError(
            f"Strategy {strategy_name!r} does not support content_type={doc_content_type!r}. "
            f"Supported: {supported}"
        )


def validate_chunk_offsets(doc: Document, chunks: list[Chunk]) -> None:
    """Raise ValueError if any chunk's text doesn't match its offsets in doc.content."""
    if _SKIP or not chunks:
        return
    n = len(doc.content)
    for c in chunks:
        if not (0 <= c.start_offset < c.end_offset <= n):
            raise ValueError(
                f"Chunk {c.id!r} offsets [{c.start_offset}:{c.end_offset}] are out of bounds "
                f"for doc {doc.id!r} (length {n})"
            )
        got = doc.content[c.start_offset : c.end_offset]
        if got != c.text:
            raise ValueError(
                f"Offset invariant violated for chunk {c.id!r} in doc {doc.id!r}: "
                f"content[{c.start_offset}:{c.end_offset}]={got[:60]!r} "
                f"!= chunk.text={c.text[:60]!r}"
            )
