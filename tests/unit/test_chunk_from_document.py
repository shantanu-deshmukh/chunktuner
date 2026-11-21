"""Chunk.from_document factory (B4-2)."""

import pytest

from chunktuner.models import Chunk, Document


def test_chunk_from_document_slices_text() -> None:
    doc = Document(id="d1", content="hello world", content_type="text")
    c = Chunk.from_document(doc, id="c1", start_offset=0, end_offset=5)
    assert c.text == "hello"
    assert c.document_id == "d1"


def test_chunk_from_document_rejects_bad_offsets() -> None:
    doc = Document(id="d1", content="short", content_type="text")
    with pytest.raises(ValueError, match="out of bounds"):
        Chunk.from_document(doc, id="c1", start_offset=0, end_offset=99)


def test_chunk_from_document_passes_optional_fields() -> None:
    doc = Document(id="d1", content="abcdefghij", content_type="text")
    c = Chunk.from_document(
        doc,
        id="c1",
        start_offset=2,
        end_offset=5,
        tokens=3,
        metadata={"k": "v"},
    )
    assert c.text == "cde"
    assert c.tokens == 3
    assert c.metadata == {"k": "v"}
