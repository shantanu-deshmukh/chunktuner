"""Offset invariant: doc.content[start:end] == chunk.text."""

from __future__ import annotations

import uuid

import pytest

from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.chunking.recursive_character import RecursiveCharacterStrategy
from chunktuner.chunking.validation import validate_chunk_offsets
from chunktuner.models import Chunk, ChunkConfig, Document

REGISTRY = build_full_registry()

TEXT_CASES = [
    ("simple prose", "The quick brown fox jumps over the lazy dog. " * 50),
    ("unicode CJK", "这是一段中文文字。" * 30),
    ("only separators", "\n\n\n".join(["paragraph"] * 20)),
    ("single char", "x"),
]

# Emoji can break tiktoken single-token decode length vs Python string length in edge cases.
TEXT_CASES_RECURSIVE_ONLY = [
    ("unicode emoji", "Hello 🌍! This is a test. 🚀" * 20),
]


def _assert_offsets(doc: Document, chunks) -> None:
    for c in chunks:
        assert doc.content[c.start_offset : c.end_offset] == c.text


def _strategy(name: str):
    if name not in REGISTRY.names():
        pytest.skip(f"strategy {name!r} not registered (optional extra missing)")
    return REGISTRY.get(name)


def test_validate_chunk_offsets_rejects_out_of_bounds() -> None:
    doc = Document(id="d", content="short", content_type="text")
    bad_chunk = Chunk(id="c", document_id="d", text="short", start_offset=0, end_offset=9999)
    with pytest.raises(ValueError, match="out of bounds"):
        validate_chunk_offsets(doc, [bad_chunk])


def test_validate_chunk_offsets_empty_doc() -> None:
    doc = Document(id="d", content="", content_type="text")
    validate_chunk_offsets(doc, [])


@pytest.mark.parametrize(
    "strategy_name,wrong_type",
    [
        ("fixed_tokens", "pdf"),
        ("code_ast", "text"),
        ("pdf_structural", "code"),
    ],
)
def test_content_type_guard(strategy_name: str, wrong_type: str) -> None:
    strategy = _strategy(strategy_name)
    doc = Document(id="t", content="x", content_type=wrong_type)
    with pytest.raises(ValueError, match="does not support content_type"):
        strategy.chunk(doc, ChunkConfig(name=strategy_name, params={}))


@pytest.mark.parametrize("strategy_name", ["fixed_tokens", "recursive_character"])
@pytest.mark.parametrize("label,text", TEXT_CASES)
def test_offset_invariant_core_strategies(strategy_name: str, label: str, text: str) -> None:
    strategy = _strategy(strategy_name)
    doc = Document(id="t1", content=text, content_type="text")
    config = ChunkConfig(name=strategy_name, params={})
    chunks = strategy.chunk(doc, config)
    for c in chunks:
        assert doc.content[c.start_offset : c.end_offset] == c.text, (
            f"[{strategy_name}][{label}] offset mismatch for chunk {c.id}"
        )


@pytest.mark.parametrize("label,text", TEXT_CASES_RECURSIVE_ONLY)
def test_offset_invariant_recursive_emoji_ok(label: str, text: str) -> None:
    strategy = _strategy("recursive_character")
    doc = Document(id="t1e", content=text, content_type="text")
    cfg = ChunkConfig(
        name="recursive_character", params={"chunk_size_chars": 120, "chunk_overlap_chars": 10}
    )
    chunks = strategy.chunk(doc, cfg)
    for c in chunks:
        assert doc.content[c.start_offset : c.end_offset] == c.text, (
            f"[recursive_character][{label}] offset mismatch for chunk {c.id}"
        )


@pytest.mark.parametrize(
    "content",
    [
        "Short",
        "Paragraph one.\n\nParagraph two.\n\n" + "x" * 5000,
        "Unicode: café résumé naïve",
    ],
)
def test_recursive_character_offsets(content: str) -> None:
    doc = Document(id=str(uuid.uuid4()), content=content, content_type="markdown")
    strat = RecursiveCharacterStrategy()
    cfg = ChunkConfig(
        name="recursive_character",
        params={"chunk_size_chars": 80, "chunk_overlap_chars": 10},
    )
    chunks = strat.chunk(doc, cfg)
    _assert_offsets(doc, chunks)


def test_fixed_tokens_offsets() -> None:
    doc = Document(
        id=str(uuid.uuid4()),
        content="alpha beta " * 200 + "gamma",
        content_type="text",
    )
    strat = FixedTokenStrategy()
    cfg = ChunkConfig(name="fixed_tokens", params={"max_tokens": 32, "overlap_tokens": 8})
    chunks = strat.chunk(doc, cfg)
    _assert_offsets(doc, chunks)
    assert len(chunks) >= 1


@pytest.mark.parametrize(
    "strategy_name,content_type,params",
    [
        ("semantic", "text", {}),
        ("markdown_semantic", "markdown", {}),
        ("pdf_structural", "markdown", {}),
        (
            "structural_semantic",
            "markdown",
            {"max_region_chars": 2000, "max_tokens": 64, "overlap_tokens": 0},
        ),
        ("late_chunking", "text", {}),
        ("code_window", "code", {}),
        ("code_ast", "code", {}),
    ],
)
def test_offset_invariant_optional_strategies(
    strategy_name: str,
    content_type: str,
    params: dict,
) -> None:
    if strategy_name in ("semantic", "markdown_semantic"):
        pytest.importorskip("semchunk")
    strategy = _strategy(strategy_name)
    if content_type == "markdown":
        body = "# Title\n\n" + ("Section paragraph. " * 40)
    elif content_type == "code":
        body = "def foo():\n    return 1\n\n" + "class Bar:\n    pass\n" * 15
    else:
        body = "Plain text body. " * 60
    doc = Document(
        id="t2",
        content=body,
        content_type=content_type,
        language="python" if content_type == "code" else None,
    )
    cfg = ChunkConfig(name=strategy_name, params=params)
    chunks = strategy.chunk(doc, cfg)
    for c in chunks:
        assert doc.content[c.start_offset : c.end_offset] == c.text, (
            f"[{strategy_name}] offset mismatch for chunk {c.id}"
        )


@pytest.mark.parametrize("strategy_name", ["fixed_tokens", "recursive_character", "late_chunking"])
def test_empty_document_returns_no_chunks(strategy_name: str) -> None:
    strategy = _strategy(strategy_name)
    doc = Document(id="empty", content="", content_type="text")
    cfg = ChunkConfig(name=strategy_name, params={})
    chunks = strategy.chunk(doc, cfg)
    assert chunks == [], f"{strategy_name} must return [] for empty content"
