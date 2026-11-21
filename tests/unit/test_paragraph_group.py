"""D11: ``paragraph_group`` strategy."""

import pytest

from chunktuner.chunking.paragraph_group import ParagraphGroupStrategy
from chunktuner.models import ChunkConfig, Document

STRAT = ParagraphGroupStrategy()


def _doc(content: str, content_type: str = "text") -> Document:
    return Document(id="t1", content=content, content_type=content_type)


def _chunk(doc: Document, **params):
    chunks = STRAT.chunk(doc, ChunkConfig(name="paragraph_group", params=params))
    for c in chunks:
        assert doc.content[c.start_offset : c.end_offset] == c.text
    return chunks


PARAS = [f"Paragraph {i} talks about topic {i}." for i in range(7)]
SEVEN = "\n\n".join(PARAS)


@pytest.mark.parametrize("params", STRAT.default_param_grid())
def test_default_grid_offsets_and_cap(params: dict) -> None:
    doc = _doc(SEVEN)
    chunks = _chunk(doc, **params)
    assert chunks
    assert chunks[0].start_offset == 0 and chunks[-1].end_offset == len(SEVEN)
    assert all(c.tokens <= params["max_tokens"] for c in chunks)


def test_groups_without_overlap() -> None:
    chunks = _chunk(_doc(SEVEN), paragraphs_per_chunk=3, overlap_paragraphs=0)
    assert [c.text for c in chunks] == [
        "\n\n".join(PARAS[0:3]),
        "\n\n".join(PARAS[3:6]),
        PARAS[6],
    ]


def test_groups_with_overlap() -> None:
    chunks = _chunk(_doc(SEVEN), paragraphs_per_chunk=3, overlap_paragraphs=1)
    assert [c.text for c in chunks] == [
        "\n\n".join(PARAS[0:3]),
        "\n\n".join(PARAS[2:5]),
        "\n\n".join(PARAS[4:7]),
    ]


def test_cap_shortens_group() -> None:
    long_para = "word " * 40  # ~40 tokens
    content = "\n\n".join([long_para.strip()] * 3)
    chunks = _chunk(_doc(content), paragraphs_per_chunk=3, max_tokens=60)
    assert len(chunks) == 3
    assert all(c.tokens <= 60 for c in chunks)


def test_oversized_paragraph_is_token_split() -> None:
    big = " ".join(f"w{i}" for i in range(300))
    content = f"Intro line.\n\n{big}\n\nOutro line."
    chunks = _chunk(_doc(content), paragraphs_per_chunk=2, max_tokens=50)
    assert all(c.tokens <= 50 for c in chunks)
    covered = "".join(c.text for c in chunks)
    assert big.replace(" ", "") in covered.replace(" ", "").replace("\n", "")


def test_crlf_whitespace_lines_and_multiple_blanks() -> None:
    content = "  First para.\r\n \t\r\n\r\n\nSecond para.\n\n\n\nThird para.  \n"
    chunks = _chunk(_doc(content), paragraphs_per_chunk=1)
    assert [c.text for c in chunks] == ["First para.", "Second para.", "Third para."]


def test_no_blank_lines_is_one_chunk() -> None:
    content = "line one\nline two\nline three"
    chunks = _chunk(_doc(content), paragraphs_per_chunk=2)
    assert [c.text for c in chunks] == [content]


@pytest.mark.parametrize("content", ["", "   \n\n \t\n"])
def test_empty_or_whitespace_doc(content: str) -> None:
    assert _chunk(_doc(content)) == []


@pytest.mark.parametrize(
    "params",
    [
        {"paragraphs_per_chunk": 2, "overlap_paragraphs": 2},
        {"paragraphs_per_chunk": 0},
        {"overlap_paragraphs": -1},
        {"max_tokens": 0},
    ],
)
def test_invalid_params_raise(params: dict) -> None:
    with pytest.raises(ValueError):
        _chunk(_doc(SEVEN), **params)


def test_rejects_unsupported_content_type() -> None:
    with pytest.raises(ValueError):
        _chunk(_doc("def f():\n    pass\n", content_type="code"))
