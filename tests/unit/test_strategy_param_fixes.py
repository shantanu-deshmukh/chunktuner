"""D21 code_ast merge_small, D22 semantic min_tokens, D26 structural_semantic on docx/pptx."""

import pytest
import tiktoken

from chunktuner.chunking.structural_semantic import StructuralSemanticStrategy
from chunktuner.models import Chunk, ChunkConfig, Document
from tests._extras import require_extra

ENC = tiktoken.get_encoding("cl100k_base")


def _check_invariant(doc: Document, chunks: list[Chunk]) -> None:
    assert chunks
    for c in chunks:
        assert doc.content[c.start_offset : c.end_offset] == c.text


def _uncovered(doc: Document, chunks: list[Chunk]) -> str:
    covered = [False] * len(doc.content)
    for c in chunks:
        for j in range(c.start_offset, c.end_offset):
            covered[j] = True
    return "".join(ch for ch, cov in zip(doc.content, covered, strict=True) if not cov).strip()


# --- D26: structural_semantic on docx / pptx -------------------------------------------------

MD_BODY = "\n\n".join(
    f"# Section {i}\n\n" + " ".join(f"Paragraph {i} sentence {j}." for j in range(60))
    for i in range(4)
)


@pytest.mark.parametrize("content_type", ["docx", "pptx"])
def test_structural_semantic_office_types(content_type: str) -> None:
    doc = Document(id="o1", content=MD_BODY, content_type=content_type)
    cfg = ChunkConfig(name="structural_semantic", params={"max_tokens": 128})
    chunks = StructuralSemanticStrategy().chunk(doc, cfg)
    _check_invariant(doc, chunks)
    assert all(c.document_id == "o1" for c in chunks)
    md = Document(id="o1", content=MD_BODY, content_type="markdown")
    expected = StructuralSemanticStrategy().chunk(md, cfg)
    assert [(c.start_offset, c.end_offset) for c in chunks] == [
        (c.start_offset, c.end_offset) for c in expected
    ]
    assert doc.content_type == content_type  # caller's doc is not mutated


# --- D21: code_ast merge_small ---------------------------------------------------------------

SMALL_FUNCS = "\n\n".join(f"def f{i}(x):\n    return x + {i}" for i in range(6))
BIG_FUNC = "def big(x):\n" + "".join(
    f"    v{i} = some_function_name(x, {i}) * another_call({i})\n" for i in range(120)
)
PY_SRC = (
    '"""Module docstring."""\n'
    "import os\n"
    "import sys\n\n"
    "CONSTANT = 42\n\n"
    f"{SMALL_FUNCS}\n\n"
    "print(CONSTANT)\n\n"
    f"{BIG_FUNC}\n"
    "@decorator\n"
    "def decorated():\n"
    "    return 1\n"
)


@pytest.fixture
def code_ast():
    require_extra("code", "tree_sitter_python")
    from chunktuner.chunking.code_ast import CodeASTStrategy

    strat = CodeASTStrategy()
    if strat._parser is None:
        pytest.skip("tree-sitter parser unavailable")
    return strat


def _code_chunks(strat, params: dict) -> tuple[Document, list[Chunk]]:
    doc = Document(id="py", content=PY_SRC, content_type="code", language="python")
    return doc, strat.chunk(doc, ChunkConfig(name="code_ast", params=params))


def test_merge_small_covers_module_code(code_ast) -> None:
    doc, chunks = _code_chunks(code_ast, {"max_tokens": 128, "merge_small": True})
    _check_invariant(doc, chunks)
    assert _uncovered(doc, chunks) == ""
    assert "import os" in chunks[0].text and "CONSTANT = 42" in chunks[0].text
    assert any("print(CONSTANT)" in c.text for c in chunks)
    assert any("@decorator" in c.text for c in chunks)
    starts = [c.start_offset for c in chunks]
    assert starts == sorted(starts)


def test_merge_small_merges_small_functions(code_ast) -> None:
    _, chunks = _code_chunks(code_ast, {"max_tokens": 128, "merge_small": True})
    merged = [c for c in chunks if "def f0" in c.text]
    assert len(merged) == 1 and "def f5" in merged[0].text
    big_start = PY_SRC.index("def big")
    big_end = big_start + len(BIG_FUNC.rstrip("\n"))
    for c in chunks:
        if not (big_start <= c.start_offset and c.end_offset <= big_end):  # not a fallback piece
            assert len(ENC.encode(c.text)) <= 128


def test_merge_small_splits_big_function(code_ast) -> None:
    doc, chunks = _code_chunks(code_ast, {"max_tokens": 128, "merge_small": True})
    big_start = PY_SRC.index("def big")
    big_end = big_start + len(BIG_FUNC.rstrip("\n"))
    pieces = [c for c in chunks if big_start <= c.start_offset and c.end_offset <= big_end]
    assert len(pieces) > 1
    assert pieces[0].start_offset == big_start and pieces[-1].end_offset == big_end


def test_merge_small_false_keeps_definitions_only(code_ast) -> None:
    doc, chunks = _code_chunks(code_ast, {"max_tokens": 128, "merge_small": False})
    _check_invariant(doc, chunks)
    assert not any("import os" in c.text for c in chunks)
    assert sum("def f" in c.text and "def big" not in c.text for c in chunks) == 6
    # Absent key keeps the legacy (merge_small=False) behaviour.
    _, legacy = _code_chunks(code_ast, {"max_tokens": 128})
    assert [(c.start_offset, c.end_offset) for c in legacy] == [
        (c.start_offset, c.end_offset) for c in chunks
    ]


# --- D22: semantic min_tokens ----------------------------------------------------------------

SEM_TEXT = "\n\n".join(
    ["Tiny."]
    + [" ".join(f"Sentence {i}.{j} about topic {i}." for j in range(12)) for i in range(6)]
    + ["End."]
)


def _semantic(params: dict) -> tuple[Document, list[Chunk]]:
    require_extra("semantic", "semchunk")
    from chunktuner.chunking.semantic import SemanticStrategy

    doc = Document(id="s1", content=SEM_TEXT, content_type="text")
    return doc, SemanticStrategy().chunk(doc, ChunkConfig(name="semantic", params=params))


def test_min_tokens_zero_is_noop() -> None:
    _, base = _semantic({"max_tokens": 64})
    _, zero = _semantic({"max_tokens": 64, "min_tokens": 0})
    assert [(c.start_offset, c.end_offset) for c in base] == [
        (c.start_offset, c.end_offset) for c in zero
    ]


def test_min_tokens_merges_small_chunks() -> None:
    doc, base = _semantic({"max_tokens": 64})
    assert any(c.tokens < 10 for c in base)  # fixture sanity: small first/last pieces
    doc, merged = _semantic({"max_tokens": 64, "min_tokens": 10})
    _check_invariant(doc, merged)
    assert len(merged) < len(base)
    assert all(c.tokens >= 10 for c in merged)
    assert merged[0].start_offset == base[0].start_offset
    assert merged[-1].end_offset == base[-1].end_offset
    assert "Tiny." in merged[0].text and "Sentence 0.0" in merged[0].text
    assert "End." in merged[-1].text and merged[-1].tokens > 10
    assert [c.id for c in merged] == [f"s1_sem_{i}" for i in range(len(merged))]


def test_min_tokens_with_overlap_keeps_offsets() -> None:
    doc, merged = _semantic({"max_tokens": 64, "overlap_ratio": 0.2, "min_tokens": 40})
    _check_invariant(doc, merged)
    starts = [c.start_offset for c in merged]
    assert starts == sorted(starts)
    assert _uncovered(doc, merged) == ""
