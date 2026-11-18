"""Offset invariant for every registered (non-LLM) strategy x default grid x golden fixture."""

import pytest

from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.models import ChunkConfig
from tests.golden.snapshot import EXCLUDED_STRATEGIES, load_docs

_REG = build_full_registry()
_DOCS = load_docs()
_CASES = [
    (name, i, doc.id)
    for name in sorted(_REG.names())
    if name not in EXCLUDED_STRATEGIES
    for i, _ in enumerate(_REG.get(name).default_param_grid())
    for doc in _DOCS
    if doc.content_type in _REG.get(name).supported_content_types
]


@pytest.mark.parametrize(("name", "grid_idx", "doc_id"), _CASES)
def test_offset_invariant(name: str, grid_idx: int, doc_id: str) -> None:
    strat = _REG.get(name)
    params = strat.default_param_grid()[grid_idx]
    doc = next(d for d in _DOCS if d.id == doc_id)
    chunks = strat.chunk(doc, ChunkConfig(name=name, params=params))
    assert chunks, f"{name} produced no chunks for {doc_id}"
    for c in chunks:
        assert doc.content[c.start_offset : c.end_offset] == c.text
        assert c.document_id == doc.id
