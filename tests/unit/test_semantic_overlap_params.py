"""D8: ``semantic`` overlap params (``overlap_ratio`` with deprecated ``similarity_threshold``)."""

import warnings

import pytest

from chunktuner.models import ChunkConfig, Document
from tests._extras import require_extra

require_extra("semantic", "semchunk")

from chunktuner.chunking.semantic import SemanticStrategy  # noqa: E402

TEXT = " ".join(
    f"Sentence number {i} talks about topic {i % 7} in some detail." for i in range(400)
)
DOC = Document(id="t1", content=TEXT, content_type="text")


def _spans(params: dict) -> list[tuple[int, int]]:
    chunks = SemanticStrategy().chunk(DOC, ChunkConfig(name="semantic", params=params))
    for c in chunks:
        assert DOC.content[c.start_offset : c.end_offset] == c.text
    return [(c.start_offset, c.end_offset) for c in chunks]


@pytest.mark.parametrize("ratio", [0.0, 0.1, 0.2])
def test_overlap_ratio_matches_deprecated_alias(ratio: float) -> None:
    new = _spans({"max_tokens": 128, "overlap_ratio": ratio})
    with pytest.warns(DeprecationWarning):
        old = _spans({"max_tokens": 128, "similarity_threshold": ratio})
    assert new == old
    assert new


def test_similarity_threshold_emits_deprecation_and_log(caplog) -> None:
    with caplog.at_level("WARNING", logger="chunktuner.chunking.semantic"):
        with pytest.warns(DeprecationWarning, match="overlap_ratio"):
            _spans({"max_tokens": 128, "similarity_threshold": 0.1})
    assert sum("similarity_threshold" in r.message for r in caplog.records) == 1


def test_overlap_ratio_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _spans({"max_tokens": 128, "overlap_ratio": 0.1})


def test_overlap_tokens_wins_over_ratio_and_alias() -> None:
    explicit = _spans({"max_tokens": 128, "overlap_tokens": 5})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mixed = _spans(
            {
                "max_tokens": 128,
                "overlap_tokens": 5,
                "overlap_ratio": 0.2,
                "similarity_threshold": 0.2,
            }
        )
    assert mixed == explicit
    assert explicit != _spans({"max_tokens": 128, "overlap_ratio": 0.2})


def test_overlap_ratio_wins_over_alias() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        both = _spans({"max_tokens": 128, "overlap_ratio": 0.0, "similarity_threshold": 0.2})
    assert both == _spans({"max_tokens": 128})


def test_default_grid_uses_overlap_ratio() -> None:
    grid = SemanticStrategy().default_param_grid()
    assert grid == [
        {"max_tokens": mt, "overlap_ratio": r} for mt in (512, 1024) for r in (0.0, 0.1, 0.2)
    ]
    assert "overlap_ratio" in SemanticStrategy().param_schema()
