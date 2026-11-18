"""Semantic-ish chunking via ``semchunk`` (optional extra ``chunktuner[semantic]``)."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import tiktoken

from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document

logger = logging.getLogger(__name__)


def _require_semchunk():
    try:
        import semchunk  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "Install the semantic extra: uv add 'chunktuner[semantic]' or pip install semchunk"
        ) from e
    return __import__("semchunk")


class SemanticStrategy:
    """Token-budget chunks via optional ``semchunk`` (install ``chunktuner[semantic]``)."""

    name = "semantic"
    supported_content_types = ["text", "markdown", "html"]
    description = (
        "Token-budget recursive splitting via semchunk (tiktoken sizing); does not use embeddings."
    )

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Chunk ``doc.content`` with semchunk using ``max_tokens`` / overlap settings."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        semchunk = _require_semchunk()
        max_tokens = max(16, int(config.params.get("max_tokens", 512)))
        overlap = self._resolve_overlap(config.params, max_tokens)
        text = doc.content
        if not text:
            validate_chunk_offsets(doc, [])
            return []
        out = semchunk.chunk(
            text,
            max_tokens,
            self._count,
            offsets=True,
            overlap=overlap,
        )
        pieces, offsets = out
        chunks: list[Chunk] = []
        for idx, (piece, (a, b)) in enumerate(zip(pieces, offsets, strict=True)):
            slice_text = text[a:b]
            if slice_text != piece:
                raise ValueError(
                    f"semchunk offset mismatch at [{a}:{b}]: "
                    f"expected {piece[:40]!r}, got {slice_text[:40]!r}. "
                    "This is a semchunk bug — please report it."
                )
            chunks.append(
                Chunk.from_document(
                    doc,
                    id=f"{doc.id}_sem_{idx}",
                    start_offset=a,
                    end_offset=b,
                    tokens=self._count(slice_text),
                )
            )
        validate_chunk_offsets(doc, chunks)
        return chunks

    def _resolve_overlap(self, params: dict[str, Any], max_tokens: int) -> int | None:
        """Overlap in tokens: ``overlap_tokens`` > ``overlap_ratio`` > ``similarity_threshold``.

        ``similarity_threshold`` is a deprecated alias for ``overlap_ratio`` (it was never a
        similarity threshold; this strategy does not use embeddings).
        """
        ot = params.get("overlap_tokens")
        if ot is not None:
            return int(ot)
        if params.get("overlap_ratio") is not None:
            ratio = float(params["overlap_ratio"] or 0.0)
        elif params.get("similarity_threshold") is not None:
            msg = (
                "semantic: 'similarity_threshold' is deprecated and is interpreted as an "
                "overlap fraction of max_tokens; use 'overlap_ratio' instead."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
            logger.warning(msg)
            ratio = float(params["similarity_threshold"] or 0.0)
        else:
            ratio = 0.0
        return int(max_tokens * ratio) if ratio > 0 else None

    def param_schema(self) -> dict[str, Any]:
        return {
            "max_tokens": {"type": "integer", "minimum": 16},
            "min_tokens": {"type": "integer", "minimum": 0},
            "overlap_tokens": {
                "type": "integer",
                "minimum": 0,
                "description": "Overlap in tokens; takes precedence over overlap_ratio.",
            },
            "overlap_ratio": {
                "type": "number",
                "minimum": 0,
                "description": "Overlap fraction of max_tokens (used if overlap_tokens unset).",
            },
            "similarity_threshold": {
                "type": "number",
                "deprecated": True,
                "description": "Deprecated alias for overlap_ratio (not a similarity threshold).",
            },
        }

    def default_param_grid(self) -> list[dict]:
        grid: list[dict] = []
        for mt in (512, 1024):
            for ratio in (0.0, 0.1, 0.2):
                grid.append({"max_tokens": mt, "overlap_ratio": ratio})
        return grid
