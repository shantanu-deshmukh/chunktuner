"""Recursive character splitting with overlap (LangChain-style)."""

from typing import Any

import tiktoken

from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveCharacterStrategy:
    """Hierarchical character splits (paragraphs, lines, sentences) with overlap."""

    name = "recursive_character"
    supported_content_types = ["text", "markdown", "html"]
    description = "Splits on hierarchical separators up to chunk_size_chars with overlap."

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Produce overlapping spans by splitting on ``separators`` up to ``chunk_size_chars``."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        chunk_size = int(config.params.get("chunk_size_chars", 1600))
        overlap = int(config.params.get("chunk_overlap_chars", 0))
        separators: list[str] = list(config.params.get("separators", _DEFAULT_SEPARATORS))
        chunk_size = max(1, chunk_size)
        if overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap_chars ({overlap}) must be < chunk_size_chars ({chunk_size}). "
                "Otherwise the sliding window cannot advance."
            )
        overlap = max(0, min(overlap, chunk_size - 1)) if chunk_size > 1 else 0

        text = doc.content
        if not text:
            return []

        raw_ranges: list[tuple[int, int]] = []
        start = 0
        n = len(text)
        while start < n:
            end = self._find_break(text, start, min(start + chunk_size, n), separators)
            if end <= start:
                end = min(start + chunk_size, n)
            # Separator within the overlap zone: hard cut so the window
            # always advances by at least overlap+1 chars.
            if end - start <= overlap:
                end = min(start + chunk_size, n)
            raw_ranges.append((start, end))
            if end >= n:
                break
            start = max(start + 1, end - overlap)

        chunks: list[Chunk] = []
        for idx, (a, b) in enumerate(raw_ranges):
            slice_text = text[a:b]
            toks = len(self._enc.encode(slice_text))
            chunks.append(
                Chunk.from_document(
                    doc,
                    id=f"{doc.id}_rc_{idx}",
                    start_offset=a,
                    end_offset=b,
                    tokens=toks,
                )
            )
        validate_chunk_offsets(doc, chunks)
        return chunks

    def _find_break(self, text: str, start: int, end: int, separators: list[str]) -> int:
        """Prefer breaking at last separator occurrence inside ``text[start:end]``."""
        if end >= len(text):
            return len(text)
        window = text[start:end]
        for sep in separators:
            if sep == "":
                return end
            pos = window.rfind(sep)
            if pos != -1:
                return start + pos + len(sep)
        return end

    def param_schema(self) -> dict[str, Any]:
        return {
            "chunk_size_chars": {"type": "integer", "minimum": 1},
            "chunk_overlap_chars": {"type": "integer", "minimum": 0},
            "separators": {"type": "array", "items": {"type": "string"}},
        }

    def default_param_grid(self) -> list[dict]:
        grid: list[dict] = []
        for size in (800, 1600, 3200):
            for ov in (0, 100, 200):
                if ov < size:
                    grid.append({"chunk_size_chars": size, "chunk_overlap_chars": ov})
        return grid
