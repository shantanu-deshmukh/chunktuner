"""Groups of consecutive paragraphs with paragraph-level overlap, under a token cap.

Paragraph Group Chunking was the top-performing strategy in Shaukat et al., "A Systematic
Investigation of Document Chunking Strategies and Embedding Sensitivity" (arXiv 2603.06976).
"""

import re
from typing import Any

import tiktoken

from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document

# One or more blank lines (optionally holding spaces/tabs), LF or CRLF.
_PARAGRAPH_BREAK = re.compile(r"\r?\n(?:[ \t]*\r?\n)+")


class ParagraphGroupStrategy:
    """Windows of ``paragraphs_per_chunk`` blank-line paragraphs, capped at ``max_tokens``."""

    name = "paragraph_group"
    supported_content_types = ["text", "markdown", "html"]
    description = (
        "Groups paragraphs_per_chunk consecutive blank-line paragraphs with overlap_paragraphs "
        "of overlap; a group is shortened to fit max_tokens, and a single paragraph over the "
        "cap is split into token windows."
    )

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)
        self._token_splitter = FixedTokenStrategy(encoding_name=encoding_name)

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Group paragraph spans of ``doc.content`` into chunks with exact char offsets."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        group = int(config.params.get("paragraphs_per_chunk", 3))
        overlap = int(config.params.get("overlap_paragraphs", 0))
        max_tokens = int(config.params.get("max_tokens", 512))
        if group < 1:
            raise ValueError(f"paragraphs_per_chunk must be >= 1, got {group}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if not 0 <= overlap < group:
            raise ValueError(
                f"overlap_paragraphs ({overlap}) must be >= 0 and < paragraphs_per_chunk "
                f"({group}). Otherwise the window cannot advance."
            )

        text = doc.content
        units = self._units(doc, max_tokens)
        if not units:
            return []

        chunks: list[Chunk] = []
        i = 0
        n = len(units)
        while i < n:
            j = min(i + group, n)
            # Shorten the window until it fits the cap; one unit always fits (see _units).
            while j > i + 1 and self._count(text[units[i][0] : units[j - 1][1]]) > max_tokens:
                j -= 1
            start, end = units[i][0], units[j - 1][1]
            chunks.append(
                Chunk.from_document(
                    doc,
                    id=f"{doc.id}_pg_{len(chunks)}",
                    start_offset=start,
                    end_offset=end,
                    tokens=self._count(text[start:end]),
                )
            )
            if j >= n:
                break
            i = max(i + 1, j - overlap)
        validate_chunk_offsets(doc, chunks)
        return chunks

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _units(self, doc: Document, max_tokens: int) -> list[tuple[int, int]]:
        """Paragraph spans (outer whitespace trimmed); paragraphs over the cap are token-split."""
        text = doc.content
        units: list[tuple[int, int]] = []
        pos = 0
        for brk in _PARAGRAPH_BREAK.finditer(text):
            units.extend(self._paragraph_units(doc, pos, brk.start(), max_tokens))
            pos = brk.end()
        units.extend(self._paragraph_units(doc, pos, len(text), max_tokens))
        return units

    def _paragraph_units(
        self, doc: Document, start: int, end: int, max_tokens: int
    ) -> list[tuple[int, int]]:
        segment = doc.content[start:end]
        stripped = segment.strip()
        if not stripped:
            return []
        a = start + segment.index(stripped)
        b = a + len(stripped)
        if self._count(stripped) <= max_tokens:
            return [(a, b)]
        sub = Document(id=f"{doc.id}__para", content=stripped, content_type="text")
        pieces = self._token_splitter.chunk(
            sub, ChunkConfig(name="fixed_tokens", params={"max_tokens": max_tokens})
        )
        return [(a + c.start_offset, a + c.end_offset) for c in pieces]

    def param_schema(self) -> dict[str, Any]:
        return {
            "paragraphs_per_chunk": {"type": "integer", "minimum": 1},
            "overlap_paragraphs": {"type": "integer", "minimum": 0},
            "max_tokens": {"type": "integer", "minimum": 1},
        }

    def default_param_grid(self) -> list[dict]:
        # G=2/O=1 and G=3/O=1 repeat a third or more of their text; the negative
        # duplication_ratio score weight penalizes that, so they are kept in the grid.
        return [
            {"paragraphs_per_chunk": g, "overlap_paragraphs": o, "max_tokens": 512}
            for g in (2, 3, 4)
            for o in (0, 1)
        ]
