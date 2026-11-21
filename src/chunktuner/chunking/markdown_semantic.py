"""Markdown: split on headings, then ``semchunk`` within large sections."""

import re
from typing import Any

import tiktoken

from chunktuner.chunking.semantic import _require_semchunk
from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document

_HEADING = re.compile(r"(?m)^#{1,3}\s+\S")


class MarkdownSemanticStrategy:
    """Markdown headings define sections; each section is semchunk-split by token budget."""

    name = "markdown_semantic"
    supported_content_types = ["markdown"]
    description = "Heading-aware sections, then semchunk token windows inside each section."

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _section_spans(self, text: str) -> list[tuple[int, int]]:
        matches = list(_HEADING.finditer(text))
        if not matches:
            return [(0, len(text))]
        spans: list[tuple[int, int]] = []
        if matches[0].start() > 0:
            spans.append((0, matches[0].start()))
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            spans.append((start, end))
        return spans

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Chunk per heading section then merge semchunk spans with stable offsets."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        semchunk = _require_semchunk()
        max_tokens = max(16, int(config.params.get("max_tokens", 512)))
        overlap = int(config.params.get("overlap_tokens", 0))
        text = doc.content
        if not text:
            validate_chunk_offsets(doc, [])
            return []
        chunks: list[Chunk] = []
        idx = 0
        for sec_a, sec_b in self._section_spans(text):
            section = text[sec_a:sec_b]
            if not section.strip():
                continue
            out = semchunk.chunk(
                section,
                max_tokens,
                self._count,
                offsets=True,
                overlap=overlap or None,
            )
            pieces, offsets = out
            for piece, (la, lb) in zip(pieces, offsets, strict=True):
                a = sec_a + la
                b = sec_a + lb
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
                        id=f"{doc.id}_mdsem_{idx}",
                        start_offset=a,
                        end_offset=b,
                        tokens=self._count(slice_text),
                        metadata={"section_start": sec_a},
                    )
                )
                idx += 1
        validate_chunk_offsets(doc, chunks)
        return chunks

    def param_schema(self) -> dict[str, Any]:
        return {
            "max_tokens": {"type": "integer", "minimum": 16},
            "overlap_tokens": {"type": "integer", "minimum": 0},
        }

    def default_param_grid(self) -> list[dict]:
        return [
            {"max_tokens": 512, "overlap_tokens": 0},
            {"max_tokens": 1024, "overlap_tokens": 50},
        ]
