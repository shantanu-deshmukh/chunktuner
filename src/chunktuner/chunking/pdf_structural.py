"""Layout-inspired splits for PDF-derived markdown (``## Page N`` markers optional)."""

import re
from typing import Any

import tiktoken

from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document

_PAGE = re.compile(r"(?m)^##\s*Page\s+(\d+)\s*$")
_SECTION = re.compile(r"(?m)^#{1,2}\s+\S.*$")


class PdfStructuralStrategy:
    """Page or section headings define regions; long regions are split by character cap."""

    name = "pdf_structural"
    supported_content_types = ["pdf", "markdown", "html"]
    description = (
        "Splits on page headings or major markdown headings, then sub-splits long regions."
    )

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)

    def _region_spans(self, text: str) -> list[tuple[int, int]]:
        markers = list(_PAGE.finditer(text)) or list(_SECTION.finditer(text))
        if not markers:
            return [(0, len(text))]
        spans: list[tuple[int, int]] = []
        if markers[0].start() > 0:
            spans.append((0, markers[0].start()))
        for i, m in enumerate(markers):
            a = m.start()
            b = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            spans.append((a, b))
        return spans

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Emit one chunk per sub-region up to ``max_region_chars`` within each structural span."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        text = doc.content
        if not text:
            return []
        max_chars = int(config.params.get("max_region_chars", 4000))
        chunks: list[Chunk] = []
        idx = 0
        for a, b in self._region_spans(text):
            pos = a
            while pos < b:
                end = min(pos + max_chars, b)
                piece = text[pos:end]
                meta: dict = {}
                if doc.page_number is not None:
                    meta["page_number"] = doc.page_number
                chunks.append(
                    Chunk.from_document(
                        doc,
                        id=f"{doc.id}_pdf_{idx}",
                        start_offset=pos,
                        end_offset=end,
                        tokens=len(self._enc.encode(piece)),
                        metadata=meta,
                    )
                )
                idx += 1
                pos = end
        validate_chunk_offsets(doc, chunks)
        return chunks

    def param_schema(self) -> dict[str, Any]:
        return {"max_region_chars": {"type": "integer", "minimum": 200}}

    def default_param_grid(self) -> list[dict]:
        return [{"max_region_chars": x} for x in (2000, 4000, 8000)]
