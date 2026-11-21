"""Structural regions then token-sized windows (pdf/docx/pptx markdown)."""

from typing import Any

import tiktoken

from chunktuner.chunking.pdf_structural import PdfStructuralStrategy
from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document


class StructuralSemanticStrategy:
    """Coarse `PdfStructuralStrategy` regions refined with fixed-token sub-windows."""

    name = "structural_semantic"
    supported_content_types = ["pdf", "docx", "pptx", "markdown"]
    description = "PdfStructural regions, subdivided by fixed token windows when regions are large."

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)
        self._struct = PdfStructuralStrategy(encoding_name=encoding_name)

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Map structural regions to absolute-offset sub-chunks using `FixedTokenStrategy`."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        max_tokens = max(32, int(config.params.get("max_tokens", 512)))
        overlap = int(config.params.get("overlap_tokens", 0))
        coarse = dict(config.params)
        coarse.setdefault("max_region_chars", 8000)
        # PdfStructuralStrategy only accepts pdf/markdown/html; docx/pptx are ingested as
        # markdown-like text, so hand it a markdown-typed copy (same content -> same offsets).
        struct_doc = doc
        if doc.content_type not in self._struct.supported_content_types:
            struct_doc = doc.model_copy(update={"content_type": "markdown"})
        regions = self._struct.chunk(struct_doc, ChunkConfig(name="pdf_structural", params=coarse))
        out: list[Chunk] = []
        idx = 0
        for r in regions:
            subdoc = Document(
                id=doc.id,
                content=r.text,
                content_type="markdown",
                path=doc.path,
                page_number=doc.page_number,
                metadata=dict(doc.metadata),
            )
            from chunktuner.chunking.fixed_tokens import FixedTokenStrategy

            ft = FixedTokenStrategy(encoding_name=self._encoding_name)
            subchunks = ft.chunk(
                subdoc,
                ChunkConfig(
                    name="fixed_tokens",
                    params={"max_tokens": max_tokens, "overlap_tokens": overlap},
                ),
            )
            for sc in subchunks:
                abs_start = r.start_offset + sc.start_offset
                abs_end = r.start_offset + sc.end_offset
                out.append(
                    Chunk.from_document(
                        doc,
                        id=f"{doc.id}_ss_{idx}",
                        start_offset=abs_start,
                        end_offset=abs_end,
                        tokens=sc.tokens,
                        metadata={**r.metadata, **sc.metadata},
                    )
                )
                idx += 1
        validate_chunk_offsets(doc, out)
        return out

    def param_schema(self) -> dict[str, Any]:
        return {
            "max_tokens": {"type": "integer"},
            "overlap_tokens": {"type": "integer"},
            "max_region_chars": {"type": "integer"},
        }

    def default_param_grid(self) -> list[dict]:
        return [
            {"max_tokens": 512, "overlap_tokens": 0, "max_region_chars": 8000},
            {"max_tokens": 1024, "overlap_tokens": 50, "max_region_chars": 12000},
        ]
