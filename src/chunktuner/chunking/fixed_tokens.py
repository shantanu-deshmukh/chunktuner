"""Fixed-size token windows with overlap (tiktoken)."""

from typing import Any

import tiktoken

from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document


class FixedTokenStrategy:
    """Sliding tiktoken windows with optional overlap (baseline RAG chunker)."""

    name = "fixed_tokens"
    supported_content_types = ["text", "markdown", "html"]
    description = "Sliding windows of max_tokens with overlap_tokens (tiktoken)."

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Split ``doc.content`` into fixed-size token spans with correct char offsets."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        max_tokens = int(config.params.get("max_tokens", 512))
        overlap = int(config.params.get("overlap_tokens", 0))
        max_tokens = max(1, max_tokens)
        overlap = max(0, min(overlap, max_tokens - 1)) if max_tokens > 1 else 0
        step = max(1, max_tokens - overlap)

        text = doc.content
        ids = self._enc.encode(text)
        if not ids:
            return []

        char_boundaries = self._token_char_boundaries(text, ids)
        chunks: list[Chunk] = []
        i = 0
        idx = 0
        while i < len(ids):
            j = min(i + max_tokens, len(ids))
            start_char = char_boundaries[i]
            end_char = char_boundaries[j] if j < len(ids) else len(text)
            chunks.append(
                Chunk.from_document(
                    doc,
                    id=f"{doc.id}_ft_{idx}",
                    start_offset=start_char,
                    end_offset=end_char,
                    tokens=j - i,
                )
            )
            idx += 1
            if j >= len(ids):
                break
            i += step
        validate_chunk_offsets(doc, chunks)
        return chunks

    def _token_char_boundaries(self, text: str, ids: list[int]) -> list[int]:
        """char_boundaries[t] = start char index in ``text`` for token ``t``."""
        boundaries: list[int] = []
        pos = 0
        for tid in ids:
            boundaries.append(pos)
            piece = self._enc.decode([tid])
            pos += len(piece)
        if pos != len(text):
            # Rare tiktoken round-trip mismatch — fall back to whole doc as one window
            raise ValueError("Token decode length mismatch vs document; check encoding")
        return boundaries

    def param_schema(self) -> dict[str, Any]:
        return {
            "max_tokens": {"type": "integer", "minimum": 1},
            "overlap_tokens": {"type": "integer", "minimum": 0},
        }

    def default_param_grid(self) -> list[dict]:
        grid: list[dict] = []
        for mt in (256, 512, 1024):
            for ov in (0, 50, 100):
                if ov < mt:
                    grid.append({"max_tokens": mt, "overlap_tokens": ov})
        return grid
