"""Line-respecting sliding windows for code documents."""

from __future__ import annotations

from typing import Any

import tiktoken

from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document


class CodeWindowStrategy:
    """Greedy batches of source lines capped by tiktoken count (code baseline)."""

    name = "code_window"
    supported_content_types = ["code"]
    description = "Greedy line batches capped by max_tokens (tiktoken)."

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Walk lines accumulating tokens until ``max_tokens``, with ``overlap_lines`` rewind."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        max_tokens = max(16, int(config.params.get("max_tokens", 512)))
        overlap_lines = max(0, int(config.params.get("overlap_lines", 2)))
        lines = doc.content.splitlines(keepends=True)
        if not lines:
            return []
        starts: list[int] = [0]
        for ln in lines:
            starts.append(starts[-1] + len(ln))
        chunks: list[Chunk] = []
        i = 0
        idx = 0
        while i < len(lines):
            tok = 0
            j = i
            while j < len(lines):
                t = len(self._enc.encode(lines[j]))
                if tok + t > max_tokens and j > i:
                    break
                tok += t
                j += 1
            if j == i:
                j = i + 1
            a, b = starts[i], starts[j]
            piece = doc.content[a:b]
            chunks.append(
                Chunk(
                    id=f"{doc.id}_cw_{idx}",
                    document_id=doc.id,
                    text=piece,
                    start_offset=a,
                    end_offset=b,
                    tokens=len(self._enc.encode(piece)),
                )
            )
            idx += 1
            if j >= len(lines):
                break
            i = max(i + 1, j - overlap_lines)
        validate_chunk_offsets(doc, chunks)
        return chunks

    def param_schema(self) -> dict[str, Any]:
        return {
            "max_tokens": {"type": "integer", "minimum": 16},
            "overlap_lines": {"type": "integer", "minimum": 0},
        }

    def default_param_grid(self) -> list[dict]:
        return [{"max_tokens": m, "overlap_lines": 2} for m in (256, 512, 1024)]
