"""``late_chunking``: fixed-token fallback; true late chunking is not implemented.

Late chunking is an embedding-side technique (pooling per-token vectors from a whole-document
forward pass), so it cannot be expressed as a pure text splitter. This strategy is excluded
from default all-strategy tuner runs and must be requested explicitly.
"""

from __future__ import annotations

from typing import Any

from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.chunking.validation import validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document


class LateChunkingStrategy:
    """Fixed-token fallback: delegates to `FixedTokenStrategy` (no late pooling)."""

    name = "late_chunking"
    supported_content_types = ["text", "markdown"]
    description = (
        "Fixed-token fallback (identical to fixed_tokens windows). True late chunking is NOT "
        "implemented: it is an embedding-side technique, not a text-splitting one. "
        "Excluded from default tuner runs; request it explicitly."
    )

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._inner = FixedTokenStrategy(encoding_name=encoding_name)

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Chunk with inner `FixedTokenStrategy` using ``chunk_size_tokens`` / overlap."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        max_tokens = int(
            config.params.get("chunk_size_tokens", config.params.get("max_tokens", 256))
        )
        overlap = int(config.params.get("overlap_tokens", 0))
        return self._inner.chunk(
            doc,
            ChunkConfig(
                name="fixed_tokens",
                params={"max_tokens": max(16, max_tokens), "overlap_tokens": overlap},
            ),
        )

    def param_schema(self) -> dict[str, Any]:
        return {
            "chunk_size_tokens": {"type": "integer", "minimum": 16},
            "overlap_tokens": {"type": "integer", "minimum": 0},
            "model": {"type": "string"},
        }

    def default_param_grid(self) -> list[dict]:
        return [{"chunk_size_tokens": t, "overlap_tokens": 0} for t in (256, 512)]
