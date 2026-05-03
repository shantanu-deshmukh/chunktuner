"""Late chunking placeholder: token pooling needs per-token embedding APIs."""

from __future__ import annotations

from typing import Any

from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.chunking.validation import validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document


class LateChunkingStrategy:
    """Placeholder: delegates to fixed token windows until true late pooling is wired."""

    name = "late_chunking"
    supported_content_types = ["text", "markdown"]
    description = (
        "Fallback: fixed token windows. True late chunking (document-level attention pooling) "
        "requires embedding models that expose per-token vectors (e.g. Jina v2)."
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
