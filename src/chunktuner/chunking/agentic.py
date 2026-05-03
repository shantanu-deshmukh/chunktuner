"""LLM-proposed chunk boundaries (expensive; opt-in via strategy selection)."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import tiktoken

from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document

logger = logging.getLogger(__name__)

MAX_CHARS = 50_000


def _fuzzy_token_match(
    actual: str, intended: str, enc: tiktoken.Encoding, *, threshold: float
) -> bool:
    """True when token-id Jaccard similarity between ``actual`` and ``intended`` is >= threshold."""
    if not intended.strip():
        return True
    ta = set(enc.encode(actual))
    tb = set(enc.encode(intended))
    if not ta or not tb:
        return actual.strip() == intended.strip()
    inter = len(ta & tb)
    union = len(ta | tb)
    return (inter / union) >= threshold


class AgenticStrategy:
    """LLM proposes UTF-8 character spans; validates offsets against ``doc.content``."""

    name = "agentic"
    supported_content_types = ["text", "markdown"]
    description = "Uses an LLM to propose chunk spans; requires API access and ``litellm``."

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Call LiteLLM JSON mode to obtain ``start_offset`` / ``end_offset`` chunk list."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        import litellm

        model = str(config.params.get("model", "gpt-4o-mini"))
        max_props = int(config.params.get("max_propositions", 40))
        content = doc.content
        truncated = len(content) > MAX_CHARS
        if truncated:
            logger.warning(
                "AgenticStrategy: doc %r truncated from %d to %d chars. "
                "Content beyond offset %d will have no chunks.",
                doc.id,
                len(doc.content),
                MAX_CHARS,
                MAX_CHARS,
            )
            content = content[:MAX_CHARS]

        prompt = (
            "Split the following document into coherent RAG chunks. "
            "Return JSON object with key chunks: array of "
            '{"start_offset": int, "end_offset": int} using UTF-16? NO — use character offsets '
            "into the exact input string (Python slicing). Max chunks: "
            f"{max_props}.\n\nDOCUMENT:\n{content}"
        )
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        raw_items = data.get("chunks", data) if isinstance(data, dict) else data
        if isinstance(raw_items, dict):
            raw_items = raw_items.get("chunks", [])
        chunks: list[Chunk] = []
        for item in raw_items[:max_props]:
            if not isinstance(item, dict):
                continue
            raw_a = int(item.get("start_offset", item.get("start", 0)))
            raw_b = int(item.get("end_offset", item.get("end", 0)))
            a = max(0, min(raw_a, len(content)))
            b = max(a, min(raw_b, len(content)))
            if a != raw_a or b != raw_b:
                logger.warning(
                    "AgenticStrategy: clamped LLM offsets [%d:%d] → [%d:%d] for doc %r",
                    raw_a,
                    raw_b,
                    a,
                    b,
                    doc.id,
                )
            piece = content[a:b]
            if not piece.strip():
                continue
            intended = item.get("text") or item.get("chunk_text") or item.get("content") or ""
            if isinstance(intended, str) and intended.strip():
                if not _fuzzy_token_match(piece, intended, self._enc, threshold=0.5):
                    logger.warning("AgenticStrategy: rejecting chunk with poor offset match")
                    continue
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=doc.id,
                    text=piece,
                    start_offset=a,
                    end_offset=b,
                    tokens=len(self._enc.encode(piece)),
                )
            )
        if not chunks:
            from chunktuner.chunking.recursive_character import RecursiveCharacterStrategy

            out = RecursiveCharacterStrategy(encoding_name=self._enc.name).chunk(
                doc,
                ChunkConfig(
                    name="recursive_character",
                    params={"chunk_size_chars": 1200, "chunk_overlap_chars": 100},
                ),
            )
            if truncated:
                for c in out:
                    c.metadata["agentic_truncated"] = True
                    c.metadata["agentic_truncated_at"] = MAX_CHARS
            return out
        if truncated:
            for c in chunks:
                c.metadata["agentic_truncated"] = True
                c.metadata["agentic_truncated_at"] = MAX_CHARS
        validate_chunk_offsets(doc, chunks)
        return chunks

    def param_schema(self) -> dict[str, Any]:
        return {
            "model": {"type": "string"},
            "max_propositions": {"type": "integer", "minimum": 1, "maximum": 200},
        }

    def default_param_grid(self) -> list[dict]:
        return [{"model": "gpt-4o-mini", "max_propositions": 30}]
