"""AST-aware chunks for Python (``tree-sitter`` optional extra)."""

from __future__ import annotations

from typing import Any

import tiktoken

from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document


class CodeASTStrategy:
    """Python tree-sitter top-level definitions, with token-capped fallback splits."""

    name = "code_ast"
    supported_content_types = ["code"]
    description = "Top-level functions/classes as chunks (Python via tree-sitter)."

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)
        self._parser = None
        try:
            import tree_sitter_python as tsp
            from tree_sitter import Language, Parser

            self._parser = Parser(Language(tsp.language()))
        except ImportError:
            pass

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        """Emit function/class chunks for Python AST; non-Python falls back to line windows."""
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        max_tokens = int(config.params.get("max_tokens", 512))
        if self._parser is None or (doc.language or "").lower() not in ("", "python", "py"):
            out = self._fallback().chunk(doc, config)
            validate_chunk_offsets(doc, out)
            return out
        b = doc.content.encode("utf8")
        tree = self._parser.parse(b)
        root = tree.root_node
        targets = ("function_definition", "class_definition")
        chunks: list[Chunk] = []
        idx = 0
        for child in root.children:
            if child.type not in targets:
                continue
            start_b, end_b = child.start_byte, child.end_byte
            start_c = len(b[:start_b].decode("utf8"))
            end_c = len(b[:end_b].decode("utf8"))
            text = doc.content[start_c:end_c]
            if len(self._enc.encode(text)) > max_tokens:
                subdoc = Document(
                    id=doc.id,
                    content=text,
                    content_type="code",
                    path=doc.path,
                    language=doc.language,
                    metadata=doc.metadata,
                )
                for c in self._fallback().chunk(
                    subdoc,
                    ChunkConfig(
                        name="code_window", params={"max_tokens": max_tokens, "overlap_lines": 2}
                    ),
                ):
                    chunks.append(
                        Chunk(
                            id=f"{doc.id}_ast_{idx}",
                            document_id=doc.id,
                            text=c.text,
                            start_offset=start_c + c.start_offset,
                            end_offset=start_c + c.end_offset,
                            tokens=c.tokens,
                        )
                    )
                    idx += 1
                continue
            chunks.append(
                Chunk(
                    id=f"{doc.id}_ast_{idx}",
                    document_id=doc.id,
                    text=text,
                    start_offset=start_c,
                    end_offset=end_c,
                    tokens=len(self._enc.encode(text)),
                )
            )
            idx += 1
        if not chunks:
            out = self._fallback().chunk(doc, config)
            validate_chunk_offsets(doc, out)
            return out
        validate_chunk_offsets(doc, chunks)
        return chunks

    def _fallback(self):
        from chunktuner.chunking.code_window import CodeWindowStrategy

        return CodeWindowStrategy(encoding_name=self._encoding_name)

    def param_schema(self) -> dict[str, Any]:
        return {
            "max_tokens": {"type": "integer", "minimum": 16},
            "merge_small": {"type": "boolean"},
        }

    def default_param_grid(self) -> list[dict]:
        return [{"max_tokens": m, "merge_small": True} for m in (512, 1024)]
