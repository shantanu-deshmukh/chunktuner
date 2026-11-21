"""AST-aware chunks for Python (``tree-sitter`` optional extra)."""

from typing import Any

import tiktoken

from chunktuner.chunking.validation import validate_chunk_offsets, validate_content_type
from chunktuner.models import Chunk, ChunkConfig, Document


class CodeASTStrategy:
    """Python tree-sitter top-level definitions, with token-capped fallback splits."""

    name = "code_ast"
    supported_content_types = ["code"]
    description = (
        "Top-level functions/classes as chunks (Python via tree-sitter); merge_small "
        "covers module-level code and merges small units up to max_tokens."
    )

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
        """Emit function/class chunks for Python AST; non-Python falls back to line windows.

        With ``merge_small=True`` the whole file is covered: top-level definitions and the runs
        of other top-level code between them (imports, constants, statements) are greedily
        merged while the merged span stays within ``max_tokens``.
        """
        validate_content_type(self.name, self.supported_content_types, doc.content_type)
        max_tokens = int(config.params.get("max_tokens", 512))
        merge_small = bool(config.params.get("merge_small", False))
        lang = (doc.language or "").lower()
        use_ast = self._parser is not None and lang in ("", "python", "py")
        if not use_ast:
            out = self._fallback().chunk(doc, config)
            validate_chunk_offsets(doc, out)
            return out
        b = doc.content.encode("utf8")
        tree = self._parser.parse(b)
        root = tree.root_node

        def to_char(byte_off: int) -> int:
            return len(b[:byte_off].decode("utf8"))

        if merge_small:
            chunks = self._merged_chunks(doc, root, to_char, max_tokens)
        else:
            chunks = self._definition_chunks(doc, root, to_char, max_tokens)
        if not chunks:
            out = self._fallback().chunk(doc, config)
            validate_chunk_offsets(doc, out)
            return out
        validate_chunk_offsets(doc, chunks)
        return chunks

    def _definition_chunks(
        self, doc: Document, root: Any, to_char: Any, max_tokens: int
    ) -> list[Chunk]:
        """One chunk per top-level function/class (legacy behaviour, ``merge_small=False``)."""
        chunks: list[Chunk] = []
        for child in root.children:
            if child.type not in ("function_definition", "class_definition"):
                continue
            self._emit_span(
                doc, to_char(child.start_byte), to_char(child.end_byte), max_tokens, chunks
            )
        return chunks

    def _merged_chunks(
        self, doc: Document, root: Any, to_char: Any, max_tokens: int
    ) -> list[Chunk]:
        """Cover every top-level node: definitions and in-between code runs, greedily merged."""
        targets = ("function_definition", "class_definition", "decorated_definition")
        units: list[tuple[int, int]] = []
        run: list[int] | None = None  # [start_c, end_c] of the current non-definition run
        for child in root.children:
            start_c, end_c = to_char(child.start_byte), to_char(child.end_byte)
            if end_c <= start_c:
                continue
            if child.type in targets:
                if run is not None:
                    units.append((run[0], run[1]))
                    run = None
                units.append((start_c, end_c))
            elif run is None:
                run = [start_c, end_c]
            else:
                run[1] = end_c
        if run is not None:
            units.append((run[0], run[1]))

        chunks: list[Chunk] = []
        cur: tuple[int, int] | None = None
        for start_c, end_c in units:
            if cur is not None and self._count(doc.content[cur[0] : end_c]) <= max_tokens:
                cur = (cur[0], end_c)
                continue
            if cur is not None:
                self._emit_span(doc, cur[0], cur[1], max_tokens, chunks)
                cur = None
            if self._count(doc.content[start_c:end_c]) > max_tokens:
                self._emit_span(doc, start_c, end_c, max_tokens, chunks)
            else:
                cur = (start_c, end_c)
        if cur is not None:
            self._emit_span(doc, cur[0], cur[1], max_tokens, chunks)
        return chunks

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _emit_span(
        self, doc: Document, start_c: int, end_c: int, max_tokens: int, chunks: list[Chunk]
    ) -> None:
        """Append ``doc.content[start_c:end_c]`` as one chunk, or code_window pieces if too big."""
        text = doc.content[start_c:end_c]
        n_tokens = self._count(text)
        if n_tokens > max_tokens:
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
                    Chunk.from_document(
                        doc,
                        id=f"{doc.id}_ast_{len(chunks)}",
                        start_offset=start_c + c.start_offset,
                        end_offset=start_c + c.end_offset,
                        tokens=c.tokens,
                    )
                )
            return
        chunks.append(
            Chunk.from_document(
                doc,
                id=f"{doc.id}_ast_{len(chunks)}",
                start_offset=start_c,
                end_offset=end_c,
                tokens=n_tokens,
            )
        )

    def _fallback(self):
        from chunktuner.chunking.code_window import CodeWindowStrategy

        return CodeWindowStrategy(encoding_name=self._encoding_name)

    def param_schema(self) -> dict[str, Any]:
        return {
            "max_tokens": {"type": "integer", "minimum": 16},
            "merge_small": {
                "type": "boolean",
                "description": (
                    "Cover the whole file (incl. imports/module code) and merge consecutive "
                    "top-level units up to max_tokens. Default false: definitions only."
                ),
            },
        }

    def default_param_grid(self) -> list[dict]:
        return [{"max_tokens": m, "merge_small": True} for m in (512, 1024)]
