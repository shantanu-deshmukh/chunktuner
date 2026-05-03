"""Load documents from filesystem paths."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from chunktuner.ingestion.preprocessor import preprocess
from chunktuner.models import ContentType, Document

_EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
}

logger = logging.getLogger(__name__)


class FileIngestor:
    """Load `Document` records from filesystem paths (single file or directory tree)."""

    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".md": "markdown",
        ".mdx": "markdown",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".go": "code",
        ".java": "code",
        ".rs": "code",
        ".cpp": "code",
        ".c": "code",
    }

    def __init__(self, root: Path | None = None):
        self.root = root.resolve() if root else None

    def _ensure_under_root(self, path: Path) -> Path:
        path = path.resolve()
        if self.root is not None:
            try:
                path.relative_to(self.root)
            except ValueError as e:
                raise ValueError(f"Path {path} is not under root {self.root}") from e
        return path

    def ingest_path(
        self,
        path: Path,
        *,
        content_type_override: str | None = None,
    ) -> list[Document]:
        """Ingest a single file or expand a directory via `ingest_dir`."""
        path = self._ensure_under_root(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            return self.ingest_dir(path)
        return self._ingest_file_multi(path, content_type_override=content_type_override)

    def ingest_dir(self, path: Path, glob: str = "**/*") -> list[Document]:
        """Walk ``path`` with ``glob`` and ingest every file with a supported extension."""
        path = self._ensure_under_root(path)
        docs: list[Document] = []
        for p in sorted(path.glob(glob)):
            if not p.is_file():
                continue
            if p.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            try:
                docs.extend(self._ingest_file_multi(p, content_type_override=None))
            except (ImportError, NotImplementedError, OSError):
                continue
        return docs

    def _ingest_file_multi(
        self,
        path: Path,
        *,
        content_type_override: str | None = None,
    ) -> list[Document]:
        ct = content_type_override or self.SUPPORTED_EXTENSIONS[path.suffix.lower()]
        if ct in ("pdf", "docx", "pptx"):
            return self._ingest_docling(path, ct)
        return [self._ingest_plain(path, ct)]

    def _read_text(self, path: Path) -> tuple[str, str]:
        """Return ``(text, encoding_used)``."""
        try:
            return path.read_text(encoding="utf-8"), "utf-8"
        except UnicodeDecodeError:
            logger.warning(
                "File %s is not valid UTF-8; retrying with latin-1. "
                "Character offsets may differ from byte offsets.",
                path,
            )
            return path.read_text(encoding="latin-1"), "latin-1"

    def _ingest_plain(self, path: Path, detected: str) -> Document:
        raw, encoding = self._read_text(path)
        content = preprocess(raw, "html" if detected == "html" else detected)
        content_type: ContentType
        if detected == "text":
            content_type = "text"
        elif detected == "markdown":
            content_type = "markdown"
        elif detected == "html":
            content_type = "html"
        elif detected == "code":
            content_type = "code"
        else:
            content_type = "markdown"
        lang = _EXT_LANG.get(path.suffix.lower())
        meta: dict = {"filename": path.name}
        if encoding != "utf-8":
            meta["source_encoding"] = encoding
        return Document(
            id=str(uuid.uuid4()),
            content=content,
            content_type=content_type,
            path=str(path),
            language=lang,
            metadata=meta,
        )

    def _ingest_docling(self, path: Path, detected: str) -> list[Document]:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Install docling extra: uv sync --extra docling  (or pip install docling)"
            ) from e
        conv = DocumentConverter()
        res = conv.convert(str(path))
        md = res.document.export_to_markdown()
        content = preprocess(md, "markdown")
        doc_ct: ContentType
        if detected == "pdf":
            doc_ct = "pdf"
        elif detected == "docx":
            doc_ct = "docx"
        else:
            doc_ct = "pptx"
        return [
            Document(
                id=str(uuid.uuid4()),
                content=content,
                content_type=doc_ct,
                path=str(path),
                metadata={"filename": path.name, "parser": "docling"},
            )
        ]
