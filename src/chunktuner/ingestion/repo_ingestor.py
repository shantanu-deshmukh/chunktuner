"""Walk a repository tree into per-file ``Document`` records."""

import uuid
from pathlib import Path

from chunktuner.ingestion.content_type import detect_content_type
from chunktuner.ingestion.preprocessor import preprocess
from chunktuner.models import ContentType, Document

_BAD_PARTS = frozenset({"node_modules", "__pycache__", ".git", "dist", "build", ".venv", "venv"})

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


class RepoIngestor:
    """Walk a repository tree into code or prose `Document` records (skips vendor dirs)."""

    def _skip(self, path: Path, root: Path) -> bool:
        rel = path.relative_to(root)
        if _BAD_PARTS.intersection(rel.parts):
            return True
        if path.name.endswith(".min.js"):
            return True
        return False

    def ingest_repo(self, root: Path) -> list[Document]:
        """Recursively ingest supported extensions under ``root``."""
        root = root.resolve()
        docs: list[Document] = []
        exts = {".py", ".js", ".ts", ".tsx", ".go", ".java", ".rs", ".cpp", ".c", ".md", ".txt"}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in exts:
                continue
            if self._skip(path, root):
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            detected = detect_content_type(path, raw)
            if detected in ("markdown", "text"):
                ct: ContentType = "markdown" if detected == "markdown" else "text"
                content = preprocess(raw, detected)
            else:
                ct = "code"
                content = raw
            lang = _EXT_LANG.get(path.suffix.lower())
            docs.append(
                Document(
                    id=str(uuid.uuid4()),
                    content=content,
                    content_type=ct,
                    path=str(path),
                    language=lang,
                    metadata={"repo_root": str(root), "rel_path": str(path.relative_to(root))},
                )
            )
        return docs
