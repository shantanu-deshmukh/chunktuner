"""Content-type detection from path and optional text sample."""

import re
from pathlib import Path

_EXTENSION_MAP: dict[str, str] = {
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
    ".tsx": "code",
    ".go": "code",
    ".java": "code",
    ".rs": "code",
    ".cpp": "code",
    ".c": "code",
}


def detect_content_type(path: Path, text_sample: str | None = None) -> str:
    """Map extension → type; optional markdown heuristic; fallback ``text``."""
    suf = path.suffix.lower()
    if suf in _EXTENSION_MAP:
        return _EXTENSION_MAP[suf]
    sample = text_sample if text_sample is not None else ""
    if not sample and path.is_file():
        try:
            sample = path.read_text(encoding="utf-8", errors="ignore")[:8000]
        except OSError:
            sample = ""
    if re.search(r"(?m)^#{1,3}\s+\S", sample):
        return "markdown"
    return "text"
