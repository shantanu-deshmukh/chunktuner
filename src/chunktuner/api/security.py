"""Validate user-supplied corpus paths against ``CHUNK_TUNER_BASE_DIR``."""

import os
from pathlib import Path


def _resolved_base_dir() -> Path:
    raw = os.environ.get("CHUNK_TUNER_BASE_DIR")
    if not raw or not raw.strip():
        return Path.cwd().resolve()
    return Path(raw).expanduser().resolve()


def require_under_base(path_str: str) -> Path:
    """Resolve ``path_str`` to an absolute path and ensure it stays under the workspace base."""
    base = _resolved_base_dir()
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as e:
        raise ValueError(f"path must be under CHUNK_TUNER_BASE_DIR ({base}): {candidate}") from e
    return candidate
