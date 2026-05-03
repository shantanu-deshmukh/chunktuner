"""SQLite-backed embedding cache (SHA256 key)."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from pathlib import Path


class EmbeddingCache:
    """Persistent cache: ``SHA256(model + '|' + text)`` → float vector."""

    def __init__(self, db_path: Path, model: str):
        self.model = model
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                k TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL
            )
            """
        )
        self._conn.commit()

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}|{text}".encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT dim, vec FROM embeddings WHERE k = ?",
            (self._key(text),),
        ).fetchone()
        if row is None:
            return None
        dim, blob = row
        return list(struct.unpack(f"{dim}f", blob))

    def set(self, text: str, embedding: list[float]) -> None:
        dim = len(embedding)
        blob = struct.pack(f"{dim}f", *embedding)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (k, dim, vec) VALUES (?, ?, ?)",
            (self._key(text), dim, blob),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM embeddings")
        self._conn.commit()

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(dim*4+32),0) FROM embeddings"
        ).fetchone()
        return {"rows": int(row[0] or 0), "approx_bytes": int(row[1] or 0)}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def default_embedding_db_path(cache_dir: Path) -> Path:
    return cache_dir / "chunktuner_cache.sqlite"
