"""SQLite-backed embedding cache (SHA256 key)."""

import hashlib
import sqlite3
import struct
from pathlib import Path
from typing import Self


class EmbeddingCache:
    """Persistent cache: ``SHA256(model + '|' + text)`` → float vector."""

    def __init__(self, db_path: Path, model: str):
        self.db_path = Path(db_path).resolve()
        self.model = model
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                k TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL,
                model TEXT
            )
            """
        )
        # Migrate pre-0.3 tables (no ``model`` column); legacy rows keep ``model`` NULL.
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(embeddings)")}
        if "model" not in cols:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN model TEXT")
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
        # float64 rows (current) round-trip exactly; float32 rows come from older versions.
        fmt = "d" if len(blob) == dim * 8 else "f"
        return list(struct.unpack(f"{dim}{fmt}", blob))

    def set(self, text: str, embedding: list[float]) -> None:
        dim = len(embedding)
        # Store float64 so cached vectors equal the provider's values exactly (results must not
        # change between a cold and a warm cache).
        blob = struct.pack(f"{dim}d", *embedding)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (k, dim, vec, model) VALUES (?, ?, ?, ?)",
            (self._key(text), dim, blob, self.model),
        )
        self._conn.commit()

    def clear(self, model: str | None = None) -> None:
        """Delete all rows, or only rows written for ``model``.

        Rows from older versions have no recorded model and are only removed by ``clear()``.
        """
        if model is None:
            self._conn.execute("DELETE FROM embeddings")
        else:
            self._conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
        self._conn.commit()

    def stats(self, model: str | None = None) -> dict:
        """Row count and approximate size for all rows, or only rows written for ``model``."""
        sql = "SELECT COUNT(*), COALESCE(SUM(LENGTH(vec)+32),0) FROM embeddings"
        if model is None:
            row = self._conn.execute(sql).fetchone()
        else:
            row = self._conn.execute(sql + " WHERE model = ?", (model,)).fetchone()
        return {"rows": int(row[0] or 0), "approx_bytes": int(row[1] or 0)}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def default_embedding_db_path(cache_dir: Path) -> Path:
    return cache_dir / "chunktuner_cache.sqlite"
