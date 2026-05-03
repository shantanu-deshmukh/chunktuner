"""SQLite-backed chunking result cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from chunktuner.models import Chunk, ChunkConfig, Document


class ChunkCache:
    """Cache keyed by ``SHA256(content + strategy + params_json)``."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                k TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _key(self, doc: Document, strategy_name: str, config: ChunkConfig) -> str:
        params = json.dumps(config.params, sort_keys=True)
        raw = f"{doc.content}\0{strategy_name}\0{params}".encode()
        return hashlib.sha256(raw).hexdigest()

    def get(self, doc: Document, strategy_name: str, config: ChunkConfig) -> list[Chunk] | None:
        row = self._conn.execute(
            "SELECT payload FROM chunks WHERE k = ?",
            (self._key(doc, strategy_name, config),),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return [Chunk.model_validate(x) for x in data]

    def set(
        self, doc: Document, strategy_name: str, config: ChunkConfig, chunks: list[Chunk]
    ) -> None:
        payload = json.dumps([c.model_dump() for c in chunks])
        self._conn.execute(
            "INSERT OR REPLACE INTO chunks (k, payload) VALUES (?, ?)",
            (self._key(doc, strategy_name, config), payload),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)),0) FROM chunks"
        ).fetchone()
        return {"rows": int(row[0] or 0), "approx_payload_bytes": int(row[1] or 0)}

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ChunkCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
