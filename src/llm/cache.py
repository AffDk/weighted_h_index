"""SQLite-backed cache for LLM relevance responses.

Avoids redundant API calls on identical (citing_abstract, cited_abstract, model)
tuples by storing parsed weight + reasoning keyed by a SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from .types import RelevanceResponse


def _cache_key(citing_abstract: str, cited_abstract: str, model: str) -> str:
    payload = f"{model}\n{citing_abstract}\n{cited_abstract}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    """Persistent cache for LLM relevance scores."""

    def __init__(self, db_path: str | Path = "data/llm_cache.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                weight REAL NOT NULL,
                reasoning TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        self._conn.commit()

    def get(
        self, citing_abstract: str, cited_abstract: str, model: str
    ) -> RelevanceResponse | None:
        key = _cache_key(citing_abstract, cited_abstract, model)
        cur = self._conn.execute(
            "SELECT weight, reasoning FROM llm_cache WHERE cache_key = ?",
            (key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return RelevanceResponse(weight=float(row[0]), reasoning=row[1])

    def put(
        self,
        citing_abstract: str,
        cited_abstract: str,
        model: str,
        response: RelevanceResponse,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        key = _cache_key(citing_abstract, cited_abstract, model)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache
            (cache_key, model, weight, reasoning, input_tokens, output_tokens)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                model,
                response.weight,
                response.reasoning,
                input_tokens,
                output_tokens,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def stats(self) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT COUNT(*), SUM(input_tokens), SUM(output_tokens) FROM llm_cache"
        )
        rows = cur.fetchone()
        return {
            "entries": rows[0] or 0,
            "input_tokens": rows[1] or 0,
            "output_tokens": rows[2] or 0,
        }
