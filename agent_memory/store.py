"""SQLite persistence with an FTS5 index for lexical recall."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .models import MemoryRecord, MemoryType, from_iso, to_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    entity TEXT NOT NULL,
    content TEXT NOT NULL,
    structured TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    superseded_by TEXT,
    importance REAL NOT NULL,
    confidence REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    source TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_entity ON memories(entity);
CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_mem_valid ON memories(valid_to);
"""


class Store:
    """A tiny SQLite-backed store. Pass ``":memory:"`` for tests."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.fts_enabled = self._init_fts()
        self.conn.commit()

    def _init_fts(self) -> bool:
        # trigram gives real substring matching for CJK; fall back to the
        # default tokenizer on old SQLite builds.
        for tokenizer in ("tokenize='trigram'", ""):
            try:
                self.conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
                    "USING fts5(id UNINDEXED, content, entity"
                    + (", " + tokenizer if tokenizer else "")
                    + ")"
                )
                return True
            except sqlite3.OperationalError:
                continue
        return False

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def insert(self, rec: MemoryRecord) -> None:
        self.conn.execute(
            "INSERT INTO memories (id,type,entity,content,structured,valid_from,"
            "valid_to,superseded_by,importance,confidence,access_count,"
            "last_accessed,source,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec.id, rec.type.value, rec.entity, rec.content,
                json.dumps(rec.structured, ensure_ascii=False),
                to_iso(rec.valid_from), to_iso(rec.valid_to), rec.superseded_by,
                rec.importance, rec.confidence, rec.access_count,
                to_iso(rec.last_accessed), rec.source, to_iso(rec.created_at),
            ),
        )
        if self.fts_enabled:
            self.conn.execute(
                "INSERT INTO memories_fts (id, content, entity) VALUES (?,?,?)",
                (rec.id, rec.content, rec.entity),
            )
        self.conn.commit()

    def update(self, rec: MemoryRecord) -> None:
        self.conn.execute(
            "UPDATE memories SET type=?,entity=?,content=?,structured=?,"
            "valid_from=?,valid_to=?,superseded_by=?,importance=?,confidence=?,"
            "access_count=?,last_accessed=?,source=? WHERE id=?",
            (
                rec.type.value, rec.entity, rec.content,
                json.dumps(rec.structured, ensure_ascii=False),
                to_iso(rec.valid_from), to_iso(rec.valid_to), rec.superseded_by,
                rec.importance, rec.confidence, rec.access_count,
                to_iso(rec.last_accessed), rec.source, rec.id,
            ),
        )
        if self.fts_enabled:
            self.conn.execute("DELETE FROM memories_fts WHERE id=?", (rec.id,))
            self.conn.execute(
                "INSERT INTO memories_fts (id, content, entity) VALUES (?,?,?)",
                (rec.id, rec.content, rec.entity),
            )
        self.conn.commit()

    def delete(self, rec_id: str) -> None:
        self.conn.execute("DELETE FROM memories WHERE id=?", (rec_id,))
        if self.fts_enabled:
            self.conn.execute("DELETE FROM memories_fts WHERE id=?", (rec_id,))
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def get(self, rec_id: str) -> Optional[MemoryRecord]:
        row = self.conn.execute("SELECT * FROM memories WHERE id=?", (rec_id,)).fetchone()
        return self._row(row) if row else None

    def all(self, include_superseded: bool = True) -> List[MemoryRecord]:
        sql = "SELECT * FROM memories"
        if not include_superseded:
            sql += " WHERE valid_to IS NULL"
        return [self._row(r) for r in self.conn.execute(sql).fetchall()]

    def query(
        self,
        entity: Optional[str] = None,
        types: Optional[List[str]] = None,
        include_history: bool = False,
    ) -> List[MemoryRecord]:
        clauses, params = [], []
        if entity:
            clauses.append("entity = ?")
            params.append(entity)
        if types:
            values = [MemoryType(t).value for t in types]
            clauses.append("type IN (" + ",".join("?" for _ in values) + ")")
            params.extend(values)
        if not include_history:
            clauses.append("valid_to IS NULL")
        sql = "SELECT * FROM memories"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [self._row(r) for r in self.conn.execute(sql, params).fetchall()]

    def search_fts(self, query: str, limit: int = 100) -> List[MemoryRecord]:
        if not self.fts_enabled or not query.strip():
            return []
        try:
            rows = self.conn.execute(
                "SELECT m.* FROM memories_fts f JOIN memories m ON m.id = f.id "
                "WHERE memories_fts MATCH ? LIMIT ?",
                (self._match_expr(query), limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [self._row(r) for r in rows]

    @staticmethod
    def _match_expr(text: str) -> str:
        """Turn free text into a trigram OR expression.

        With the trigram tokenizer a quoted phrase requires an exact substring,
        which almost never matches a natural-language query. OR-ing the query's
        trigrams gives real fuzzy matching for CJK.
        """
        chars = [c for c in text if not c.isspace()]
        if len(chars) < 3:
            return '"' + text.replace('"', '""') + '"'
        seen, grams = set(), []
        for i in range(len(chars) - 2):
            gram = "".join(chars[i:i + 3])
            if gram not in seen:
                seen.add(gram)
                grams.append(gram)
        return " OR ".join('"' + g.replace('"', '""') + '"' for g in grams)

    # ------------------------------------------------------------------ #
    def _row(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            type=MemoryType(row["type"]),
            entity=row["entity"],
            content=row["content"],
            structured=json.loads(row["structured"] or "{}"),
            valid_from=from_iso(row["valid_from"]),
            valid_to=from_iso(row["valid_to"]),
            superseded_by=row["superseded_by"],
            importance=row["importance"],
            confidence=row["confidence"],
            access_count=row["access_count"],
            last_accessed=from_iso(row["last_accessed"]),
            source=row["source"],
            created_at=from_iso(row["created_at"]),
        )

    def close(self) -> None:
        self.conn.close()
