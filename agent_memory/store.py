"""SQLite persistence with an FTS5 index for lexical recall.

Two details carry most of the retrieval quality and are easy to get wrong:

* ``entity`` is stored **UNINDEXED** in the FTS table. Indexing it makes every
  record of the queried entity match every query, which silently turns the
  lexical path into a duplicate of the structured path.
* the FTS query is ordered by ``bm25()``. Without an ``ORDER BY`` SQLite returns
  rows in rowid order, i.e. the "ranking" is insertion order.

``action_type`` is a real column with an index: it is the axis the intervention
API queries on ("what happened last time we switched a supplier"), and it must
work across entities.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .models import MemoryRecord, MemoryType, from_iso, to_iso, utcnow

#: Bumped when the table layout changes; old databases are migrated on open.
SCHEMA_VERSION = 3
#: Bumped when the FTS layout changes; older databases are re-indexed once.
FTS_VERSION = 2

_COLUMNS = (
    "id,type,entity,content,structured,action_type,valid_from,valid_to,"
    "superseded_by,importance,confidence,access_count,last_accessed,source,created_at"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    entity TEXT NOT NULL,
    content TEXT NOT NULL,
    structured TEXT NOT NULL,
    action_type TEXT NOT NULL DEFAULT '',
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
# NB: the index on `action_type` is created in ``_ensure_columns``, not here --
# on a database from before 0.3 the column does not exist yet and a CREATE INDEX
# in this script would abort the migration before it could add it.

#: Lexical matching needs at least three characters: the trigram tokenizer has
#: no tokens for shorter input, and CJK terms like "断货" / "交期" are two.
_MIN_TRIGRAM_CHARS = 3

#: A ``LIKE`` fallback hit has no bm25 score; it is treated as a fixed amount of
#: evidence so short queries still outrank records that do not contain the term.
_LIKE_EVIDENCE = 0.5


class Store:
    """A tiny SQLite-backed store. Pass ``":memory:"`` for tests."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.fts_enabled = False
        self.fts_tokenizer: Optional[str] = None
        version = self._get_version()
        self._ensure_columns()
        self._init_fts(version)
        self._set_version(SCHEMA_VERSION)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # schema / migration
    # ------------------------------------------------------------------ #
    def _get_version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    def _set_version(self, version: int) -> None:
        self.conn.execute("PRAGMA user_version = %d" % version)

    def _ensure_columns(self) -> None:
        """Add ``action_type`` to databases created before 0.3."""
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(memories)")}
        if "action_type" not in columns:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN action_type TEXT NOT NULL DEFAULT ''"
            )
            self._backfill_action_type()
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_action ON memories(action_type)"
        )

    def _backfill_action_type(self) -> None:
        """Pull the value out of ``structured`` for rows written by 0.1/0.2."""
        rows = self.conn.execute("SELECT id, structured FROM memories").fetchall()
        updates = []
        for row in rows:
            try:
                structured = json.loads(row["structured"] or "{}")
            except (TypeError, ValueError):  # pragma: no cover - corrupt row
                continue
            action = (structured.get("action_type") or "").strip()
            if action:
                updates.append((action, row["id"]))
        if updates:
            self.conn.executemany("UPDATE memories SET action_type = ? WHERE id = ?", updates)

    def _init_fts(self, version: int) -> None:
        """Create (or rebuild) the FTS index, keeping old databases working."""
        existed = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone() is not None

        rebuild = False
        if existed and version < FTS_VERSION:
            # Older layout indexed `entity`, which poisons lexical recall.
            self.conn.execute("DROP TABLE memories_fts")
            existed, rebuild = False, True

        for tokenizer in ("trigram", None):
            try:
                self.conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
                    "USING fts5(id UNINDEXED, entity UNINDEXED, content"
                    + (", tokenize='%s'" % tokenizer if tokenizer else "")
                    + ")"
                )
                self.fts_enabled = True
                self.fts_tokenizer = tokenizer or "unicode61"
                break
            except sqlite3.OperationalError:  # pragma: no cover - old sqlite
                continue

        if self.fts_enabled and (rebuild or not existed):
            self._rebuild_fts()

    def _rebuild_fts(self) -> None:
        self.conn.execute(
            "INSERT INTO memories_fts (id, entity, content) "
            "SELECT id, entity, content FROM memories"
        )

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    @staticmethod
    def _values(rec: MemoryRecord) -> tuple:
        return (
            rec.id, rec.type.value, rec.entity, rec.content,
            json.dumps(rec.structured, ensure_ascii=False), rec.action_type,
            to_iso(rec.valid_from), to_iso(rec.valid_to), rec.superseded_by,
            rec.importance, rec.confidence, rec.access_count,
            to_iso(rec.last_accessed), rec.source, to_iso(rec.created_at),
        )

    _INSERT_SQL = (
        "INSERT INTO memories (" + _COLUMNS + ") "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )

    def insert(self, rec: MemoryRecord) -> None:
        self.insert_many([rec])

    def insert_many(self, recs: Sequence[MemoryRecord]) -> None:
        """Bulk insert: one commit and one FTS write per chunk, not per record."""
        if not recs:
            return
        for start in range(0, len(recs), 500):
            chunk = recs[start:start + 500]
            self.conn.executemany(self._INSERT_SQL, [self._values(r) for r in chunk])
            if self.fts_enabled:
                self.conn.executemany(
                    "INSERT INTO memories_fts (id, entity, content) VALUES (?,?,?)",
                    [(r.id, r.entity, r.content) for r in chunk],
                )
            self.conn.commit()

    def update(self, rec: MemoryRecord) -> None:
        self.conn.execute(
            "UPDATE memories SET type=?,entity=?,content=?,structured=?,action_type=?,"
            "valid_from=?,valid_to=?,superseded_by=?,importance=?,confidence=?,"
            "access_count=?,last_accessed=?,source=? WHERE id=?",
            (
                rec.type.value, rec.entity, rec.content,
                json.dumps(rec.structured, ensure_ascii=False), rec.action_type,
                to_iso(rec.valid_from), to_iso(rec.valid_to), rec.superseded_by,
                rec.importance, rec.confidence, rec.access_count,
                to_iso(rec.last_accessed), rec.source, rec.id,
            ),
        )
        if self.fts_enabled:
            self.conn.execute("DELETE FROM memories_fts WHERE id=?", (rec.id,))
            self.conn.execute(
                "INSERT INTO memories_fts (id, entity, content) VALUES (?,?,?)",
                (rec.id, rec.entity, rec.content),
            )
        self.conn.commit()

    def delete(self, rec_id: str) -> None:
        self.delete_many([rec_id])

    def delete_many(self, rec_ids: Sequence[str]) -> None:
        if not rec_ids:
            return
        marks = ",".join("?" for _ in rec_ids)
        self.conn.execute("DELETE FROM memories WHERE id IN (%s)" % marks, list(rec_ids))
        if self.fts_enabled:
            self.conn.execute("DELETE FROM memories_fts WHERE id IN (%s)" % marks, list(rec_ids))
        self.conn.commit()

    def touch_many(self, rec_ids: Sequence[str]) -> None:
        """Record one access for each id in a single transaction (recall is hot)."""
        if not rec_ids:
            return
        now = to_iso(utcnow())
        self.conn.executemany(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            [(now, rid) for rid in rec_ids],
        )
        self.conn.commit()

    def clear_superseded_by(self, rec_ids: Sequence[str]) -> None:
        """Never leave a pointer at a row that no longer exists."""
        if not rec_ids:
            return
        marks = ",".join("?" for _ in rec_ids)
        self.conn.execute(
            "UPDATE memories SET superseded_by = NULL WHERE superseded_by IN (%s)" % marks,
            list(rec_ids),
        )
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
        action_type: Optional[str] = None,
    ) -> List[MemoryRecord]:
        clauses, params = [], []
        if entity:
            clauses.append("entity = ?")
            params.append(entity)
        if action_type:
            clauses.append("action_type = ?")
            params.append(action_type)
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

    # ------------------------------------------------------------------ #
    # lexical search
    # ------------------------------------------------------------------ #
    def search_fts_scored(
        self,
        query: str,
        limit: int = 200,
        entity: Optional[str] = None,
        include_history: bool = True,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Return ``(record, bm25)`` ordered by relevance (lower bm25 is better).

        ``include_history=False`` filters superseded rows in SQL, before the
        limit. Filtering after the limit lets 200 old versions of a fact crowd
        the current row out of the result set entirely.
        """
        if not (query or "").strip():
            return []
        if self._needs_like(query):
            return self._search_like(query, limit, entity, include_history)
        if not self.fts_enabled:  # pragma: no cover - no fts5 available
            return self._search_like(query, limit, entity, include_history)
        try:
            sql = (
                "SELECT m.*, bm25(memories_fts) AS _bm25 FROM memories_fts f "
                "JOIN memories m ON m.id = f.id WHERE memories_fts MATCH ?"
            )
            params: list = [self._match_expr(query)]
            if entity:
                sql += " AND f.entity = ?"
                params.append(entity)
            if not include_history:
                sql += " AND m.valid_to IS NULL"
            sql += " ORDER BY _bm25 LIMIT ?"
            params.append(limit)
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:  # pragma: no cover - malformed match
            return self._search_like(query, limit, entity, include_history)
        return [(self._row(r), float(r["_bm25"])) for r in rows]

    def search_fts(
        self,
        query: str,
        limit: int = 200,
        entity: Optional[str] = None,
        include_history: bool = True,
    ) -> List[MemoryRecord]:
        return [
            rec for rec, _ in self.search_fts_scored(
                query, limit, entity, include_history=include_history,
            )
        ]

    @staticmethod
    def _needs_like(text: str) -> bool:
        return len([c for c in text if not c.isspace()]) < _MIN_TRIGRAM_CHARS

    def _search_like(
        self, query: str, limit: int, entity: Optional[str],
        include_history: bool = True,
    ) -> List[Tuple[MemoryRecord, float]]:
        needle = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql = "SELECT * FROM memories WHERE content LIKE ? ESCAPE '\\'"
        params: list = ["%" + needle + "%"]
        if entity:
            sql += " AND entity = ?"
            params.append(entity)
        if not include_history:
            sql += " AND valid_to IS NULL"
        sql += " LIMIT ?"
        params.append(limit)
        return [(self._row(r), -_LIKE_EVIDENCE) for r in self.conn.execute(sql, params).fetchall()]

    @staticmethod
    def _match_expr(text: str) -> str:
        """Turn free text into a trigram OR expression.

        With the trigram tokenizer a quoted phrase requires an exact substring,
        which almost never matches a natural-language query. OR-ing the query's
        trigrams gives real fuzzy matching for CJK.
        """
        chars = [c for c in text if not c.isspace()]
        if len(chars) < _MIN_TRIGRAM_CHARS:
            return '"' + text.replace('"', '""') + '"'
        seen, grams = set(), []
        for i in range(len(chars) - 2):
            gram = "".join(chars[i:i + 3])
            if gram not in seen:
                seen.add(gram)
                grams.append(gram)
            if len(grams) >= 48:  # long questions must stay cheap
                break
        return " OR ".join('"' + g.replace('"', '""') + '"' for g in grams)

    # ------------------------------------------------------------------ #
    def _row(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            type=MemoryType(row["type"]),
            entity=row["entity"],
            content=row["content"],
            structured=json.loads(row["structured"] or "{}"),
            action_type=row["action_type"] or "",
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
