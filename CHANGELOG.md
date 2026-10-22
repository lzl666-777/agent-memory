# Changelog

## 0.2.0

Retrieval and forgetting were reworked after measuring that the 0.1 implementation
did not do what its README claimed. See `docs/adr/`.

### Changed

- **Ranking** is an explicit weighted sum (`relevance`, `decision_value`,
  `importance`, `freshness`) instead of RRF + multiplicative rerank. Every hit
  exposes `signals`. (ADR 0001)
- **Freshness is type-aware**: facts decay over 180 days, episodes over 365,
  decisions over 1095 with a floor of 0.6, so an old lesson is never buried.
- **A `query` is also a filter.** Scope narrows what may be recalled; the query
  decides whether anything should be. A question that matches nothing returns
  nothing instead of the newest records of the entity.
- **`forget()` protects decisions and superseded history by default** and clears
  `superseded_by` pointers on deletion. (ADR 0002)
- **Evaluation** uses 24 distractors per case, reports recall@5 / MRR / negative
  results / p95 latency, and runs five ablations of itself. (ADR 0003)

### Fixed

- FTS5 results are ordered by `bm25()`; previously the "ranking" was rowid order.
- `entity` is `UNINDEXED` in the FTS table; previously every record of the queried
  entity matched every query through the entity column.
- Two-character CJK queries (`断货`, `交期`) match again via an indexed `LIKE`
  fallback — a trigram index cannot represent them.
- Writing the same fact value twice is idempotent instead of creating duplicate
  current rows.
- `recall()` batches its access bookkeeping into one transaction (was: one commit
  per hit); semantic vectors are cached in-process (was: re-embedded per query).
- Existing databases are migrated automatically (`PRAGMA user_version` → FTS rebuild).

### Breaking

- `forget()` no longer deletes `decision` records or superseded history unless
  `include_decisions=True` / `include_history=True`.
- `Memory.recall(entity=..., query=...)` can now return `[]` when nothing matches
  the query. Callers that want the full entity history should omit `query`.
- `RetrievedMemory.to_dict()` gained a `signals` field.
- Python 3.10+ only (was already the case, now enforced by the type hints in use).

### Known gaps

- No synonym matching without an embedder; no vector persistence.
- SQLite without WAL: one process, one writer.
- Importance is rule-based; `WritePolicy.score()` is a documented seam.

## 0.1.0

First release: facts / episodes / decisions, outcome + lesson, temporal validity
via `valid_from` / `valid_to`, FTS5 lexical recall, decay-based forgetting,
MCP server.
