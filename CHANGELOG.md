# Changelog

## 0.3.0

The library stops being only a search API and becomes an intervention API.

### Added

- `Memory.before(action, context=..., entity=...)` returns a `Precedent`: a verdict
  (`no_history` / `unverified` / `clear` / `caution` / `high_risk`), the lessons from
  earlier attempts, the precedents themselves, and a bounded `to_prompt()` card.
- `action_type` is a first-class indexed field on a decision. `before()` keys on it,
  so a lesson learned on one entity fires when a *different* entity is about to be
  treated the same way.
- `record_outcome(..., status=...)` — `success` / `failure` / `mixed` / `unknown`.
  Declared and validated; never inferred from free text.
- `recall(action_type=...)`, and MCP tool `before_decision`.

### Changed

- Schema 3: `action_type` column with an index. Databases written by 0.1/0.2 are
  altered and backfilled from `structured["action_type"]` on open, and the FTS index
  is rebuilt once if it predates 0.2.
- Evaluation gained an 8-case intervention suite (verdicts, cross-entity transfer,
  lesson de-duplication, card bounds, migration).
- Retrieval: an optional `floor=False` mode used by `before()` — a precedent is
  shown, not searched for, so `context` ranks the history instead of filtering it.

### Notes

- Backwards compatible: `action_type` falls back to `structured["action_type"]`, and
  `record_outcome` without a status still records the outcome (as unclassified).
- A decision with an outcome but no status is reported as unclassified rather than
  counted as a success.

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
