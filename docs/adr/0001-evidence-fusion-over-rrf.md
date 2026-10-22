# 0001 — Replace RRF with explicit evidence fusion

- Status: accepted (0.2.0)
- Context: `Retriever.recall`

## Context

0.1 fused two rank lists with Reciprocal Rank Fusion (`1 / (60 + rank)`):
a "structured" list (the SQL `WHERE` result, i.e. arbitrary rowid order) and a
"lexical" list from FTS5. A multiplicative rerank was applied on top:
`RRF × (0.5 + 0.5·importance) × 0.5^(-age/180) × (1.15 if outcome)`.

## What was measured

Three problems, all reproducible:

1. **RRF is scale-dependent and the scale was wrong.** RRF's `k=60` is tuned for
   candidate sets in the thousands. With 7 candidates the entire rank prior spans
   1.6% (`1/61` vs `1/67`), while the metadata multipliers span 150–400%.
   The ranking was therefore *decided by age*, not by relevance.
2. **The lexical path was a copy of the structured path.** `entity` was indexed in
   the FTS table, so a single-entity query matched every record of that entity on
   the trigram `供应商`. Both rank lists were then near-identical.
3. **`search_fts` had no `ORDER BY`.** FTS5 without `ORDER BY bm25()` returns rowid
   order. Measured on a 4-document corpus: the correct document scored bm25
   `-0.8539`, the rest `-0.0000` — the signal existed and was discarded.

Consequence: the flagship scenario ("decide in March, ask again in September")
ranked the correct memory **last** from ~65 days of age.

## Decision

Each signal becomes a number in `[0, 1]` with an explicit weight:

```
score = 0.55·relevance + 0.20·decision_value + 0.10·importance + 0.15·freshness
```

- `relevance` = `0.5·trigram_coverage + 0.5·(1 - exp(-2·max(0, -bm25)))`, with bm25
  ordered in SQL and `entity` marked `UNINDEXED`.
- `freshness` is type-aware (see ADR 0002).
- The full component vector is returned as `signals` on every hit.

The coverage term exists because bm25 measures *discrimination*: on a corpus of one
document every term has idf ≈ 0, so a perfect match would score zero evidence.
Coverage means the same thing at 1 record and at 100k.

## Consequences

- 13-case harness: recall@5 100%, MRR 1.000 (0.1: 12.5%, 0.042).
- Removing the lexical path now costs 87.5 points, so the signal is verifiably
  load-bearing (`test_lexical_path_is_load_bearing`).
- Scores are absolute and comparable across queries, unlike RRF sums.
- Cost: weights are hand-tuned on 13 cases. A larger benchmark is required before
  claiming generalisation — recorded as a limitation in the README.
