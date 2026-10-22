# 0004 — Key the precedent on the action, not on the entity

- Status: accepted (0.3.0)
- Context: `intervention.py`, `Memory.before`, schema 3

## Context

Every memory API in this project was keyed on `entity`: `recall(entity=...)` is a
hard filter on the supplier / product / channel, and that is the right default for
"what do we know about X".

It is the wrong default for the question that actually changes behaviour. Measured
in 0.2: a decision filed under `供应商B` with an outcome and a lesson returned **0
hits** for `recall(entity="供应商A", ...)`. The agent asking about supplier A, about
to make the same mistake, could not see it. A memory that only answers questions
about the same noun cannot prevent a repeated mistake — it can only describe it
afterwards, again.

The second problem was outcome classification. `outcome` is free text. Any verdict
computed from it would require keyword guessing or an LLM call, and both fail
silently: a mis-guessed "success" is worse than no verdict at all.

## Decision

1. **Add a first-class `action_type`** (indexed column, not a JSON key) describing
   what was done: `supplier_switch`, `price_cut`, `channel_expand`...
2. **`before(action, context, entity)` keys on the action** and searches across
   entities. `entity` is context for the card, not a filter. This is the feature:
   lessons transfer.
3. **`context` ranks, it does not filter** (`floor=False`). A precedent is *shown*,
   not *searched for*; an unrelated phrasing must not hide a warning.
4. **Outcome status is declared** by the caller (`success` / `failure` / `mixed` /
   `unknown`), validated, and defaulted to unclassified. The library never infers it,
   and never counts an unclassified outcome as a success.
5. **The verdict is a small, readable rule**: `no_history` / `unverified` / `clear` /
   `caution` / `high_risk` (≥2 outcomes and ≥50% non-clean). Fewer than two outcomes
   never yield a rate.
6. **The card is bounded** (3 lessons, 3 precedents, ~1k characters) because it is
   meant to be injected into a prompt on every risky action.

## Consequences

- The flagship failure mode is now addressed: a lesson learned on B fires for C
  (`test_lesson_transfers_across_entities`, and case 1 of the intervention suite).
- Cost: callers must label `action_type` for decisions. Without it, `before()` reports
  `no_history` — an honest, visible failure rather than a silent one.
- The verdict is a heuristic over declared statuses, not a calibrated risk model.
  Recorded in the README's limitations rather than dressed up.
- Schema migration is required (`ALTER TABLE` + backfill from
  `structured["action_type"]`). Guarded by a test that opens a pre-0.3 database.
