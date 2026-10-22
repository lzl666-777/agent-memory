# 0002 — Type-aware forgetting; protect lessons and history

- Status: accepted (0.2.0)
- Context: `forgetting.forget`

## Context

0.1 applied one decay curve (`importance × 2^(-age/180) × log(1+access)`) to every
row and deleted everything below `0.05`, including superseded rows
(`include_superseded=True`).

## What was measured

- A decision with a recorded outcome and lesson, never accessed, reaches
  `0.0436` at **800 days** and is deleted. It is deleted *because* nobody has needed
  it yet — which is the definition of a memory that has no backup.
- Superseded rows are deleted too, so `forget()` destroys the version history that
  `include_history=True` is advertised to provide: measured 2 rows before, 1 after.
- Deleting a record that other rows point at via `superseded_by` leaves a dangling
  identifier. There is no foreign key and `PRAGMA foreign_keys` is off.

Forgetting was therefore optimised against the product thesis: it removed exactly
the memories the project exists to keep.

## Decision

1. `decision` records are never deleted by default (`include_decisions=True` to force).
2. Superseded rows are never deleted by default (`include_history=True` to force).
3. Every deletion clears `superseded_by` pointers that would otherwise dangle.
4. `decay_score()` stays a pure, type-agnostic number; the *policy* lives in one
   place (`forget`) so it is testable on its own.

Freshness in retrieval changed for the same reason: facts half-life 180 d,
episodes 365 d, decisions 1095 d with a floor of 0.6 — an old lesson is demoted at
most 40%, never buried.

## Consequences

- An old decision is still recovered at 3 years (`test_old_lesson_ranks_first_after_two_years`
  checks 90/365/730 days).
- Storage grows with decision count instead of shrinking. For this product that is
  the correct trade-off; teams that disagree can pass `include_decisions=True`.
- Breaking change: `forget()` no longer removes history that 0.1 removed.
