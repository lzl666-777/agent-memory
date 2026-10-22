# 0003 — An evaluation that is allowed to fail

- Status: accepted (0.2.0)
- Context: `evaluation.py`, CI

## Context

0.1 shipped 12 "cases" that scored 100%. Each seeded one or two records, so every
candidate fitted inside `top_k` and no ordering could be wrong.

## What was measured

Replacing `Memory.recall` with a function that ignored scoring entirely — same
candidates, insertion order — still scored **12/12**. Ignoring the `entity` and
`type` filters as well scored 9/12. The suite was testing `WHERE` clauses, not
retrieval, while the README claimed it prevented "another *seems fine* demo".

## Decision

1. Every case seeds **24 distractor records for the same entity**, half carrying
   outcomes and lessons, so the target must outrank plausible neighbours.
2. Cases are of three kinds: **ranked** (target must be rank 1), **structural**
   (counts, type isolation, `include_history`), **negative** (the right answer is
   an empty result).
3. The harness reports `recall@5`, `MRR`, negative results and p95 latency.
4. It runs **ablations of itself**: no memory, random order, newest-first,
   importance-only, bm25-only. A random ranker must not be able to pass.
5. Two acceptance criteria are encoded as tests:
   - random order must score ≥30 points below full;
   - removing the lexical path must cost ≥15 points (actual: 87.5).

## Consequences

- The suite can fail, and does when ranking is broken.
- The distractor corpus is now part of the specification: changing it changes the
  numbers, so it must be changed deliberately.
- 13 cases is a regression suite, not a benchmark. Absolute numbers are evidence of
  ranking behaviour; they are not a generalisation claim. A public benchmark
  (e.g. LongMemEval-style) is the next step, not a substitute.
