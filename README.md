# agent-memory

> Give your AI agent a memory of **decisions** — and stop it from repeating them.

## The problem

Agents forget. Every session starts blank, so the same mistakes get repeated:

- You switch to a cheaper supplier → it stock-outs → three months later you switch again.
- You cut price into a margin disaster → next quarter you do it again.

A vector store remembers *what was said*. It does not remember **what was decided,
why, what happened next, and what that should change about the next decision**.

## What it remembers

| type | example |
|---|---|
| `fact` | 供应商A 交期 45 天 |
| `episode` | 2026-03 断货两周 |
| `decision` | 换成供应商B — 理由：交期延迟 — 结果：B 断货 — 教训：先评估备选产能 |

Decisions are the point: `record_outcome()` attaches what actually happened, and
`recall()` surfaces the lesson the next time a similar situation appears.

## Quickstart

```bash
git clone https://github.com/you/agent-memory.git
cd agent-memory
pip install -e .
```

```python
from agent_memory import Memory

mem = Memory("memory.db")

# 1. record a decision
rec = mem.remember(
    type="decision",
    entity="供应商A",
    content="决定更换供应商：把供应商A换成供应商B",
    action_type="supplier_switch",          # what was done, not who it was done to
    structured={
        "decision": "换成供应商B",
        "rationale": "A 交期延迟、产能不足",
        "context": "3 月断货两周",
    },
)

# 2. later, record what actually happened
mem.record_outcome(
    rec.id,
    outcome="供应商B 在 6 月断货两周",
    lesson="先评估备选产能",
    status="failure",                      # declared, never inferred
)

# 3. next time, recall the experience
for hit in mem.recall(entity="供应商A", query="要不要换供应商", types=["decision"]):
    print(hit.score, hit.why, hit.record.structured["lesson"])
    print(hit.signals)
# 0.55 structured+lexical+outcome
# {'lexical': 0.2, 'value': 1.0, 'importance': 0.9, 'freshness': 1.0}
```

`score` is a weighted sum of four signals in `[0, 1]`, and `signals` tells you
which one carried the hit. `why` lists the paths that matched
(`structured` / `lexical` / `semantic`) plus `outcome` when the record has a
recorded result. Nothing about a recall is a black box.

## Before the decision

`recall()` answers *"what do we know about X?"*. The question that changes
behaviour is *"we are about to do X again — what happened last time?"*.
`before()` answers that one, and it is keyed on the **action**, not the entity:

```python
card = mem.before("supplier_switch", context="要不要把供应商C换成供应商D")
print(card.verdict)      # caution
print(card.to_prompt())
```

```
PRECEDENT CHECK · supplier_switch
verdict: CAUTION — 1 recorded outcome, 1 failed
lessons from earlier attempts:
  - 换供应商前必须先评估备选产能，不能只看交期
precedents:
  1. [供应商A · 2026-03] 换成供应商B
     -> 供应商B 在 6 月断货两周 (failure)
Before acting: say how each failure mode above is avoided or accepted.
```

The lesson was learned on supplier A and it fires when the agent is about to
switch supplier **C**. That is the point: **a lesson belongs to the action, not
to the object it happened to.** A recall API keyed on `entity` can never surface
it, which is why the previous version could not.

| verdict | when |
|---|---|
| `no_history` | nothing recorded for this action |
| `unverified` | decisions exist, none has an outcome |
| `clear` | at least one outcome, no failure |
| `caution` | some failure, or too few attempts to quote a rate |
| `high_risk` | ≥2 recorded outcomes and ≥50% of them non-clean |

`status` is **declared, not inferred**. The library never guesses "failure" from
free text, and an unclassified outcome is never counted as a success:

```python
mem.record_outcome(rec.id, outcome="B 断货两周",
                   lesson="先评估备选产能",
                   status="failure")   # success | failure | mixed | unknown
```

`to_prompt()` is bounded (3 lessons, 3 precedents, ~1k chars) so an intervention
cannot flood a context window. `context` only *ranks* the history — it never
filters it away, because a precedent is shown, not searched for.

### Does it change the outcome?

`examples/agent_loop.py` runs the same agent twice against the same simulated
world: identical suppliers, identical request, identical policy. The only
difference is whether the first quarter's result was recorded.

```
                       stockouts   second decision
no memory              2/2         switch on lead time (repeat the mistake)
with agent-memory      1/2         verify capacity first -> decline
```

The first mistake still happens — memory is not clairvoyance. It prevents the
second one. The full prompt is printed, and the policy is a deterministic
stand-in so the demo runs offline; swap it for a real model and the loop is
unchanged. `tests/test_agent_loop.py` asserts the causal claim, including that a
supplier which *can* carry the demand is still switched to (the check is a
check, not a blanket refusal).

### Conflict resolution

Facts are versioned, not overwritten:

```python
mem.remember(type="fact", entity="供应商A", content="交期 30 天",
             structured={"key": "lead_time_days", "value": 30})
mem.remember(type="fact", entity="供应商A", content="交期 45 天",
             structured={"key": "lead_time_days", "value": 45})

# current value wins ...
assert mem.recall(entity="供应商A", types=["fact"])[0].record.structured["value"] == 45
# ... but the old value is still there when you ask for history
assert len(mem.recall(entity="供应商A", types=["fact"], include_history=True)) == 2
```

Writing the *same* value twice is a no-op, not a duplicate row.

## How recall ranks

Four signals, explicit weights, no hidden magic:

| signal | weight | what it measures |
|---|---|---|
| relevance | 0.55 | bm25 from FTS5 (ordered in SQL, not in Python) blended with query-trigram coverage |
| decision value | 0.20 | a decision with an outcome + lesson > a bare decision > an episode > a fact |
| importance | 0.10 | write-time score, pluggable |
| freshness | 0.15 | **type-aware decay**: facts go stale, lessons do not |

Two choices in there are deliberate and were made after measuring:

- **bm25 is ordered in SQL and `entity` is `UNINDEXED` in the FTS table.** Indexing
  `entity` makes every record of the queried entity match every query, which turns
  the lexical path into a silent copy of the SQL filter.
- **Reciprocal Rank Fusion was removed.** RRF is built for candidate sets in the
  thousands; with 5–50 candidates its rank prior spans 1.6% while any metadata
  multiplier spans 100%+. It ranked by recency and called it relevance. See
  [`docs/adr/0001`](docs/adr/0001-evidence-fusion-over-rrf.md).

A two-character CJK query (`断货`, `交期`, `涨价` — the common case) cannot be served
by a trigram index at all, so it falls back to an indexed `LIKE` scan.

When a `query` is given it also acts as a filter: scope (`entity` / `types`) says
*what may be recalled*, the query says *whether anything should be*. Asking about
refunds inside a supplier's memory returns nothing instead of the ten most recent
unrelated records.

## Forgetting

```python
mem.forget(dry_run=True)   # preview
```

Facts decay (`importance × 2^(-age/180) × log(1+access)`) and are dropped below a
threshold. Two categories are protected by default:

- **decisions** — an old lesson is *more* valuable, not less; the previous version
  deleted them at ~2.2 years, i.e. exactly the ones nobody had needed yet;
- **superseded rows** — they are the audit trail that conflict resolution exists to keep.

Both can be forced with `include_decisions=True` / `include_history=True`. Deleting a
row that another row points at never leaves a dangling `superseded_by`.
See [`docs/adr/0002`](docs/adr/0002-type-aware-forgetting.md).

## Evaluation

```bash
python -m agent_memory.evaluation
```

13 cases, each seeded with **24 distractor records for the same entity** (payment
terms, MOQ, packaging, logistics, inspection — half of them carrying outcomes), so
the answer has to beat its neighbours. The harness reports `recall@5`, `MRR`,
negative-case results and p95 latency, and it runs five ablations of itself:

| configuration | recall@5 | MRR | negative/structural |
|---|---|---|---|
| **full** | **100.0%** | **1.000** | **5/5** |
| no memory | 0.0% | 0.000 | 2/5 |
| random order | 50.0% | 0.500 | 5/5 |
| newest first | 50.0% | 0.500 | 5/5 |
| importance only | 50.0% | 0.500 | 5/5 |
| bm25 only | 100.0% | 0.938 | 5/5 |
| full, lexical path removed | 12.5% | 0.025 | — |

Two of those rows are the point:

- **random order scores 50%, not 100%.** The previous version of this harness used
  one or two records per case, so every candidate fit in `top_k` and it reported
  100% *even with scoring removed entirely*. A test that cannot fail is decoration.
- **removing the lexical path costs 87.5 points.** If someone breaks the ranking,
  CI notices.

The same 13 cases run against the 0.1 implementation scored **recall@5 12.5% /
MRR 0.042 / 4-of-13 passed**; its flagship scenario (record a decision, come back
three months later) ranked the correct memory *last* from ~65 days on.

p95 recall latency: **4.3 ms** at 5k records, **18 ms** at 20k (down from 189 ms).

A second suite covers the intervention layer — verdict escalation, cross-entity
transfer, lesson de-duplication, card bounds, and the migration of a database
written before `action_type` existed: **8/8 cases**.

## What this is not

Stated plainly, because the gaps are real:

- **No vector database, no orchestration framework, no web server.** The hard part is
  the memory lifecycle, not the infrastructure. An optional `embedder` enables a
  semantic signal; vectors are cached in-process, not persisted.
- **No synonym matching without an embedder.** `缺货` will not find `断货` — that is
  what the semantic path is for.
- **The verdict is a heuristic over declared statuses, not a risk model.** It counts
  what `record_outcome(status=...)` was told; it cannot detect an outcome the caller
  never recorded, and `high_risk` is a threshold (`≥2 outcomes, ≥50% non-clean`),
  not a calibrated probability.
- **No LLM-based importance scoring or extraction.** `WritePolicy.score()` is a
  rule-based placeholder with a documented seam; it does not read content.
- **Single-process.** SQLite without WAL; concurrent writers from multiple threads
  are not supported yet.
- The 13 cases are a regression suite, not a benchmark. Weights and `BM25_SCALE`
  were calibrated on them, so treat the absolute numbers as *evidence of ranking
  behaviour*, not as generalisation.

## MCP server

Expose the memory to Claude Desktop, Cursor, or any MCP client:

```bash
pip install -e ".[mcp]"
agent-memory-mcp
```

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory-mcp",
      "env": { "AGENT_MEMORY_DB": "/path/to/memory.db" }
    }
  }
}
```

Tools: `before_decision`, `remember`, `recall`, `record_outcome`.

`before_decision` is the one to wire into an agent loop: it returns the verdict,
the precedents and a ready-to-inject `prompt`. `recall` returns `why` and
`signals` with every hit, so the agent can say *why* it brought something up.

## Development

```bash
pip install -e ".[dev]"
pytest -q
python examples/ecommerce_ops.py
python examples/agent_loop.py
python -m agent_memory.evaluation
```

CI runs the tests, the evaluation and both examples on Python 3.10–3.12.

## License

MIT
