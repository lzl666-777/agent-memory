# agent-memory

> Give your AI agent a memory of **decisions** — not just of facts.

## The problem

Agents forget. Every session starts blank, so the same mistakes get repeated:

- You switch to a cheaper supplier → it stock-outs → three months later you switch again.
- You cut price into a margin disaster → next quarter you do it again.

A vector store remembers *what was said*. It does not remember **what was decided, why, and what happened next**.

## What it remembers

| type | example |
|---|---|
| `fact` | 供应商A 交期 45 天 |
| `episode` | 2026-03 断货两周 |
| `decision` | 换成供应商B — 理由：交期延迟 — 结果：B 断货 — 教训：先评估备选产能 |

Decisions are the point: `record_outcome()` attaches what actually happened, and `recall()` surfaces the lesson the next time a similar situation appears.

## Quickstart

```bash
pip install agent-memory
```

```python
from agent_memory import Memory

mem = Memory("memory.db")

# 1. record a decision
rec = mem.remember(
    type="decision",
    entity="供应商A",
    content="决定把供应商A换成供应商B",
    structured={
        "decision": "换成供应商B",
        "rationale": "A 交期延迟、产能不足",
        "context": "3 月断货两周",
    },
)

# 2. later, record what actually happened
mem.record_outcome(rec.id, outcome="B 在 6 月断货两周", lesson="先评估备选产能")

# 3. next time, recall the experience
for hit in mem.recall(entity="供应商A", query="要不要换供应商", types=["decision"]):
    print(hit.score, hit.why, hit.record.structured["lesson"])
# 0.87 structured+lexical  先评估备选产能
```

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

## MCP server

Expose the memory to Claude Desktop, Cursor, or any MCP client:

```bash
pip install "agent-memory[mcp]"
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

Tools: `remember`, `recall`, `record_outcome`.

## Design

Every part exists for a reason:

| module | solves | without it |
|---|---|---|
| `writer.py` | importance scoring + conflict detection | memory bloat, contradictory facts |
| `store.py` | SQLite + FTS5 (trigram) | no lexical recall for CJK |
| `retrieval.py` | structured + lexical (+ optional semantic) → RRF → rerank | one path misses what matters |
| `forgetting.py` | exponential decay × access | unbounded growth, stale noise |
| `evaluation.py` | recall / conflict / staleness cases | another "seems fine" demo |

Deliberately **no vector database, no orchestration framework, no web server** — the hard part is the memory lifecycle, not the infrastructure.

## Evaluation

```bash
python -m agent_memory.evaluation
```

12 built-in cases over recall, conflict resolution and staleness.

## Development

```bash
git clone https://github.com/<you>/agent-memory.git
cd agent-memory
pip install -e ".[dev]"
pytest -q
python examples/ecommerce_ops.py
```

## License

MIT
