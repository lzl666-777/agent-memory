"""End-to-end demo: the agent remembers a failed supplier switch.

Run:
    python examples/ecommerce_ops.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import Memory

mem = Memory(":memory:")

print("=" * 60)
print("场景 1（3 月）：运营决定换供应商")
print("=" * 60)
decision = mem.remember(
    type="decision",
    entity="供应商A",
    content="决定把供应商A换成供应商B",
    structured={
        "decision_type": "supplier_switch",
        "decision": "换成供应商B",
        "rationale": "A 交期延迟、产能不足",
        "context": "3 月因 A 延迟断货两周",
    },
    source="session-2026-03-15",
)
print(f"已记录决策: {decision.id} | 重要性: {decision.importance:.2f}")

print()
print("=" * 60)
print("场景 2（6 月）：结果回来了")
print("=" * 60)
mem.record_outcome(
    decision.id,
    outcome="供应商B 在 6 月断货两周",
    lesson="换供应商前必须先评估备选产能，不能只看交期",
    outcome_at="2026-06-10",
)
print("已补充结果与教训，重要性提升")

print()
print("=" * 60)
print("场景 3（9 月）：运营又问「要不要再换供应商」")
print("=" * 60)
hits = mem.recall(
    entity="供应商A",
    query="现在要不要换供应商",
    types=["decision"],
    top_k=3,
)
for hit in hits:
    print(f"\n得分: {hit.score:.4f} | 命中路径: {hit.why}")
    print(f"决策: {hit.record.structured.get('decision')}")
    print(f"结果: {hit.record.structured.get('outcome')}")
    print(f"教训: {hit.record.structured.get('lesson')}")

print()
print("=" * 60)
print("场景 4（10 月）：供应商A 交期更新为 45 天（冲突消解）")
print("=" * 60)
mem.remember(
    type="fact", entity="供应商A", content="交期 30 天",
    structured={"key": "lead_time_days", "value": 30},
)
mem.remember(
    type="fact", entity="供应商A", content="交期 45 天",
    structured={"key": "lead_time_days", "value": 45},
)
current = mem.recall(entity="供应商A", types=["fact"])
print(f"当前生效事实: 交期 {current[0].record.structured['value']} 天（旧的 30 天已进入历史）")

print()
print("=" * 60)
print("场景 5：遗忘预览（默认 180 天半衰期）")
print("=" * 60)
would_forget = mem.forget(dry_run=True)
print(f"将被遗忘: {len(would_forget)} 条（刚写入的记忆都还新鲜）")
