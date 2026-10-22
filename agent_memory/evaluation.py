"""Evaluation harness: recall, conflict resolution and staleness.

Run::

    python -m agent_memory.evaluation
"""

from __future__ import annotations

from typing import Any, Dict, List

from .memory import Memory

CASES: List[Dict[str, Any]] = [
    # -------------------------- recall -------------------------- #
    {
        "name": "按实体召回决策",
        "setup": [dict(type="decision", entity="供应商A", content="决定换成供应商B",
                       structured={"decision": "换B", "rationale": "交期延迟"})],
        "recall": dict(entity="供应商A", types=["decision"]),
        "expect": {"top_entity": "供应商A", "count": 1},
    },
    {
        "name": "按问题召回历史决策",
        "setup": [dict(type="decision", entity="供应商A", content="决定换成供应商B",
                       structured={"decision": "换B", "rationale": "交期延迟"})],
        "recall": dict(entity="供应商A", query="要不要换供应商", types=["decision"]),
        "expect": {"top_entity": "供应商A"},
    },
    {
        "name": "按实体召回事实",
        "setup": [dict(type="fact", entity="供应商A", content="交期30天",
                       structured={"key": "lead_time_days", "value": 30})],
        "recall": dict(entity="供应商A", types=["fact"]),
        "expect": {"top_value": 30, "count": 1},
    },
    {
        "name": "按关键词召回情景",
        "setup": [dict(type="episode", entity="供应商A", content="3月断货两周",
                       structured={"event": "断货", "impact": "2周"})],
        "recall": dict(entity="供应商A", query="断货", types=["episode"]),
        "expect": {"top_entity": "供应商A"},
    },
    # ------------------------ conflict ------------------------- #
    {
        "name": "冲突后新值覆盖旧值",
        "setup": [
            dict(type="fact", entity="供应商A", content="交期30天",
                 structured={"key": "lead_time_days", "value": 30}),
            dict(type="fact", entity="供应商A", content="交期45天",
                 structured={"key": "lead_time_days", "value": 45}),
        ],
        "recall": dict(entity="供应商A", types=["fact"]),
        "expect": {"top_value": 45, "count": 1},
    },
    {
        "name": "不同实体的同名事实不冲突",
        "setup": [
            dict(type="fact", entity="供应商A", content="交期30天",
                 structured={"key": "lead_time_days", "value": 30}),
            dict(type="fact", entity="供应商B", content="交期45天",
                 structured={"key": "lead_time_days", "value": 45}),
        ],
        "recall": dict(entity="供应商B", types=["fact"]),
        "expect": {"top_value": 45, "count": 1},
    },
    # ------------------------ staleness ------------------------ #
    {
        "name": "默认不返回过期事实",
        "setup": [
            dict(type="fact", entity="供应商A", content="交期30天",
                 structured={"key": "lead_time_days", "value": 30}),
            dict(type="fact", entity="供应商A", content="交期45天",
                 structured={"key": "lead_time_days", "value": 45}),
        ],
        "recall": dict(entity="供应商A", types=["fact"]),
        "expect": {"not_value": 30, "count": 1},
    },
    {
        "name": "历史可查（include_history）",
        "setup": [
            dict(type="fact", entity="供应商A", content="交期30天",
                 structured={"key": "lead_time_days", "value": 30}),
            dict(type="fact", entity="供应商A", content="交期45天",
                 structured={"key": "lead_time_days", "value": 45}),
        ],
        "recall": dict(entity="供应商A", types=["fact"], include_history=True, top_k=10),
        "expect": {"count": 2},
    },
    # --------------------- outcome feedback -------------------- #
    {
        "name": "有结果的决策优先召回",
        "setup": [
            dict(type="decision", entity="供应商A", content="决定换成供应商B",
                 structured={"decision": "换B", "rationale": "交期延迟"}),
            dict(type="decision", entity="供应商A", content="决定维持供应商A",
                 structured={"decision": "不换", "rationale": "价格便宜"}),
        ],
        "recall": dict(entity="供应商A", query="换供应商", types=["decision"]),
        "expect": {"top_entity": "供应商A"},
    },
    {
        "name": "决策记忆保存教训",
        "setup": [dict(type="decision", entity="供应商A", content="决定换成供应商B",
                       structured={"decision": "换B", "outcome": "B断货两周",
                                   "lesson": "先评估备选产能"})],
        "recall": dict(entity="供应商A", types=["decision"]),
        "expect": {"top_has_lesson": True},
    },
    # ------------------------ isolation ------------------------ #
    {
        "name": "实体过滤隔离",
        "setup": [
            dict(type="decision", entity="供应商A", content="换供应商B"),
            dict(type="decision", entity="供应商C", content="换供应商D"),
        ],
        "recall": dict(entity="供应商C", types=["decision"]),
        "expect": {"top_entity": "供应商C", "count": 1},
    },
    {
        "name": "类型过滤隔离",
        "setup": [
            dict(type="decision", entity="供应商A", content="换供应商B"),
            dict(type="episode", entity="供应商A", content="3月断货"),
        ],
        "recall": dict(entity="供应商A", types=["episode"]),
        "expect": {"top_type": "episode", "count": 1},
    },
]


def _check(hits, expect: Dict[str, Any]) -> bool:
    if not hits:
        return False
    top = hits[0].record
    if "top_entity" in expect and top.entity != expect["top_entity"]:
        return False
    if "top_type" in expect and top.type.value != expect["top_type"]:
        return False
    if "top_value" in expect and top.structured.get("value") != expect["top_value"]:
        return False
    if "not_value" in expect and any(h.record.structured.get("value") == expect["not_value"] for h in hits):
        return False
    if "top_has_lesson" in expect and bool(top.structured.get("lesson")) != expect["top_has_lesson"]:
        return False
    if "count" in expect and len(hits) != expect["count"]:
        return False
    return True


def run_eval() -> Dict[str, Any]:
    """Run every built-in case against a fresh in-memory store."""
    results = []
    passed = 0
    for case in CASES:
        mem = Memory(":memory:")
        try:
            for item in case["setup"]:
                mem.remember(**item)
            hits = mem.recall(**case["recall"])
            ok = _check(hits, case["expect"])
        finally:
            mem.close()
        passed += ok
        results.append({"name": case["name"], "passed": ok})

    total = len(CASES)
    return {
        "total": total,
        "passed": passed,
        "accuracy": round(passed / total * 100, 1) if total else 0.0,
        "results": results,
    }


if __name__ == "__main__":
    report = run_eval()
    for item in report["results"]:
        print(("PASS" if item["passed"] else "FAIL"), item["name"])
    print()
    print(f"准确率: {report['accuracy']}% ({report['passed']}/{report['total']})")
