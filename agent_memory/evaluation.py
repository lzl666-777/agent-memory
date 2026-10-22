"""Evaluation harness: recall, conflict resolution, staleness -- with distractors.

Why this file looks like this
-----------------------------
The first version of this harness used one or two records per case. It reported
100% accuracy, and it reported 100% accuracy *for a retriever with scoring
removed entirely* -- every candidate fits in ``top_k``, so nothing could fail.
A test that cannot fail is decoration.

So each case here seeds a realistic corpus for the *same entity* (24 distractor
records: payment terms, MOQ, packaging, logistics, inspection...), and the
answer has to beat them. Cases with no target are negative tests: the correct
answer is an empty result, or an absence of a specific record.

Run::

    python -m agent_memory.evaluation
"""

from __future__ import annotations

import random
import time
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .memory import Memory
from .models import MemoryRecord, utcnow
from .retrieval import Retriever

ENTITY = "供应商A"
DISTRACTORS = 24

#: Realistic colleagues of the answer: same supplier, other topics, mostly recent,
#: half of them carrying an outcome + lesson (so "value" alone cannot separate
#: them from the answer -- only relevance can).
_DISTRACTORS: List[Dict[str, Any]] = [
    dict(type="decision", content="{s} 的付款账期调整为 30 天",
         structured=dict(decision="账期30天", rationale="改善现金流",
                         outcome="正常执行", lesson="账期调整要同步财务")),
    dict(type="decision", content="要求 {s} 把起订量提到 500 件",
         structured=dict(decision="起订量500", rationale="降低物流成本",
                         outcome="仓容紧张", lesson="提 MOQ 前先看仓容")),
    dict(type="decision", content="{s} 的包装改为双层纸箱",
         structured=dict(decision="双层纸箱", rationale="降低破损率",
                         outcome="破损率下降", lesson="包装升级要算单件成本")),
    dict(type="episode", content="{s} 到货物流延误 3 天",
         structured=dict(event="物流延误", impact="3天")),
    dict(type="fact", content="{s} 的报价为 12.5 元",
         structured=dict(key="unit_price", value=12)),
    dict(type="decision", content="{s} 的质检抽检比例改为 10%",
         structured=dict(decision="抽检10%", rationale="来料波动",
                         outcome="漏检率下降", lesson="抽检要覆盖新批次")),
    dict(type="episode", content="{s} 到货破损率升到 2%",
         structured=dict(event="到货破损", impact="2%")),
    dict(type="decision", content="{s} 的发票改为电子发票",
         structured=dict(decision="电子发票", rationale="提速",
                         outcome="正常", lesson="先确认对方接收方式")),
    dict(type="fact", content="{s} 的账期为 30 天",
         structured=dict(key="payment_days", value=30)),
    dict(type="episode", content="{s} 对账出现 1200 元差异",
         structured=dict(event="对账差异", impact="1200元")),
    dict(type="decision", content="{s} 的仓配迁到新仓",
         structured=dict(decision="迁新仓", rationale="扩容",
                         outcome="上架延迟", lesson="搬迁要留缓冲期")),
    dict(type="decision", content="与 {s} 续签一年合同",
         structured=dict(decision="续签一年", rationale="锁价",
                         outcome="价格未变", lesson="续签前先比价")),
]

CASES: List[Dict[str, Any]] = [
    # ---------------- the flagship scenario: the lesson must survive age ----
    dict(
        name="3 个月后仍召回换供应商决策",
        target=dict(
            type="decision",
            content="决定更换供应商：把供应商A换成供应商B",
            structured=dict(decision="换成供应商B", rationale="交期延迟、产能不足",
                            outcome="供应商B 在 6 月断货两周", lesson="换供应商前先评估备选产能"),
            age_days=90,
        ),
        query=dict(entity=ENTITY, query="现在要不要更换供应商", types=["decision"]),
        max_rank=1,
    ),
    dict(
        name="1 年后仍召回换供应商决策",
        target=dict(
            type="decision",
            content="决定更换供应商：把供应商A换成供应商B",
            structured=dict(decision="换成供应商B", rationale="交期延迟、产能不足",
                            outcome="供应商B 在 6 月断货两周", lesson="换供应商前先评估备选产能"),
            age_days=365,
        ),
        query=dict(entity=ENTITY, query="现在要不要更换供应商", types=["decision"]),
        max_rank=1,
    ),
    dict(
        name="2 年后仍召回换供应商决策",
        target=dict(
            type="decision",
            content="决定更换供应商：把供应商A换成供应商B",
            structured=dict(decision="换成供应商B", rationale="交期延迟、产能不足",
                            outcome="供应商B 在 6 月断货两周", lesson="换供应商前先评估备选产能"),
            age_days=730,
        ),
        query=dict(entity=ENTITY, query="现在要不要更换供应商", types=["decision"]),
        max_rank=1,
    ),
    # ---------------- CJK: two-character terms are the common case ---------
    dict(
        name="两字中文词召回断货事件",
        target=dict(
            type="episode",
            content="2026 年 3 月因供应商A交期延迟导致断货两周",
            structured=dict(event="断货", impact="2周"),
            age_days=120,
        ),
        query=dict(entity=ENTITY, query="断货", types=["episode"]),
        max_rank=1,
    ),
    dict(
        name="三字中文词召回交期决策",
        target=dict(
            type="decision",
            content="供应商A 交期延迟，决定改用空运补货",
            structured=dict(decision="改用空运", rationale="交期延迟",
                            outcome="运费上涨", lesson="空运只做应急"),
            age_days=200,
        ),
        query=dict(entity=ENTITY, query="交期延迟", types=["decision"]),
        max_rank=1,
    ),
    dict(
        name="涨价决策召回",
        target=dict(
            type="decision",
            content="记录一下要不要涨价的讨论：最终决定对 SKU-12 涨价 5%",
            structured=dict(decision="涨价5%", rationale="原料成本上涨",
                            outcome="销量下滑 8%", lesson="涨价要配合活动"),
            age_days=150,
        ),
        query=dict(entity=ENTITY, query="要不要涨价"),
        max_rank=1,
    ),
    # ---------------- outcome-bearing decisions win ties -------------------
    dict(
        name="有 outcome 的决策优先于同主题无 outcome 的决策",
        target=dict(
            type="decision",
            content="决定更换供应商：把供应商A换成供应商B",
            structured=dict(decision="换成供应商B", rationale="交期延迟",
                            outcome="B 断货两周", lesson="先评估备选产能"),
            age_days=30,
        ),
        extras=[
            dict(type="decision", content="决定更换供应商，但暂不执行",
                 structured=dict(decision="暂缓", rationale="价格便宜"), age_days=0),
        ],
        query=dict(entity=ENTITY, query="更换供应商", types=["decision"]),
        max_rank=1,
    ),
    # ---------------- conflict resolution / temporal validity --------------
    dict(
        name="冲突后当前值胜出（旧值不可见）",
        extras=[
            dict(type="fact", content="交期 30 天", structured=dict(key="lead_time_days", value=30), age_days=40),
            dict(type="fact", content="交期 45 天", structured=dict(key="lead_time_days", value=45), age_days=0),
        ],
        query=dict(entity=ENTITY, types=["fact"], top_k=50),
        must_include=["交期 45 天"],
        forbidden=["交期 30 天"],
    ),
    dict(
        name="历史可查（include_history）",
        extras=[
            dict(type="fact", content="交期 30 天", structured=dict(key="lead_time_days", value=30), age_days=40),
            dict(type="fact", content="交期 45 天", structured=dict(key="lead_time_days", value=45), age_days=0),
        ],
        query=dict(entity=ENTITY, types=["fact"], include_history=True, top_k=50),
        must_include=["交期 30 天", "交期 45 天"],
    ),
    # ---------------- isolation / negatives --------------------------------
    dict(
        name="实体隔离：不泄漏其他实体的内容",
        extras=[
            dict(type="decision", entity="供应商C", content="决定更换供应商：把供应商C换成供应商D",
                 structured=dict(decision="换D"), age_days=20),
        ],
        query=dict(entity="供应商C", query="更换供应商"),
        forbidden=["供应商A"],
        expect_count=1,
    ),
    dict(
        name="无关问题不返回垃圾（相关性下限）",
        query=dict(entity=ENTITY, query="退款政策怎么申请"),
        expect_empty=True,
    ),
    dict(
        name="既无 scope 也无问题时不返回任何东西",
        query=dict(),
        expect_empty=True,
    ),
    dict(
        name="类型隔离",
        target=dict(
            type="episode",
            content="供应商A 断货两天",
            structured=dict(event="断货", impact="2天"),
            age_days=10,
        ),
        query=dict(entity=ENTITY, query="断货", types=["episode"]),
        expect_types=["episode"],
        expect_count=1,
    ),
]


# ---------------------------------------------------------------------- #
# intervention ("what happened last time we did this?")
# ---------------------------------------------------------------------- #
#: Each case answers one question: given this history, is the card correct?
#: Note that the setup entity and the entity asked about differ where it matters:
#: a lesson is keyed on the action, not on the supplier it happened to.
INTERVENTION_CASES: List[Dict[str, Any]] = [
    dict(
        name="跨实体教训可见：B 的坑在 C 上仍然报警",
        setup=[dict(
            type="decision", entity="供应商B", action_type="supplier_switch",
            content="决定更换供应商：把供应商B换成供应商C",
            structured=dict(decision="换成供应商C"),
            outcome=dict(outcome="供应商C 断货两周", lesson="换供应商前先评估备选产能", status="failure"),
            age_days=120,
        )],
        action="supplier_switch",
        entity="供应商A",
        # context shares no vocabulary with the record: a precedent is shown,
        # not searched for, so it must still come back.
        context="价格更便宜",
        expect=dict(verdict="caution", attempts=1, failures=1,
                    lesson_contains="先评估备选产能", precedent_entity="供应商B",
                    precedent_count_min=1),
    ),
    dict(
        name="失败占多数升级为 high_risk",
        setup=[
            dict(type="decision", entity="供应商B", action_type="supplier_switch",
                 content="决定换成供应商C", structured=dict(decision="换C"),
                 outcome=dict(outcome="C 断货", status="failure"), age_days=200),
            dict(type="decision", entity="供应商D", action_type="supplier_switch",
                 content="决定换成供应商E", structured=dict(decision="换E"),
                 outcome=dict(outcome="E 涨价 20%", status="failure"), age_days=100),
            dict(type="decision", entity="供应商F", action_type="supplier_switch",
                 content="决定换成供应商G", structured=dict(decision="换G"),
                 outcome=dict(outcome="交付正常", status="success"), age_days=50),
        ],
        action="supplier_switch",
        context="要不要把供应商A换成供应商B",
        expect=dict(verdict="high_risk", attempts=3, failures=2, successes=1,
                    prompt_contains=["HIGH_RISK", "2 failed"], needs_attention=True),
    ),
    dict(
        name="没有历史时明确说没有",
        setup=[],
        action="price_cut",
        context="要不要降价 5%",
        expect=dict(verdict="no_history", attempts=0, needs_attention=False),
    ),
    dict(
        name="只成功不报警",
        setup=[
            dict(type="decision", entity="供应商A", action_type="supplier_switch",
                 content="决定换成供应商B", structured=dict(decision="换B"),
                 outcome=dict(outcome="交付正常", status="success"), age_days=60),
            dict(type="decision", entity="供应商C", action_type="supplier_switch",
                 content="决定换成供应商D", structured=dict(decision="换D"),
                 outcome=dict(outcome="成本下降", status="success"), age_days=30),
        ],
        action="supplier_switch",
        context="要不要换供应商",
        expect=dict(verdict="clear", failures=0, successes=2, needs_attention=False),
    ),
    dict(
        name="有决策但无结果时标为 unverified",
        setup=[dict(type="decision", entity="供应商A", action_type="price_cut",
                    content="决定降价 5%", structured=dict(decision="降5%"), age_days=10)],
        action="price_cut",
        context="要不要降价",
        expect=dict(verdict="unverified", attempts=0),
    ),
    dict(
        name="不同动作不串味",
        setup=[dict(type="decision", entity="供应商A", action_type="supplier_switch",
                    content="决定换成供应商B", structured=dict(decision="换B"),
                    outcome=dict(outcome="断货", lesson="先评估产能", status="failure"), age_days=90)],
        action="price_cut",
        context="要不要降价",
        expect=dict(verdict="no_history", attempts=0),
    ),
    dict(
        name="教训去重",
        setup=[
            dict(type="decision", entity="供应商B", action_type="supplier_switch",
                 content="决定换成供应商C", structured=dict(decision="换C"),
                 outcome=dict(outcome="断货", lesson="先评估备选产能", status="failure"), age_days=300),
            dict(type="decision", entity="供应商D", action_type="supplier_switch",
                 content="决定换成供应商E", structured=dict(decision="换E"),
                 outcome=dict(outcome="断货", lesson="先评估备选产能", status="failure"), age_days=200),
            dict(type="decision", entity="供应商F", action_type="supplier_switch",
                 content="决定换成供应商G", structured=dict(decision="换G"),
                 outcome=dict(outcome="断货", lesson="先评估备选产能", status="failure"), age_days=100),
        ],
        action="supplier_switch",
        context="要不要换供应商",
        expect=dict(verdict="high_risk", lesson_count=1),
    ),
    dict(
        name="卡片可直接注入提示词且长度有界",
        setup=[dict(type="decision", entity="供应商B", action_type="supplier_switch",
                    content="决定更换供应商：把供应商B换成供应商C" + "，细节" * 40,
                    structured=dict(decision="换成供应商C"),
                    outcome=dict(outcome="C 断货两周", lesson="换供应商前先评估备选产能",
                                 status="failure"), age_days=120)],
        action="supplier_switch",
        context="要不要换供应商",
        expect=dict(verdict="caution",
                    prompt_contains=["PRECEDENT CHECK", "supplier_switch", "先评估备选产能",
                                     "Before acting"],
                    prompt_max_len=1200),
    ),
]


# ---------------------------------------------------------------------- #
# seeding
# ----------------------------------------------------------------------#
def _age(mem: Memory, rec_id: str, days: float) -> None:
    if not days:
        return
    rec = mem.store.get(rec_id)
    rec.valid_from = utcnow() - timedelta(days=days)
    mem.store.update(rec)


def _seed(mem: Memory, case: Dict[str, Any]) -> Optional[str]:
    """Seed the distractor corpus, extras and the target; return the target id."""
    target_id: Optional[str] = None
    n = case.get("distractors", DISTRACTORS)
    for i in range(n):
        tpl = _DISTRACTORS[i % len(_DISTRACTORS)]
        structured = dict(tpl["structured"])
        if tpl["type"] == "fact":
            structured["key"] = "%s_%d" % (structured["key"], i)
            structured["value"] = int(structured.get("value", 0)) + i
        content = tpl["content"].format(s=ENTITY) + "（记录 %d）" % i
        rec = mem.remember(type=tpl["type"], entity=ENTITY, content=content, structured=structured)
        _age(mem, rec.id, i % 20)

    for extra in case.get("extras", []):
        _add(mem, extra)

    if case.get("target"):
        target_id = _add(mem, case["target"])
    return target_id


def _add(mem: Memory, spec: Dict[str, Any]) -> str:
    rec = mem.remember(
        type=spec["type"],
        entity=spec.get("entity", ENTITY),
        content=spec["content"],
        structured=dict(spec.get("structured") or {}),
        importance=spec.get("importance"),
    )
    _age(mem, rec.id, spec.get("age_days", 0))
    return rec.id


# ---------------------------------------------------------------------- #
# judging
# ---------------------------------------------------------------------- #
def _target_rank(hits, target_id: Optional[str]) -> Optional[int]:
    if target_id is None:
        return None
    for rank, hit in enumerate(hits, start=1):
        if hit.record.id == target_id:
            return rank
    return None


def _judge(hits, case: Dict[str, Any], target_id: Optional[str]) -> bool:
    if case.get("expect_empty") and hits:
        return False
    if "expect_count" in case and len(hits) != case["expect_count"]:
        return False
    if case.get("expect_types"):
        if any(hit.record.type.value not in case["expect_types"] for hit in hits):
            return False
    if case.get("forbidden"):
        blob = " ".join(hit.record.content for hit in hits)
        if any(word in blob for word in case["forbidden"]):
            return False
    if case.get("must_include"):
        blob = " ".join(hit.record.content for hit in hits)
        if any(word not in blob for word in case["must_include"]):
            return False
    if target_id is not None:
        rank = _target_rank(hits, target_id)
        if rank is None or rank > case.get("max_rank", 1):
            return False
    return True


def run_case(case: Dict[str, Any], memory_factory: Callable[[], Memory]) -> Tuple[bool, Optional[int], Optional[str]]:
    mem = memory_factory()
    try:
        target_id = _seed(mem, case)
        hits = mem.recall(**case["query"])
        return _judge(hits, case, target_id), _target_rank(hits, target_id), None
    finally:
        mem.close()


# ---------------------------------------------------------------------- #
# baselines
# ---------------------------------------------------------------------- #
class _NoMemory(Retriever):
    """Memory disabled: the floor for every metric."""

    def recall(self, **kwargs):
        return []


class _RandomOrder(Retriever):
    """Same candidates, no ranking: proves the harness can fail."""

    def recall(self, top_k: int = 5, **kwargs):
        hits = super().recall(top_k=10 ** 6, **kwargs)
        random.Random(7).shuffle(hits)
        return hits[:top_k]


BASELINES: Dict[str, Callable[[Any], Retriever]] = {
    "no_memory": lambda store: _NoMemory(store),
    "random": lambda store: _RandomOrder(store),
    "newest_first": lambda store: Retriever(store, weights={"freshness": 1.0}),
    "importance_only": lambda store: Retriever(store, weights={"importance": 1.0}),
    "keyword_only": lambda store: Retriever(store, weights={"lexical": 1.0}),
}


# ---------------------------------------------------------------------- #
# runner
# ---------------------------------------------------------------------- #
def _summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    targeted = [r for r in results if r["has_target"]]
    negative = [r for r in results if not r["has_target"]]
    ranks = [r["rank"] for r in targeted if r["rank"] is not None]
    return {
        "total": total,
        "passed": passed,
        "accuracy": round(passed / total * 100, 1) if total else 0.0,
        "recall_at_5": round(sum(1 for r in ranks if r <= 5) / len(targeted) * 100, 1)
        if targeted else 0.0,
        "mrr": round(sum(1.0 / r for r in ranks) / len(targeted), 3) if targeted else 0.0,
        "negatives_passed": sum(1 for r in negative if r["passed"]),
        "negatives_total": len(negative),
    }


def _evaluate(
    memory_factory: Optional[Callable[[], Memory]],
    retriever_factory: Optional[Callable[[Any], Retriever]] = None,
) -> Dict[str, Any]:
    def _make() -> Memory:
        mem = memory_factory() if memory_factory else Memory(":memory:")
        if retriever_factory is not None:
            mem.retriever = retriever_factory(mem.store)
        return mem

    results = []
    for case in CASES:
        try:
            passed, rank, error = run_case(case, _make)
        except Exception as exc:  # a broken case must fail loudly, not crash the run
            passed, rank, error = False, None, repr(exc)
        results.append({
            "name": case["name"],
            "passed": bool(passed),
            "rank": rank,
            "has_target": bool(case.get("target")),
            "error": error,
        })
    return {"results": results, **_summarise(results)}


# ---------------------------------------------------------------------- #
# intervention runner
# ---------------------------------------------------------------------- #
def run_intervention_case(case: Dict[str, Any], memory_factory=None):
    """Seed the case history, then ask for a precedent check."""
    mem = memory_factory() if memory_factory else Memory(":memory:")
    try:
        for item in case.get("setup", []):
            rec = mem.remember(
                type=item["type"],
                entity=item["entity"],
                content=item["content"],
                structured=dict(item.get("structured") or {}),
                action_type=item.get("action_type"),
            )
            _age(mem, rec.id, item.get("age_days", 0))
            if item.get("outcome"):
                mem.record_outcome(rec.id, **item["outcome"])
        return mem.before(case["action"], context=case.get("context"), entity=case.get("entity"))
    finally:
        mem.close()


def _judge_intervention(precedent, expect: Dict[str, Any]) -> bool:
    if "verdict" in expect and precedent.verdict != expect["verdict"]:
        return False
    if "attempts" in expect and precedent.attempts != expect["attempts"]:
        return False
    if "failures" in expect and precedent.failures != expect["failures"]:
        return False
    if "successes" in expect and precedent.successes != expect["successes"]:
        return False
    if "needs_attention" in expect and precedent.needs_attention != expect["needs_attention"]:
        return False
    if "lesson_contains" in expect:
        if not any(expect["lesson_contains"] in lesson for lesson in precedent.lessons):
            return False
    if "lesson_count" in expect and len(precedent.lessons) != expect["lesson_count"]:
        return False

    payload = precedent.to_dict()
    if "precedent_entity" in expect:
        if not payload["precedents"] or payload["precedents"][0]["entity"] != expect["precedent_entity"]:
            return False
    if "precedent_count_min" in expect:
        if len(payload["precedents"]) < expect["precedent_count_min"]:
            return False
    if "prompt_contains" in expect:
        if any(token not in payload["prompt"] for token in expect["prompt_contains"]):
            return False
    if "prompt_max_len" in expect and len(payload["prompt"]) > expect["prompt_max_len"]:
        return False
    return True


def run_intervention_eval(memory_factory=None) -> Dict[str, Any]:
    results = []
    for case in INTERVENTION_CASES:
        try:
            precedent = run_intervention_case(case, memory_factory)
            passed, error = _judge_intervention(precedent, case["expect"]), None
        except Exception as exc:
            passed, error = False, repr(exc)
        results.append({"name": case["name"], "passed": bool(passed), "error": error})

    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    return {
        "total": total,
        "passed": passed,
        "accuracy": round(passed / total * 100, 1) if total else 0.0,
        "results": results,
    }


def latency_p95(memory_factory: Optional[Callable[[], Memory]] = None,
                n: int = 5000, runs: int = 50) -> Optional[float]:
    """p95 recall latency over ``n`` records (200 entities, 25 records each)."""
    mem = memory_factory() if memory_factory else Memory(":memory:")
    try:
        if not hasattr(mem.store, "insert_many"):
            return None
        records = [
            MemoryRecord.create(
                "episode", "供应商%d" % (i % 200),
                "第 %d 次到货延迟记录，需要跟进" % i,
                structured={"event": "到货延迟", "impact": "1天"}, importance=0.6,
            )
            for i in range(n)
        ]
        mem.store.insert_many(records)
        for _ in range(5):
            mem.recall(entity="供应商7", query="到货延迟", top_k=5)
        samples = []
        for _ in range(runs):
            started = time.perf_counter()
            mem.recall(entity="供应商7", query="到货延迟", top_k=5)
            samples.append((time.perf_counter() - started) * 1000.0)
        samples.sort()
        return round(samples[max(0, int(len(samples) * 0.95) - 1)], 2)
    finally:
        mem.close()


def run_eval(
    memory_factory: Optional[Callable[[], Memory]] = None,
    include_baselines: bool = True,
    latency: bool = True,
    intervention: bool = True,
) -> Dict[str, Any]:
    """Run every case against a fresh in-memory store."""
    report = _evaluate(memory_factory)
    report["p95_latency_ms"] = latency_p95(memory_factory) if latency else None
    report["intervention"] = run_intervention_eval(memory_factory) if intervention else None

    if include_baselines:
        report["baselines"] = {
            name: _evaluate(memory_factory, factory)
            for name, factory in BASELINES.items()
        }
    return report


def _print_report(report: Dict[str, Any]) -> None:
    for item in report["results"]:
        mark = "PASS" if item["passed"] else "FAIL"
        where = "" if item["rank"] is None else "  (rank %d)" % item["rank"]
        print("%s  %s%s" % (mark, item["name"], where))
    print()
    print("recall@5 : %5.1f%%" % report["recall_at_5"])
    print("MRR      : %5.3f" % report["mrr"])
    print("negatives: %d/%d" % (report["negatives_passed"], report["negatives_total"]))
    print("cases    : %d/%d passed" % (report["passed"], report["total"]))
    if report.get("p95_latency_ms") is not None:
        print("p95 recall latency (5k records): %.2f ms" % report["p95_latency_ms"])

    if report.get("baselines"):
        print()
        print("%-16s %-12s %-8s %-10s" % ("ablation", "recall@5", "MRR", "negatives"))
        print("-" * 48)
        print("%-16s %-12s %-8s %-10s" % (
            "full", "%.1f%%" % report["recall_at_5"], "%.3f" % report["mrr"],
            "%d/%d" % (report["negatives_passed"], report["negatives_total"])))
        for name, base in report["baselines"].items():
            print("%-16s %-12s %-8s %-10s" % (
                name, "%.1f%%" % base["recall_at_5"], "%.3f" % base["mrr"],
                "%d/%d" % (base["negatives_passed"], base["negatives_total"])))

    if report.get("intervention"):
        block = report["intervention"]
        print()
        print("intervention (before-the-decision cards): %d/%d passed" % (
            block["passed"], block["total"]))
        for item in block["results"]:
            print(("PASS  " if item["passed"] else "FAIL  ") + item["name"])


if __name__ == "__main__":
    _print_report(run_eval())
