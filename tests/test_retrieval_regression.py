"""Regression tests for the retrieval, evaluation and forgetting fixes.

Each test here corresponds to a failure that the first version of the project
had and that the 12-case evaluation could not see.
"""

import pytest

from agent_memory import Memory
from agent_memory.evaluation import run_eval
from agent_memory.models import utcnow
from agent_memory.retrieval import Retriever

from datetime import timedelta

ENTITY = "供应商A"


def _aged(mem: Memory, rec_id: str, days: float) -> None:
    rec = mem.store.get(rec_id)
    rec.valid_from = utcnow() - timedelta(days=days)
    mem.store.update(rec)


# --------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------- #
def test_old_lesson_ranks_first_after_two_years():
    """The flagship scenario: an old decision with an outcome must surface.

    The first version ranked this memory last from ~65 days on, because every
    same-entity record matched the FTS entity column and the fused scores tied.
    """
    mem = Memory(":memory:")
    lesson = mem.remember(
        type="decision", entity=ENTITY, content="决定更换供应商：把供应商A换成供应商B",
        structured={"decision": "换成供应商B", "outcome": "B 断货两周", "lesson": "先评估备选产能"},
    )
    _aged(mem, lesson.id, 730)
    for i in range(6):
        mem.remember(
            type="decision", entity=ENTITY, content="决定把付款账期从%d天调整为%d天" % (i, i + 5),
            structured={"decision": "账期%d" % i},
        )

    for age in (90, 365, 730):
        _aged(mem, lesson.id, age)
        hits = mem.recall(entity=ENTITY, query="现在要不要更换供应商", types=["decision"], top_k=7)
        assert hits[0].record.id == lesson.id, "lesson buried at age %dd" % age
    mem.close()


def test_entity_is_not_indexed_in_fts():
    """Indexing `entity` made every same-entity record match every query."""
    mem = Memory(":memory:")
    mem.remember(type="decision", entity=ENTITY, content="决定调整付款账期")
    assert mem.store.search_fts("供应商") == []
    mem.close()


def test_fts_is_ordered_by_bm25():
    mem = Memory(":memory:")
    mem.remember(type="decision", entity=ENTITY, content="决定更换供应商：甲换成乙")
    mem.remember(type="decision", entity=ENTITY, content="供应商甲的报价单")
    hits = mem.store.search_fts_scored("更换供应商")
    assert hits, "expected lexical matches"
    assert hits[0][0].content.startswith("决定更换供应商")
    assert all(hits[i][1] <= hits[i + 1][1] for i in range(len(hits) - 1)), "not sorted by bm25"
    mem.close()


def test_two_character_cjk_query_matches():
    """`断货` / `交期` are two characters; the trigram tokenizer indexes nothing."""
    mem = Memory(":memory:")
    rec = mem.remember(type="episode", entity=ENTITY, content="2026 年 3 月因交期延迟断货两周")
    for query in ("断货", "交期"):
        hits = mem.recall(entity=ENTITY, query=query, types=["episode"])
        assert [h.record.id for h in hits] == [rec.id], query
    mem.close()


def test_unrelated_query_returns_nothing():
    """Scope (entity/type) narrows; the query decides whether anything should come back."""
    mem = Memory(":memory:")
    mem.remember(type="decision", entity=ENTITY, content="决定更换供应商", structured={"decision": "换"})
    assert mem.recall(entity=ENTITY, query="退款政策怎么申请") == []
    mem.close()


def test_score_is_explainable():
    mem = Memory(":memory:")
    rec = mem.remember(
        type="decision", entity=ENTITY, content="决定更换供应商",
        structured={"decision": "换", "outcome": "断货", "lesson": "先评估产能"},
    )
    hit = mem.recall(entity=ENTITY, query="更换供应商")[0]
    assert hit.record.id == rec.id
    assert "lexical" in hit.why and "outcome" in hit.why
    assert 0.0 <= hit.score <= 1.0
    assert set(hit.signals) == {"lexical", "value", "importance", "freshness"}
    mem.close()


# --------------------------------------------------------------------- #
# write policy
# --------------------------------------------------------------------- #
def test_duplicate_fact_is_idempotent():
    mem = Memory(":memory:")
    first = mem.remember(type="fact", entity=ENTITY, content="交期 45 天",
                         structured={"key": "lead_time_days", "value": 45})
    for _ in range(2):
        again = mem.remember(type="fact", entity=ENTITY, content="交期 45 天",
                             structured={"key": "lead_time_days", "value": 45})
        assert again.id == first.id
    assert len(mem.recall(entity=ENTITY, types=["fact"])) == 1
    mem.close()


# --------------------------------------------------------------------- #
# forgetting
# --------------------------------------------------------------------- #
def test_forget_protects_decisions():
    mem = Memory(":memory:")
    rec = mem.remember(type="decision", entity=ENTITY, content="去年的教训",
                       structured={"decision": "x", "outcome": "y", "lesson": "z"})
    _aged(mem, rec.id, 3 * 365)
    assert mem.forget(dry_run=True) == []
    mem.forget()
    assert mem.get(rec.id) is not None
    mem.close()


def test_forget_protects_superseded_history():
    mem = Memory(":memory:")
    old = mem.remember(type="fact", entity=ENTITY, content="交期 30 天",
                       structured={"key": "k", "value": 30})
    mem.remember(type="fact", entity=ENTITY, content="交期 45 天",
                 structured={"key": "k", "value": 45})
    _aged(mem, old.id, 3 * 365)
    mem.forget()
    assert mem.get(old.id) is not None, "audit trail was deleted"
    mem.close()


def test_forget_leaves_no_dangling_pointer():
    mem = Memory(":memory:")
    old = mem.remember(type="fact", entity=ENTITY, content="交期 30 天",
                       structured={"key": "k", "value": 30})
    new = mem.remember(type="fact", entity=ENTITY, content="交期 45 天",
                       structured={"key": "k", "value": 45})
    _aged(mem, old.id, 3 * 365)
    _aged(mem, new.id, 3 * 365)   # the current value is now stale enough to drop

    mem.forget()
    assert mem.get(new.id) is None
    surviving = mem.get(old.id)
    assert surviving is not None and surviving.superseded_by is None
    mem.close()


def test_forget_can_be_forced():
    mem = Memory(":memory:")
    rec = mem.remember(type="decision", entity=ENTITY, content="很久以前的决定",
                       structured={"decision": "x"})
    _aged(mem, rec.id, 3 * 365)
    assert mem.forget(include_decisions=True) != []
    assert mem.get(rec.id) is None
    mem.close()


# --------------------------------------------------------------------- #
# the evaluation itself
# --------------------------------------------------------------------- #
def test_evaluation_passes():
    report = run_eval(include_baselines=False, latency=False)
    assert report["accuracy"] == 100.0, [
        r["name"] for r in report["results"] if not r["passed"]
    ]


def test_random_ranking_cannot_pass_the_evaluation():
    """The first harness scored 100% with a ranker that had no scoring at all."""
    report = run_eval(include_baselines=True, latency=False)
    assert report["recall_at_5"] - report["baselines"]["random"]["recall_at_5"] >= 30


def test_lexical_path_is_load_bearing():
    """Acceptance criterion: removing lexical recall must cost >= 15 points."""
    full = run_eval(include_baselines=False, latency=False)
    try:
        Retriever.LEXICAL_WEIGHT = 0.0
        Retriever.RELEVANCE_FLOOR = False
        ablated = run_eval(include_baselines=False, latency=False)
    finally:
        Retriever.LEXICAL_WEIGHT = 0.55
        Retriever.RELEVANCE_FLOOR = True
    assert full["recall_at_5"] - ablated["recall_at_5"] >= 15
