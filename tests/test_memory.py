import pytest

from agent_memory import Memory
from agent_memory.evaluation import run_eval


def test_remember_and_recall():
    mem = Memory(":memory:")
    mem.remember(type="decision", entity="供应商A", content="换成供应商B",
                 structured={"decision": "换B"})
    hits = mem.recall(entity="供应商A", types=["decision"])
    assert len(hits) == 1
    assert hits[0].record.entity == "供应商A"


def test_conflict_supersedes_old():
    mem = Memory(":memory:")
    old = mem.remember(type="fact", entity="供应商A", content="交期30天",
                       structured={"key": "lead_time_days", "value": 30})
    new = mem.remember(type="fact", entity="供应商A", content="交期45天",
                       structured={"key": "lead_time_days", "value": 45})

    stored_old = mem.get(old.id)
    assert stored_old.valid_to is not None       # closed, not deleted
    assert stored_old.superseded_by == new.id

    hits = mem.recall(entity="供应商A", types=["fact"])
    assert len(hits) == 1
    assert hits[0].record.structured["value"] == 45


def test_different_entities_do_not_conflict():
    mem = Memory(":memory:")
    mem.remember(type="fact", entity="A", content="30",
                 structured={"key": "lead_time_days", "value": 30})
    mem.remember(type="fact", entity="B", content="45",
                 structured={"key": "lead_time_days", "value": 45})
    assert len(mem.recall(entity="A", types=["fact"])) == 1
    assert len(mem.recall(entity="B", types=["fact"])) == 1


def test_include_history_returns_superseded():
    mem = Memory(":memory:")
    mem.remember(type="fact", entity="A", content="30",
                 structured={"key": "k", "value": 30})
    mem.remember(type="fact", entity="A", content="45",
                 structured={"key": "k", "value": 45})
    hits = mem.recall(entity="A", types=["fact"], include_history=True, top_k=10)
    assert len(hits) == 2


def test_record_outcome_raises_importance():
    mem = Memory(":memory:")
    rec = mem.remember(type="decision", entity="A", content="换B",
                       structured={"decision": "换B"})
    before = rec.importance
    mem.record_outcome(rec.id, outcome="断货两周", lesson="先评估备选产能")
    updated = mem.get(rec.id)
    assert updated.structured["outcome"] == "断货两周"
    assert updated.structured["lesson"] == "先评估备选产能"
    assert updated.importance > before
    assert updated.has_outcome


def test_record_outcome_missing_id():
    mem = Memory(":memory:")
    with pytest.raises(KeyError):
        mem.record_outcome("does-not-exist", outcome="x")


def test_forgetting_dry_run_keeps_data():
    mem = Memory(":memory:")
    mem.remember(type="fact", entity="A", content="x", importance=0.01)
    victims = mem.forget(dry_run=True)
    assert len(victims) >= 1
    assert len(mem.all()) >= 1


def test_evaluation_runs():
    report = run_eval()
    assert report["total"] > 0
    assert 0 <= report["accuracy"] <= 100
