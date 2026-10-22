"""Tests for the intervention layer: a precedent check *before* the decision.

The claim under test is not "recall works" -- that is test_retrieval_regression.py.
It is: the memory can stop the agent from repeating a mistake it made on a
*different* entity, and it says so without guessing.
"""

import json
import os
import sqlite3
import tempfile

import pytest

from agent_memory import Memory
from agent_memory.evaluation import run_intervention_eval


def _memory_with_history() -> Memory:
    mem = Memory(":memory:")
    # the mistake happened on supplier B
    rec = mem.remember(
        type="decision", entity="供应商B", content="决定更换供应商：把供应商B换成供应商C",
        structured={"decision": "换成供应商C", "rationale": "价格更低"},
        action_type="supplier_switch",
    )
    mem.record_outcome(
        rec.id, outcome="供应商C 断货两周", lesson="换供应商前先评估备选产能", status="failure",
    )
    return mem


# --------------------------------------------------------------------- #
# the point of the whole module
# --------------------------------------------------------------------- #
def test_lesson_transfers_across_entities():
    mem = _memory_with_history()
    # the agent is now about to switch supplier A, not B
    card = mem.before("supplier_switch", context="现在要不要把供应商A换成供应商D")
    assert card.verdict == "caution"
    assert card.needs_attention
    assert "换供应商前先评估备选产能" in card.lessons
    assert card.to_dict()["precedents"][0]["entity"] == "供应商B"
    mem.close()


def test_context_ranks_but_does_not_filter():
    """A precedent is shown, not searched for: unrelated wording must not hide it."""
    mem = _memory_with_history()
    card = mem.before("supplier_switch", context="预算怎么算")
    assert card.records and card.to_dict()["precedents"]
    mem.close()


def test_action_types_do_not_mix():
    mem = _memory_with_history()
    assert mem.before("price_cut").verdict == "no_history"
    mem.close()


# --------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------- #
def test_verdict_escalates_when_failures_dominate():
    mem = Memory(":memory:")
    outcomes = [("failure", 200), ("failure", 100), ("success", 50)]
    for index, (status, age) in enumerate(outcomes):
        rec = mem.remember(
            type="decision", entity="供应商%d" % index, content="决定换成另一家供应商",
            structured={"decision": "换"}, action_type="supplier_switch",
        )
        mem.record_outcome(rec.id, outcome="结果 %d" % index, status=status)

    card = mem.before("supplier_switch")
    assert (card.attempts, card.failures, card.successes) == (3, 2, 1)
    assert card.failure_rate == 0.667
    assert card.verdict == "high_risk"
    assert "HIGH_RISK" in card.to_prompt()
    mem.close()


def test_single_failure_is_caution_not_high_risk():
    """One data point is not a rate."""
    mem = _memory_with_history()
    assert mem.before("supplier_switch").verdict == "caution"
    mem.close()


def test_successes_do_not_warn():
    mem = Memory(":memory:")
    for index in range(2):
        rec = mem.remember(
            type="decision", entity="供应商%d" % index, content="决定换供应商",
            structured={"decision": "换"}, action_type="supplier_switch",
        )
        mem.record_outcome(rec.id, outcome="交付正常", status="success")
    card = mem.before("supplier_switch")
    assert card.verdict == "clear"
    assert not card.needs_attention
    mem.close()


def test_decision_without_outcome_is_unverified():
    mem = Memory(":memory:")
    mem.remember(type="decision", entity="供应商A", content="决定降价 5%",
                 structured={"decision": "降5%"}, action_type="price_cut")
    card = mem.before("price_cut")
    assert card.verdict == "unverified"
    assert card.attempts == 0
    assert "UNVERIFIED" in card.to_prompt()
    mem.close()


def test_no_history_is_stated_explicitly():
    mem = Memory(":memory:")
    card = mem.before("channel_expand")
    assert card.verdict == "no_history"
    assert card.to_prompt().startswith("PRECEDENT CHECK")
    assert "NO HISTORY" in card.to_prompt()
    mem.close()


# --------------------------------------------------------------------- #
# card quality
# --------------------------------------------------------------------- #
def test_lessons_are_deduplicated():
    mem = Memory(":memory:")
    for index in range(3):
        rec = mem.remember(
            type="decision", entity="供应商%d" % index, content="决定换供应商",
            structured={"decision": "换"}, action_type="supplier_switch",
        )
        mem.record_outcome(rec.id, outcome="断货", lesson="先评估备选产能", status="failure")
    assert mem.before("supplier_switch").lessons == ["先评估备选产能"]
    mem.close()


def test_card_is_bounded_even_with_verbose_records():
    mem = Memory(":memory:")
    rec = mem.remember(
        type="decision", entity="供应商B", content="决定换供应商" + "，细节" * 200,
        structured={"decision": "换成供应商C" + "很长的理由" * 50},
        action_type="supplier_switch",
    )
    mem.record_outcome(rec.id, outcome="断货" * 100, lesson="先评估产能", status="failure")
    card = mem.before("supplier_switch")
    assert len(card.to_prompt()) < 1200
    mem.close()


def test_to_dict_is_prompt_ready():
    mem = _memory_with_history()
    payload = mem.before("supplier_switch").to_dict()
    assert payload["action_type"] == "supplier_switch"
    assert payload["failure_rate"] == 1.0
    assert payload["prompt"].startswith("PRECEDENT CHECK")
    assert payload["precedents"][0]["status"] == "failure"
    json.dumps(payload)  # must be serialisable for MCP
    mem.close()


# --------------------------------------------------------------------- #
# write-side contract
# --------------------------------------------------------------------- #
def test_outcome_status_is_validated():
    mem = Memory(":memory:")
    rec = mem.remember(type="decision", entity="A", content="决定换供应商", action_type="supplier_switch")
    with pytest.raises(ValueError):
        mem.record_outcome(rec.id, outcome="x", status="probably-fine")
    mem.close()


def test_action_type_is_persisted_and_queryable():
    mem = Memory(":memory:")
    mem.remember(type="decision", entity="供应商A", content="换供应商",
                 action_type="supplier_switch")
    mem.remember(type="decision", entity="供应商A", content="降价", action_type="price_cut")
    hits = mem.recall(types=["decision"], action_type="supplier_switch")
    assert len(hits) == 1
    assert hits[0].record.action_type == "supplier_switch"
    mem.close()


def test_action_type_falls_back_to_structured():
    """Old call sites put it in `structured`; that must keep working."""
    mem = Memory(":memory:")
    rec = mem.remember(type="decision", entity="A", content="换供应商",
                       structured={"action_type": "supplier_switch", "decision": "换"})
    assert rec.action_type == "supplier_switch"
    mem.close()


# --------------------------------------------------------------------- #
# schema migration
# --------------------------------------------------------------------- #
_OLD_SCHEMA = """
CREATE TABLE memories (
    id TEXT PRIMARY KEY, type TEXT NOT NULL, entity TEXT NOT NULL,
    content TEXT NOT NULL, structured TEXT NOT NULL,
    valid_from TEXT NOT NULL, valid_to TEXT, superseded_by TEXT,
    importance REAL NOT NULL, confidence REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0, last_accessed TEXT,
    source TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
"""


def test_migrates_a_database_written_before_action_type_existed():
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "old.db")
    try:
        conn = sqlite3.connect(path)
        conn.executescript(_OLD_SCHEMA)
        conn.execute(
            "INSERT INTO memories (id,type,entity,content,structured,valid_from,valid_to,"
            "superseded_by,importance,confidence,access_count,last_accessed,source,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("old1", "decision", "供应商A", "决定更换供应商",
             json.dumps({"decision": "换", "action_type": "supplier_switch"}),
             "2026-01-01T00:00:00+00:00", None, None, 0.75, 1.0, 0, None, "",
             "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        mem = Memory(path)
        try:
            record = mem.get("old1")
            assert record is not None
            assert record.action_type == "supplier_switch", "structured.action_type was not backfilled"
            # the FTS index is rebuilt from scratch for a database that never had it
            assert len(mem.store.search_fts("供应商")) == 1
        finally:
            mem.close()
    finally:
        for suffix in ("", "-wal", "-shm"):
            candidate = path + suffix
            if os.path.exists(candidate):
                try:
                    os.remove(candidate)
                except OSError:  # pragma: no cover - windows file lock
                    pass
        try:
            os.rmdir(directory)
        except OSError:  # pragma: no cover
            pass


# --------------------------------------------------------------------- #
# the intervention suite itself
# --------------------------------------------------------------------- #
def test_intervention_suite_passes():
    report = run_intervention_eval()
    assert report["accuracy"] == 100.0, [
        item["name"] for item in report["results"] if not item["passed"]
    ]
