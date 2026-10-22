"""Intervene before the decision, instead of searching after it.

``recall()`` answers *"what do we know about X?"*. A precedent check answers the
only question that changes behaviour: *"we are about to do X again — what
happened last time?"*.

The difference is the axis. Recall is keyed on the entity; a precedent is keyed
on the **action**. That is what makes a lesson transferable: "we switched
supplier B and it stock-outed" is about ``supplier_switch``, so it is visible
when the agent is about to switch supplier C — which is exactly the mistake this
project exists to prevent.

Outcome status is *declared*, never inferred: the caller passes
``status="failure"`` to :meth:`Memory.record_outcome`. The library does not guess
from free text, and a decision with an outcome but no status is reported as
unclassified rather than silently counted as a success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from .models import MemoryRecord

if TYPE_CHECKING:  # pragma: no cover
    from .memory import Memory

#: Declared outcome statuses. ``unknown`` means "recorded, not classified".
OUTCOME_STATUSES = ("success", "failure", "mixed", "unknown")

#: Below this many recorded outcomes, a rate is not worth quoting.
MIN_ATTEMPTS_FOR_RATE = 2
#: At or above this share of non-clean outcomes, the verdict escalates.
HIGH_RISK_SHARE = 0.5

_VERDICT_ORDER = {
    "no_history": 0,
    "unverified": 1,
    "clear": 2,
    "caution": 3,
    "high_risk": 4,
}

#: Card budgets, so an intervention cannot flood a context window.
MAX_LESSONS = 3
MAX_PRECEDENTS = 3
MAX_TEXT = 180


@dataclass
class Precedent:
    """The answer to "we are about to do this again — what happened last time?"."""

    action_type: str
    verdict: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    mixed: int = 0
    lessons: List[str] = field(default_factory=list)
    records: List[MemoryRecord] = field(default_factory=list)
    hits: List[Any] = field(default_factory=list)
    entity: Optional[str] = None

    # ------------------------------------------------------------------ #
    @property
    def failure_rate(self) -> Optional[float]:
        """Share of recorded outcomes that failed; ``None`` before any outcome."""
        return round(self.failures / self.attempts, 3) if self.attempts else None

    @property
    def needs_attention(self) -> bool:
        return self.verdict in ("caution", "high_risk")

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "verdict": self.verdict,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "mixed": self.mixed,
            "failure_rate": self.failure_rate,
            "lessons": list(self.lessons),
            "precedents": [
                {
                    "id": rec.id,
                    "entity": rec.entity,
                    "decision": rec.structured.get("decision", ""),
                    "outcome": rec.structured.get("outcome", ""),
                    "lesson": rec.structured.get("lesson", ""),
                    "status": rec.structured.get("outcome_status", "unknown"),
                    "at": _month(rec),
                }
                for rec in self._card_records()
            ],
            "prompt": self.to_prompt(),
        }

    # ------------------------------------------------------------------ #
    def to_prompt(self) -> str:
        """A ready-to-inject card. Bounded so it cannot flood the context."""
        head = "PRECEDENT CHECK · %s" % (self.action_type or "(unspecified action)")
        if self.verdict == "no_history":
            return (
                head + "\n"
                "verdict: NO HISTORY — no decision of this kind has been recorded yet.\n"
                "No constraint from memory. Consider recording the outcome afterwards."
            )
        if self.verdict == "unverified":
            return (
                head + "\n"
                "verdict: UNVERIFIED — %d decision%s of this kind, no recorded outcome yet.\n"
                "Rationale on file, but nothing learned from it. Treat as weak evidence."
                % (len(self.records), "" if len(self.records) == 1 else "s")
            )

        lines = [head, "verdict: %s — %s" % (self.verdict.upper(), self._summary())]
        if self.lessons:
            lines.append("lessons from earlier attempts:")
            for lesson in self.lessons[:MAX_LESSONS]:
                lines.append("  - " + _clip(lesson, MAX_TEXT))
        precedents = self._card_records()[:MAX_PRECEDENTS]
        if precedents:
            lines.append("precedents:")
            for index, rec in enumerate(precedents, start=1):
                status = rec.structured.get("outcome_status", "unknown")
                outcome = rec.structured.get("outcome") or "(no outcome recorded)"
                lines.append(
                    "  %d. [%s · %s] %s"
                    % (index, rec.entity, _month(rec), _clip(str(rec.structured.get("decision") or rec.content), MAX_TEXT))
                )
                lines.append("     -> %s (%s)" % (_clip(str(outcome), MAX_TEXT), status))
        if self.needs_attention:
            lines.append("Before acting: say how each failure mode above is avoided or accepted.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def _summary(self) -> str:
        if not self.attempts:
            return "no recorded outcome"
        parts = [
            "%d recorded outcome%s" % (self.attempts, "" if self.attempts == 1 else "s")
        ]
        if self.failures:
            parts.append("%d failed" % self.failures)
        if self.mixed:
            parts.append("%d partly" % self.mixed)
        if self.successes:
            parts.append("%d succeeded" % self.successes)
        unclassified = self.attempts - self.failures - self.mixed - self.successes
        if unclassified:
            parts.append("%d unclassified" % unclassified)
        return ", ".join(parts)

    def _card_records(self) -> List[MemoryRecord]:
        """Failures first, then relevance order: a warning belongs at the top."""
        def _ranked(rec: MemoryRecord) -> int:
            return 0 if rec.structured.get("outcome_status") in ("failure", "mixed") else 1

        ordered = sorted(self.hits, key=lambda hit: _ranked(hit.record))
        seen, out = set(), []
        for hit in ordered:
            if hit.record.id in seen:
                continue
            seen.add(hit.record.id)
            out.append(hit.record)
        return out


# ---------------------------------------------------------------------- #
# builder
# ---------------------------------------------------------------------- #
def build(
    memory: "Memory",
    action: str,
    context: Optional[str] = None,
    entity: Optional[str] = None,
    top_k: int = MAX_PRECEDENTS,
) -> Precedent:
    """Collect every recorded decision of ``action`` and judge the next one."""
    action = (action or "").strip()
    records = memory.store.query(types=["decision"], action_type=action) if action else []

    executed = [rec for rec in records if rec.structured.get("outcome")]
    failures = [rec for rec in executed if rec.structured.get("outcome_status") == "failure"]
    mixed = [rec for rec in executed if rec.structured.get("outcome_status") == "mixed"]
    successes = [rec for rec in executed if rec.structured.get("outcome_status") == "success"]

    precedent = Precedent(
        action_type=action,
        verdict=_verdict(records, executed, failures, mixed),
        attempts=len(executed),
        successes=len(successes),
        failures=len(failures),
        mixed=len(mixed),
        lessons=_lessons(records, failures, mixed),
        records=records,
        entity=entity,
    )

    if records:
        # `floor=False`: a precedent is shown, not searched for. Context ranks
        # the history; it never filters it away.
        precedent.hits = memory.retriever.recall(
            types=["decision"],
            action_type=action,
            query=context,
            top_k=top_k,
            floor=False,
        )
    return precedent


def _verdict(
    records: Sequence[MemoryRecord],
    executed: Sequence[MemoryRecord],
    failures: Sequence[MemoryRecord],
    mixed: Sequence[MemoryRecord],
) -> str:
    if not records:
        return "no_history"
    if not executed:
        return "unverified"
    if not failures and not mixed:
        return "clear"
    bad = len(failures) + len(mixed)
    if len(executed) >= MIN_ATTEMPTS_FOR_RATE and bad / len(executed) >= HIGH_RISK_SHARE:
        return "high_risk"
    return "caution"


def _lessons(
    records: Sequence[MemoryRecord],
    failures: Sequence[MemoryRecord],
    mixed: Sequence[MemoryRecord],
) -> List[str]:
    """Lessons from failures first, then the rest, most recent, de-duplicated."""
    flagged = {rec.id for rec in list(failures) + list(mixed)}
    ordered = sorted(
        records,
        key=lambda rec: (rec.id not in flagged, -rec.valid_from.timestamp()),
    )
    seen, lessons = set(), []
    for rec in ordered:
        lesson = (rec.structured.get("lesson") or "").strip()
        if not lesson or lesson in seen:
            continue
        seen.add(lesson)
        lessons.append(lesson)
        if len(lessons) >= MAX_LESSONS:
            break
    return lessons


def _month(rec: MemoryRecord) -> str:
    try:
        return rec.valid_from.strftime("%Y-%m")
    except (AttributeError, ValueError):  # pragma: no cover - naive/odd datetime
        return "?"


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
