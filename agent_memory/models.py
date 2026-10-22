"""Memory data model: facts, episodes and decisions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class MemoryType(str, Enum):
    """The three kinds of long-term memory."""

    FACT = "fact"          # stable attribute: supplier lead time, product cost
    EPISODE = "episode"    # a concrete event: a stock-out, a price change
    DECISION = "decision"  # decision + rationale + outcome + lesson


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def from_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class MemoryRecord:
    """A single memory row.

    ``structured`` conventions::

        fact     -> {"key": "lead_time_days", "value": 45}
        episode  -> {"event": "...", "impact": "...", "at": "2026-03-15"}
        decision -> {"decision_type": "...", "decision": "...",
                     "rationale": "...", "context": "...",
                     "outcome": "...", "lesson": "...", "outcome_at": "..."}
    """

    id: str
    type: MemoryType
    entity: str
    content: str
    structured: Dict[str, Any] = field(default_factory=dict)

    # temporal validity: when this memory became true, and when it stopped
    valid_from: datetime = field(default_factory=utcnow)
    valid_to: Optional[datetime] = None
    superseded_by: Optional[str] = None

    # scoring
    importance: float = 0.5
    confidence: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    # provenance
    source: str = ""
    created_at: datetime = field(default_factory=utcnow)

    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        type: Any,
        entity: str,
        content: str,
        structured: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        source: str = "",
    ) -> "MemoryRecord":
        return cls(
            id=uuid.uuid4().hex[:12],
            type=MemoryType(type),
            entity=entity.strip(),
            content=content.strip(),
            structured=dict(structured or {}),
            importance=float(importance),
            confidence=float(confidence),
            source=source,
        )

    # ------------------------------------------------------------------ #
    @property
    def is_current(self) -> bool:
        """True when the memory has not been superseded (still believed)."""
        return self.valid_to is None

    @property
    def has_outcome(self) -> bool:
        """Decision memories with a recorded outcome are 'experience'."""
        return bool(self.structured.get("outcome"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "entity": self.entity,
            "content": self.content,
            "structured": self.structured,
            "valid_from": to_iso(self.valid_from),
            "valid_to": to_iso(self.valid_to),
            "superseded_by": self.superseded_by,
            "importance": self.importance,
            "confidence": self.confidence,
            "access_count": self.access_count,
            "last_accessed": to_iso(self.last_accessed),
            "source": self.source,
            "created_at": to_iso(self.created_at),
        }
