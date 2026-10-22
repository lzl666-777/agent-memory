"""Write policy: importance scoring and conflict detection."""

from __future__ import annotations

from typing import List

from .models import MemoryRecord, MemoryType, utcnow
from .store import Store


class WritePolicy:
    """Decides what deserves to be remembered and resolves contradictions."""

    def score(self, type: MemoryType, content: str, structured: dict) -> float:
        """Rule-based importance in ``[0, 1]``; an LLM scorer can replace it."""
        if type is MemoryType.DECISION:
            base = 0.9 if structured.get("outcome") else 0.75
            if structured.get("lesson"):
                base = min(1.0, base + 0.05)
            return base
        if type is MemoryType.EPISODE:
            return 0.6
        return 0.5

    # ------------------------------------------------------------------ #
    def find_conflicts(self, store: Store, new: MemoryRecord) -> List[MemoryRecord]:
        """Facts about the same entity + key with a different value conflict.

        Only *current* memories are checked: an older value that was already
        superseded stays in history untouched.
        """
        key = new.structured.get("key")
        if key is None:
            return []
        candidates = store.query(
            entity=new.entity, types=[MemoryType.FACT.value], include_history=False
        )
        return [
            old
            for old in candidates
            if old.id != new.id
            and old.structured.get("key") == key
            and old.structured.get("value") != new.structured.get("value")
        ]

    def supersede(self, store: Store, old: MemoryRecord, new: MemoryRecord) -> None:
        """Close the validity interval of ``old`` instead of deleting it."""
        old.valid_to = utcnow()
        old.superseded_by = new.id
        store.update(old)
