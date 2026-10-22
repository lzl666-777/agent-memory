"""The Memory facade: the only class most users need."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .forgetting import forget as _forget
from .models import MemoryRecord, MemoryType
from .retrieval import RetrievedMemory, Retriever
from .store import Store
from .writer import WritePolicy


class Memory:
    """Long-term decision memory backed by SQLite.

    Args:
        path: SQLite file path, or ``":memory:"`` for an ephemeral store.
        embedder: optional ``callable(text) -> vector`` to enable semantic recall.
        weights: optional override of the retrieval signal weights, used by
            ablations (see :mod:`agent_memory.evaluation`).
    """

    def __init__(
        self,
        path: str = "memory.db",
        embedder: Optional[Callable[[str], List[float]]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.store = Store(path)
        self.policy = WritePolicy()
        self.retriever = Retriever(self.store, embedder=embedder, weights=weights)

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    def remember(
        self,
        type: Any,
        entity: str,
        content: str,
        structured: Optional[dict] = None,
        importance: Optional[float] = None,
        confidence: float = 1.0,
        source: str = "",
    ) -> MemoryRecord:
        """Store a memory, resolving conflicts with existing facts."""
        kind = MemoryType(type)
        structured = dict(structured or {})
        if importance is None:
            importance = self.policy.score(kind, content, structured)

        record = MemoryRecord.create(
            kind, entity, content, structured=structured,
            importance=importance, confidence=confidence, source=source,
        )
        existing = self.policy.find_redundant(self.store, record)
        if existing is not None:
            return existing
        conflicts = self.policy.find_conflicts(self.store, record)
        self.store.insert(record)
        for old in conflicts:
            self.policy.supersede(self.store, old, record)
        return record

    def record_outcome(
        self,
        memory_id: str,
        outcome: str,
        lesson: Optional[str] = None,
        outcome_at: Optional[str] = None,
    ) -> MemoryRecord:
        """Attach what actually happened to a past decision."""
        record = self.store.get(memory_id)
        if record is None:
            raise KeyError(f"memory not found: {memory_id}")
        record.structured["outcome"] = outcome
        if lesson:
            record.structured["lesson"] = lesson
        if outcome_at:
            record.structured["outcome_at"] = str(outcome_at)
        # a decision with a known outcome is far more valuable than a bare one
        record.importance = min(1.0, record.importance + 0.15)
        self.store.update(record)
        return record

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def recall(
        self,
        entity: Optional[str] = None,
        query: Optional[str] = None,
        types: Optional[List[str]] = None,
        top_k: int = 5,
        include_history: bool = False,
    ) -> List[RetrievedMemory]:
        return self.retriever.recall(
            entity=entity, query=query, types=types,
            top_k=top_k, include_history=include_history,
        )

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        return self.store.get(memory_id)

    def all(self, include_history: bool = True) -> List[MemoryRecord]:
        return self.store.all(include_superseded=include_history)

    # ------------------------------------------------------------------ #
    # maintenance
    # ------------------------------------------------------------------ #
    def forget(
        self,
        threshold: float = 0.05,
        half_life_days: float = 180.0,
        dry_run: bool = False,
        include_decisions: bool = False,
        include_history: bool = False,
    ) -> List[MemoryRecord]:
        """Drop stale facts. Decisions and superseded history are protected."""
        return _forget(
            self.store,
            threshold=threshold,
            half_life_days=half_life_days,
            dry_run=dry_run,
            include_decisions=include_decisions,
            include_history=include_history,
        )

    def close(self) -> None:
        self.store.close()
