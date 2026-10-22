"""Multi-path recall: structured + lexical (+ optional semantic) -> RRF -> rerank."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

from .models import MemoryRecord, MemoryType, utcnow
from .store import Store


@dataclass
class RetrievedMemory:
    record: MemoryRecord
    score: float
    why: str  # which paths matched, e.g. "structured+lexical"

    def to_dict(self) -> dict:
        data = self.record.to_dict()
        data["score"] = round(self.score, 4)
        data["why"] = self.why
        return data


class Retriever:
    """Recall is the product: one path is never enough."""

    RRF_K = 60

    def __init__(self, store: Store, embedder: Optional[Callable[[str], List[float]]] = None):
        self.store = store
        self.embedder = embedder

    def recall(
        self,
        entity: Optional[str] = None,
        query: Optional[str] = None,
        types: Optional[List[str]] = None,
        top_k: int = 5,
        include_history: bool = False,
    ) -> List[RetrievedMemory]:
        rank_lists: List[List[str]] = []
        reasons: dict = {}

        # path 1: structured (entity / type / current validity)
        structured = self.store.query(entity=entity, types=types, include_history=include_history)
        if entity or types:
            rank_lists.append([r.id for r in structured])
            for r in structured:
                reasons.setdefault(r.id, []).append("structured")

        # path 2: lexical (FTS5)
        if query and query.strip():
            lexical = self.store.search_fts(query)
            if entity:
                lexical = [r for r in lexical if r.entity == entity]
            if types:
                allowed = {MemoryType(t) for t in types}
                lexical = [r for r in lexical if r.type in allowed]
            if not include_history:
                lexical = [r for r in lexical if r.is_current]
            rank_lists.append([r.id for r in lexical])
            for r in lexical:
                reasons.setdefault(r.id, []).append("lexical")

        # path 3: semantic (opt-in)
        if self.embedder and query and query.strip():
            semantic = self._semantic(query, entity, types, include_history)
            rank_lists.append([r.id for r in semantic])
            for r in semantic:
                reasons.setdefault(r.id, []).append("semantic")

        if not rank_lists:
            return []

        # RRF fusion
        fused: dict = {}
        for ranked in rank_lists:
            for rank, rid in enumerate(ranked):
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (self.RRF_K + rank + 1)

        records = {r.id: r for r in self.store.all(include_superseded=include_history)}

        # rerank: relevance * importance * recency * outcome bonus
        hits: List[RetrievedMemory] = []
        for rid, base in fused.items():
            rec = records.get(rid)
            if rec is None:
                continue
            score = base * (0.5 + 0.5 * rec.importance) * self._recency(rec)
            if rec.has_outcome:
                score *= 1.15
            hits.append(RetrievedMemory(rec, score, "+".join(reasons.get(rid, []))))

        hits.sort(key=lambda h: h.score, reverse=True)
        top = hits[:top_k]

        # access bookkeeping feeds the forgetting policy
        for hit in top:
            hit.record.access_count += 1
            hit.record.last_accessed = utcnow()
            self.store.update(hit.record)
        return top

    # ------------------------------------------------------------------ #
    @staticmethod
    def _recency(rec: MemoryRecord) -> float:
        age_days = max(0.0, (utcnow() - rec.valid_from).total_seconds() / 86400)
        return 0.5 ** (age_days / 180.0)

    def _semantic(self, query, entity, types, include_history) -> List[MemoryRecord]:
        try:
            query_vec = self.embedder(query)
        except Exception:
            return []
        scored = []
        for rec in self.store.query(entity=entity, types=types, include_history=include_history):
            try:
                scored.append((self._cosine(query_vec, self.embedder(rec.content)), rec))
            except Exception:
                continue
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [rec for _, rec in scored[:50]]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
