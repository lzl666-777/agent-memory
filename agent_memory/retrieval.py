"""Recall as evidence fusion: lexical (bm25) + decision value + importance + freshness.

Why not RRF any more
--------------------
The previous implementation fused a *structured* rank list (a SQL ``WHERE``
result, i.e. arbitrary rowid order) with a lexical list using
``1 / (60 + rank)``. Reciprocally ranked fusion is built for candidate sets in
the thousands: with 7 candidates the whole rank prior spans 1.6%, while any
metadata multiplier spans 100%+. The result was a retriever that ranked by
recency and called it relevance, and it buried the on-topic lesson within three
months (see ``docs/adr/0001-evidence-fusion-over-rrf.md``).

Here every signal is a number in ``[0, 1]`` with an explicit weight, so what
drives a hit is visible in ``RetrievedMemory.signals``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .models import MemoryRecord, MemoryType, utcnow
from .store import Store

#: Weight sets. They sum to 1.0; an embedder moves 0.15 from lexical to semantic.
_WEIGHTS = {
    "lexical": 0.55,
    "value": 0.20,
    "importance": 0.10,
    "freshness": 0.15,
}
_WEIGHTS_WITH_EMBEDDER = {
    "lexical": 0.40,
    "semantic": 0.15,
    "value": 0.20,
    "importance": 0.10,
    "freshness": 0.15,
}


@dataclass
class RetrievedMemory:
    record: MemoryRecord
    score: float
    why: str  # which paths matched, e.g. "lexical+structured+outcome"
    signals: Dict[str, float] = field(default_factory=dict)  # component scores

    def to_dict(self) -> dict:
        data = self.record.to_dict()
        data["score"] = round(self.score, 4)
        data["why"] = self.why
        data["signals"] = {k: round(v, 4) for k, v in self.signals.items()}
        return data


class Retriever:
    """Recall is the product: it must put the memory that changes the next action first."""

    # Class attributes so ablations can be measured from the outside.
    LEXICAL_WEIGHT = _WEIGHTS["lexical"]
    SEMANTIC_WEIGHT = 0.15
    VALUE_WEIGHT = _WEIGHTS["value"]
    IMPORTANCE_WEIGHT = _WEIGHTS["importance"]
    FRESHNESS_WEIGHT = _WEIGHTS["freshness"]

    #: When a query is given, only records that actually match it are returned.
    #: Scope (entity/type) narrows *what may be recalled*; the query decides
    #: *whether anything should be*. Without this floor a question about refunds
    #: returns every record of the entity, because the entity filter matched.
    RELEVANCE_FLOOR = True
    SEMANTIC_FLOOR = 0.30

    #: bm25 is negative and unbounded; this squashes it into [0, 1) monotonically.
    #: Calibrated on the evaluation set (`python -m agent_memory.evaluation`).
    BM25_SCALE = 2.0

    #: bm25 measures *discrimination*, so on a one-record corpus every term has
    #: idf ~ 0 and relevance collapses to zero even for a perfect match. Half the
    #: relevance comes from query-trigram coverage instead, which means the same
    #: thing at 1 record and at 100k.
    COVERAGE_WEIGHT = 0.5

    #: (half-life in days, floor) per type. Facts go stale; a lesson does not --
    #: an old decision is *more* valuable, not less.
    FRESHNESS = {
        MemoryType.FACT: (180.0, 0.0),
        MemoryType.EPISODE: (365.0, 0.20),
        MemoryType.DECISION: (1095.0, 0.60),
    }

    def __init__(
        self,
        store: Store,
        embedder: Optional[Callable[[str], List[float]]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.weights = self._default_weights(embedder is not None)
        if weights is not None:
            self.weights = {k: float(weights.get(k, 0.0)) for k in self.weights}
        self._vec_cache: Dict[tuple, List[float]] = {}

    def _default_weights(self, with_embedder: bool) -> Dict[str, float]:
        if not with_embedder:
            return {
                "lexical": self.LEXICAL_WEIGHT,
                "value": self.VALUE_WEIGHT,
                "importance": self.IMPORTANCE_WEIGHT,
                "freshness": self.FRESHNESS_WEIGHT,
            }
        return {
            "lexical": 0.40,
            "semantic": self.SEMANTIC_WEIGHT,
            "value": self.VALUE_WEIGHT,
            "importance": self.IMPORTANCE_WEIGHT,
            "freshness": self.FRESHNESS_WEIGHT,
        }

    # ------------------------------------------------------------------ #
    def recall(
        self,
        entity: Optional[str] = None,
        query: Optional[str] = None,
        types: Optional[List[str]] = None,
        top_k: int = 5,
        include_history: bool = False,
    ) -> List[RetrievedMemory]:
        query = (query or "").strip()
        allowed = {MemoryType(t) for t in types} if types else None

        candidates: Dict[str, MemoryRecord] = {}
        reasons: Dict[str, List[str]] = {}

        # scope path: entity / type / validity
        if entity or allowed:
            for rec in self.store.query(entity=entity, types=types, include_history=include_history):
                if allowed and rec.type not in allowed:
                    continue
                candidates[rec.id] = rec
                reasons.setdefault(rec.id, []).append("structured")

        # lexical path: bm25, filtered in SQL so the limit applies after filtering
        lexical_raw: Dict[str, float] = {}
        if query:
            for rec, raw in self.store.search_fts_scored(query, entity=entity):
                if allowed and rec.type not in allowed:
                    continue
                if not include_history and not rec.is_current:
                    continue
                candidates[rec.id] = rec
                lexical_raw[rec.id] = raw
                reasons.setdefault(rec.id, []).append("lexical")

        # semantic path: opt-in, in-process vector cache
        semantic: Dict[str, float] = {}
        if self.embedder and query:
            semantic = self._semantic_scores(query, candidates, entity, types, include_history)

        if query and self.RELEVANCE_FLOOR:
            candidates = {
                rid: rec for rid, rec in candidates.items()
                if rid in lexical_raw or semantic.get(rid, 0.0) >= self.SEMANTIC_FLOOR
            }
            reasons = {rid: r for rid, r in reasons.items() if rid in candidates}

        if not candidates:
            return []

        query_grams = self._query_trigrams(query) if query else set()

        hits: List[RetrievedMemory] = []
        for rid, rec in candidates.items():
            relevance = 0.0
            if rid in lexical_raw:
                coverage = self._coverage(query, query_grams, rec.content)
                relevance = self._relevance(lexical_raw[rid], coverage)
            signals = {
                "lexical": relevance,
                "value": self._value(rec),
                "importance": max(0.0, min(1.0, rec.importance)),
                "freshness": self._freshness(rec),
            }
            if "semantic" in self.weights:
                signals["semantic"] = semantic.get(rid, 0.0)
            score = sum(self.weights.get(k, 0.0) * v for k, v in signals.items())
            why = list(reasons.get(rid, []))
            if rec.has_outcome:
                why.append("outcome")
            hits.append(RetrievedMemory(rec, score, "+".join(why), signals))

        hits.sort(key=lambda h: h.score, reverse=True)
        top = hits[:top_k]

        # access bookkeeping feeds the forgetting policy; one commit for the batch
        for hit in top:
            hit.record.access_count += 1
            hit.record.last_accessed = utcnow()
        self.store.touch_many([hit.record.id for hit in top])
        return top

    # ------------------------------------------------------------------ #
    # signals
    # ------------------------------------------------------------------ #
    def _relevance(self, bm25: float, coverage: float) -> float:
        """Map lexical evidence to [0, 1): bm25 magnitude + trigram coverage."""
        bm25_part = 1.0 - math.exp(-self.BM25_SCALE * max(0.0, -bm25))
        return self.COVERAGE_WEIGHT * coverage + (1.0 - self.COVERAGE_WEIGHT) * bm25_part

    @staticmethod
    def _query_trigrams(query: str) -> set:
        chars = [c for c in query if not c.isspace()]
        if len(chars) < 3:
            return set()
        grams, seen = [], set()
        for i in range(len(chars) - 2):
            gram = "".join(chars[i:i + 3])
            if gram not in seen:
                seen.add(gram)
                grams.append(gram)
            if len(grams) >= 48:
                break
        return set(grams)

    @staticmethod
    def _coverage(query: str, grams: set, text: str) -> float:
        if not grams:  # two-character query: the store used a LIKE match
            return 1.0 if query in text else 0.0
        matched = sum(1 for gram in grams if gram in text)
        return matched / len(grams)

    @staticmethod
    def _value(rec: MemoryRecord) -> float:
        """How much this memory can change the next decision."""
        if rec.type is MemoryType.DECISION:
            has_outcome = bool(rec.structured.get("outcome"))
            has_lesson = bool(rec.structured.get("lesson"))
            if has_outcome and has_lesson:
                return 1.0
            if has_outcome:
                return 0.9
            if has_lesson:
                return 0.8
            return 0.4
        if rec.type is MemoryType.EPISODE:
            return 0.3
        return 0.2

    def _freshness(self, rec: MemoryRecord) -> float:
        half_life, floor = self.FRESHNESS[rec.type]
        age_days = max(0.0, (utcnow() - rec.valid_from).total_seconds() / 86400)
        decay = 0.5 ** (age_days / half_life)
        return floor + (1.0 - floor) * decay

    # ------------------------------------------------------------------ #
    # optional semantic path
    # ------------------------------------------------------------------ #
    def _embed(self, rec: MemoryRecord) -> Optional[List[float]]:
        key = (rec.id, hash(rec.content))
        if key not in self._vec_cache:
            try:
                self._vec_cache[key] = self.embedder(rec.content)
            except Exception:  # pragma: no cover - user supplied embedder
                return None
        return self._vec_cache[key]

    def _semantic_scores(
        self, query: str, candidates: Dict[str, MemoryRecord],
        entity: Optional[str], types: Optional[List[str]], include_history: bool,
    ) -> Dict[str, float]:
        try:
            query_vec = self.embedder(query)
        except Exception:  # pragma: no cover - user supplied embedder
            return {}
        pool = dict(candidates)
        if not pool:
            # semantic-only search: no scope was given, so scan (bounded)
            for rec in self.store.query(entity=entity, types=types, include_history=include_history):
                pool[rec.id] = rec
        scores: Dict[str, float] = {}
        for rid, rec in pool.items():
            vec = self._embed(rec)
            if vec is None:
                continue
            scores[rid] = max(0.0, self._cosine(query_vec, vec))
        return scores

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
