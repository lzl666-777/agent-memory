"""Forgetting: memories decay by age and lose to access, not by disk pressure."""

from __future__ import annotations

import math
from typing import List

from .models import MemoryRecord, utcnow
from .store import Store


def decay_score(rec: MemoryRecord, half_life_days: float = 180.0) -> float:
    """importance x recency x access, the value of keeping this memory."""
    age_days = max(0.0, (utcnow() - rec.valid_from).total_seconds() / 86400)
    recency = 0.5 ** (age_days / half_life_days)
    access = 1.0 + math.log1p(rec.access_count)
    return rec.importance * recency * access


def forget(
    store: Store,
    threshold: float = 0.05,
    half_life_days: float = 180.0,
    dry_run: bool = False,
) -> List[MemoryRecord]:
    """Delete (or preview) memories whose decay score fell below ``threshold``."""
    victims = [
        rec
        for rec in store.all(include_superseded=True)
        if decay_score(rec, half_life_days) < threshold
    ]
    if not dry_run:
        for rec in victims:
            store.delete(rec.id)
    return victims
