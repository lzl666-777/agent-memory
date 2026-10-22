"""Forgetting: stale *facts* fade. Lessons and audit history do not.

The previous implementation applied one decay curve to everything and deleted
whatever fell below the threshold, which meant the highest-value rows -- an
old decision whose outcome nobody had needed yet -- were the first to go, at
~2.2 years. It also deleted superseded rows, destroying the version history
that conflict resolution exists to preserve.

See ``docs/adr/0002-type-aware-forgetting.md``.
"""

from __future__ import annotations

import math
from typing import List, Set

from .models import MemoryRecord, MemoryType, utcnow
from .store import Store

#: A decision carries a lesson; a lesson does not expire. Opt in explicitly.
LESSON_TYPES: Set[MemoryType] = {MemoryType.DECISION}


def decay_score(rec: MemoryRecord, half_life_days: float = 180.0) -> float:
    """importance x recency x access: how much this memory is still worth.

    Type-agnostic on purpose -- it is a pure number, the policy lives in
    :func:`forget`.
    """
    age_days = max(0.0, (utcnow() - rec.valid_from).total_seconds() / 86400)
    recency = 0.5 ** (age_days / half_life_days)
    access = 1.0 + math.log1p(rec.access_count)
    return rec.importance * recency * access


def forget(
    store: Store,
    threshold: float = 0.05,
    half_life_days: float = 180.0,
    dry_run: bool = False,
    include_decisions: bool = False,
    include_history: bool = False,
) -> List[MemoryRecord]:
    """Delete (or preview) memories whose decay score fell below ``threshold``.

    Protected by default:

    * ``decision`` records -- they hold the outcome/lesson the product exists for;
    * superseded records -- they are the audit trail (``valid_to`` set).

    Both can be included explicitly. Deleting a row that another row points at
    through ``superseded_by`` is not allowed to leave a dangling pointer.
    """
    victims: List[MemoryRecord] = []
    for rec in store.all(include_superseded=True):
        if rec.valid_to is not None and not include_history:
            continue
        if rec.type in LESSON_TYPES and not include_decisions:
            continue
        if decay_score(rec, half_life_days) < threshold:
            victims.append(rec)

    if victims and not dry_run:
        ids = [rec.id for rec in victims]
        store.delete_many(ids)
        store.clear_superseded_by(ids)
    return victims
