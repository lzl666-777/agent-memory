"""agent-memory: decision memory for AI agents.

Remember what was decided, record what happened, recall the lesson before
the next decision::

    from agent_memory import Memory

    mem = Memory("memory.db")
    rec = mem.remember(
        type="decision",
        entity="供应商A",
        content="决定换成供应商B",
        structured={"decision": "换B", "rationale": "交期延迟"},
    )
    mem.record_outcome(rec.id, outcome="B 断货两周", lesson="换之前先评估备选产能")

    for hit in mem.recall(entity="供应商A", query="要不要换供应商"):
        print(hit.score, hit.why, hit.record.structured.get("lesson"))
"""

from .memory import Memory
from .models import MemoryRecord, MemoryType
from .retrieval import RetrievedMemory
from .store import Store

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "MemoryRecord",
    "MemoryType",
    "Store",
    "RetrievedMemory",
    "__version__",
]
