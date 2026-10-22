"""MCP server: expose agent-memory to Claude Desktop, Cursor, and any MCP client.

Run::

    pip install "agent-memory[mcp]"
    agent-memory-mcp

Claude Desktop config::

    {
      "mcpServers": {
        "agent-memory": {
          "command": "agent-memory-mcp",
          "env": {"AGENT_MEMORY_DB": "/path/to/memory.db"}
        }
      }
    }

Tools: ``before_decision`` (check the record before acting), ``remember``,
``recall``, ``record_outcome``.
"""

from __future__ import annotations

import os


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "The mcp package is required: pip install \"agent-memory[mcp]\""
        ) from exc

    from .memory import Memory

    db_path = os.getenv("AGENT_MEMORY_DB", "memory.db")
    mem = Memory(db_path)
    mcp = FastMCP("agent-memory")

    @mcp.tool()
    def remember(
        type: str,
        entity: str,
        content: str,
        structured: dict | None = None,
        action_type: str | None = None,
    ) -> dict:
        """Store a memory.

        Args:
            type: one of "fact", "episode", "decision".
            entity: the subject, e.g. a supplier, product or channel.
            content: natural-language description.
            structured: optional fields; for a decision use
                {"decision": ..., "rationale": ..., "context": ...}.
            action_type: for a decision, what was done, as a stable slug
                ("supplier_switch", "price_cut"). This is what makes the lesson
                transferable to other entities.
        """
        record = mem.remember(
            type=type, entity=entity, content=content,
            structured=structured, action_type=action_type,
        )
        return record.to_dict()

    @mcp.tool()
    def before_decision(
        action_type: str,
        context: str | None = None,
        entity: str | None = None,
        top_k: int = 3,
    ) -> dict:
        """Check the recorded history of an action *before* doing it.

        Call this before an irreversible or previously-regretted action. The
        answer is keyed on the action, so lessons from other entities apply.
        Returns a verdict (no_history / unverified / clear / caution /
        high_risk), the lessons, the precedents, and a ready-to-inject `prompt`.
        """
        return mem.before(action_type, context=context, entity=entity, top_k=top_k).to_dict()

    @mcp.tool()
    def recall(
        entity: str | None = None,
        query: str | None = None,
        types: list[str] | None = None,
        top_k: int = 5,
        include_history: bool = False,
        action_type: str | None = None,
    ) -> list[dict]:
        """Recall memories relevant to the current situation.

        Returns each memory with ``why`` (which paths matched) and ``signals``
        (how much each signal contributed to the score).
        """
        hits = mem.recall(
            entity=entity, query=query, types=types,
            top_k=top_k, include_history=include_history,
            action_type=action_type,
        )
        return [hit.to_dict() for hit in hits]

    @mcp.tool()
    def record_outcome(
        memory_id: str,
        outcome: str,
        lesson: str | None = None,
        status: str | None = None,
    ) -> dict:
        """Attach what actually happened to a past decision.

        Args:
            status: "success" | "failure" | "mixed" | "unknown". Declared, not
                inferred -- `before_decision` only counts what was classified.
        """
        return mem.record_outcome(
            memory_id, outcome=outcome, lesson=lesson, status=status,
        ).to_dict()

    mcp.run()


if __name__ == "__main__":
    main()
