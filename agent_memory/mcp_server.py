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
    def remember(type: str, entity: str, content: str, structured: dict | None = None) -> dict:
        """Store a memory.

        Args:
            type: one of "fact", "episode", "decision".
            entity: the subject, e.g. a supplier, product or channel.
            content: natural-language description.
            structured: optional fields; for a decision use
                {"decision": ..., "rationale": ..., "context": ...}.
        """
        record = mem.remember(type=type, entity=entity, content=content, structured=structured)
        return record.to_dict()

    @mcp.tool()
    def recall(
        entity: str | None = None,
        query: str | None = None,
        types: list[str] | None = None,
        top_k: int = 5,
        include_history: bool = False,
    ) -> list[dict]:
        """Recall memories relevant to the current situation.

        Returns each memory with a ``why`` field explaining which retrieval
        paths matched (structured / lexical / semantic).
        """
        hits = mem.recall(
            entity=entity, query=query, types=types,
            top_k=top_k, include_history=include_history,
        )
        return [hit.to_dict() for hit in hits]

    @mcp.tool()
    def record_outcome(memory_id: str, outcome: str, lesson: str | None = None) -> dict:
        """Attach what actually happened to a past decision."""
        return mem.record_outcome(memory_id, outcome=outcome, lesson=lesson).to_dict()

    mcp.run()


if __name__ == "__main__":
    main()
