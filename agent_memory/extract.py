"""Turn free text (a conversation, a note) into structured memories.

Two extractors:

* :class:`RuleExtractor` - zero dependency, regex heuristics, good default.
* :class:`LLMExtractor`  - plug in any OpenAI-compatible completion function
  for much higher recall.

Run::

    from agent_memory import Memory
    from agent_memory.extract import RuleExtractor

    mem = Memory("memory.db")
    mem.remember_from_text("供应商A 交期 45 天，决定换成供应商B", extractor=RuleExtractor("供应商A"))
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Protocol

EXTRACT_PROMPT = """你是记忆抽取器。从下面的文本中抽取值得长期记住的内容。
只输出一个 JSON 数组，不要解释、不要 markdown 围栏。每项格式：
{"type": "fact|episode|decision", "entity": "主体（供应商/商品/渠道）",
 "content": "一句话描述", "action_type": "动作 slug（仅 decision，可空）",
 "structured": {}}

规则：
- fact：稳定属性，structured 用 {"key": "lead_time_days", "value": 45}
- episode：具体事件，structured 用 {"event": "...", "impact": "..."}
- decision：决策，structured 用 {"decision": "...", "rationale": "...", "context": "..."}；
  action_type 必须是稳定的动作 slug，例如 supplier_switch / price_cut / price_raise /
  restock / listing_change / channel_expand。它描述“做了什么”，不是“对谁做的”，
  这样同一类动作的教训才能跨实体复用。
只抽取文本中明确出现的信息，不要编造。

文本：
"""

VALID_TYPES = ("fact", "episode", "decision")


class Extractor(Protocol):
    def extract(self, text: str) -> List[Dict[str, Any]]:
        ...


#: Heuristics for the zero-dependency extractor.  The LLM extractor asks the
#: model for the same slugs; this mapping is deliberately small and explicit.
_ACTION_PATTERNS = (
    (r"(?:更换|换(?:成)?)供应商|供应商切换", "supplier_switch"),
    (r"降价|价格下调|打折", "price_cut"),
    (r"涨价|提价|上调价格", "price_raise"),
    (r"备货|补货", "restock"),
    (r"下架|上架|停售", "listing_change"),
    (r"扩渠道|开渠道|新渠道", "channel_expand"),
)


def infer_action_type(text: str) -> str:
    """Return a stable action slug for a decision sentence (or empty string)."""
    for pattern, action in _ACTION_PATTERNS:
        if re.search(pattern, text):
            return action
    return ""


class RuleExtractor:
    """Zero-dependency heuristic extraction. Good enough for demos and tests."""

    def __init__(self, default_entity: str = "业务"):
        self.default_entity = default_entity

    def extract(self, text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        for match in re.finditer(r"交期\s*(\d+)\s*天", text):
            days = int(match.group(1))
            items.append({
                "type": "fact",
                "entity": self.default_entity,
                "content": f"交期 {days} 天",
                "structured": {"key": "lead_time_days", "value": days},
            })

        for match in re.finditer(r"(决定[^，。；\n]{1,30})", text):
            decision = match.group(1).strip()
            action = infer_action_type(decision)
            items.append({
                "type": "decision",
                "entity": self.default_entity,
                "content": decision,
                "action_type": action,
                "structured": {"decision": decision},
            })

        for match in re.finditer(r"(断货[^，。；\n]{0,12})", text):
            event = match.group(1).strip()
            items.append({
                "type": "episode",
                "entity": self.default_entity,
                "content": event,
                "structured": {"event": event},
            })

        return items


class LLMExtractor:
    """LLM extraction with an injected completion function.

    ``complete`` takes a prompt and returns the model's raw text. This keeps
    the core dependency-free::

        from openai import OpenAI
        client = OpenAI(api_key=..., base_url=...)
        complete = lambda prompt: client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content
        mem.remember_from_text(text, extractor=LLMExtractor(complete))
    """

    def __init__(self, complete: Callable[[str], str]):
        self.complete = complete

    def extract(self, text: str) -> List[Dict[str, Any]]:
        raw = self.complete(EXTRACT_PROMPT + text)
        return parse_json_array(raw)


def parse_json_array(raw: str) -> List[Dict[str, Any]]:
    """Extract the first JSON array from a model response, tolerating fences."""
    if not raw:
        return []
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [
        item
        for item in data
        if isinstance(item, dict)
        and item.get("type") in VALID_TYPES
        and item.get("entity")
        and item.get("content")
    ]
