from agent_memory import Memory
from agent_memory.extract import LLMExtractor, RuleExtractor, parse_json_array


def test_rule_extractor_finds_all_three_types():
    text = "供应商A 交期 45 天，决定换成供应商B，结果 B 断货两周"
    items = RuleExtractor("供应商A").extract(text)
    assert {i["type"] for i in items} == {"fact", "decision", "episode"}
    fact = next(i for i in items if i["type"] == "fact")
    assert fact["structured"] == {"key": "lead_time_days", "value": 45}


def test_parse_json_array_strips_code_fence():
    raw = '```json\n[{"type":"fact","entity":"A","content":"x","structured":{}}]\n```'
    items = parse_json_array(raw)
    assert len(items) == 1
    assert items[0]["entity"] == "A"


def test_parse_json_array_drops_invalid_items():
    raw = (
        '[{"type":"nope","entity":"A","content":"x"},'
        '{"type":"fact","entity":"A","content":"y"}]'
    )
    items = parse_json_array(raw)
    assert len(items) == 1
    assert items[0]["type"] == "fact"


def test_parse_json_array_handles_garbage():
    assert parse_json_array("not json at all") == []
    assert parse_json_array("") == []


def test_llm_extractor_calls_complete_with_prompt():
    seen = []

    def fake_complete(prompt):
        seen.append(prompt)
        return '[{"type":"decision","entity":"供应商A","content":"换成B","structured":{"decision":"换成B"}}]'

    items = LLMExtractor(fake_complete).extract("随便一段对话")
    assert len(items) == 1
    assert seen and "文本" in seen[0]


def test_remember_from_text_stores_and_recalls():
    mem = Memory(":memory:")
    created = mem.remember_from_text(
        "供应商A 交期 45 天，决定换成供应商B",
        extractor=RuleExtractor("供应商A"),
    )
    assert len(created) >= 2
    hits = mem.recall(entity="供应商A")
    assert len(hits) >= 1
