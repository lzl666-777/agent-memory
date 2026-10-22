"""Closed loop: memory changes a decision, and the decision changes the outcome.

Most memory demos stop at retrieval. This one shows the consequence: the same
agent, the same suppliers, the same economics -- and a different result, because
what happened the first time was recorded.

The model is ``scripted_policy``: a deterministic stand-in so this runs offline
with no API key. It reads exactly the string a real model would read (printed by
``build_prompt``), and its single rule is "obey the instruction the card adds".
Swap it for a real model and the loop is unchanged::

    def llm_policy(prompt, request):
        return Decision(**json.loads(my_model(prompt)))

Run:
    python examples/agent_loop.py
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import Memory  # noqa: E402

DEMAND_PER_MONTH = 5000

#: This demo replays two quarters, so the simulated clock is explicit.
MARCH = datetime(2026, 3, 15, tzinfo=timezone.utc)
SEPTEMBER = datetime(2026, 9, 10, tzinfo=timezone.utc)


# ---------------------------------------------------------------------- #
# the world (deliberately simple, and independent of the memory)
# ---------------------------------------------------------------------- #
@dataclass
class Supplier:
    name: str
    lead_time_days: int
    capacity_per_month: int


@dataclass
class Request:
    goal: str
    target: str


@dataclass
class Decision:
    action: str      # "switch" | "verify" | "decline"
    target: str
    reason: str


@dataclass
class Outcome:
    text: str
    failed: bool


class World:
    """A switch only works if the new supplier can carry the demand."""

    def __init__(self, suppliers: List[Supplier], demand: int = DEMAND_PER_MONTH):
        self.suppliers: Dict[str, Supplier] = {s.name: s for s in suppliers}
        self.demand = demand

    def capacity(self, name: str) -> int:
        return self.suppliers[name].capacity_per_month

    def lead_time(self, name: str) -> int:
        return self.suppliers[name].lead_time_days

    def execute(self, decision: Decision) -> Outcome:
        if decision.action != "switch":
            return Outcome("未执行更换（供应商未变）", failed=False)
        capacity = self.capacity(decision.target)
        if capacity < self.demand:
            return Outcome(
                "%s 产能 %d < 需求 %d，断货两周" % (decision.target, capacity, self.demand),
                failed=True,
            )
        return Outcome(
            "%s 产能 %d ≥ 需求 %d，交付正常" % (decision.target, capacity, self.demand),
            failed=False,
        )


# ---------------------------------------------------------------------- #
# the model (stand-in)
# ---------------------------------------------------------------------- #
def build_prompt(request: Request, card, world: World) -> str:
    candidate = request.target
    return "\n".join([
        "You are an e-commerce operations agent.",
        "Goal: " + request.goal,
        "Candidate: %s (lead time %d days, lower price)"
        % (candidate, world.lead_time(candidate)),
        "",
        card.to_prompt() if card is not None else "(no precedent check available)",
    ])


def scripted_policy(prompt: str, request: Request) -> Decision:
    """An instruction-following agent, reduced to its one relevant behaviour.

    The card ends with "Before acting: ..." only when the recorded history
    warrants it. A model would comply with that instruction; so does this.
    """
    if "Before acting" in prompt:
        return Decision("verify", request.target, "卡片要求在动作前回应失败模式")
    return Decision("switch", request.target, "交期更短、价格更低")


# ---------------------------------------------------------------------- #
# the loop
# ---------------------------------------------------------------------- #
def run_episode(
    memory: Memory,
    world: World,
    request: Request,
    learn: bool = True,
    when: Optional[datetime] = None,
) -> dict:
    card = memory.before("supplier_switch", context=request.goal, entity=request.target)
    prompt = build_prompt(request, card, world)
    decision = scripted_policy(prompt, request)

    if decision.action == "verify":
        capacity = world.capacity(decision.target)
        if capacity < world.demand:
            decision = Decision(
                "decline", decision.target,
                "产能核查：%d < 需求 %d" % (capacity, world.demand),
            )
        else:
            decision = Decision(
                "switch", decision.target,
                "产能核查通过：%d ≥ 需求 %d" % (capacity, world.demand),
            )

    outcome = world.execute(decision)

    if learn and decision.action == "switch":
        record = memory.remember(
            type="decision",
            entity=request.target,
            content="决定更换供应商：" + request.goal,
            structured={"decision": "换 " + request.target, "rationale": decision.reason},
            action_type="supplier_switch",
        )
        # the lesson is written by the operator in this demo; a real agent would
        # summarise the outcome itself.
        memory.record_outcome(
            record.id,
            outcome=outcome.text,
            lesson="换供应商前先评估备选产能",
            status="failure" if outcome.failed else "success",
        )
        if when is not None:
            _at(memory, record.id, when)

    return {"request": request, "card": card, "prompt": prompt,
            "decision": decision, "outcome": outcome}


def _at(memory: Memory, record_id: str, when: datetime) -> None:
    """Backdate the record so the card's date matches the simulated quarter."""
    record = memory.store.get(record_id)
    record.valid_from = when
    memory.store.update(record)


def run_scenario(learn: bool, world: World) -> List[dict]:
    """Two quarters of the same agent. `learn` is the only difference."""
    memory = Memory(":memory:")
    try:
        return [
            run_episode(memory, world, Request("把供应商A换成供应商B", "供应商B"),
                        learn=learn, when=MARCH),
            run_episode(memory, world, Request("把供应商C换成供应商D", "供应商D"),
                        learn=learn, when=SEPTEMBER),
        ]
    finally:
        memory.close()


# ---------------------------------------------------------------------- #
# reporting
# ---------------------------------------------------------------------- #
def _print_episode(label: str, episode: dict, show_prompt: bool = False) -> None:
    request, card, decision, outcome = (
        episode["request"], episode["card"], episode["decision"], episode["outcome"],
    )
    print("[%s] %s" % (label, request.goal))
    print("  记忆: verdict=%s" % (card.verdict if card is not None else "未查询"))
    if show_prompt:
        print("  模型看到的提示词:")
        for line in episode["prompt"].splitlines():
            print("    │ " + line)
    print("  决策: %s（%s）" % (decision.action, decision.reason))
    print("  结果: %s" % outcome.text)
    print()


def main() -> None:
    world = World([
        Supplier("供应商A", lead_time_days=45, capacity_per_month=8000),
        Supplier("供应商B", lead_time_days=20, capacity_per_month=3000),
        Supplier("供应商C", lead_time_days=50, capacity_per_month=9000),
        Supplier("供应商D", lead_time_days=18, capacity_per_month=2800),
    ])

    print("=" * 68)
    print("场景 1：不记录结果的 agent（3 月踩坑，9 月再踩一次）")
    print("=" * 68)
    blind = run_scenario(learn=False, world=world)
    _print_episode("2026-03", blind[0])
    _print_episode("2026-09", blind[1], show_prompt=True)

    print("=" * 68)
    print("场景 2：用 agent-memory 记录结果的 agent（同样的世界、同样的策略）")
    print("=" * 68)
    learning = run_scenario(learn=True, world=world)
    _print_episode("2026-03", learning[0])
    _print_episode("2026-09", learning[1], show_prompt=True)

    print("=" * 68)
    print("对比")
    print("=" * 68)
    print("%-18s %-12s %s" % ("", "断货次数", "第二次的依据"))
    print("%-18s %-12s %s" % (
        "不记录结果",
        "%d/%d" % (sum(e["outcome"].failed for e in blind), len(blind)),
        blind[1]["decision"].reason,
    ))
    print("%-18s %-12s %s" % (
        "记录结果",
        "%d/%d" % (sum(e["outcome"].failed for e in learning), len(learning)),
        learning[1]["decision"].reason,
    ))
    print()
    print("记忆不能阻止第一次踩坑；它阻止第二次。")


if __name__ == "__main__":
    main()
