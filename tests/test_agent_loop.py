"""The closed-loop demo makes a causal claim, so the claim is tested.

Claim: with the same world, the same request and the same policy, recording the
first outcome changes the second decision -- and therefore the second outcome.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _demo():
    path = ROOT / "examples" / "agent_loop.py"
    spec = importlib.util.spec_from_file_location("agent_loop_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _world(demo, capacity_of_d: int = 2800):
    return demo.World([
        demo.Supplier("供应商A", lead_time_days=45, capacity_per_month=8000),
        demo.Supplier("供应商B", lead_time_days=20, capacity_per_month=3000),
        demo.Supplier("供应商C", lead_time_days=50, capacity_per_month=9000),
        demo.Supplier("供应商D", lead_time_days=18, capacity_per_month=capacity_of_d),
    ])


def test_without_learning_the_mistake_repeats():
    demo = _demo()
    episodes = demo.run_scenario(learn=False, world=_world(demo))
    assert [e["outcome"].failed for e in episodes] == [True, True]
    assert episodes[1]["card"].verdict == "no_history"
    assert episodes[1]["decision"].action == "switch"


def test_with_learning_the_second_mistake_is_stopped():
    demo = _demo()
    episodes = demo.run_scenario(learn=True, world=_world(demo))
    # the first mistake still happens: memory is not clairvoyance
    assert episodes[0]["outcome"].failed is True
    # the second does not
    assert episodes[1]["card"].verdict == "caution"
    assert episodes[1]["decision"].action == "decline"
    assert episodes[1]["outcome"].failed is False
    assert "产能" in episodes[1]["decision"].reason


def test_the_card_is_the_only_difference_between_the_two_runs():
    demo = _demo()
    world = _world(demo)
    blind = demo.run_scenario(learn=False, world=world)[1]["prompt"]
    learning = demo.run_scenario(learn=True, world=world)[1]["prompt"]

    # same request, same candidate, same instruction scaffolding
    assert blind.splitlines()[:3] == learning.splitlines()[:3]
    # ... and the one difference is the memory block
    assert "Before acting" not in blind
    assert "Before acting" in learning
    assert "换供应商前先评估备选产能" in learning


def test_memory_does_not_refuse_a_capable_supplier():
    """The check must be a check, not a blanket refusal."""
    demo = _demo()
    episodes = demo.run_scenario(learn=True, world=_world(demo, capacity_of_d=8000))
    assert episodes[1]["decision"].action == "switch"
    assert episodes[1]["outcome"].failed is False
    assert "产能核查通过" in episodes[1]["decision"].reason


def test_demo_is_deterministic():
    demo = _demo()
    first = demo.run_scenario(learn=True, world=_world(demo))
    second = demo.run_scenario(learn=True, world=_world(demo))
    assert [e["decision"].action for e in first] == [e["decision"].action for e in second]
    assert [e["outcome"].text for e in first] == [e["outcome"].text for e in second]
