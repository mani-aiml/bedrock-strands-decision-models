"""Offline tests: the notebook's routing rules, Jev cache and router hook, against a fake Jev and scripted models.
No API calls."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nbformat
import pytest
from strands.models import Model

from request_set import REQUESTS, TIERS


class FakeClaude(Model):
    """Answers every call with fixed text and counts the calls."""

    def __init__(self, name: str) -> None:
        self.name, self.calls = name, 0

    def update_config(self, **_: Any) -> None:
        pass

    def get_config(self) -> dict:
        return {}

    async def structured_output(self, *_: Any, **__: Any):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **_: Any):
        self.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": f"answered by {self.name}"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120}, "metrics": {"latencyMs": 1}}}


class FakeJev:
    """Returns a scripted decision per request text and counts the calls."""

    def __init__(self, decisions: dict[str, tuple[str, float, float]]) -> None:
        self.decisions, self.calls = decisions, 0

    def system_one(self, state: dict, questions: dict) -> SimpleNamespace:
        self.calls += 1
        tier, confidence, one_way = self.decisions.get(state["REQUEST"], ("general", 0.9, 0.0))
        return SimpleNamespace(
            answers={"tier": SimpleNamespace(choice=tier, confidence=confidence, probabilities={tier: confidence}),
                     "one_way_door": SimpleNamespace(noul=one_way)},
            usage=SimpleNamespace(input_tokens=50),
        )


@pytest.fixture
def nb(tmp_path: Path) -> dict[str, Any]:
    """The notebook's setup and definition cells, everything before the first cell that calls Jev or Claude."""
    cells = [c.source for c in nbformat.read(Path(__file__).parent / "routing.ipynb", as_version=4).cells if c.cell_type == "code"]
    scope: dict[str, Any] = {}
    for source in cells[: next(i for i, s in enumerate(cells) if s.startswith("decisions = ask_jev"))]:
        # The test's purpose is to run this repo's own notebook code; no external input reaches exec.
        exec(source, scope)  # nosec B102  # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
    scope["JEV_CACHE"] = tmp_path / "jev_answers.json"
    return scope


def decision(tier: str = "general", confidence: float = 0.9, one_way: float = 0.0) -> dict[str, Any]:
    return {"tier": tier, "confidence": confidence, "one_way_door": one_way}


@pytest.mark.parametrize("d, expected", [
    (decision("general", 0.95), "sonnet"),
    (decision("big_decision", 0.95), "opus"),
    (decision("general", 0.65), "opus"),             # 50 to 70 percent: the stronger model
    (decision("general", 0.45), "human"),            # below 50 percent: a person
    (decision("one_way_door", 0.99), "human"),       # a one-way door, whatever the confidence
    (decision("general", 0.99, one_way=0.8), "human"),  # the separate flag catches what the tier missed
])
def test_routing_rules(nb, d, expected) -> None:
    assert nb["route"](d)[0] == expected


def test_jev_is_called_once_per_request_then_cached(nb) -> None:
    jev = FakeJev({})
    first = nb["ask_jev"](REQUESTS[:5], jev)
    again = nb["ask_jev"](REQUESTS[:5], jev)
    assert jev.calls == 5 and first == again
    assert set(json.loads(nb["JEV_CACHE"].read_text())) == {r.id for r in REQUESTS[:5]}


def test_router_hook_sends_each_route_to_the_right_place(nb) -> None:
    by_id = nb["BY_ID"]
    decisions = {"R-01": decision("general", 0.9), "R-21": decision("big_decision", 0.9), "R-34": decision("one_way_door", 0.9)}
    models = {"sonnet": FakeClaude("sonnet"), "opus": FakeClaude("opus")}
    router = nb["JevRouter"](decisions, models)
    runs = {rid: nb["answer"](by_id[rid], router) for rid in decisions}
    assert runs["R-01"].answer == "answered by sonnet" and runs["R-01"].usd > 0
    assert runs["R-21"].answer == "answered by opus"
    assert runs["R-34"].route == "human" and runs["R-34"].usd == 0 and runs["R-34"].answer.startswith("Escalated to a human")
    assert models["sonnet"].calls == 1 and models["opus"].calls == 1   # the human route never reached a model
    assert [t["request"] for t in router.tickets] == ["R-34"]


def test_request_set_is_labelled_and_unique() -> None:
    assert len(REQUESTS) == 40 and len({r.id for r in REQUESTS}) == 40
    assert {t: sum(r.tier == t for r in REQUESTS) for t in TIERS} == {"general": 20, "big_decision": 12, "one_way_door": 8}
