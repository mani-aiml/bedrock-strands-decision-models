"""Offline tests: the notebook's agents and hooks, run against a scripted Claude and a fake Jev. No API calls."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nbformat
import pytest
from strands.models import Model

import tools
from agent_jev_helper import candidate_calls, question

TASK = "Where is order O-1042?"
LATENCY = 0.2


class FakeClaude(Model):
    """Replies from a script: ("tools", [(name, arg, value), ...]) asks for lookups, ("text", str) answers."""

    def __init__(self, script: list[tuple[str, Any]]) -> None:
        self.script, self.seen = list(script), []

    def update_config(self, **_: Any) -> None:
        pass

    def get_config(self) -> dict:
        return {}

    async def structured_output(self, *_: Any, **__: Any):
        raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **_: Any):
        self.seen.append((json.loads(json.dumps(messages, default=str)), tool_specs))
        kind, body = self.script.pop(0)
        yield {"messageStart": {"role": "assistant"}}
        if kind == "tools":
            for i, (name, arg, value) in enumerate(body):
                yield {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": f"t{len(self.seen)}-{i}"}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({arg: value})}}}}
                yield {"contentBlockStop": {}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": body}}}
            yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use" if kind == "tools" else "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110}, "metrics": {"latencyMs": 1}}}


class FakeJev:
    """Answers every question with a fixed probability per tool name."""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities

    def system_one(self, state: dict, questions: dict) -> SimpleNamespace:
        answers = {k: SimpleNamespace(noul=self.probabilities.get(q.instructions.split("(")[0], 0.0)) for k, q in questions.items()}
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=100))


@pytest.fixture
def nb(monkeypatch) -> dict[str, Any]:
    """The notebook's setup and definition cells, everything before the first cell that calls an API."""
    monkeypatch.setattr(tools, "TOOL_LATENCY_S", LATENCY)
    cells = [c.source for c in nbformat.read(Path(__file__).parent / "speed_gain.ipynb", as_version=4).cells if c.cell_type == "code"]
    scope: dict[str, Any] = {}
    for source in cells[: next(i for i, s in enumerate(cells) if s.startswith("base = run_agent"))]:
        # The test's purpose is to run this repo's own notebook code; no external input reaches exec.
        exec(source, scope)  # nosec B102  # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
    return scope


def script_claude(nb: dict[str, Any], *scripts: list) -> list[FakeClaude]:
    """Each new agent in the run gets the next script."""
    queue, made = iter(scripts), []
    nb["make_model"] = lambda model=None: made.append(FakeClaude(next(queue))) or made[-1]
    return made


def test_candidates_come_from_ids_in_state_and_skip_called() -> None:
    calls = candidate_calls({"task": TASK, "lookups_so_far": []}, {("get_order", "O-1042")})
    assert ("get_shipment", "O-1042") in calls and ("get_order", "O-1042") not in calls
    assert not any(name == "get_customer" for name, _ in calls)
    assert ("get_refund_policy", "EU") in calls


def test_question_names_the_tool_and_asks_for_one_of_now() -> None:
    q = question(("get_shipment", "O-1042"))
    assert q.instructions.startswith("get_shipment(order_id=O-1042) is one of the lookups")
    assert "possibly alongside others" in q.instructions


def test_claude_alone_counts_calls_lookups_and_tokens(nb) -> None:
    script_claude(nb, [
        ("tools", [("get_order", "order_id", "O-1042"), ("get_shipment", "order_id", "O-1042")]),
        ("tools", [("get_refund_policy", "region", "EU")]),
        ("text", "Held at customs."),
    ])
    run = nb["run_agent"](TASK)
    assert (run.claude_calls, run.tool_calls, run.input_tokens, run.answer) == (3, 3, 300, "Held at customs.")
    assert run.seconds < 3 * LATENCY, "the two lookups of the first turn run in parallel"


def test_step_limit_stops_a_model_that_never_answers(nb) -> None:
    script_claude(nb, [("tools", [("get_order", "order_id", "O-1042")])] * 20)
    run = nb["run_agent"](TASK)
    assert run.claude_calls == nb["MAX_STEPS"] and run.answer.startswith("Stopped after")


def test_jev_hook_fetches_facts_and_claude_writes_once_without_tools(nb) -> None:
    nb["TypeSafeClient"] = lambda: FakeJev({"get_shipment": 0.9, "get_order": 0.8})
    claude = script_claude(nb, [("text", "Held at customs in Leipzig.")])
    run, stats, fell_back = nb["run_agent_jev_first"](TASK)
    messages, tool_specs = claude[0].seen[0]
    prompt = messages[0]["content"][0]["text"]
    assert (run.claude_calls, run.tool_calls, fell_back, stats.jev_calls) == (1, 2, False, 2)
    assert not tool_specs and len(messages) == 1
    assert prompt.startswith(f"Task: {TASK}") and "held at customs" in prompt


def test_nothing_is_fetched_below_threshold(nb) -> None:
    nb["TypeSafeClient"] = lambda: FakeJev({"get_shipment": nb["FETCH_AT"] - 0.01})
    script_claude(nb, [("text", "I have no facts.")])
    run, stats, _ = nb["run_agent_jev_first"](TASK)
    assert run.tool_calls == 0 and stats.jev_calls == 1


def test_missing_reply_falls_back_to_claude_alone(nb) -> None:
    nb["TypeSafeClient"] = lambda: FakeJev({"get_shipment": 0.9})
    script_claude(
        nb,
        [("text", "MISSING: the invoice")],
        [("tools", [("get_invoice", "order_id", "O-1042")]), ("text", "Paid, and held at customs.")],
    )
    run, _, fell_back = nb["run_agent_jev_first"](TASK)
    assert fell_back and run.answer == "Paid, and held at customs."
    assert (run.claude_calls, run.tool_calls, run.input_tokens) == (3, 2, 300)
    assert run.seconds >= 2 * LATENCY, "the clock covers Jev's lookups and the fallback"


def test_every_seed_plants_the_situations_the_tasks_need() -> None:
    from generate_data import check, generate

    for seed in range(200):
        check(generate(seed, extra=seed % 30))   # raises on a broken reference or a missing situation


def test_the_catalog_is_reproducible_and_seeds_differ() -> None:
    from generate_data import generate

    assert generate(3) == generate(3)
    assert generate(3)["customers"]["C-17"]["name"] != generate(4)["customers"]["C-17"]["name"]
