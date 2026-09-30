"""The Claude calls behind the training data, on Amazon Bedrock through Strands: the generator
(Haiku) that writes items and the reference judge (Sonnet) that grades them, both replying with
typed Strands structured output, plus `SpendCap`, the hook that keeps generation under a dollar cap.

Used by build_train_set.py and build_hybrid.py, and by the notebook's live generation step."""

import os
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from strands import Agent
from strands.hooks import AfterInvocationEvent, BeforeInvocationEvent, HookProvider, HookRegistry
from strands.models import BedrockModel

from judge_task import CRITERIA

_env = Path(__file__).parent / ".env"
for _line in _env.read_text().splitlines() if _env.exists() else []:  # same reader as nb_helpers._load_env
    if "=" in _line and not _line.startswith("#"):
        _key, _, _value = _line.partition("=")
        os.environ.setdefault(_key.strip(), _value.strip())

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
GOLD_MODEL = os.getenv("GOLD_MODEL_ID", "global.anthropic.claude-sonnet-5")
REGION = os.getenv("AWS_REGION", "us-east-1")
HAIKU_USD = (1.0 / 1e6, 5.0 / 1e6)  # Haiku 4.5 list price per token, in / out


class QA(BaseModel):
    """One generated item."""

    question: str
    answer: str


class Grade(BaseModel):
    """The reference judge's two decisions for one item."""

    verdict: Literal["pass", "rework"]
    criterion: Literal[tuple(CRITERIA)] = Field(description="the single best-fitting label, 'none' if nothing is wrong")


def ask(
    model_id: str, prompt: str, output: type[BaseModel], hooks: list | None = None
) -> tuple[BaseModel | None, dict[str, int]]:
    """One typed call on a fresh agent, plus what it cost in tokens. The reply is None if a hook
    cancelled the call or the call failed, so a resumable stage simply picks the item up next run."""
    agent = Agent(
        model=BedrockModel(model_id=model_id, region_name=REGION, max_tokens=2000),
        hooks=hooks or [], callback_handler=None,
    )
    try:
        result = agent(prompt, structured_output_model=output)
    except Exception as err:
        print(f"{model_id}: {type(err).__name__} {str(err)[:160]}", flush=True)
        return None, {"in": 0, "out": 0}
    usage = result.metrics.accumulated_usage
    return result.structured_output, {"in": usage["inputTokens"], "out": usage["outputTokens"]}


class SpendCap(HookProvider):
    """Keeps a run of Claude calls under a dollar cap without touching the code that makes them.

    Before each invocation: once the cap is reached, cancel it, so Claude is never called.
    After each invocation: add what the call cost. Shared across threads, so calls already in
    flight when the cap is reached still finish and can overshoot it by a few cents."""

    def __init__(self, cap_usd: float, usd_in: float, usd_out: float, spent: float = 0.0) -> None:
        self.cap_usd, self.usd_in, self.usd_out = cap_usd, usd_in, usd_out
        self.spent, self.cancelled, self._lock = spent, 0, threading.Lock()

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self.check)
        registry.add_callback(AfterInvocationEvent, self.bill)

    def check(self, event: BeforeInvocationEvent) -> None:
        with self._lock:
            if self.spent >= self.cap_usd:
                self.cancelled += 1
                event.cancel = f"spend cap ${self.cap_usd:.2f} reached"

    def bill(self, event: AfterInvocationEvent) -> None:
        if event.result is None:
            return
        usage = event.result.metrics.accumulated_usage  # a fresh agent per call, so this is one call
        with self._lock:
            self.spent += usage["inputTokens"] * self.usd_in + usage["outputTokens"] * self.usd_out
