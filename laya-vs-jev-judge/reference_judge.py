"""The two Claude roles in this demo, on Amazon Bedrock through Strands: the generator that writes
the items and the reference judge that grades them. Both reply with typed Strands structured
output, so there is no JSON to fish out of the text.

Used by build_dataset.py to build the corpus and by the notebook to grade live."""

import os
from typing import Any, Literal

from pydantic import BaseModel, Field
from strands import Agent
from strands.models import BedrockModel

from judge_task import CRITERIA
from run_engines import load_env

load_env()

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
GOLD_MODEL = os.getenv("GOLD_MODEL_ID", "global.anthropic.claude-sonnet-5")
REGION = os.getenv("AWS_REGION", "us-east-1")

GOLD_PROMPT = """You are grading an AI-generated answer.

QUESTION: {question}
ANSWER: {answer}

Decide two things:
1. verdict: "pass" if the answer is correct, responsive and free of invented detail; "rework" otherwise.
2. criterion: the single best-fitting label from this list for what is wrong ("none" if nothing is):
{criteria}"""

LISTING = "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())


class QA(BaseModel):
    """One generated item."""

    question: str
    answer: str


class Grade(BaseModel):
    """The reference judge's two decisions for one item."""

    verdict: Literal["pass", "rework"]
    criterion: Literal[tuple(CRITERIA)] = Field(description="the single best-fitting label, 'none' if nothing is wrong")


def bedrock(model_id: str) -> BedrockModel:
    return BedrockModel(model_id=model_id, region_name=REGION, max_tokens=2000)


def gold_prompt(item: dict[str, Any]) -> str:
    return GOLD_PROMPT.format(question=item["question"], answer=item["answer"], criteria=LISTING)


def ask(
    model_id: str, prompt: str, output: type[BaseModel], hooks: list | None = None, state: dict | None = None
) -> tuple[BaseModel | None, dict[str, int]]:
    """One typed call on a fresh agent, plus what it cost in tokens. `state` reaches the hooks as invocation_state.
    The reply is None if a hook cancelled the call."""
    agent = Agent(model=bedrock(model_id), hooks=hooks or [], callback_handler=None)
    result = agent(prompt, invocation_state=state, structured_output_model=output)
    usage = result.metrics.accumulated_usage
    return result.structured_output, {"in": usage["inputTokens"], "out": usage["outputTokens"]}
