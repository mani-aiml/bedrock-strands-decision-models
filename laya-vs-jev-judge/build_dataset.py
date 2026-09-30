"""Build the judged corpus in two resumable stages, with Claude on Bedrock through Strands.

    stage 1  generate (question, answer) with a planted defect  -> data/generated.jsonl
    stage 2  gold-label each one blind                          -> data/judged.jsonl

Nothing here is shipped: the notebook calls `main()` on its first run, and since both stages skip
ids they already hold, a later run (or a failed call) costs time and never money twice.
Which topic and which defect each id gets is seeded, so a rebuild asks for the same items; the
wording Haiku writes will differ.
Strands retries throttled Bedrock calls on its own (6 attempts, backing off from 4 s).

Two independent references come out of this:
  planted  - the defect the generator was told to introduce (a controlled label)
  gold     - what a frontier judge says, reading only question and answer

The video quotes agreement with `gold`. `planted` is the honesty check: if the frontier
judge cannot recover a defect it was never told about, no small model will either.
"""

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from judge_task import CRITERIA
from reference_judge import GENERATOR_MODEL, GOLD_MODEL, QA, Grade, ask, gold_prompt

DATA = Path(__file__).parent / "data"
GENERATED, JUDGED = DATA / "generated.jsonl", DATA / "judged.jsonl"
SEED = 20260922

TOPICS = [
    "LLM inference and serving", "retrieval and vector search", "AI agents and tool use",
    "Python performance", "distributed systems", "SQL and data modelling",
    "Kubernetes and deployment", "security and authentication", "statistics for ML",
    "GPU hardware", "evaluation and benchmarking", "networking",
]

GEN_PROMPT = """Write one realistic technical QUESTION about {topic}, then an ANSWER to it.

The answer must contain exactly this defect: {defect} ({description})

Make the defect realistic and not obvious, the kind a competent model actually produces.
The answer should be 2-4 sentences and otherwise fluent and confident."""

CLEAN_PROMPT = """Write one realistic technical QUESTION about {topic}, then a genuinely
correct, complete ANSWER to it, 2-4 sentences."""

LOCK = threading.Lock()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def safe_ask(*args: Any) -> tuple[Any, dict[str, int]]:
    """A failed item is skipped and picked up by the next run."""
    try:
        return ask(*args)
    except Exception:
        return None, {}


def generate(target: int, concurrency: int = 10) -> None:
    """Top the generated corpus up to `target` items."""
    have = {row["id"] for row in read_jsonl(GENERATED)}
    todo = [i for i in range(target) if i not in have]
    print(f"stage 1: have {len(have)}, generating {len(todo)}")
    defects = [k for k in CRITERIA if k != "none"]

    def one(index: int) -> None:
        rng = random.Random(f"{SEED}:{index}")  # nosec B311: seeded synthetic data, not security
        topic = rng.choice(TOPICS)
        planted = "none" if index % 3 == 0 else rng.choice(defects)
        prompt = (
            CLEAN_PROMPT.format(topic=topic) if planted == "none"
            else GEN_PROMPT.format(topic=topic, defect=planted, description=CRITERIA[planted])
        )
        pair, _ = safe_ask(GENERATOR_MODEL, prompt, QA)
        if pair:
            append_jsonl(GENERATED, {"id": index, "topic": topic, "planted": planted, **pair.model_dump()})

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(one, todo))
    print(f"stage 1 complete: {len(read_jsonl(GENERATED))} generated")


def gold_label(concurrency: int = 4) -> None:
    """Grade every generated item that has no gold label yet."""
    done = {row["id"] for row in read_jsonl(JUDGED)}
    todo = [row for row in read_jsonl(GENERATED) if row["id"] not in done]
    print(f"stage 2: have {len(done)}, labelling {len(todo)}")

    def one(item: dict[str, Any]) -> bool:
        grade, usage = safe_ask(GOLD_MODEL, gold_prompt(item), Grade)
        if grade is None:
            return False
        append_jsonl(JUDGED, {
            **item, "gold_verdict": grade.verdict, "gold_criterion": grade.criterion,
            "gold_tokens_in": usage["in"], "gold_tokens_out": usage["out"],
        })
        return True

    with ThreadPoolExecutor(concurrency) as pool:
        failures = sum(not ok for ok in pool.map(one, todo))
    print(f"stage 2 complete: {len(read_jsonl(JUDGED))} judged, {failures} failed")


def report() -> None:
    rows = read_jsonl(JUDGED)
    recovered = sum(1 for r in rows if r["planted"] == r["gold_criterion"])
    agreed = sum(1 for r in rows if (r["planted"] == "none") == (r["gold_verdict"] == "pass"))
    rework = sum(1 for r in rows if r["gold_verdict"] == "rework")
    print(f"\nitems: {len(rows)}   gold says rework: {rework} ({rework/len(rows):.1%})")
    print(f"gold recovered the planted criterion: {recovered}/{len(rows)} ({recovered/len(rows):.1%})")
    print(f"gold verdict matched planted:         {agreed}/{len(rows)} ({agreed/len(rows):.1%})")
    print(f"distinct gold criteria present: {len({r['gold_criterion'] for r in rows})} of {len(CRITERIA)}")
    tin = sum(r.get("gold_tokens_in", 0) for r in rows) / len(rows)
    tout = sum(r.get("gold_tokens_out", 0) for r in rows) / len(rows)
    usd = (tin * 3.0 + tout * 15.0) / 1_000_000
    print(f"measured gold judge: {tin:.0f} in / {tout:.0f} out per item = ${usd*1000:.2f} per 1k items")


def main(target: int = 500) -> None:
    generate(target)
    gold_label()
    report()


if __name__ == "__main__":
    main()
