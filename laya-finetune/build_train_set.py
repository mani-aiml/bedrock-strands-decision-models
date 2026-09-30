"""Build the held-out TEST set and the TRAINING corpus the same way, with Claude on Bedrock through Strands.

    test   ids 0-499        Haiku writes, the reference judge grades  -> data/test_generated.jsonl, data/test_judged.jsonl
    train  ids 100,000+     Haiku writes; planted labels only          -> data/train_generated.jsonl

Training and test items come from one generator and one set of prompts, so they share a
distribution; ids never collide. Training needs no judge: the planted label is the answer key
(`python build_train_set.py 1000 --judge` adds reference labels anyway, to data/train_judged.jsonl).

Nothing here is shipped. Every stage skips ids it already holds, so a rerun or a failed call
costs time and never money twice, and Strands retries throttled Bedrock calls on its own. Which
topic and defect each id gets is seeded, so a rebuild asks for the same items; the wording Haiku
writes will differ.

Two independent references come out of the test build:
  planted  - the defect the generator was told to introduce (a controlled label)
  gold     - what a frontier judge says, reading only question and answer

    python build_train_set.py test 500
    python build_train_set.py 1000
"""

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from claude_bedrock import GENERATOR_MODEL, GOLD_MODEL, QA, Grade, ask
from judge_task import CRITERIA

DATA = Path(__file__).parent / "data"
GENERATED, JUDGED = DATA / "train_generated.jsonl", DATA / "train_judged.jsonl"
TEST_GENERATED, TEST_JUDGED = DATA / "test_generated.jsonl", DATA / "test_judged.jsonl"
ID_OFFSET = 100_000
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

GOLD_PROMPT = """You are grading an AI-generated answer.

QUESTION: {question}
ANSWER: {answer}

Decide two things:
1. verdict: "pass" if the answer is correct, responsive and free of invented detail; "rework" otherwise.
2. criterion: the single best-fitting label from this list for what is wrong ("none" if nothing is):
{criteria}"""

_LOCK = threading.Lock()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Thread-safe: the stages below write from a thread pool."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def gen_prompt(topic: str, planted: str) -> str:
    return (CLEAN_PROMPT.format(topic=topic) if planted == "none"
            else GEN_PROMPT.format(topic=topic, defect=planted, description=CRITERIA[planted]))


def generate(ids: range, out: Path, concurrency: int = 10) -> None:
    """Generate every id in `ids` that `out` does not hold yet. One in three is planted clean."""
    have = {row["id"] for row in read_jsonl(out)}
    todo = [i for i in ids if i not in have]
    print(f"stage 1: have {len(have)}, generating {len(todo)}")
    defects = [k for k in CRITERIA if k != "none"]

    def one(index: int) -> None:
        rng = random.Random(f"{SEED}:{index}")  # nosec B311: seeded synthetic data, not security
        topic = rng.choice(TOPICS)
        planted = "none" if index % 3 == 0 else rng.choice(defects)
        pair, _ = ask(GENERATOR_MODEL, gen_prompt(topic, planted), QA)
        if pair:
            append_jsonl(out, {"id": index, "topic": topic, "planted": planted, **pair.model_dump()})

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(one, todo))
    print(f"stage 1 complete: {len(read_jsonl(out))} generated in {out.name}")


def gold_label(source: Path, out: Path, concurrency: int = 4) -> None:
    """Grade every item in `source` that `out` has no gold label for yet."""
    done = {row["id"] for row in read_jsonl(out)}
    todo = [row for row in read_jsonl(source) if row["id"] not in done]
    print(f"stage 2: have {len(done)}, labelling {len(todo)}")
    listing = "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())

    def one(item: dict[str, Any]) -> bool:
        prompt = GOLD_PROMPT.format(question=item["question"], answer=item["answer"], criteria=listing)
        grade, usage = ask(GOLD_MODEL, prompt, Grade)
        if grade is None:
            return False
        append_jsonl(out, {
            **item, "gold_verdict": grade.verdict, "gold_criterion": grade.criterion,
            "gold_tokens_in": usage["in"], "gold_tokens_out": usage["out"],
        })
        return True

    with ThreadPoolExecutor(concurrency) as pool:
        failures = sum(not ok for ok in pool.map(one, todo))
    print(f"stage 2 complete: {len(read_jsonl(out))} judged in {out.name}, {failures} failed")


def report(judged: Path) -> None:
    rows = read_jsonl(judged)
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


def build_test_set(n: int = 500) -> None:
    """The held-out items both engines are scored on, with the reference judge's labels."""
    if len(read_jsonl(TEST_JUDGED)) >= n:  # already labelled, possibly copied in by make_data --test-from
        print(f"test set: {TEST_JUDGED.name} holds {n} labelled items")
        return
    generate(range(n), TEST_GENERATED)
    gold_label(TEST_GENERATED, TEST_JUDGED)
    report(TEST_JUDGED)


def build_train_set(n: int, judge: bool = False) -> None:
    generate(range(ID_OFFSET, ID_OFFSET + n), GENERATED)
    if judge:
        gold_label(GENERATED, JUDGED)
        report(JUDGED)


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "test":
        build_test_set(int(sys.argv[2]) if len(sys.argv) > 2 else 500)
    else:
        build_train_set(int(sys.argv[1]), judge="--judge" in sys.argv)
