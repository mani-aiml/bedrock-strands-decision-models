"""Build extra training data with planted labels only: no judge model anywhere.

    stage 1  Haiku writes clean pairs and the 28 semantic defects   -> data/hybrid_haiku.jsonl
    stage 2  code plants the 11 surface defects on clean pairs      -> data/hybrid.jsonl (with stage 1)

Every Haiku prompt names a topic AND an angle, because Haiku repeats itself when given a topic
alone (284 of the first 998 questions were near-copies of test questions). Stage 1 records
each call's tokens and stops at a dollar cap: a `SpendCap` Strands hook on every Haiku agent
cancels the call once the cap is reached. Rows near any test question are dropped before
anything is written to data/hybrid.jsonl.

    python build_hybrid.py 1.50            # spend cap in USD for stage 1
    python build_hybrid.py 0.80 natural    # Haiku-written versions of the 11 code-made types
"""

import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from build_train_set import TOPICS, append_jsonl, gen_prompt, read_jsonl
from claude_bedrock import GENERATOR_MODEL, HAIKU_USD, QA, SpendCap, ask
from inject_defects import INJECTORS, SWAPPED
from judge_task import CRITERIA
from prepare_data import drop_test_copies

DATA = Path(__file__).parent / "data"
HAIKU_OUT, OUT = DATA / "hybrid_haiku.jsonl", DATA / "hybrid.jsonl"
HAIKU_ID, CODE_ID, NATURAL_ID = 200_000, 300_000, 400_000
N_CLEAN, PER_DEFECT, PER_NATURAL, SEED = 400, 25, 40, 20260925
USD_IN, USD_OUT = HAIKU_USD
SEMANTIC = [k for k in CRITERIA if k != "none" and k not in INJECTORS and k not in SWAPPED]
ANGLES = ["a production incident", "a cost trade-off", "a configuration default", "a migration",
          "a benchmark result", "a security review", "an interview question", "a debugging session",
          "a capacity plan", "a design review", "an on-call runbook", "a vendor comparison",
          "a code review comment", "a postmortem finding", "a scaling limit", "a beginner's confusion"]


def tasks(plan: list[str], id_base: int) -> list[dict[str, Any]]:
    """Deterministic, so a resumed run asks the same prompts under the same ids."""
    rng = random.Random(SEED)  # nosec B311: seeded synthetic data, not security
    return [{"id": id_base + i, "planted": p, "topic": rng.choice(TOPICS), "angle": rng.choice(ANGLES)}
            for i, p in enumerate(plan)]


PLANS = {  # name: (plan, id base, raw Haiku file, filtered output)
    "hybrid": (["none"] * N_CLEAN + [d for d in SEMANTIC for _ in range(PER_DEFECT)], HAIKU_ID, HAIKU_OUT, OUT),
    # Haiku-written versions of the types first taught by code templates, which transferred only partly
    "natural": ([d for d in (*INJECTORS, *SWAPPED) for _ in range(PER_NATURAL)], NATURAL_ID,
                DATA / "natural_haiku.jsonl", DATA / "natural.jsonl"),
}


def stage_haiku(cap_usd: float, plan: list[dict[str, Any]], out: Path, concurrency: int = 10) -> None:
    have = {r["id"] for r in read_jsonl(out)}
    cap = SpendCap(cap_usd, USD_IN, USD_OUT, spent=sum(r["usd"] for r in read_jsonl(out)))
    todo = [t for t in plan if t["id"] not in have]
    print(f"stage 1: have {len(have)} (${cap.spent:.2f}), {len(todo)} to go, cap ${cap_usd:.2f}", flush=True)

    def one(t: dict[str, Any]) -> None:
        pair, usage = ask(GENERATOR_MODEL, gen_prompt(f"{t['topic']}, framed as {t['angle']}", t["planted"]), QA,
                          hooks=[cap])
        if pair:
            usd = usage["in"] * USD_IN + usage["out"] * USD_OUT
            append_jsonl(out, {**t, **pair.model_dump(), "source": "haiku", "usd": usd})

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(one, todo))
    print(f"stage 1 done: {len(read_jsonl(out))} rows, ${cap.spent:.2f} spent, "
          f"{cap.cancelled} calls stopped by the cap", flush=True)


def stage_code(clean: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """PER_DEFECT items per surface defect, each built on a clean pair used at most once."""
    rng = random.Random(SEED)  # nosec B311: seeded synthetic data, not security
    bases = rng.sample(clean, len(clean))
    out, kinds = [], [k for k in (*INJECTORS, *SWAPPED) for _ in range(PER_DEFECT)]
    for i, (kind, base) in enumerate(zip(kinds, bases)):
        q, a = base["question"], base["answer"]
        if kind in SWAPPED:
            same = kind == "answered_different_question"
            other = rng.choice([c for c in clean if c is not base and (c["topic"] == base["topic"]) == same])
            a = other["answer"]
        else:
            q, a = INJECTORS[kind](q, a, rng)
        out.append({"id": CODE_ID + i, "planted": kind, "topic": base["topic"], "question": q,
                    "answer": a, "source": "code", "base_id": base["id"]})
    return out


def main(cap_usd: float, name: str) -> None:
    plan, id_base, raw, out = PLANS[name]
    stage_haiku(cap_usd, tasks(plan, id_base), raw)
    kept, near = drop_test_copies(read_jsonl(raw))
    code = stage_code([r for r in kept if r["planted"] == "none"]) if name == "hybrid" else []
    out.write_text("")
    for r in kept + code:
        append_jsonl(out, r)
    print(f"dropped {len(near)} near-test; wrote {len(kept)} haiku + {len(code)} code rows to {out.name}")


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "hybrid")
