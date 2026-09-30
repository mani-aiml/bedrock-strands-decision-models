"""Build every data file the notebook and train.py read, from nothing. Nothing in data/ is shipped.

    1. test set     500 held-out items, Haiku writes, Sonnet 5 grades   data/test_judged.jsonl
    2. train set    1,000 items, Haiku writes, planted labels only       data/train_generated.jsonl
    3. hybrid       Haiku with angles under a $1.50 cap, plus code-made  data/hybrid_haiku.jsonl, data/hybrid.jsonl
    4. natural      Haiku versions of the 11 code-made types, $0.80 cap  data/natural_haiku.jsonl, data/natural.jsonl
    5. prepare      combine, drop test copies, check the mix             data/train_all.jsonl (+ _natural), plots/

Every step is resumable: rerun after an interruption and only the missing items are asked for.
About $8 and 20 minutes on Bedrock the first time. The test set is the same kind of corpus
laya-vs-jev-judge builds; pass `--test-from` to reuse that one instead of paying for a second.

    python make_data.py
    python make_data.py --test-from ../laya-vs-jev-judge/data/judged.jsonl
"""

import argparse
import shutil
from pathlib import Path

import build_hybrid
from build_train_set import TEST_JUDGED, build_test_set, build_train_set
from prepare_data import prepare

SOURCES = ["data/train_generated.jsonl", "data/hybrid_haiku.jsonl", "data/hybrid.jsonl"]


def main(test_from: str | None = None, train: int = 1000, hybrid_cap: float = 1.50, natural_cap: float = 0.80) -> None:
    if test_from:
        if not TEST_JUDGED.exists():
            TEST_JUDGED.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(test_from, TEST_JUDGED)
        print(f"test set: {TEST_JUDGED.name}, copied from {test_from}")
    else:
        build_test_set(500)             # before anything else: every later step drops copies of it
    build_train_set(train)
    build_hybrid.main(hybrid_cap, "hybrid")
    build_hybrid.main(natural_cap, "natural")
    prepare(SOURCES, "data/train_all.jsonl", "plots/mix_train_all.png")
    prepare(SOURCES + ["data/natural_haiku.jsonl"], "data/train_all_natural.jsonl", "plots/mix_train_all_natural.png")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")  # headless: the plots go to files
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test-from", help="an already judged 500-item corpus to use as the test set")
    ap.add_argument("--train", type=int, default=1000, help="training items to generate")
    args = ap.parse_args()
    main(args.test_from, args.train)
    print("data ready:", ", ".join(sorted(p.name for p in Path("data").glob("*.jsonl"))))
