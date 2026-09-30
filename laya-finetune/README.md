# Fine-tuning Laya on a grading job

A small open model, fine-tuned on one narrow task, can beat a general hosted one on that task. Laya (Convai, 421M
parameters, Apache-2.0) is fine-tuned to grade AI-written answers, pass or rework and which of 40 failure modes, then
scored against Jev (TypeSafe, hosted) on the same 500 held-out items. Most of the gain comes from the data, not the
training code, and the notebook shows each data check that made the difference.

## Results

From the [original demo](https://github.com/mani-aiml/jev-demos/tree/main/laya-finetune), scored against the planted
answer key (Apple M4 Pro):

| | Laya out of the box | Laya fine-tuned (2 seeds) | Jev |
|---|---|---|---|
| Pass / rework | 0.350 | 0.942 to 0.958 | 0.842 |
| Which of 40 failure modes | 0.208 | 0.628 to 0.640 | 0.654 |
| One call, median | about 40 ms (local) | about 40 ms (local) | 110 to 160 ms (hosted) |

Training and test items come from the same generator, and Jev saw none of them. Your rebuilt data will give
slightly different numbers.

## The data lessons

| Problem | Cause | Fix |
|---|---|---|
| The training set contained the test set | Haiku repeats itself within a topic; about 29% of first-pass training questions were near-copies of test questions | drop every row 0.8 or more similar to a test question, before anything else (`prepare_data.py`) |
| Dropping them broke the class mix | clean questions are generic, so most near-copies were clean; the clean share fell from 1 in 3 to about 1 in 5 | weight rows back to the generation design (`--mix design`) |
| Adding data broke it again, one level down | 40 extra examples for 11 defect types doubled their share, and the model over-called them | weight every defect type to an equal share (`--mix balanced`) |

Measure the mix after every change to the data. `train.py` prints and plots the mix it is about to train on and
refuses to start on a skewed one.

## Prerequisites

- Python 3.12 on an Apple silicon Mac (`laya-mlx`; training runs on the Mac's GPU)
- Bedrock model access to Claude Haiku 4.5 and Claude Sonnet 5, and AWS credentials in your environment
- A TypeSafe API key
- About 1 GB of disk for each fine-tuned checkpoint (about 840 MB)

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add TYPESAFE_API_KEY
jupyter lab finetune_laya.ipynb
```

The notebook's first cell builds the data. Its scoring cells expect a fine-tuned checkpoint at `runs/all2082`, which
you train yourself:

```bash
python train.py --train data/train_all.jsonl --mix balanced --out runs/all2082   # about 85 minutes on an M4 Pro
python evaluate.py base                                                       # score Laya out of the box
python evaluate.py runs/all2082                                               # score the fine-tuned checkpoint
```

| Step | Cost |
|---|---|
| Build the data, first run only | about $8 on Bedrock, about 20 minutes |
| Live generation step in the notebook | 1 cent, capped |
| Training and Laya scoring | free, local |
| Jev on 500 items | a fraction of a cent |

## Data

Nothing is committed. `make_data.py` builds every file in order, and each step skips items it already has, so reruns
are free.

```bash
python make_data.py                                                     # everything
python make_data.py --test-from ../laya-vs-jev-judge/data/judged.jsonl  # reuse that demo's test set, saves about $4
```

| Step | Model | Output |
|---|---|---|
| Test set: 500 items, graded by the reference judge | Haiku writes, Sonnet 5 grades | `data/test_judged.jsonl` |
| Training set: 1,000 items, planted labels only | Haiku | `data/train_generated.jsonl` |
| Hybrid: clean pairs and 28 semantic defects, plus 11 code-made surface defects | Haiku, $1.50 cap | `data/hybrid.jsonl` |
| Natural: Haiku versions of the 11 code-made types | Haiku, $0.80 cap | `data/natural.jsonl` |
| Prepare: combine, drop test copies, check the mix | none | `data/train_all.jsonl`, `plots/` |

Training uses the planted labels, so only the test set needs a judge. The dollar caps are enforced by `SpendCap`, a
Strands hook that cancels calls before they reach Claude once the cap is spent.

## Files

| File | Contents |
|---|---|
| `finetune_laya.ipynb` | the full flow: data, test copies, mix checks, training, scoring, Jev, scoreboard |
| `make_data.py` | builds all data, in order |
| `claude_bedrock.py` | Bedrock model ids, typed replies, `ask()`, the `SpendCap` hook |
| `build_train_set.py`, `build_hybrid.py`, `inject_defects.py` | test and training items with planted labels |
| `prepare_data.py`, `mix_report.py` | drop test copies, weight the mix, the mix table and plot |
| `finetune_data.py`, `train.py`, `calibrate.py` | training sequences, fine-tuning, temperature fit |
| `evaluate.py`, `jev_timing.py`, `nb_helpers.py` | scoring Laya and Jev, notebook helpers |
| `build_notebook.py` | regenerates `finetune_laya.ipynb` |
