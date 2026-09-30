# Laya vs Jev as graders

Grading AI output with a frontier model is accurate but costly at volume. This demo tests whether a small decision
model can take most of that load. Jev (TypeSafe, hosted) and Laya (Convai, open weights, local) grade the same 500
AI-written answers zero-shot, on two questions: pass or rework, and which of 40 failure modes. Both are scored by
agreement with a reference judge, Claude Sonnet 5 on Bedrock. A live cascade then sends only Jev's low-confidence
items to Claude.

## Results

Measured on the original demo's corpus:

| | Jev | Laya |
|---|---|---|
| Pass / rework, agreement with the reference judge | 0.872 | 0.382 (said "pass" on 492 of 500) |
| Which of 40 failure modes | 0.798 | 0.234 |
| One call, median | 114 ms (hosted, includes network) | 10.9 ms (local, Apple silicon) |
| Reference judge's bill removed by the cascade, at 95% agreement | 75.5% | 0% (no confidence cut reaches 95%) |

The latencies are measured differently and are not combined into one speedup. Your rebuilt corpus will give
slightly different numbers.

## How it works

| Piece | Where | Does |
|---|---|---|
| Reference judge | `reference_judge.py` | a Strands agent on Bedrock that returns a typed `Grade` (verdict, criterion) |
| `JevFirst` hook | notebook | on `BeforeInvocationEvent`, asks Jev first; if Jev's confidence clears the cut, keeps Jev's verdict and cancels the Claude call |
| Laya and Jev runs | `run_engines.py`, `nb_helpers.py` | both engines get the identical item and questions in one call |
| Scoring | `compare.py` | agreement, risk-coverage and the cascade maths |

## Prerequisites

- Python 3.12 on an Apple silicon Mac (`laya-mlx`)
- Bedrock model access to Claude Sonnet 5 and Claude Haiku 4.5, and AWS credentials in your environment
- A TypeSafe API key

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add TYPESAFE_API_KEY
jupyter lab laya_vs_jev.ipynb
```

| Step | Cost |
|---|---|
| Build the corpus, first run only (notebook's first cell) | about $5 on Bedrock, 10 to 20 minutes |
| Laya sections | free, local |
| Jev sections (about 1,100 calls) | a fraction of a cent |
| Live reference judge and live cascade | about $0.25 on Bedrock |

To build the corpus without the notebook:

```bash
python build_dataset.py
```

## Data

Nothing is committed. `build_dataset.py` asks Haiku for 500 questions with answers, one in three clean and the rest
with one planted defect, then has Sonnet 5 grade each blind. Topic and defect are seeded per item, and every stage
skips items it already has, so reruns are free.

| File | Contents |
|---|---|
| `data/generated.jsonl` | the 500 items and their planted defects |
| `data/judged.jsonl` | the same items with the reference judge's labels and token counts |

## Files

| File | Contents |
|---|---|
| `laya_vs_jev.ipynb` | the full flow: corpus, Laya, Jev, live reference judge, cascade maths, live cascade |
| `reference_judge.py` | Bedrock model ids, the grading prompt, the typed replies, `ask()` |
| `build_dataset.py` | generates and grades the corpus |
| `judge_task.py` | the two questions and the 40 failure modes, shared by both engines |
| `run_engines.py`, `compare.py`, `nb_helpers.py` | engine runs, scoring and notebook helpers |

## Notes

- Scores are agreement with the reference judge, not correctness. On the original corpus the judge itself matched 92% of planted verdicts and 68% of planted defects.
- Laya shares one token budget across all options of a question. 40 long descriptions overflow it; the notebook's `budget_check` shows by how much.
- The live cascade's confidence cut is chosen on the same run it is applied to, so treat its saving as optimistic.
- The typed `Grade` reply sends its schema with every call, about 1,985 input tokens per grade.
