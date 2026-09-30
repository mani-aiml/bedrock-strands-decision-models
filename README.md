# bedrock-strands-decision-models

Three demos that pair a small, fast decision model (TypeSafe's Jev, or Convai's open-weights Laya) with Claude, built
on the [Strands Agents](https://strandsagents.com) SDK with Claude on Amazon Bedrock. Each shows a cost or speed
win you can reproduce: the decision model handles the high-volume choices, and Claude does only the work that needs it.

Ported from [mani-aiml/jev-demos](https://github.com/mani-aiml/jev-demos). The harness logic lives in Strands hooks,
so the tools and the model calls stay unchanged.

| Demo | What it shows | Strands hooks |
|---|---|---|
| [jev-fetches-claude-writes](jev-fetches-claude-writes/) | A support agent where Jev picks the lookups and Claude writes once: fewer Claude calls, faster and cheaper answers | `RunResult` (metrics, step limit), `JevFetches` (prefetch, fallback) |
| [laya-vs-jev-judge](laya-vs-jev-judge/) | Jev and Laya grading 500 AI-written answers zero-shot against a Claude reference judge, and how much of the judge's bill Jev can remove | `JevFirst` (cascade: skip Claude when Jev is confident) |
| [laya-finetune](laya-finetune/) | Laya fine-tuned on the grading job, scored against Jev, and the data checks that made the difference | `SpendCap` (dollar cap on data generation) |

## Prerequisites

| Requirement | Needed for |
|---|---|
| Python 3.12 | all demos |
| AWS account with Bedrock model access to Claude Sonnet 5 and Claude Haiku 4.5 | all demos |
| AWS credentials in your environment (`aws login`, SSO or a profile) | all demos |
| TypeSafe API key | Jev, in all demos |
| Apple silicon Mac | Laya (`laya-mlx`), in the two Laya demos |

## Setup

Each folder is self-contained, with its own requirements and `.env`.

```bash
cd <demo-folder>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add TYPESAFE_API_KEY
jupyter lab
```

## Configuration

Set these in each folder's `.env`. It is gitignored.

| Variable | Default | Purpose |
|---|---|---|
| `TYPESAFE_API_KEY` | none | Jev |
| `AWS_PROFILE` | your default credentials | Bedrock credentials |
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `BEDROCK_MODEL_ID` | `global.anthropic.claude-sonnet-5` | Claude model for the support agent |
| `GOLD_MODEL_ID` | `global.anthropic.claude-sonnet-5` | Claude model for the reference judge |
| `GENERATOR_MODEL_ID` | `global.anthropic.claude-haiku-4-5-20251001-v1:0` | Claude model that writes the synthetic data |

## Data

No data is committed. Each demo builds its own on first run, and all of it is synthetic.

| Demo | Built by | Model | Cost |
|---|---|---|---|
| jev-fetches-claude-writes | `generate_data.py`, a seeded generator | none | free |
| laya-vs-jev-judge | `build_dataset.py`, from the notebook's first cell | Haiku writes, Sonnet grades | about $5, once |
| laya-finetune | `make_data.py`, from the notebook's first cell | Haiku writes, Sonnet grades the test set | about $8, once |

## Before committing

Clear notebook outputs so recorded results and printed data stay out of git:

```bash
jupyter nbconvert --clear-output --inplace */*.ipynb
```

## Licence

MIT, see [LICENSE](LICENSE).
