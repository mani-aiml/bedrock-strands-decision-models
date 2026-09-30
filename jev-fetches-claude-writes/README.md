# Jev fetches, Claude writes

A support agent answers questions from eight read-only lookups. Run the usual way, most of its Claude calls only
decide which record to fetch next. This demo hands that decision to Jev, a model that returns typed decisions with
probabilities in about 200 ms. Claude then answers once, from the facts, with no tools, no schemas and no history.

## Results

One task, both arms, on Bedrock (Claude Sonnet 5):

| | Claude alone | Jev fetches, Claude writes |
|---|---|---|
| Seconds | 16.6 | 8.0 |
| Claude calls | 4 | 1 |
| Input tokens | 6,684 | 513 |
| Cost | $0.0188 | $0.0037 + Jev $0.00012 |

In the [original demo](https://github.com/mani-aiml/jev-demos/tree/main/jev-fetches-claude-writes), over eight tasks
run three times each, the Jev arm was 1.7x faster and 5.8x cheaper, with no fallbacks in 24 runs. The notebook's
last section reruns that comparison.

## How it works

Both arms are plain Strands agents. The lookups are ordinary `@tool` functions and never change. Everything the
harness does is a hook.

| Hook | Event | Does |
|---|---|---|
| `RunResult` | `BeforeModelCallEvent` | counts Claude calls, cancels the call after 8 |
| `RunResult` | `AfterToolCallEvent` | counts lookups |
| `RunResult` | `AfterInvocationEvent` | records tokens, seconds and the answer |
| `JevFetches` | `BeforeInvocationEvent` | asks Jev which lookups to run, runs those scoring 0.7 or more in parallel, repeats while results expose new ids, then rewrites the prompt to the task plus the facts |
| `JevFetches` | `AfterInvocationEvent` | if Claude replies `MISSING:`, reruns the task with the Claude-alone agent |

Only read-only lookups are ever run on Jev's prediction.

## Prerequisites

- Python 3.12
- Bedrock model access to Claude Sonnet 5 and AWS credentials in your environment
- A TypeSafe API key

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add TYPESAFE_API_KEY
pytest test_notebook.py           # offline, no API calls
jupyter lab speed_gain.ipynb
```

| Cells | Cost |
|---|---|
| Tests | free |
| One task, each arm | a few cents |
| Eight tasks, three repeats, both arms (48 runs) | about $0.50 on Bedrock, under half a cent on Jev |

## Data

`generate_data.py` builds the catalog from a seed, with no model. It plants the situation each task in `tasks.json`
needs (an EU order held at customs past 10 days, a double charge, a switch under warranty with none in stock, and so
on) and draws names, dates, amounts, carriers and stock levels at random. `check()` verifies every cross-reference
and every planted situation. `tools.py` builds the catalog in memory from `DATA_SEED` (default `0`).

```bash
python generate_data.py                                          # summary of the seed-0 catalog
python generate_data.py --seed 7 --extra 20 --out catalog.json   # another seed, 20 distractor customers, as a file
```

## Files

| File | Contents |
|---|---|
| `speed_gain.ipynb` | both agents, the hooks, one task each way, then 8 tasks x 3 repeats with medians and charts |
| `tools.py` | the eight lookups with a simulated 1.5 s round trip, as plain functions and Strands tools |
| `agent_jev_helper.py` | candidate lookups from the ids seen so far, the question Jev is asked, Jev's cost |
| `generate_data.py` | the seeded catalog generator and its checks |
| `tasks.json` | the eight support tasks |
| `test_notebook.py` | runs the notebook's agents and hooks against a scripted Claude and a fake Jev, and the generator over 200 seeds |

## Notes

- Tool latency is simulated at 1.5 s. Faster tools shrink the time gain and leave the cost gain.
- The `MISSING` fallback catches under-fetching. It cannot catch a wrong answer written from incomplete facts.
- `FETCH_AT = 0.7` is deliberately low: a wrong yes costs one wasted read-only lookup, a wrong no costs a fallback.
- Jev reads its question literally. "Is this one of the lookups to run now, possibly alongside others?" works far better than "Is this the next lookup?"
