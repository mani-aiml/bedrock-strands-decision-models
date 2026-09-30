# Jev routes, Claude answers, humans decide

Sending every request to the strongest model wastes money on routine questions. Sending everything to a mid-size
model under-serves the few that matter. And some decisions, such as payments, contracts or deleting data, should not
be made by any model alone. This demo puts Jev, a model that returns typed decisions with probabilities in about 130
ms, in front of Claude as a Strands hook, and routes each request to one of three places.

![Jev routes requests to Sonnet 5, Opus 5 or a human](assets/jev-routes-claude-answers.gif)

The full 1080p video is [`assets/jev-routes-claude-answers.mp4`](assets/jev-routes-claude-answers.mp4).

## Results

40 labelled engineering and operations requests, each answered three ways:

| | All Sonnet 5 | All Opus 5 | Routed by Jev |
|---|---|---|---|
| One-way doors answered by a model (of 8) | 8 | 8 | **0** |
| Big decisions answered by Sonnet (of 12) | 12 | 0 | 0 |
| Sent to a human | 0 | 0 | 9 |
| Cost for 40 requests | $0.110 | $0.426 | $0.179, **58% cheaper than all Opus** (56% on a second run) |
| Median seconds per answer | 4.9 | 8.0 | 5.4 |

Jev routed 39 of 40 requests to the right tier. The one miss was on the safe side: it sent a reversible release
decision to a human. All 40 Jev calls together cost about $0.0008.

## How it works

| Tier | Route | Example |
|---|---|---|
| General question | Claude Sonnet 5 | "How do I rotate an IAM access key?" |
| Big decision, reversible | Claude Opus 5 | "Roll back this release or hotfix forward?" |
| One-way door: payments, contracts, irreversible | a human | "Pick a vendor for a 3-year, $2M contract" |

One Jev call per request asks two questions: which tier, with a confidence, and a separate yes/no for "one-way
door". The rules run in order:

| # | Rule | Route |
|---|---|---|
| 1 | one-way-door flag 0.5 or more, or Jev picks one-way door | human |
| 2 | confidence below 50% | human |
| 3 | confidence 50% to 70% | Opus 5 |
| 4 | confidence 70% or more | the tier Jev picked |

`JevRouter` is a Strands hook on `BeforeInvocationEvent`. For Sonnet or Opus it sets the agent's model for that
invocation. For a human it files an escalation ticket and cancels the invocation, so Claude is never called. The
agent, its prompt and the calling code are the same for every route.

## Prerequisites

- Python 3.12
- Bedrock model access to Claude Sonnet 5 and Claude Opus 5, and AWS credentials in your environment
- A TypeSafe API key

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add TYPESAFE_API_KEY
pytest test_routing.py            # offline, no API calls
jupyter lab routing.ipynb
```

| Step | Cost |
|---|---|
| Tests | free |
| Jev decisions for 40 requests | 40 calls on the first run, about $0.0008; cached in `data/jev_answers.json`, free after |
| All three arms on Bedrock (about 120 Claude calls) | about $0.75 |
| Threshold sweep | free, reuses cached answers and measured costs |

## The animation

`animation/routing_flow.py` is a [manim](https://www.manim.community) scene built from the recorded run.

```bash
brew install cairo pango pkg-config
pip install -r animation/requirements.txt
manim -qh --fps 60 animation/routing_flow.py RoutingFlow
```

## Files

| File | Contents |
|---|---|
| `routing.ipynb` | Jev's questions and cache, the routing rules, the `JevRouter` and `RunStats` hooks, three arms, safety scorecard, threshold sweep |
| `request_set.py` | the 40 labelled requests: 20 general, 12 big decisions, 8 one-way doors, 11 of them borderline |
| `test_routing.py` | routing rules, the Jev cache and the hook, against a fake Jev and scripted models |
| `animation/routing_flow.py` | the manim scene behind the video |
| `assets/` | the rendered video and GIF |

## Notes

- 40 synthetic requests is a small set, and the thresholds are scored on the same set they were chosen for.
- Costs use Anthropic list prices ($2/$10 per million tokens for Sonnet 5, $5/$25 for Opus 5); Bedrock bills separately. Human review time is not priced.
- Jev's latency is measured on its first call and added to the routed arm; cached reruns do not call Jev.
- The one-way-door flag decides most human routes. R-21 ("roll back or hotfix forward") scored 0.48, close to the 0.5 cut; tune `ONE_WAY_AT` on your own traffic.
