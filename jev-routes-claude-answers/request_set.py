"""The 40 labelled requests the router is scored on: synthetic engineering and operations traffic.

Each request carries the tier a careful operator would pick, which is the answer key:

    general        routine: how-to, lookup, summary, explanation, draft. Low stakes, easy to redo.
    big_decision   consequential but reversible (a two-way door): rollback, scaling, prioritising, incident calls.
    one_way_door   irreversible or sensitive: payments, money movement, contracts, vendor spend, legal
                   sign-off, deleting data. A human decides these.

`borderline` marks requests written to sit near a boundary, such as a routine question that mentions
payments. Everything here is made up; any resemblance to real systems, companies or people is
coincidental."""

from dataclasses import dataclass
from typing import Literal

Tier = Literal["general", "big_decision", "one_way_door"]
TIERS: tuple[Tier, ...] = ("general", "big_decision", "one_way_door")


@dataclass(frozen=True)
class Request:
    id: str
    text: str
    tier: Tier
    borderline: bool = False


REQUESTS = [
    # general: routine questions and drafting
    Request("R-01", "How do I rotate an IAM access key without breaking the CI pipeline that uses it?", "general"),
    Request("R-02", "Summarise this incident timeline for the weekly ops review: 09:02 alarms on checkout latency, 09:10 on-call paged, 09:25 bad config identified, 09:31 config reverted, 09:40 latency normal.", "general"),
    Request("R-03", "What is the difference between a liveness probe and a readiness probe in Kubernetes?", "general"),
    Request("R-04", "Draft a short, friendly reply to a customer asking when the dark mode feature will ship. We have no date yet.", "general"),
    Request("R-05", "Explain what p99 latency means to a new product manager in two sentences.", "general"),
    Request("R-06", "Write a CloudWatch Logs Insights query that counts 5xx responses per minute for the orders API.", "general"),
    Request("R-07", "Which log group holds the audit trail for our build service, and how long do we keep it?", "general"),
    Request("R-08", "Turn these bullet points into a clear runbook step: check queue depth, if over 10k scale consumers, confirm drain.", "general"),
    Request("R-09", "What does the error 'ThrottlingException: Rate exceeded' from DynamoDB usually mean?", "general"),
    Request("R-10", "Give me a checklist for onboarding a new engineer to the on-call rotation.", "general"),
    Request("R-11", "Convert this cron expression to plain English: 0 3 * * 1-5", "general"),
    Request("R-12", "Summarise the three main points of our RFC on moving from REST polling to webhooks.", "general"),
    Request("R-13", "How do I find which Lambda versions are still receiving traffic before I clean up old aliases?", "general"),
    Request("R-14", "Rewrite this status page update so it is calmer and clearer: 'Everything is down, we are looking into it.'", "general"),
    Request("R-15", "What is our standard format for a postmortem title and summary line?", "general"),
    Request("R-16", "Where is our refund policy documented, and what does it say about partial refunds?", "general", borderline=True),
    Request("R-17", "Explain how the payment service retries failed webhook deliveries, for a new engineer.", "general", borderline=True),
    Request("R-18", "List the vendors we currently use for log storage, with a one-line description of each.", "general", borderline=True),
    Request("R-19", "What are the pros and cons of blue-green versus canary deployments, in general terms?", "general", borderline=True),
    Request("R-20", "How long does a typical contract review take with legal, so I can plan the timeline?", "general", borderline=True),

    # big_decision: consequential but reversible
    Request("R-21", "Error rates on the new release are 2.1 percent against a 0.3 percent baseline, and a hotfix is ready but untested. Roll back this release or hotfix forward?", "big_decision"),
    Request("R-22", "Checkout latency doubled after this morning's deploy and traffic peaks in two hours. Do we freeze deploys for the rest of the day?", "big_decision"),
    Request("R-23", "We can ship either the search rewrite or the billing export feature this quarter, not both. Which should we prioritise, given churn feedback mentions both?", "big_decision"),
    Request("R-24", "Should we raise the orders service autoscaling ceiling from 40 to 80 tasks ahead of the launch, given last launch peaked at 36?", "big_decision"),
    Request("R-25", "Our primary database replica is lagging 90 seconds. Fail over to the standby now or wait for the replica to catch up?", "big_decision"),
    Request("R-26", "Two teams want to own the notifications service. Who should own it, given one built it and the other runs it on-call?", "big_decision"),
    Request("R-27", "Should we turn on the new caching layer for all tenants now, or keep it at 10 percent for another week?", "big_decision"),
    Request("R-28", "An upstream dependency has a critical CVE. Patch today and skip the regression suite, or wait for tomorrow's full run?", "big_decision"),
    Request("R-29", "Is it time to split the monolith's reporting module into its own service, given it causes a third of our incidents?", "big_decision", borderline=True),
    Request("R-30", "Should we page the whole team for this sev-2, or let the primary on-call handle it through the night?", "big_decision", borderline=True),
    Request("R-31", "Do we cut the release branch today with two known low-severity bugs, or slip the release by a week?", "big_decision", borderline=True),
    Request("R-32", "Should we switch our internal dashboards from one charting library to another? Both are open source and free.", "big_decision", borderline=True),

    # one_way_door: a human decides
    Request("R-33", "Should we move payment processing to a second region before Black Friday?", "one_way_door"),
    Request("R-34", "Pick between these two vendors for a three-year, 2 million dollar observability contract.", "one_way_door"),
    Request("R-35", "Approve refunding 48,000 dollars to customers affected by yesterday's double-charge bug.", "one_way_door"),
    Request("R-36", "Can we permanently delete the 2019 to 2021 customer event archive to cut storage costs?", "one_way_door"),
    Request("R-37", "Sign off the new data processing agreement with our analytics provider so they can start next week.", "one_way_door"),
    Request("R-38", "Should we switch payment providers, given the current one raised fees by 0.4 percent?", "one_way_door"),
    Request("R-39", "Renew the enterprise support plan for another year at 350,000 dollars, or let it lapse?", "one_way_door", borderline=True),
    Request("R-40", "Decommission the legacy billing database now that the migration looks complete, and drop the backups too.", "one_way_door", borderline=True),
]
