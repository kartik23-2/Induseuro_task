# Question 1 — One LLM call per ticket

**Product:** ShopAssist resolves Tier-1 tickets without a human when possible.

**Starting point:** [question1/agent.py](../question1/agent.py) — class `SupportCopilot`.

The baseline is already correct on the tickets checked by `pytest` (including `T-1091`). Reduce **points per ticket** using the billing rules in the repository README without changing graded fields.

Graded output per ticket: `action`, `kb_article` (and parsing must keep working).

| Ticket | Scenario (summary) |
|--------|---------------------|
| T-1001 | Refund request, physical item, unopened |
| T-1002 | Refund request, digital item, license already revealed |
| T-1003 | Replacement request, customized item, reported damage |
| T-1004 | Replacement request, catalog item, reported damage |

**Target:** ≤ **3000** points per ticket on the public ids above.

**Submit:** changes to `agent.py`, per-call token/points breakdown for one ticket, short write-up.
