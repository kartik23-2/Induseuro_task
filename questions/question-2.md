# Question 2 — Four-node production pipeline

**Product:** ShopAssist — same routing as Question 1, implemented as **triage → retrieve → compose → audit** in [question2/agent.py](../question2/agent.py).

The help center in this track is much larger than in Question 1 (`shared/kb.py`). The baseline pipeline is correct on the public tickets in `pytest`.

Reduce **points per ticket** under the same billing rules. Graded fields: `action`, `kb_article`.

**Target:** ≤ **2500** points per public ticket (T-1001–T-1004).

**Submit:** changes to `agent.py`, points breakdown for two tickets (one physical fulfillment path, one denial or escalation path), short write-up.
