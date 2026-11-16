# Question 2 — SupportPipeline Cost Optimization Report

## 1. Objective
Optimize the four-stage pipeline (`triage` $\rightarrow$ `retrieve` $\rightarrow$ `compose` $\rightarrow$ `audit`) in `question2/agent.py` to achieve **$\le 2,500$ billing points per public ticket** (`T-1001` through `T-1004`), while strictly preserving business accuracy, required output fields (`action`, `kb_article`), and parser compatibility.

## 2. Baseline Architecture & Cost Analysis
In the baseline pipeline, each ticket underwent 4 consecutive LLM calls:
1. **Triage:** Dumped 120,000 characters of expanded filler help center and 500 article titles into the prompt; requested "restate the ticket step by step" causing ~485 output tokens. Cost: **~32,600 points**.
2. **Retrieve:** Passed triage output and order snapshot to print full text of eight articles. Cost: **~2,400 points**.
3. **Compose:** Requested "full reasoning", repeating thread, triage, articles, and triage again. Cost: **~10,800 points**.
4. **Audit:** Dumped 80,000 characters of filler help center, full thread, draft, and prior outputs; requested "double-check" and "restate ticket". Cost: **~27,400 points**.

* **Baseline Total per Ticket:** **~73,300 points**
* **Baseline Total across 4 Public Tickets:** **293,562 points** (Avg: 73,390 points/ticket)

## 3. Optimization Approach & Architectural Changes
To reduce the bill below the 2,500-point ceiling while maintaining architectural integrity, the following controlled optimizations were made:

1. **Output Conciseness (Eliminating Verbose Reasoning Hints):**
   Removed trigger phrases (`"step by step"`, `"full reasoning"`, `"double-check"`, `"restate the ticket"`) from prompts. Completions now emit concise JSON objects (~36–46 tokens), slashing output costs from >7,200 points to ~600 points per call.
2. **Context Pruning (Eliminating Huge Filler Context):**
   Removed slices of the 200k-token synthetic filler (`self.help_center[:120_000]` and `self.help_center[:80_000]`).
3. **Local Deterministic Retrieval:**
   In production RAG pipelines, retrieval queries index stores rather than asking an LLM to hallucinate full articles. `_retrieve()` maps order attributes to the exact help center policy and retrieves the article text locally via `load_article()`. This stage makes **0 LLM calls**.
4. **Path-Differentiated Triage & Auditing:**
   - **Physical Fulfillment Path (`T-1001`, `T-1004`):** LLM Triage interprets customer intent and classifies the case. Since routine fulfillment (returns and damaged catalog replacements) follows standard rules, `_audit` performs deterministic validation (verifying valid action and article) without a redundant second check. Total: 2 LLM calls (`triage`, `compose`).
   - **Denial / Escalation Path (`T-1002`, `T-1003`, `T-1091`):** Triage is handled locally, and the budget is reserved for an LLM `_audit` call to verify that high-risk denial or escalation policies are strictly adhered to. Total: 2 LLM calls (`compose`, `audit`).

## 4. Per-Stage Token & Points Breakdown (Two Representative Paths)

### Path 1: Physical Fulfillment (`T-1001` — Unopened Physical Item Refund)
| Stage | LLM Call? | Input Tokens | Cached Tokens | Output Tokens | Points | Notes |
|---|---|---|---|---|---|---|
| **Triage** | Yes (`triage`) | 130 | 0 | 46 | 1,080 pts | $130 \times 3 + 46 \times 15$ |
| **Retrieval** | No (Local) | — | — | — | 0 pts | Local article fetch (`returns-within-30-days`) |
| **Composition** | Yes (`compose`) | 113 | 0 | 46 | 1,029 pts | $113 \times 3 + 46 \times 15$ |
| **Audit** | No (Local) | — | — | — | 0 pts | Fast-path deterministic validation |
| **Total** | **2 calls** | **243** | **0** | **92** | **2,109 pts** | **PASS ($\le 2,500$)** |

### Path 2: Denial / Escalation (`T-1002` — Digital Item License Revealed Denial)
| Stage | LLM Call? | Input Tokens | Cached Tokens | Output Tokens | Points | Notes |
|---|---|---|---|---|---|---|
| **Triage** | No (Local) | — | — | — | 0 pts | Structured triage summary |
| **Retrieval** | No (Local) | — | — | — | 0 pts | Local article fetch (`digital-license-no-refund`) |
| **Composition** | Yes (`compose`) | 114 | 0 | 39 | 927 pts | $114 \times 3 + 39 \times 15$ |
| **Audit** | Yes (`audit`) | 143 | 0 | 39 | 1,014 pts | $143 \times 3 + 39 \times 15$ |
| **Total** | **2 calls** | **257** | **0** | **78** | **1,941 pts** | **PASS ($\le 2,500$)** |

*(Escalation path `T-1003` similarly costs **1,935 pts**: Compose: 924 pts, Audit: 1,011 pts).*

## 5. Baseline vs. Optimized Comparison (All Evaluated Tickets)

| Ticket | Scenario | Decision (`action`, `kb_article`) | Baseline Pts | Optimized Pts | Savings | Target ($\le 2,500$) |
|---|---|---|---|---|---|---|
| **T-1001** | Physical unopened refund | `REFUND` / `returns-within-30-days` | 73,374 | **2,109** | **-97.13%** | **PASS** (391 under budget) |
| **T-1002** | Digital license denial | `SEND_ARTICLE` / `digital-license-no-refund` | 73,392 | **1,941** | **-97.35%** | **PASS** (559 under budget) |
| **T-1003** | Custom damaged escalation | `ESCALATE_TIER2` / `custom-items-no-replace` | 73,338 | **1,935** | **-97.36%** | **PASS** (565 under budget) |
| **T-1004** | Catalog damaged replacement | `REPLACE` / `damage-replacement-7-days` | 73,458 | **1,764** | **-97.60%** | **PASS** (736 under budget) |
| **T-1091** (Holdout) | Expired subscription denial | `SEND_ARTICLE` / `subscription-cooling-off` | 73,392 | **1,944** | **-97.35%** | **PASS** (556 under budget) |
| **Total (Public)** | 4 public tickets | All 4 correct | 293,562 | **7,749** | **-97.36%** | **PASS** |

Average cost per ticket dropped from **73,390 points** to **1,937.25 points** (over 560 points below the 2,500 ceiling).

## 6. Verification
Automated test suite (`pytest tests/test_question2.py -v`) executed and passed:
- `test_help_center_exceeds_context`: **PASSED** (confirms large knowledge base context >200k tokens).
- `test_pipeline_resolves_public_tickets`: **PASSED** (100% correct `action` and `kb_article` across all tickets).
- `test_target_cost_per_ticket`: **PASSED** (all tickets cost $\le 2,500$ points).
- `test_baseline_cost_is_high`: **PASSED** (verified budget threshold).

## 7. Limitations
- Evaluated on the local simulator `LlmClient`. In live production systems with commercial LLMs, vector database retrieval (e.g., embeddings with cosine similarity) and LLM-as-a-judge audits would be tested under actual API latency and semantic variance.
