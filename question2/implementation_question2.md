# Question 2 --- LLM Cost Optimization: Implementation Plan

> **Scope:** Question 2 only of the Induseuro Labs
> `intern-assignment-llm-cost-optimization` repository, focused on
> `question2/agent.py`.
>
> **Goal:** Reduce simulated billing to **2,500 points or fewer per
> public ticket**, while preserving required decisions, knowledge-base
> article IDs, output schema, and expected pipeline behavior.
>
> **Important:** This is a plan and verification guide. Fill all result
> tables with values measured from your own runs. Do not present
> estimates as measured results.

## 1. Objective and acceptance criteria

ShopAssist is a simulated e-commerce customer-support copilot. Question
2 uses a four-stage pipeline:

1.  **Triage** --- interpret the ticket and classify the issue/path.
2.  **Article retrieval** --- identify the applicable help-center
    article or policy.
3.  **Response composition** --- prepare the proposed resolution and
    customer-facing response.
4.  **Audit** --- check the proposed resolution.

The challenge is to reduce the cost of the complete pipeline---not just
one prompt---while preserving correctness.

### Acceptance checklist

-   [ ] Optimize `question2/agent.py`.
-   [ ] Preserve the supplied interfaces and intended multi-stage
    pipeline.
-   [ ] Preserve the correct `action` and `kb_article` for all public
    tickets.
-   [ ] Preserve the output schema and parsing behavior.
-   [ ] Remain compatible with `shared/llm.py`, `shared/billing.py`, and
    supplied data tools.
-   [ ] Achieve **≤ 2,500 points per public ticket**.
-   [ ] Run relevant tests, including holdout tests where present.
-   [ ] Capture per-call/per-stage token and points breakdowns for two
    tickets: one physical-fulfillment path and one denial/escalation
    path.
-   [ ] Include a short write-up with changes, measured savings, and
    correctness checks.

The public tickets are described as `T-1001` through `T-1004`; the test
suite may also exercise a holdout such as `T-1091`. Verify the exact
fixtures in your checkout.

## 2. Relevant repository files

  -----------------------------------------------------------------------
  Path                                Responsibility
  ----------------------------------- -----------------------------------
  `question2/agent.py`                Main multi-stage pipeline; primary
                                      implementation file

  `question2/run.py`                  Runs public examples and reports
                                      outcomes/costs

  `shared/llm.py`                     `LlmClient`, `ChatMessage`,
                                      simulated completions and call
                                      accounting

  `shared/billing.py`                 `CostSession` and billing formula

  `shared/tokens.py`                  Token counting

  `shared/tools.py`                   Ticket/order loading and
                                      conversation formatting

  `shared/kb.py`                      Help-center article content and
                                      indexing utilities

  `eval/reference.py`                 Reference decision/evaluation
                                      logic; read for understanding, do
                                      not import into the agent

  `tests/` and Question 2 tests       Correctness requirements

  `data/`                             Public tickets, orders and
                                      help-center content
  -----------------------------------------------------------------------

Inspect the actual files in your checked-out repository before editing.
Confirm function names, stage labels, session accessors, and test paths
from the source.

## 3. Baseline architecture

``` text
Ticket ID
   |
   v
Load ticket, order and conversation
   |
   v
Stage 1: TRIAGE
   |
   v
Stage 2: ARTICLE RETRIEVAL
   |
   v
Stage 3: RESPONSE COMPOSITION
   |
   v
Stage 4: AUDIT
   |
   v
Final structured resolution
   |
   v
Correctness tests + cost report
```

Use the exact names and result objects defined by `question2/agent.py`;
this diagram describes responsibilities, not a replacement API.

The total cost for a ticket is the sum of its calls:

``` text
ticket_points =
    triage_points
  + retrieval_points
  + composition_points
  + audit_points
```

A pipeline may repeat the ticket thread, order snapshot, help-center
content, instructions, or previous-stage explanations. Each repetition
can add input tokens. Verbose intermediate outputs also cost money even
when the next stage only needs a few fields.

## 4. Billing model

The repository's simulated billing formula is:

``` text
points = 3 × uncached_input_tokens
       + 0.3 × cached_input_tokens
       + 15 × output_tokens
```

-   **Uncached input tokens:** prompt tokens charged at the normal rate.
-   **Cached input tokens:** eligible repeated prefix tokens charged at
    the discounted rate.
-   **Output tokens:** generated completion tokens.

Use the tokenizer and `CostSession` values from the repository; do not
estimate token counts from characters.

At these rates, 100 additional output tokens cost 1,500 points, while
100 uncached input tokens cost 300 points. Therefore, verbose
intermediate completions can dominate cost.

### Simulator-specific caution

The supplied `LlmClient` is a simulator rather than a live hosted-model
API. Inspect `shared/llm.py` to confirm the exact behavior in your
version. The described implementation uses the order snapshot and
reference decision behavior to produce deterministic outcomes and
records token usage. Some verbose prompt phrases may trigger longer
simulated completions.

Do not bypass the intended interface, hard-code ticket-specific answers,
or import `eval/reference.py` into `question2/agent.py`. Simulator
results demonstrate performance under the assignment's billing model;
they do not by themselves prove equivalent quality or caching behavior
on a real hosted model.

## 5. Establish a baseline

Run the original implementation before making changes. Ideally commit
the baseline or save a patch so you can compare the final diff.

### Environment setup

From the repository root:

``` bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

``` bash
source .venv/bin/activate
```

Install dependencies:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run tests and the Question 2 runner:

``` bash
pytest -q
python question2/run.py
```

Follow the repository README if it specifies different commands.

### Record baseline results

For every public ticket, record:

  ----------------------------------------------------------------------------------------------------------------------
  Ticket        Actual action   Actual article ID             Total baseline points             Tests pass?
  ------------- --------------- ----------------------------- --------------------------------- ------------------------
  `T-1001`      `REFUND`        `returns-within-30-days`      73,374                            Yes (PASSED)

  `T-1002`      `SEND_ARTICLE`  `digital-license-no-refund`   73,392                            Yes (PASSED)

  `T-1003`      `ESCALATE_TIER2``custom-items-no-replace`     73,338                            Yes (PASSED)

  `T-1004`      `REPLACE`       `damage-replacement-7-days`   73,458                            Yes (PASSED)
  ----------------------------------------------------------------------------------------------------------------------

Read fixtures/tests to identify expected actions, article IDs, and which
tickets represent physical fulfillment versus denial/escalation. Do not
infer these from ticket IDs.

### Inspect per-call data

Inspect `shared/billing.py`, `shared/llm.py`, and `question2/agent.py`
to see how the cost session is exposed. It may be available through a
property such as `copilot.session` with records in `session.calls`, but
verify the real names in your checkout.

Capture fields that the API exposes:

-   call label/stage;
-   input token count;
-   cached input token count;
-   output token count;
-   calculated points.

If the runner prints only aggregate totals, use a temporary diagnostic
script or a small reporting addition that does not change decisions. Do
not change the billing implementation simply to make the report look
better.

## 6. Optimization principles

1.  **Load data once where possible.** Avoid reloading or reformatting
    the same ticket/order unnecessarily.
2.  **Give each stage only the context it needs.** Do not automatically
    pass the entire ticket, order and knowledge base through every
    stage.
3.  **Use compact intermediate results.** Pass structured fields instead
    of lengthy prose when downstream code only needs a few values.
4.  **Request concise outputs.** Keep only fields consumed by later
    stages or the final result.
5.  **Preserve useful auditing.** Measure and optimize the audit stage
    before considering whether it can be skipped.
6.  **Change one thing at a time.** Re-run tests and measure costs after
    each experiment.
7.  **Do not hard-code ticket answers.** The solution must work from the
    supplied ticket/order data.

## 7. Stage-by-stage implementation plan

### Stage 1 --- Triage

**Purpose:** Interpret the ticket and determine the relevant issue or
fulfillment path.

Investigate: - Long instructions or requests for chain-of-thought. -
Repeated conversation/order content. - Summaries that repeat facts
already present in the ticket. - Output fields the next stage never
consumes.

Recommended steps:

1.  Inspect the triage prompt and completion.
2.  List the fields consumed by the next stage.
3.  Ask for a concise structured result rather than an essay.
4.  Keep facts that distinguish physical products, digital products,
    refund/return conditions, denials, and escalation cases.
5.  Avoid requesting a long rationale unless the implementation actually
    uses it.

Illustrative compact schema---adapt it to the actual code's expected
fields:

``` json
{
  "intent": "return_request",
  "fulfillment_type": "physical",
  "exception": "none"
}
```

Do not introduce a new schema without updating all consumers and
preserving test compatibility.

**Validation:** Confirm the stage distinguishes the public ticket paths
correctly, especially the physical-fulfillment and denial/escalation
cases required for reporting.

### Stage 2 --- Article retrieval

**Purpose:** Identify the relevant help-center article or policy.

Investigate: - Whether the stage sends the entire help center. - Whether
the same article text is repeated in later prompts. - Whether it
generates a long selection explanation. - Whether it receives ticket
facts already summarized by triage.

Recommended steps:

1.  Check how `shared/kb.py` is used.
2.  Determine whether retrieval is local/deterministic or makes an LLM
    call.
3.  If it is an LLM call, make its prompt concise and request only the
    article ID and any required supporting fields.
4.  If it is local, optimize the data passed to it without adding
    another LLM call.
5.  Pass only the selected article and necessary policy excerpt
    downstream when safe.
6.  Preserve policy exceptions, conditions and exact article IDs.

Do not invent IDs or choose an article merely because it appears first.
If filtering is introduced, document the rule and test all public
tickets plus the holdout.

**Validation:** Compare the selected article against the expected ID for
every test case.

### Stage 3 --- Response composition

**Purpose:** Produce the proposed structured resolution and
customer-facing response.

Investigate: - Repeated full thread/order snapshots. - Repetition of the
full help-center article. - Long reasoning or policy quotations. - Extra
output fields unused by downstream code.

Recommended steps:

1.  Inspect the final output schema and its consumers.
2.  Pass only the relevant ticket facts, selected article, and necessary
    prior-stage fields.
3.  Ask for a concise customer reply.
4.  Require one valid JSON object with the expected keys and no markdown
    fences or surrounding prose.
5.  Keep action/article values consistent with policy and previous
    stages.

Illustrative schema:

``` json
{
  "action": "REPLACE",
  "kb_article": "actual-article-id",
  "customer_reply": "A concise response appropriate to this ticket."
}
```

The values above are illustrative, not expected answers for a particular
ticket. Use real article IDs and the exact schema in the repository.

**Validation:** Confirm parsing, required keys, accepted action values,
correct article IDs, and expected reply behavior.

### Stage 4 --- Audit

**Purpose:** Check that the proposed resolution is consistent with the
ticket and policy.

Investigate: - Whether the audit receives the entire conversation and
order again. - Whether it needs the full previous reasoning or only the
proposed result and evidence. - Whether it regenerates the whole answer
when only a pass/fail or correction is consumed. - Whether some checks
can be performed deterministically in existing code.

Recommended steps:

1.  Inspect the audit inputs, outputs, and how its result is used.
2.  Pass the proposed resolution and only the evidence required to
    validate it.
3.  Request a concise structured audit result if compatible with the
    current contract.
4.  First optimize the audit prompt while retaining the audit stage.
5.  Consider skipping/removing the audit only if the task permits it and
    tests demonstrate that required behavior remains correct. Document
    the trade-off.

Do not remove the audit merely to lower cost without understanding what
it protects.

**Validation:** Confirm audit failures/corrections are handled as the
existing code expects and all final resolutions remain correct.

## 8. Reduce duplicated context between stages

Create a field-usage map by reading the code:

  ---------------------------------------------------------------------------------
  Stage             Inputs it         Output used       Redundancy to investigate
                    actually needs    downstream        
  ----------------- ----------------- ----------------- ---------------------------
  Triage            Fill from code    Fill from code    Repeated
                                                        thread/order/instructions

  Retrieval         Fill from code    Article           Full KB or duplicate ticket
                                      ID/evidence       facts

  Composition       Fill from code    Final resolution  Repeated
                                                        explanations/article text

  Audit             Fill from code    Pass/fail or      Full history and repeated
                                      corrected result  policy
  ---------------------------------------------------------------------------------

For each stage, ask:

-   Does it need the raw thread, or are specific extracted facts
    sufficient?
-   Does it need the full order snapshot, or only relevant order fields?
-   Does it need every article, or just the candidate/selected article?
-   Does it need previous-stage prose, or only structured fields?
-   Are instructions repeated unnecessarily?

Do not remove a field simply because it appears redundant. Check
consumers, tests and policy conditions first. Keep all changes
data-driven and generic.

## 9. Prompt caching

The billing model may discount eligible repeated prompt-prefix tokens.
Caching is not guaranteed just because prompts look similar.

Inspect how the simulator identifies the prefix and calculates cached
tokens. A changing UUID or dynamic text at the beginning may prevent the
prefix from matching. If testing prompt order, keep stable instructions
at the start only when compatible with the provided implementation.

Report cache savings only when the recorded cached-token counts and
points demonstrate them.

## 10. Preserve the output contract

The expected final result includes the correct support action and
knowledge-base article ID. The documented action categories are:

-   `REFUND`
-   `REPLACE`
-   `SEND_ARTICLE`
-   `ESCALATE_TIER2`

Confirm exact accepted values and output fields from tests and fixtures.
Preserve current parser behavior, error handling and result types.

### Output checklist

-   [ ] Required action field is present and normalized as expected.
-   [ ] Action is an accepted value.
-   [ ] `kb_article` is the exact applicable article ID.
-   [ ] Customer reply is present if required.
-   [ ] The existing parser can read the completion.
-   [ ] Downstream stages receive the fields they require.
-   [ ] No unrelated Question 1 changes are included.

## 11. Controlled experiments

Do not change every stage at once. Use separate experiments and record
the result of each.

### Experiment A --- Shorten outputs

Make triage, retrieval, composition and audit outputs concise. Re-run
tests and record stage costs.

### Experiment B --- Remove duplicated context

Remove one repeated input at a time. Check correctness and measure
per-stage cost after each change.

### Experiment C --- Reduce help-center context

Inspect how retrieval works. Try compact article content or
deterministic selection if compatible. Validate exact article IDs for
all public and holdout tickets.

### Experiment D --- Optimize the audit

Reduce its prompt and output while preserving its checks. Consider
removing/skipping it only if permitted and validated.

### Experiment E --- Combine validated changes

Combine only changes that individually passed correctness checks. Run
the full tests and runner from a clean state, then capture final
measurements.

## 12. Required cost breakdown for two tickets

The assignment requires cost breakdowns for:

1.  One **physical-fulfillment path**.
2.  One **denial or escalation path**.

Identify actual ticket IDs from the fixtures and run output. Do not
assume the mapping.

### Per-stage report

Fill in the measured values for each representative ticket:

  --------------------------------------------------------------------------------------------------------------------
  Ticket/path           Stage         Input tokens Cached input tokens   Output tokens Points
  --------------------- ------------- ------------ ------------------- --------------- -----------------------------------
  Physical fulfillment  Triage        130          0                   46              1,080 pts (130×3 + 46×15)
  (`T-1001`)

  Physical fulfillment  Retrieval     not called   not called          not called      0 pts (local retrieval)
  (`T-1001`)

  Physical fulfillment  Composition   113          0                   46              1,029 pts (113×3 + 46×15)
  (`T-1001`)

  Physical fulfillment  Audit         not called   not called          not called      0 pts (deterministic pass)
  (`T-1001`)

  **Physical            **All         243          0                   92              **2,109 pts**
  fulfillment total**   stages**                                             

  Denial/escalation     Triage        not called   not called          not called      0 pts (deterministic triage)
  (`T-1002`)

  Denial/escalation     Retrieval     not called   not called          not called      0 pts (local retrieval)
  (`T-1002`)

  Denial/escalation     Composition   114          0                   39              927 pts (114×3 + 39×15)
  (`T-1002`)

  Denial/escalation     Audit         143          0                   39              1,014 pts (143×3 + 39×15)
  (`T-1002`)

  **Denial/escalation   **All         257          0                   78              **1,941 pts**
  total**               stages**                                             
  --------------------------------------------------------------------------------------------------------------------

If a stage does not make a call in a particular path, record "not
called"; do not invent a zero-token call. If a metric is not exposed by
the billing API, disclose that and report available values.

### Baseline-versus-optimized report

  ----------------------------------------------------------------------------------------------------------------------------
  Ticket/path                   Baseline points   Optimized points   Absolute savings   Savings %   Correct action/article?
  ----------------------------- ----------------- ------------------ ------------------ ----------- ------------------------------
  Physical fulfillment `T-1001` 73,374            2,109              -71,265 pts        -97.13%     Yes (`REFUND` / `returns-within-30-days`)

  Denial `T-1002`               73,392            1,941              -71,451 pts        -97.35%     Yes (`SEND_ARTICLE` / `digital-license-no-refund`)

  Escalation `T-1003`           73,338            1,935              -71,403 pts        -97.36%     Yes (`ESCALATE_TIER2` / `custom-items-no-replace`)

  Physical fulfillment `T-1004` 73,458            1,764              -71,694 pts        -97.60%     Yes (`REPLACE` / `damage-replacement-7-days`)

  Holdout `T-1091`              73,392            1,944              -71,448 pts        -97.35%     Yes (`SEND_ARTICLE` / `subscription-cooling-off`)

  **Total (4 Public Tickets)**  **293,562**       **7,749**          **-285,813 pts**   **-97.36%** **All 4 correct**
  ----------------------------------------------------------------------------------------------------------------------------

Adjust rows to the actual public tickets in the repository. The
requirement is **≤ 2,500 points per public ticket**, not merely a low
average.

``` text
absolute_savings = baseline_points - optimized_points

percent_savings =
    100 * (baseline_points - optimized_points) / baseline_points
```

If baseline points are zero, percentage savings is undefined; report
absolute values instead.

## 13. Testing and verification

Run tests after each meaningful change and again after combining
optimizations:

``` bash
pytest -q
python question2/run.py
```

Run any targeted Question 2 tests discovered in the repository as well.

### Correctness checklist

-   [ ] Every public ticket completes.
-   [ ] Every public ticket has the expected action.
-   [ ] Every public ticket has the expected article ID.
-   [ ] Physical-fulfillment behavior remains correct.
-   [ ] Denial/escalation behavior remains correct.
-   [ ] Holdout tests pass where present.
-   [ ] Parsing and error handling remain compatible.
-   [ ] The intended Question 2 architecture is preserved unless a
    permitted, justified change is made.
-   [ ] No ticket-specific answers are hard-coded.
-   [ ] The agent does not import `eval/reference.py`.
-   [ ] Each public ticket costs ≤ 2,500 points.

### Cost checklist

-   [ ] Baseline and optimized runs use the same tickets.
-   [ ] Per-call stage labels are recorded where available.
-   [ ] Input, cached input and output tokens are reported where
    exposed.
-   [ ] Per-stage points sum to the recorded ticket total.
-   [ ] Both required representative paths have breakdowns.
-   [ ] Savings are calculated from measured totals.
-   [ ] Missing metrics are disclosed.
-   [ ] Measured results are distinguished from hypotheses.

## 14. Required short write-up

Use the format requested by the assignment README. Include:

### Objective

State the goal: no more than 2,500 simulated points per public ticket
while preserving required behavior.

### Baseline

Describe the four-stage pipeline and include measured baseline costs.

### Changes made

For each change, identify the stage/interface, the redundant context or
output removed, why the change was safe, and how it was tested.

### Results

Include baseline and optimized points, per-stage breakdowns for the two
required tickets, available token counts, savings, target status for
each public ticket, and actual test results.

### Limitations

Explain that the supplied client is simulated. Real provider deployment
requires separate validation of quality, latency, cache behavior and
costs.

Do not claim tests passed or the target was met until you have run the
code.

## 15. Common mistakes

-   Optimizing only one stage instead of the complete ticket.
-   Reporting only average cost when each ticket has a threshold.
-   Removing the audit without considering its correctness value.
-   Passing the whole knowledge base to every stage without need.
-   Passing lengthy prose when structured fields suffice.
-   Changing schemas without updating downstream consumers.
-   Inventing article IDs or expected outcomes.
-   Hard-coding answers for public ticket IDs.
-   Importing reference logic into the agent.
-   Assuming prompt caching without checking the recorded metrics.
-   Fabricating token/cost values.
-   Changing too many things at once.
-   Making unrelated changes to Question 1.

## 16. Final submission checklist

-   [ ] `question2/agent.py` contains the optimized implementation.
-   [ ] Changes are scoped to Question 2.
-   [ ] Relevant tests pass.
-   [ ] `python question2/run.py` completes.
-   [ ] All public actions and article IDs are correct.
-   [ ] Each public ticket is at or below 2,500 points.
-   [ ] A physical-fulfillment ticket has a per-stage cost breakdown.
-   [ ] A denial/escalation ticket has a per-stage cost breakdown.
-   [ ] Baseline-versus-optimized measurements are included.
-   [ ] The short write-up explains changes, measurements and
    limitations.
-   [ ] No credentials, temporary diagnostics or unrelated changes are
    included.
-   [ ] Final changes are committed and reproducible.

## 17. Recommended execution order

1.  Read `question2/agent.py`, `question2/run.py`, `shared/llm.py`,
    `shared/billing.py`, `shared/tokens.py`, `shared/tools.py`,
    `shared/kb.py`, and relevant tests.
2.  Run the baseline tests and Question 2 runner.
3.  Record baseline decisions, article IDs and total points for each
    public ticket.
4.  Inspect per-call records to identify the most expensive stages.
5.  Shorten verbose outputs first.
6.  Remove duplicated context one change at a time.
7.  Optimize article context without losing policy exceptions or exact
    IDs.
8.  Reduce audit input/output while preserving useful checks.
9.  Run the full tests after each change.
10. Fill in measured per-stage reports for both required paths.
11. Verify the target for every public ticket.
12. Write the short report and inspect the final diff.

------------------------------------------------------------------------

**Definition of done:** Question 2 is complete when the multi-stage
pipeline preserves required decisions and article IDs, passes relevant
tests, meets the **≤ 2,500 points per public ticket** target, and
includes measured per-stage breakdowns for one physical-fulfillment path
and one denial/escalation path, plus the required short write-up.
