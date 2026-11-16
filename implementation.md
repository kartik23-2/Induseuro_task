# Question 1 --- LLM Cost Optimization: Implementation Guide

> **Scope:** This document covers **Question 1 only** of the Induseuro
> Labs `intern-assignment-llm-cost-optimization` repository: optimizing
> the single-call ShopAssist agent in `question1/agent.py`.
>
> **Primary objective:** Reduce simulated billing to **3,000 points or
> fewer per public ticket**, while preserving the required ticket
> decisions, help-center article IDs, output schema, and parsing
> behavior.
>
> **Important:** This is an implementation plan and verification guide.
> Record actual costs and test results from your local run; do not copy
> illustrative values into the submission as if they were measured.

------------------------------------------------------------------------

## 1. Objective and acceptance criteria

ShopAssist is a simulated Tier-1 e-commerce customer-support copilot.
For a supplied ticket, it reads the customer conversation and order
snapshot, consults help-center content, and returns a structured
resolution.

Question 1 is a **single-call prompt-optimization task**. The agent
should continue making one call through the provided `LlmClient`
interface. Do not replace the intended interface with a direct import of
the reference evaluator or implement a separate multi-stage pipeline.

### Required outcomes

-   [ ] Optimize `question1/agent.py`.
-   [ ] Preserve the `TicketResolution` result fields and expected
    parsing behavior.
-   [ ] Preserve the business decision for every public ticket.
-   [ ] Preserve the correct `kb_article` for every public ticket.
-   [ ] Keep the implementation compatible with the supplied
    `LlmClient`, `CostSession`, data tools, and tests.
-   [ ] Achieve **≤ 3,000 billing points for each public ticket**.
-   [ ] Run the existing test suite, including the holdout test where
    available.
-   [ ] Capture a per-call token and points breakdown for at least one
    ticket.
-   [ ] Submit a short write-up explaining the changes, measured
    savings, and correctness checks.

The public ticket set used by the repository is `T-1001`, `T-1002`,
`T-1003`, and `T-1004`. The tests may also exercise holdout ticket
`T-1091`. Treat the repository's tests and README as the source of truth
if these details differ in a later version.

## 2. Repository components relevant to Question 1

  -----------------------------------------------------------------------
  Path                    Responsibility          How Question 1 uses it
  ----------------------- ----------------------- -----------------------
  `question1/agent.py`    Main single-call agent  Main file to optimize

  `question1/run.py`      Runs the public ticket  Baseline and final
                          examples and displays   demonstration
                          outcomes/cost           

  `shared/llm.py`         `LlmClient`,            Keep using this
                          `ChatMessage`,          interface
                          simulated completion    
                          and call accounting     

  `shared/billing.py`     `CostSession` and       Measures optimization
                          billing calculation     

  `shared/tokens.py`      Token counting          Explains why character
                                                  count is not a reliable
                                                  cost proxy

  `shared/tools.py`       Ticket/order loading    Provides
                          and conversation        ticket-specific inputs
                          formatting              

  `shared/kb.py`          Loads/combines          Source of article
                          help-center articles    context

  `eval/reference.py`     Reference decision      Read for understanding
                          logic used by the       only; do not import it
                          simulator/evaluation    into the agent

  `tests/` and/or         Correctness and         Must pass after every
  `question1/` tests      assignment constraints  change

  `data/`                 Public ticket, order,   Ground truth inputs for
                          and help-center data    local runs
  -----------------------------------------------------------------------

Before editing, inspect the actual files in your checked-out repository.
Names or details can vary by repository revision.

## 3. Baseline architecture

The baseline flow is:

``` text
ticket_id
   |
   v
get_ticket(ticket_id)
   |
   +--> get_order(ticket["order_id"])
   |
   +--> format_thread(ticket)
   |
   +--> all_articles_blob()
   |
   v
build system + user messages
   |
   v
LlmClient.chat("handle_ticket", messages)   # one call
   |
   v
parse_resolution(completion)
   |
   v
TicketResolution(action, kb_article, customer_reply, raw_completion)
```

### System message

The baseline system prompt includes some or all of these items:

1.  A run ID.
2.  The ShopAssist role.
3.  Four few-shot examples.
4.  The complete help-center article collection.
5.  Instructions asking the model to reason at length and return JSON.

### User message

The baseline user prompt contains:

-   The ticket ID.
-   The formatted customer conversation.
-   The associated order snapshot serialized as JSON.

The user message contains the ticket-specific facts needed for the
decision. Avoid deleting order fields or conversation details until you
have verified that they are unnecessary for all required decisions.

## 4. Understand the simulated billing model

The supplied billing module uses points rather than real provider
dollars. The formula is:

``` text
points = 3 × uncached_input_tokens
       + 0.3 × cached_input_tokens
       + 15 × output_tokens
```

Where:

-   **Uncached input tokens** are prompt tokens billed at the normal
    input rate.
-   **Cached input tokens** are eligible prompt-prefix tokens billed at
    the discounted rate.
-   **Output tokens** are generated completion tokens.

The exact token values must come from the repository's tokenizer and
`CostSession`; do not estimate them by dividing character counts by a
constant.

### Why verbose output is expensive

At the configured rates, an additional 100 output tokens costs 1,500
points. An additional 100 uncached input tokens costs 300 points.
Consequently, removing unnecessary output can have a large effect on the
total.

### Simulator-specific behavior

The repository's `LlmClient` is a simulator, not a request to a live
hosted LLM. Inspect `shared/llm.py` to confirm the exact behavior in
your checkout. In the described baseline, the simulator extracts the
order snapshot and uses reference decision logic to produce the expected
decision. It also creates a verbose completion when certain phrases such
as "step by step" or "full reasoning" appear in the prompt; this path
has a large minimum output length.

This behavior makes a concise output instruction an especially relevant
experiment. However:

-   Keep the required `LlmClient` call path.
-   Do not import or call `eval/reference.py` directly from
    `question1/agent.py`.
-   Do not hard-code ticket IDs to return answers.
-   Do not claim that simulator results alone prove equivalent quality
    on a real hosted model.
-   If the simulator implementation differs in your checkout, follow the
    code that is actually present.

## 5. Establish a baseline before changing anything

Use a clean working tree or commit the original state first. Record the
current code revision so you can compare baseline and optimized results.

### Environment setup

From the repository root:

``` bash
python -m venv .venv
```

Activate the environment:

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

### Run the existing tests and Question 1 runner

``` bash
pytest -q
python question1/run.py
```

If the repository's README specifies a different command or the test
layout differs, use the documented command. Save the output from the
baseline run.

### Record a baseline table

Fill this table with the actual output from your environment.

  ----------------------------------------------------------------------------
  Ticket      Expected     Actual      Expected     Actual            Baseline
              action       action      article      article             points
  ----------- ------------ ----------- ------------ ----------- --------------
  `T-1001`    Verify from  Record run  Verify from  Record run      Record run
              tests/data               tests/data               

  `T-1002`    Verify from  Record run  Verify from  Record run      Record run
              tests/data               tests/data               

  `T-1003`    Verify from  Record run  Verify from  Record run      Record run
              tests/data               tests/data               

  `T-1004`    Verify from  Record run  Verify from  Record run      Record run
              tests/data               tests/data               
  ----------------------------------------------------------------------------

Do not infer expected actions or article IDs from ticket IDs. Read the
fixture data, tests, or reference output.

### Inspect individual cost records

The agent may expose the session through a property such as
`copilot.session`, with call records in `session.calls`. Verify the
actual public attributes in `shared/billing.py` and `question1/agent.py`
before using them.

For each call, capture the fields the repository exposes, such as:

-   Call label.
-   Input token count.
-   Cached input token count.
-   Output token count.
-   Calculated points.

If `question1/run.py` only prints aggregate points, create a small
temporary diagnostic script or add a non-invasive reporting block to
your local experiment. Avoid modifying the shared billing implementation
just to make the report look better.

## 6. Optimization strategy

Use controlled experiments: change one thing at a time, rerun tests, and
measure the per-ticket cost. Keep a record of each experiment so the
final write-up is evidence-based.

### Optimization 1 --- Remove requests for verbose reasoning

**Baseline issue:** Instructions such as "Think step by step" and "Write
full reasoning" encourage a long completion. The simulated client may
explicitly use these phrases to select a verbose completion path.

**Change:** Ask for only the structured fields needed by the
application. For example, use a concise instruction such as:

``` text
Choose the correct support action and the applicable help-center article.
Return one valid JSON object with exactly these keys:
action, kb_article, customer_reply.
Do not include markdown fences or text outside the JSON object.
```

This is a prompt-design example, not a guarantee that the final response
will be valid. Adapt the wording to the actual simulator and tests.

**Why it helps:** Fewer generated tokens reduce the output portion of
the bill.

**Validation:** Confirm that `action` and `kb_article` are still correct
and the completion can still be parsed.

### Optimization 2 --- Reduce unnecessary examples

The baseline includes four few-shot examples. Examples can improve
format consistency, but they also increase prompt length.

Recommended experiment:

1.  Keep all examples and measure the optimized output instruction.
2.  Remove or shorten the examples in a separate experiment.
3.  Run all public and holdout tests after each change.
4.  Keep the smaller prompt only if correctness remains intact and cost
    improves.

Do not remove an example merely because it appears repetitive. A small
set of contrasting examples may be useful in a real model setting even
if the simulator's decision logic does not depend on them.

### Optimization 3 --- Reduce help-center context carefully

The baseline uses `all_articles_blob()` to include all help-center
articles. This can make the prompt much larger than necessary.

Possible approaches, in order of simplicity:

1.  **Compact the representation** while preserving article identifiers
    and policy rules.
2.  If permitted by the assignment and compatible with the tests,
    identify relevant articles from ticket/order facts and provide only
    the necessary article content.
3.  Keep the article IDs exactly as expected by the evaluation.
4.  Ensure the agent does not select an article solely because it
    appears first in a shortened prompt.

Do not silently omit policy conditions, exceptions, or article IDs. A
lower cost is not an improvement if it produces an incorrect support
decision.

If you introduce local filtering, document the deterministic selection
rule and test every public ticket plus the holdout. Do not introduce an
extra LLM call; Question 1 is a single-call task.

### Optimization 4 --- Shorten static instructions

Replace redundant prose with concise, unambiguous instructions. Keep all
business constraints needed for a correct result, including the valid
action set and output schema.

Avoid adding lengthy descriptions that repeat information already
represented in a structured order snapshot or article.

### Optimization 5 --- Be careful with prompt caching

The billing implementation may offer a discount for eligible repeated
prefixes. A changing UUID at the beginning of the system prompt can
prevent the prompt from sharing an identical prefix across runs.

Do not assume caching is helping merely because a prompt is long.
Inspect how `CostSession` calculates cached tokens and how `LlmClient`
assigns the cached-prefix count. Only report cache savings that are
actually reflected in the recorded metrics.

Because Question 1 processes separate tickets, any caching behavior must
be evaluated using the simulator's exact session and call semantics.
Avoid redesigning the prompt solely around a caching assumption without
measuring it.

## 7. Preserve the output contract

The agent returns a `TicketResolution` object with these fields:

  -----------------------------------------------------------------------
  Field                               Requirement
  ----------------------------------- -----------------------------------
  `action`                            Required; normalize to uppercase if
                                      the existing parser does so

  `kb_article`                        Required; preserve the correct
                                      article identifier

  `customer_reply`                    Keep the field and its current
                                      fallback behavior

  `raw_completion`                    Preserve the raw model completion
                                      for debugging/auditing
  -----------------------------------------------------------------------

The allowed action values described by the assignment are:

-   `REFUND`
-   `REPLACE`
-   `SEND_ARTICLE`
-   `ESCALATE_TIER2`

Confirm the exact accepted values in the repository's tests and
reference data.

### Parser considerations

The existing parser searches for a JSON object, loads it with
`json.loads`, and constructs `TicketResolution`. Preserve compatibility
with that behavior.

A valid completion should be plain JSON, for example:

``` json
{
  "action": "SEND_ARTICLE",
  "kb_article": "article-id-from-the-repository",
  "customer_reply": "A concise customer-facing reply."
}
```

The article ID above is intentionally a placeholder. Use an actual ID
from the repository's knowledge base; do not submit a made-up ID.

Do not add a markdown code fence around the JSON if the prompt can
instruct the simulator to return plain JSON. Do not remove required
fields or silently default a missing action/article to an arbitrary
value.

## 8. Testing and verification

Run tests after each meaningful change.

``` bash
pytest -q
python question1/run.py
```

If the repository has targeted Question 1 tests, run those as well.
Inspect the test files rather than assuming a test path.

### Correctness checklist

-   [ ] All four public tickets are processed.
-   [ ] Each public ticket returns the expected action.
-   [ ] Each public ticket returns the expected help-center article.
-   [ ] The holdout test, including `T-1091` if present, still passes.
-   [ ] `parse_resolution()` still handles the completion format used by
    the simulator.
-   [ ] The result object retains all expected fields.
-   [ ] The agent still uses one `LlmClient.chat(...)` call per ticket.
-   [ ] No ticket-specific hard-coded answers were added.
-   [ ] No direct reference-evaluator import was added to the agent.
-   [ ] The measured cost is ≤ 3,000 points for **each** public ticket.

### Cost checklist

-   [ ] Input and output token counts are reported separately.
-   [ ] Cached tokens are reported if the session exposes them.
-   [ ] Points are calculated by the repository's billing
    implementation.
-   [ ] Baseline and optimized costs use the same tickets and
    environment.
-   [ ] The report distinguishes measured values from estimates.
-   [ ] Any aggregate cost is not used as a substitute for the
    per-ticket threshold.

## 9. Suggested per-call cost report

The assignment asks for a token/points breakdown for one ticket. Use the
exact numbers from your `CostSession` record.

  -----------------------------------------------------------------------
  Metric                               Baseline                 Optimized
  ------------------- ------------------------- -------------------------
  Ticket ID                             Fill in               Same ticket

  Input tokens                         Measured                  Measured

  Cached input tokens   Measured, or `0` if the                  Measured
                         implementation reports 
                                           zero 

  Output tokens                        Measured                  Measured

  Input points            Calculate from actual                 Calculate
                                 billing fields 

  Output points           Calculate from actual                 Calculate
                                 billing fields 

  Total points              Measured/calculated       Measured/calculated

  Correct action?                        Yes/No                    Yes/No

  Correct article?                       Yes/No                    Yes/No
  -----------------------------------------------------------------------

Use the repository's own `CostSession` output as the authority for
points. If you calculate the subtotals manually, document the formula
and make sure the cached/uncached split matches the implementation.

### Savings formulas

For a ticket:

``` text
absolute_savings = baseline_points - optimized_points

percent_savings =
    100 * (baseline_points - optimized_points) / baseline_points
```

For a baseline of zero, percentage savings is undefined; do not divide
by zero. Report the raw values instead.

## 10. Required short write-up

Create a short write-up in the format requested by the
assignment/README. It should contain:

1.  **Approach:** What prompt/context changes were made.
2.  **Reasoning:** Why these changes should reduce the simulated bill.
3.  **Correctness:** Which tests were run and whether they passed.
4.  **Measured results:** Baseline and optimized input/output tokens and
    points for at least one ticket.
5.  **Target status:** The cost for each public ticket and whether each
    meets the 3,000-point limit.
6.  **Limitations:** The result is measured using the provided
    simulator; real LLM quality and provider-side caching should be
    tested separately.

Do not claim a percentage saving or passing status until you have run
the implementation and recorded the output.

## 11. Common mistakes to avoid

-   **Optimizing only the aggregate:** The requirement is per public
    ticket.
-   **Removing too much context:** An incorrect action or article
    invalidates the optimization.
-   **Keeping verbose reasoning:** This can dominate output cost in the
    supplied simulator.
-   **Counting characters instead of tokens:** Use the repository
    tokenizer.
-   **Changing the architecture:** Do not turn Question 1 into a
    multi-call pipeline.
-   **Hard-coding answers:** Keep the agent generic and data-driven.
-   **Importing the reference evaluator:** The agent must use the
    supplied `LlmClient` interface.
-   **Inventing metrics:** Fill the report with actual run results.
-   **Ignoring the holdout:** Passing public examples alone may not
    catch overfitting.
-   **Assuming caching:** Verify cached-token counts and billing
    semantics in code.

## 12. Final submission checklist

Before submitting, confirm:

-   [ ] `question1/agent.py` contains the optimized single-call
    implementation.
-   [ ] The code follows the existing project structure and style.
-   [ ] The existing test suite passes.
-   [ ] All public-ticket decisions and article IDs are correct.
-   [ ] Each public ticket costs no more than 3,000 points.
-   [ ] A one-ticket token/points breakdown is included.
-   [ ] The short write-up explains the changes and measured results.
-   [ ] No temporary diagnostic files, credentials, or unrelated
    Question 2 changes are included.
-   [ ] The final changes are committed and the repository is in a
    reproducible state.

## 13. Recommended implementation sequence

1.  Read `question1/agent.py`, `question1/run.py`, `shared/llm.py`,
    `shared/billing.py`, `shared/tokens.py`, `shared/tools.py`,
    `shared/kb.py`, and the relevant tests.
2.  Run and save the baseline test output and cost report.
3.  Make only the output-instruction change first.
4.  Run tests and measure all public tickets.
5.  Experiment with shortening examples.
6.  Experiment with help-center context reduction only if needed.
7.  Keep the smallest correct version that meets the target.
8.  Run the full tests and runner one final time.
9.  Fill in the measured cost table and write-up.
10. Review the final diff to ensure only intended Question 1 changes are
    present.

------------------------------------------------------------------------

**Definition of done:** Question 1 is complete when the single-call
agent preserves all required outputs, passes the repository's relevant
tests, meets the **≤ 3,000 points per public ticket** target, and
includes an honest, reproducible token/points breakdown plus the
requested short explanation.
