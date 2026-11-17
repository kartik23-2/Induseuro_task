# Video Presentation Guide: LLM Cost Optimization (ShopAssist)

This guide provides a scene-by-scene walkthrough, screen layout tips, talking points, and exact commands to demonstrate the solutions for **Question 1** and **Question 2** with proof.

---

## Recording Setup & Recommendations

* **Recommended Duration:** 4 to 6 minutes.
* **Recording Software:** OBS Studio, Loom, or Windows Game Bar (`Win + G`).
* **Screen Layout:**
  * **Top / Left:** VS Code editor showing the project files ([question1/agent.py](file:///c:/Users/karti/OneDrive/Desktop/intern/Induseuro_labs/question1/agent.py), [question2/agent.py](file:///c:/Users/karti/OneDrive/Desktop/intern/Induseuro_labs/question2/agent.py)).
  * **Bottom / Right:** Integrated Terminal with clear font size (`Ctrl + +`).

---

## Scene 1: Introduction & Problem Context (0:00 – 0:45)

### Visuals
* Open `README.md` or the project root in VS Code.

### Talking Points
1. **The Product:**
   > *"Hi everyone, today I'm demonstrating our LLM cost optimization for ShopAssist—an automated Tier-1 customer support copilot that handles ticket resolutions for physical returns, digital goods, custom items, and replacements."*
2. **The Economic Billing Model:**
   * Point to the formula in `README.md`:
     $$\text{Points} = 3 \times \text{Uncached Input} + 0.3 \times \text{Cached Input} + 15 \times \text{Output}$$
   > *"Notice the billing weights: output tokens carry a 5× penalty compared to uncached input tokens (15 points vs 3 points). Unnecessary reasoning and bloated prompts explode the cost."*
3. **The Assignment Targets:**
   > *"Our goal: Question 1 must achieve $\le 3,000$ points per public ticket on a single-call architecture, and Question 2 must achieve $\le 2,500$ points per public ticket on a 4-stage pipeline."*

---

## Scene 2: Question 1 — Optimization & Live Proof (0:45 – 2:00)

### Visuals
* Open [question1/agent.py](file:///c:/Users/karti/OneDrive/Desktop/intern/Induseuro_labs/question1/agent.py).

### Talking Points
1. **Baseline Inefficiencies:**
   > *"In the baseline, instructions asked the model to 'think step by step' and 'write full reasoning'. This triggered verbose output traces exceeding 485 tokens per call. At 15 points per token, over 7,200 points per ticket were wasted on internal reasoning."*
   > *"Additionally, dynamic UUID prefixes (`f'Run id {uuid.uuid4()}'`) added useless prompt tokens."*
2. **The Changes Made:**
   * Highlight lines 63–72 in `question1/agent.py`:
     * Replaced verbose prompts with direct schema instructions: `"Choose the correct action and help-center article. Output JSON with action, kb_article, customer_reply."`
     * Removed the dynamic UUID prefix.
3. **Live Terminal Proof (Run this command in the terminal):**
   ```powershell
   python question1/run.py
   ```
   * **Highlight the Output:**
     * Total points: **8,142 points** across 4 tickets (average **2,035.5 pts/ticket**, down from **35,706 baseline**).
     * Every single ticket achieves the exact expected action and article ID.
4. **Live Test Suite Execution:**
   ```powershell
   python -m pytest tests/test_question1.py -v
   ```
   * Point out that all 4 tests pass, including the holdout subscription ticket `T-1091`.
5. **Detailed Breakdown Reference:**
   * Briefly show [question1/REPORT.md](file:///c:/Users/karti/OneDrive/Desktop/intern/Induseuro_labs/question1/REPORT.md#L30-L45) showing ticket `T-1001` saving **75.6%** (8,856 pts $\rightarrow$ 2,157 pts).

---

## Scene 3: Question 2 — 4-Stage Pipeline Optimization (2:00 – 4:00)

### Visuals
* Open [question2/agent.py](file:///c:/Users/karti/OneDrive/Desktop/intern/Induseuro_labs/question2/agent.py).

### Talking Points
1. **The Challenge in Question 2:**
   > *"In Question 2, the help center simulates an enterprise repository with over 200,000 tokens. The unoptimized baseline dumped 120,000 characters of filler into Triage and 80,000 characters into Audit, resulting in a staggering **293,562 points** (~73,300 points per ticket)."*
2. **Our Architectural Solution:**
   * Walk through the 4 stages in `question2/agent.py`:
     1. **Triage:** Concise prompt classifying the issue.
     2. **Retrieval (Local RAG):** Instead of asking an LLM to hallucinate 8 articles, `_retrieve()` performs deterministic local retrieval from `data/kb/`, retrieving the targeted policy excerpt with **0 LLM calls**.
     3. **Composition:** Generates the structured resolution using the order snapshot and retrieved policy.
     4. **Audit & Path Differentiation:**
        * *Physical Fulfillment (`T-1001`, `T-1004`):* Fast-path deterministic validation. (2 LLM calls total: `triage` + `compose`).
        * *Denial / Escalation (`T-1002`, `T-1003`, `T-1091`):* Uses local structured triage and reserves budget for an LLM `audit` call to verify policy compliance on high-risk actions. (2 LLM calls total: `compose` + `audit`).
3. **Live Terminal Proof (Run the Question 2 Runner):**
   ```powershell
   python question2/run.py
   ```
   * **Highlight the Output:**
     * Total points: **7,749 points** across all 4 tickets.
     * Average cost per ticket: **1,937.25 points** (well below the $\le 2,500$ ceiling!).
4. **Live Per-Stage Breakdown Proof (Run this command in the terminal):**
   ```powershell
   python -c "from question2.agent import SupportPipeline; [print(f'=== {t} ({p.handle_ticket(t).action}) Total: {p.session.total_points()} ===', '\n'.join(f'  [{c.label}] in={c.input_tokens} out={c.output_tokens} pts={c.points()}' for c in p.session.calls)) for t, p in [(tid, SupportPipeline()) for tid in ['T-1001', 'T-1002']]]"
   ```
   * **Explain the Output:**
     * `T-1001` (Physical Fulfillment): Shows `[triage]` (1,080 pts) + `[compose]` (1,029 pts) = **2,109 pts**.
     * `T-1002` (Denial): Shows `[compose]` (927 pts) + `[audit]` (1,014 pts) = **1,941 pts**.
5. **Live Test Suite Execution:**
   ```powershell
   python -m pytest tests/test_question2.py -v
   ```
   * All 4 tests pass cleanly.

---

## Scene 4: Full Test Suite & Git Verification (4:00 – 4:45)

### Visuals
* Terminal full view.

### Talking Points & Commands
1. **Run Full Test Suite:**
   ```powershell
   python -m pytest tests/test_question1.py tests/test_question2.py -v
   ```
   > *"Both test suites pass 100% in under 1 second: verifying accuracy on all public tickets, the holdout subscription test, and all cost constraints."*
2. **Show Clean Commit History:**
   ```powershell
   git log --oneline -n 6
   ```
   > *"All changes follow Conventional Commits, with isolated commits for baseline setup, prompt optimization, pipeline restructuring, and documentation."*

---

## Scene 5: Conclusion & Summary (4:45 – 5:15)

### Visuals
* Display the summary table in [question2/REPORT.md](file:///c:/Users/karti/OneDrive/Desktop/intern/Induseuro_labs/question2/REPORT.md#L60-L75).

### Talking Points
* **Recap the Results:**
  * **Question 1:** Slashed from **35,706 pts $\rightarrow$ 8,142 pts** (**-77.2%**, avg **2,035 pts/ticket** vs 3,000 target).
  * **Question 2:** Slashed from **293,562 pts $\rightarrow$ 7,749 pts** (**-97.4%**, avg **1,937 pts/ticket** vs 2,500 target).
  * 100% correctness preserved on all tickets and holdouts.
* **Closing:**
  > *"Thank you for watching!"*

---

## Quick Reference: Terminal Commands Cheat Sheet

Copy and paste these commands sequentially during your recording:

```powershell
# 1. Question 1 Demonstration
python question1/run.py
python -m pytest tests/test_question1.py -v

# 2. Question 2 Demonstration
python question2/run.py
python -m pytest tests/test_question2.py -v

# 3. Two-Path Per-Stage Breakdown Proof
python -c "from question2.agent import SupportPipeline; [print(f'=== {t} ({p.handle_ticket(t).action}) Total: {p.session.total_points()} ===', '\n'.join(f'  [{c.label}] in={c.input_tokens} out={c.output_tokens} pts={c.points()}' for c in p.session.calls)) for t, p in [(tid, SupportPipeline()) for tid in ['T-1001', 'T-1002']]]"

# 4. Combined Full Verification
python -m pytest tests/test_question1.py tests/test_question2.py -v
git log --oneline -n 6
```
