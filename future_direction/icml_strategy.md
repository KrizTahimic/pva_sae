# Strategic Analysis: Maximizing ICML Acceptance

## Current State Summary

| Aspect | Status |
|--------|--------|
| **ICLR Rating** | 4.0 average (below threshold, but 7JAK gave 6) |
| **Universal Criticism** | Single model (Gemma-2B) + single benchmark (MBPP) |
| **Code Infrastructure** | Ready for LLAMA, Gemma-9B, HumanEval |
| **LLAMA/Gemma-9B Data** | Only 100-sample test runs completed |
| **Selective Steering** | Implemented, needs presentation |

---

## Priority Actions (Ranked by Impact)

### CRITICAL: Generalization Experiments (All 4 Reviewers)

**This is the single most important fix.** Every reviewer cited lack of generalization.

| Experiment | Purpose | Compute Time |
|------------|---------|--------------|
| LLAMA-3.1-8B + MBPP (full) | Cross-architecture validation | ~2-3 days |
| Gemma-2-2B + HumanEval | Cross-benchmark validation | ~1 day |
| LLAMA-3.1-8B + HumanEval | Both generalization axes | ~1 day |

**What to report:**
1. Do similar latent directions exist in LLAMA? (Phase 2.5)
2. Do AUROC/F1 metrics transfer? (Phase 3.8)
3. Does steering work similarly? (Phase 4.8)

**Minimum viable result:** Even if LLAMA shows different patterns, that's a publishable finding ("architecture-specific mechanisms").

---

### HIGH IMPACT: Visualizations & Presentation

**1. Feature-Selection Landscape (Reviewer jwL5, 7JAK)**
```
Scatter plot:
- X-axis: separation score
- Y-axis: t-statistic
- All 16k latents as gray dots
- Selected latents as colored outliers
- Show >3σ threshold lines
```
**Impact:** Makes feature selection "auditable at a glance" - directly addresses reviewer concern about hand-picking.

**2. Top-10 Latents Table**
| Layer | Latent | Sep. Score | t-stat | AUROC | Role |
|-------|--------|------------|--------|-------|------|
| 19 | 12345 | 0.85 | 8.2 | 0.82 | Correct-steering |
| 16 | 67890 | -0.72 | -7.5 | 0.79 | Incorrect-predicting |

**3. Steering Coefficient Search Plots (Reviewer 7JAK)**
- X: coefficient value
- Y: correction/corruption rate
- Shows coefficient wasn't arbitrary

---

### MEDIUM IMPACT: Methodological Clarifications

**Writing Fixes (low effort, high clarity):**
- [ ] Add missing references (2 SAE survey papers)
- [ ] Fix Figure 3 values (placeholder -> actual data)
- [ ] Define MBPP at first mention (line 56)
- [ ] Fix Elhage 2022 broken link
- [ ] Clarify equations 5-7 indexing
- [ ] Clarify AUROC/F1 threshold methodology

**Feature Threshold Sensitivity (Reviewer RXZd):**
- Test 1%, 2%, 5% thresholds on pile-10k
- Report: "Of top-20 candidates, X filtered at 2% threshold"
- Quick computation, demonstrates robustness

---

### STRATEGIC FRAMING (No Compute Required)

**1. Reframe Asymmetry as Core Contribution:**

> "We discover a fundamental asymmetry: LLMs encode incorrect code as detectable anomalies (F1=0.821) but lack corresponding representations for correctness (F1=0.504). This reveals that models perform anomaly detection rather than validity assessment—a critical insight for deployment strategies."

**2. Position SAE Limitations Honestly:**

> "We employ SAEs as established interpretability tools with known limitations (polysemanticity, imperfect reconstruction), not as solutions to the entanglement problem. Our contribution is applying existing methods to the previously unexplored domain of code correctness."

**3. Selective Steering as Practical Solution:**

> "Universal steering produces 4.04% correction but 14.66% corruption. We demonstrate that combining predictor directions with steering (applying intervention only when predictors indicate errors) eliminates corruption while preserving correction benefits."

---

### NOVEL EXPERIMENT: CoT Faithfulness (High Risk/High Reward)

From `cot_analysis_methods_v2.md`:

**Research Question:** When incorrect-predicting features activate during reasoning, does the model's language match its internal state?

| Scenario | Internal State | Generated Text | Interpretation |
|----------|---------------|----------------|----------------|
| A | Error detected (HIGH) | "wait, that's wrong" | Faithful |
| B | Error detected (HIGH) | "this works perfectly" | Overconfident |
| C | No error (LOW) | "oops, mistake" | Overcautious |
| D | No error (LOW) | "this is correct" | Faithful |

**Why this could be impactful:**
- No existing work connects SAE features to CoT faithfulness
- Addresses AI safety concerns about reasoning model honesty
- Uses DeepSeek-R1-Distill (timely/relevant model)

**Risks:**
- LlamaScope SAE transfer to distilled model is uncertain
- ~2-3 days additional compute
- May not find significant results

**Recommendation:** Run this if LLAMA experiments show positive results. It could differentiate your paper significantly at ICML.

---

## Recommended Action Plan

### Phase 1: Critical Fixes (1-2 weeks)

| Task | Days | Priority |
|------|------|----------|
| Run LLAMA full pipeline (Phase 1->4.8) | 3-4 | Critical |
| Run HumanEval on Gemma-2B (Phase 1->3.8) | 2 | Critical |
| Feature-Selection Landscape visualization | 0.5 | High |
| Top-10 latents table | 0.5 | High |
| Writing fixes (references, Figure 3, etc.) | 1 | Medium |

### Phase 2: Strengthening (1 week)

| Task | Days | Priority |
|------|------|----------|
| Pile threshold sensitivity analysis | 0.5 | Medium |
| Steering coefficient plots | 0.5 | Medium |
| Present selective steering results | 0.5 | High |
| SAE limitations discussion | 0.5 | Medium |

### Phase 3: Novel Contribution (Optional, 1 week)

| Task | Days | Priority |
|------|------|----------|
| CoT faithfulness experiment | 3-4 | High risk/reward |

---

## Summary: What Changes Acceptance Probability Most?

| Change | Probability Boost | Effort |
|--------|-------------------|--------|
| LLAMA + MBPP experiments | +15-20% | High |
| HumanEval experiments | +10% | Medium |
| Feature-Selection Landscape viz | +5% | Low |
| Selective steering presentation | +5% | Low |
| Writing/reference fixes | +3% | Low |
| CoT faithfulness (if positive) | +10-15% | High |

**Bottom line:** The generalization experiments (LLAMA + HumanEval) are non-negotiable for ICML. Everything else is valuable but secondary.

---

## Optional: Potential Blind Spots (Not in ICLR Feedback)

These are gaps that ICLR reviewers didn't explicitly flag, but ICML reviewers might.

### 1. Baseline Comparisons (Significant Gap)

ICML is methods-focused. The implicit question: "Does SAE-based detection add value over simpler approaches?"

| Baseline | What it tests | Effort |
|----------|---------------|--------|
| Linear probe on raw activations (no SAE) | Is SAE decomposition necessary? | 1-2 days |
| Output confidence/perplexity | Can you just use model's own uncertainty? | 1 day |
| Random latent directions | Is the specific direction important? | 0.5 days |

**If linear probes work equally well**, the contribution shrinks. If SAE directions work better, that's a strong argument.

### 2. Error Type Breakdown

Reviewer RXZd asked, marked as "defend." But **having the data is stronger.**

| Error Type | Count | Predictor F1 | Steering Effect |
|------------|-------|--------------|-----------------|
| Syntax errors | ? | ? | ? |
| Runtime errors | ? | ? | ? |
| Wrong logic | ? | ? | ? |

**Effort:** Low (~1 day). Code execution results exist, just need categorization.

### 3. Qualitative Examples (Zero Compute)

Missing entirely. Would make paper more compelling:
- 2-3 examples of successful correction (before/after code)
- 2-3 examples of corruption (why did steering break it?)
- What tokens does the latent "see"? (Logit lens on specific examples)

### 4. Incorrect-Steering Failure Investigation

Correct-steering works, but incorrect-steering produces garbage ("8888..."). Paper acknowledges but doesn't investigate why.

Deeper analysis could reveal:
- Is separation score methodology flawed for one direction?
- Is there something fundamentally different about how models encode "incorrect"?

Could become a **novel finding** rather than a limitation.

### 5. Statistical Completeness

Check if reporting:
- [ ] Confidence intervals on AUROC/F1
- [ ] Standard deviation across problem splits
- [ ] Effect sizes (Cohen's d) for steering
- [ ] Multiple comparison corrections if testing many latents

### Optional Tasks Summary

| Gap | Impact if Missing | Effort |
|-----|-------------------|--------|
| Baseline comparisons | Could be fatal ("why SAE?") | Medium |
| Error type breakdown | Weakens defense | Low |
| Qualitative examples | Less compelling | Very low |
| Incorrect-steering investigation | Missed opportunity | Medium |
| Statistical completeness | Minor unless flagged | Low |
