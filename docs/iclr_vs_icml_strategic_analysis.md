# Strategic Analysis: ICLR 2026 Rebuttal vs. ICML 2026 Submission

**Date:** November 26, 2025
**Context:** Deciding whether to pursue ICLR rebuttal or pivot to ICML with extended research (including circuit analysis)

---

## Executive Summary

**My recommendation: Pivot to ICML 2026 with circuit analysis.**

This isn't just about buying time—it's about transforming a "good application paper" into a "significant mechanistic interpretability contribution." Your current work discovers that code correctness directions exist; circuit analysis would reveal *how* they're computed. That's a fundamentally more interesting story.

---

## Current State Assessment

### ICLR 2026 Reviews Summary

| Reviewer | Rating | Core Concern |
|----------|--------|--------------|
| RXZd | 2 (reject) | Single model/benchmark; 14.66% corruption |
| vRko | 4 (marginal below) | Generalization; practical utility |
| 7JAK | 6 (marginal above) | Methodology clarity; robustness |
| jwL5 | 4 (marginal below) | Documentation; single benchmark |

**Average: 4.0** — All reviewers cite single-model limitation as major weakness.

### Research Completion Status

| Experiment | Status | Notes |
|------------|--------|-------|
| Gemma-2-2B + MBPP | 100% | Full pipeline complete |
| Gemma-2-2B + HumanEval | ~70% | Most phases done |
| LLAMA-3.1-8B + MBPP | ~5% | Only 10-sample test run |
| LLAMA + HumanEval | 0% | Not started |
| Circuit Analysis | 0% | Mentioned as future work |

---

## Option A: ICLR 2026 Rebuttal

### Effort Required
- **Total:** 87-124 hours
- **Timeline:** 2-3 weeks (rebuttal period)
- **Critical path:** LLAMA experiments (40-60 hours alone)

### Success Probability: ~35-45%

**Why so low?**
1. RXZd (rating 2) has a fundamental concern about single-model validation—hard to move
2. You can't add new experiments in a rebuttal, only argue with existing results
3. Even if you run LLAMA now, it won't be in the paper—only mentioned in rebuttal text
4. The 4.04% correction vs 14.66% corruption tradeoff is real and won't disappear

### What You'd Need to Do
1. Complete selective steering analysis (showing it reduces corruption)
2. Create feature visualization scatter plots
3. Reframe asymmetric findings positively
4. Add missing references and fix presentation issues
5. Write ~800-word rebuttal carefully addressing each reviewer

### Risk Assessment
- **High risk:** You do 100+ hours of work for a ~40% acceptance chance
- **Opportunity cost:** Time spent on rebuttal is time not spent improving the research
- **Even if accepted:** Paper has a weaker narrative than it could have

---

## Option B: ICML 2026 with Extended Research

### Why This Path is Superior

#### 1. **Stronger Narrative Arc**

Current story (ICLR):
> "We found code correctness directions exist and can be used for steering"

Extended story (ICML):
> "We found code correctness directions, traced their computational origins through circuit analysis, and validated across multiple models/benchmarks"

The second is a more complete scientific contribution.

#### 2. **Circuit Analysis Adds Real Value**

From your thesis Chapter 5:
> "Circuit Analysis: We identified code correctness directions but did not investigate what upstream features activate them. Circuit analysis could trace which prior features (syntax, semantics, test case matching) trigger code correctness latents."

This directly addresses the mechanistic interpretability community's core interest: not just *what* features exist, but *how* they're computed.

#### 3. **Multi-Model Validation Becomes a Strength, Not a Fix**

Instead of scrambling to add LLAMA results to satisfy reviewers, you can:
- Properly design cross-model experiments
- Compare circuit structures across Gemma vs LLAMA
- Make generalization a core contribution

#### 4. **Better Venue Fit**

ICML has strong representation of:
- Mechanistic interpretability researchers (Anthropic, DeepMind, etc.)
- SAE methodology developers
- Circuit analysis practitioners

Your work fits naturally in this community.

---

## Circuit Analysis: Is It Worth Adding?

### What Circuit Analysis Would Involve

**Goal:** Trace which upstream features activate your code correctness latents

**Methods:**
1. **Attribution patching** — Which earlier features causally influence correctness features?
2. **Path tracing** — What's the computational path from input → correctness detection?
3. **Component ablation** — Which heads/MLPs are necessary for correctness features?

### Difficulty Assessment

| Aspect | Difficulty | Notes |
|--------|------------|-------|
| Infrastructure | **Medium** | You have activation hooks; need path patching |
| Compute | **Medium-High** | Many ablation experiments |
| Novelty | **High** | No one has done this for code correctness |
| Interpretability | **Medium** | Results may be messy/hard to interpret |

### Estimated Timeline

| Task | Hours | Weeks |
|------|-------|-------|
| Literature review (TransformerLens, circuit papers) | 15-20 | 1 |
| Implement attribution patching | 25-35 | 2 |
| Run circuit experiments | 30-40 | 2-3 |
| Analysis and visualization | 20-25 | 1-2 |
| Cross-model comparison (Gemma vs LLAMA) | 30-40 | 2-3 |
| **Total** | **120-160** | **8-11** |

### Is It Worth It?

**Yes, for these reasons:**

1. **Unique contribution:** No one has traced code correctness circuits before
2. **Connects your findings:** Your attention analysis (test cases matter) becomes the *starting point* for circuit tracing
3. **Builds on existing infrastructure:** You already have:
   - Multi-layer activation extraction
   - SAE feature identification
   - Steering hooks infrastructure
4. **Natural research progression:** Discovery → Validation → Mechanism (you're missing mechanism)

**Potential concerns:**

1. **Results might be messy:** Circuits can be complex and hard to interpret
2. **Scope creep risk:** Circuit analysis can become an entire PhD thesis
3. **New methodology to learn:** TransformerLens, path patching, etc.

**Mitigation:** Scope it tightly:
- Focus on ONE circuit: "What activates the incorrect-predicting feature (L19-5441)?"
- Don't try to map entire model, just code correctness pathway
- Use attention analysis as guide (test case processing → correctness)

---

## Concrete ICML Plan

### Phase 1: Complete Multi-Model Validation (4-6 weeks)
- Finish LLAMA + MBPP full pipeline
- Add HumanEval + LLAMA
- Create proper cross-model comparison tables

### Phase 2: Circuit Analysis (6-8 weeks)
- Implement attribution patching using TransformerLens
- Trace upstream features for incorrect-predicting direction
- Compare circuit structure: Gemma vs LLAMA
- Visualize computational graph

### Phase 3: Paper Revision (3-4 weeks)
- Restructure paper with circuit analysis as key contribution
- Add all reviewer-requested fixes (references, methodology clarity)
- New figures: circuit diagrams, cross-model comparison

### Phase 4: Polish & Submit (2 weeks)
- Internal review
- Supplementary materials
- ICML submission

**Total Timeline:** ~4 months (fitting ICML 2026 deadline)

---

## Decision Framework

### Choose ICLR Rebuttal If:
- Your advisor strongly prefers faster publication
- You have <40 hours available in next 2 weeks
- You're risk-tolerant (40% acceptance is acceptable)
- You need this publication for a specific deadline (job market, graduation)

### Choose ICML With Circuit Analysis If:
- You want to maximize impact and learning
- You have 4+ months available
- You enjoy the research (you said "I'm enjoying this so much")
- You want a stronger publication for your portfolio
- You're interested in mechanistic interpretability long-term

---

## What You'd Gain From Each Path

### ICLR Rebuttal Path
- **Best case:** Publication in top venue, but with known limitations
- **Worst case:** Rejection + same paper needs more work anyway
- **Learning:** Limited (mostly writing/documentation)

### ICML + Circuit Analysis Path
- **Best case:** Stronger publication + new methodology expertise
- **Worst case:** More work but better paper regardless of outcome
- **Learning:** Significant (circuit analysis, TransformerLens, cross-model MI)

---

## My Honest Opinion

You said:
> "I'm enjoying this research so much"

and

> "So it won't be plain direction finding but also circuit analysis!!!!"

That enthusiasm matters. Research is hard, and working on something you find exciting leads to better outcomes than grinding through a rebuttal for a paper you know is incomplete.

**Circuit analysis is the right move because:**

1. **It completes the scientific story.** Right now you've found "what" (correctness directions exist) and "whether they work" (steering has effects). Circuit analysis answers "how"—which is the most interesting question.

2. **It differentiates your work.** Many papers apply SAEs to new domains. Fewer trace the computational circuits. This positions you uniquely.

3. **It builds skills.** Circuit analysis is a growth area in mechanistic interpretability. Learning it now pays dividends.

4. **It's more enjoyable.** You're excited about it. That matters.

The main risk is scope creep—circuit analysis can explode in complexity. My advice: scope it tightly to tracing ONE circuit (the incorrect-predicting feature's upstream activators), validate across two models, and call it done.

---

## Next Steps If You Choose ICML

1. **This week:** Read key circuit analysis papers
   - [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
   - [Interpretability in the Wild](https://arxiv.org/abs/2211.00593)
   - TransformerLens documentation

2. **Week 2-3:** Implement attribution patching on your existing code
   - Start with Gemma + incorrect-predicting feature
   - Use your attention analysis as hypothesis guide

3. **Month 2:** Full LLAMA pipeline + initial circuit results

4. **Month 3:** Cross-model circuit comparison + paper restructuring

5. **Month 4:** Polish and submit

---

## Summary

| Factor | ICLR Rebuttal | ICML + Circuits |
|--------|---------------|-----------------|
| Time required | 87-124 hours | 200-280 hours |
| Timeline | 2-3 weeks | 4 months |
| Success probability | 35-45% | 60-75% |
| Enjoyment | Low (fixing) | High (discovering) |
| Learning | Minimal | Significant |
| Impact | Incremental | Substantial |
| Career value | One publication | Publication + expertise |

**Bottom line:** If you have the time and interest (and you clearly do), ICML with circuit analysis is the stronger path. You'll end up with a better paper, more skills, and likely more enjoyment along the way.

The ICLR reviewers did you a favor by identifying that your work, while solid, needs the multi-model validation and deeper mechanistic understanding to reach its full potential. Take that feedback and use the extra time to deliver on it properly.

---

## Questions for You

Before finalizing this recommendation, I'd like to understand:

1. **What's your timeline pressure?** Do you have graduation/job deadlines that require a publication by a certain date?

2. **Advisor preference?** Does your advisor have strong feelings about ICLR vs ICML?

3. **Compute availability?** Circuit analysis requires many experiments—do you have dedicated GPU access?

4. **TransformerLens familiarity?** Have you used it before, or would this be new learning?
