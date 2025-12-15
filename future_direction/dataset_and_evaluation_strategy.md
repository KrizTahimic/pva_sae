# Dataset and Evaluation Strategy Discussion

## Current Situation

### Pass Rates (Zero-Shot, MBPP)
| Model | Pass Rate | Expected (0-shot) | Status |
|-------|-----------|-------------------|--------|
| Gemma-2B | 32% | 25-30% | Normal |
| Gemma-9B | 53% | 45-55% | Normal |
| Llama-8B | 51% | 45-55% | Normal |

### Current Split Sizes (MBPP, 974 total)
| Split | Tasks | Purpose |
|-------|-------|---------|
| Selection (sae) | 489 | Direction discovery |
| Tuning | 97 | Hyperparameter selection |
| Analysis | 388 | Final evaluation |

---

## The Core Question: Is ~50% Pass Rate a Problem?

**For standard benchmarking**: Yes, we'd want higher to compare with literature.

**For OUR research (SAE correctness directions)**: **Maybe not!**

We need:
1. Enough **correct** samples to find correct-predicting latents
2. Enough **incorrect** samples to find incorrect-predicting latents
3. Balanced classes for steering experiments

**50/50 is actually ideal for binary classification research.**

### What Matters More: Pass Rate or Sample Size?

| Metric | Impact on Research |
|--------|-------------------|
| Pass rate 70% vs 50% | Minimal - both give enough of each class |
| Sample size 400 vs 1000 | **Significant** - affects statistical power |

---

## Option 1: Test Instruct Models First (Phase 7.3)

### What This Tells Us
- How much instruction-tuning affects code correctness
- Whether our findings generalize across model variants
- Quick sanity check before bigger changes

### Expected Pass Rates
| Model | Base | Instruct (expected) |
|-------|------|---------------------|
| Gemma-2B | 32% | 40-50% |
| Gemma-9B | 53% | 60-70% |
| Llama-8B | 51% | 60-70% |

### Pros
- Quick to run (same infrastructure)
- Validates experimental setup
- Doesn't require dataset changes

### Cons
- Doesn't address sample size concerns
- May have different activation patterns (instruction tuning changes representations)

### Recommendation
**Do this first** - it's low-cost and informative.

---

## Option 2: Switch to MBPP+ (EvalPlus)

### What is MBPP+?
- **399 tasks** (sanitized from original 974)
- Removes ambiguous/broken problems
- **80 additional test cases per task** (more rigorous)
- What Meta/Google report in papers

### Split Sizes with MBPP+
| Split | Tasks | Concern |
|-------|-------|---------|
| Selection | 199 | Acceptable |
| Tuning | 39 | **Very small** |
| Analysis | 159 | Marginal |

### Pros
- Comparable to published benchmarks
- Cleaner problems (less noise)
- More rigorous test cases

### Cons
- **60% fewer tasks** (399 vs 974)
- Tuning split too small (39 tasks)
- May not have enough statistical power

### Recommendation
**Not recommended as primary dataset** due to size. Could use for validation only.

---

## Option 3: Few-Shot Prompting

### Impact on Pass Rates
| Model | 0-shot | 3-shot (expected) |
|-------|--------|-------------------|
| Gemma-9B | 53% | 65-70% |
| Llama-8B | 51% | 65-70% |

### Pros
- Matches standard benchmark methodology
- Higher pass rates if that matters

### Cons
- **Changes what we're measuring**
- Few-shot examples affect activations
- Not clear if SAE directions transfer between 0-shot and few-shot

### Research Question
Are correctness-predicting latents the same for 0-shot vs few-shot? This could be a separate experiment.

### Recommendation
**Don't change primary setup** - but could study as separate question.

---

## Option 4: Use Larger/Different Dataset (APPS, etc.)

### Dataset Comparison

| Dataset | Tasks | Difficulty | Pass Rate (est.) |
|---------|-------|------------|------------------|
| MBPP | 974 | Easy-Medium | 50% |
| MBPP+ | 399 | Easy-Medium | 55% |
| HumanEval | 164 | Medium | 40% |
| HumanEval+ | 164 | Medium-Hard | 30% |
| APPS | 10,000 | Hard | 10-20% |
| CodeContests | 13,328 | Very Hard | 5-10% |
| DS-1000 | 1,000 | Medium | 30-40% |
| MTPB | 115 | Multi-turn | varies |

### APPS Concerns
- **Very low pass rates** (~10-20% for these models)
- Would give mostly incorrect samples
- More complex setup (different test format)

### Potential Combination Strategy
| Dataset | Tasks | Use For |
|---------|-------|---------|
| MBPP | 974 | Primary (selection, tuning) |
| HumanEval | 164 | Cross-validation |
| MBPP+ | 399 | Rigorous validation |

### Recommendation
**Stick with MBPP as primary** - it's the right difficulty level. Use HumanEval for generalization testing (already implemented).

---

## Option 5: Combine Multiple Datasets

### Proposed Combined Dataset
| Source | Tasks | After Filtering |
|--------|-------|-----------------|
| MBPP | 974 | 974 |
| HumanEval | 164 | 164 |
| **Total** | 1,138 | ~1,100 |

### Combined Splits
| Split | Tasks |
|-------|-------|
| Selection | 550 |
| Tuning | 110 |
| Analysis | 440 |

### Pros
- More statistical power
- Tests generalization across problem types
- HumanEval already implemented

### Cons
- Different prompt formats may confuse model
- More complexity in evaluation
- Results harder to compare with literature

### Recommendation
**Keep separate** - Use MBPP for training, HumanEval for generalization testing.

---

## Key Statistical Considerations

### Sample Size Requirements for SAE Analysis

For t-statistic latent selection (Phase 2.10):
- Minimum 10 samples per class per layer
- With 50% pass rate and 489 selection tasks: ~245 correct, ~245 incorrect
- **Sufficient for statistical analysis**

For steering experiments (Phase 4.x):
- Need enough tasks to detect effect sizes
- With 388 analysis tasks: ~194 each class
- **Sufficient for statistical significance**

### What Would Break Our Analysis?
| Scenario | Problem |
|----------|---------|
| <50 tasks per class | Unreliable t-statistics |
| >90% or <10% pass rate | Class imbalance |
| <100 tasks total | Insufficient power for steering |

**Current setup avoids all these problems.**

---

## Recommended Strategy

### Immediate Actions (Low Cost)

1. **Run Phase 7.3 with instruct models**
   - Test Gemma-2B-IT, Gemma-9B-IT, Llama-8B-IT
   - Compare pass rates to base models
   - Check if SAE directions transfer

2. **Continue with current MBPP setup**
   - 50% pass rate is fine for our research
   - Sample sizes are adequate

### Future Validation (If Time Permits)

3. **Cross-validate on HumanEval**
   - Already implemented (Phase 0.2)
   - Tests generalization of SAE directions

4. **MBPP+ for publication**
   - Run final results on MBPP+ for comparability
   - Report both MBPP and MBPP+ numbers

### Don't Do (Unless Necessary)

- Switch to APPS (too hard, wrong difficulty)
- Add few-shot prompting (changes what we measure)
- Combine datasets (unnecessary complexity)

---

## Decision Matrix

| Option | Effort | Benefit | Risk | Recommend? |
|--------|--------|---------|------|------------|
| Instruct models (7.3) | Low | Medium | Low | **Yes** |
| MBPP+ only | Medium | Low | High (size) | No |
| Few-shot prompting | Medium | Low | Medium | No |
| APPS/larger dataset | High | Low | High | No |
| Combined datasets | High | Medium | Medium | Maybe later |
| Continue as-is | None | High | None | **Yes** |

---

## Questions to Resolve

1. **Is 50% pass rate acceptable for the paper?**
   - If yes: Continue as-is
   - If no: Need to justify or change setup

2. **Do we need to match published benchmarks exactly?**
   - If yes: Need MBPP+ or few-shot
   - If no: Current setup is fine

3. **Is the goal to maximize pass rate or maximize research insight?**
   - Pass rate: Change prompting/models
   - Research insight: Current setup is optimal

---

## Summary

**The "problem" of 50% pass rate may not be a problem at all.**

For SAE correctness research:
- We need balanced correct/incorrect samples: ✓
- We need sufficient sample sizes: ✓
- We need consistent evaluation: ✓

**Recommended next step**: Run Phase 7.3 with instruct models to see if pass rates improve and whether SAE directions transfer. This is low-cost and informative.

---

## Appendix: Failure Mode Analysis (December 2024)

### Investigation Results

We investigated whether failures are legitimate (true negatives) or evaluation artifacts (false negatives).

### Failure Breakdown (Gemma-2B, n=68 failures)

| Category | % | Verdict |
|----------|---|---------|
| WRONG_LOGIC | 50% | True negative ✓ |
| TYPE_MISMATCH | 15% | Mostly true negative |
| RUNTIME_ERROR | 9% | True negative ✓ |
| WRONG_FUNCTION_NAME | 6% | True negative |
| SYNTAX_ERROR | 2% | True negative ✓ |

### False Negative Rate: ~1%

Only **1 clear false negative** found across 100 tasks:
- Task 2: Returns `[4, 5]` instead of `(4, 5)`
- Values correct, type wrong
- Prompt is ambiguous ("tuple lists")

All other suspicious cases had **wrong logic** that happened to pass the first test by coincidence.

### MBPP Test Quality

Out of 257 sanitized MBPP tasks:
- 19 (7.4%) have ambiguous tuple/list language
- 17 (6.6%) have floating point comparisons
- 13 (5.1%) use sets (order may vary)

In practice, these rarely cause false negatives because models that get the type wrong usually also have wrong logic.

### Conclusion: MBPP Is Fine For Our Research

**Do NOT switch to MBPP+** because:
1. False negative rate is ~1%, not a significant concern
2. MBPP has 974 tasks vs MBPP+ 399 tasks (more statistical power)
3. Failures are legitimate algorithm errors, not evaluation bugs
4. 50% pass rate provides ideal class balance for SAE research

**MBPP+ advantages don't apply to us:**
- More tests per task → we only need binary pass/fail
- Sanitized problems → our failures are real errors
- 60% fewer tasks → worse for our analysis
