# CRITICAL BUG: Double Assert in Prompt Generation

## Discovery Date
2025-11-21

## Severity
🔴 **CRITICAL** - Affects all MBPP + Gemma-2-2B experiments requiring complete pipeline re-run

## Root Cause

### The Bug
Phases 3.5, 3.6, and 7.3 were adding `assert` prefix to test_list items that already contained `assert`:

```python
# test_list in Phase 0.1/0.2 data:
['assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8', ...]

# Buggy code in Phases 3.5, 3.6, 7.3:
test_cases_str = "\n".join([
    f"assert {test.strip()}" for test in row['test_list']  # ❌ ADDS ANOTHER ASSERT
])

# Result in generated prompts:
assert assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8  # ❌ Invalid Python syntax!
```

### Why It Happened
- **Incorrect Assumption**: Code assumed test_list items didn't include `assert` prefix
- **Actual Data Format**: Both MBPP (Phase 0.1) and HumanEval (Phase 0.2) already store complete assertions with `assert` keyword

### Example of Malformed Prompt

```python
Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].

assert assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8
assert assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12
assert assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16

# Solution:
```

## Impact

### Directly Affected Experiments (Built Malformed Prompts)
- ❌ **Phase 3.5** (Temperature Robustness) - All experiments invalid
- ❌ **Phase 3.6** (Hyperparameter Tuning) - All experiments invalid
- ❌ **Phase 7.3** (Instruction-Tuned Baseline) - All experiments invalid
- ❌ **Phase 8.2** (Threshold Optimizer) - All experiments invalid

### Cascade Affected Experiments (Depend on Buggy Phases)

**Validation & Evaluation Phases** (depend on Phase 3.5/3.6):
- ❌ **Phase 3.8** (AUROC/F1 Evaluation) - Uses test results from Phase 3.5/3.6
- ❌ **Phase 3.10** (Temperature AUROC) - Depends on Phase 3.5/3.8
- ❌ **Phase 3.11** (Temperature Visualization) - Depends on Phase 3.10
- ❌ **Phase 3.12** (Difficulty AUROC) - Depends on Phase 3.5/3.8

**Steering Coefficient Phases** (depend on Phase 3.6):
- ❌ **Phase 4.5** (Coefficient Selection) - Loads test data from Phase 3.6
- ❌ **Phase 4.6** (Golden Section Refinement) - Depends on Phase 4.5
- ❌ **Phase 4.7** (Coefficient Visualization) - Depends on Phase 4.5/4.6

**Steering Effect Phases** (depend on Phase 3.5):
- ❌ **Phase 4.8** (Steering Effect Analysis) - Uses test data from Phase 3.5
- ❌ **Phase 4.12** (Zero-Disc Steering) - Uses test data from Phase 3.5
- ❌ **Phase 4.14** (Significance Testing) - Depends on Phase 4.8/4.12

**Weight Orthogonalization Phases** (depend on Phase 3.5):
- ❌ **Phase 5.3** (Weight Orthogonalization) - Uses test data from Phase 3.5
- ❌ **Phase 5.6** (Zero-Disc Orthogonalization) - Uses test data from Phase 3.5
- ❌ **Phase 5.9** (Orthogonalization Significance) - Depends on Phase 4.8/5.3/5.6

**Attention & Mechanistic Phases** (depend on Phase 3.5/4.8):
- ❌ **Phase 6.3** (Attention Analysis) - Uses activations from Phase 3.5/4.8

**Instruction-Tuned Model Phases** (depend on Phase 7.3):
- ❌ **Phase 7.6** (Instruct Steering) - Uses test data from Phase 7.3
- ❌ **Phase 7.9** (Universality Analysis) - Depends on Phase 3.5/4.8/7.3/7.6
- ❌ **Phase 7.12** (Instruct AUROC/F1) - Uses test data from Phase 7.3

**Threshold & Selective Steering Phases** (depend on Phase 3.5/3.6/3.8):
- ❌ **Phase 8.1** (Threshold Calculator) - Depends on Phase 3.6
- ❌ **Phase 8.3** (Selective Steering) - Depends on Phase 3.5/3.8

**Note**: These 20 phases don't have the double assert bug in their code, but they all depend on test results (test_passed labels, pass rates, steering effects) from the 4 buggy phases. Since the upstream test results are invalid, all downstream analyses must be recomputed.

### Unaffected Phases (No Re-run Needed)
- ✅ **Phase 0, 0.1, 0.2, 0.3** (Data Preparation) - No test execution
- ✅ **Phase 1** (Baseline Generation) - Correct format: `'\n'.join(task['test_list'])`
- ✅ **Phase 2.2, 2.5, 2.10, 2.15** (SAE Analysis) - Uses activations only, no test execution
- ✅ **Phase 4.10** (Zero-Disc Feature Selection) - Statistical analysis of features, doesn't execute tests

**Total unaffected**: 9 out of 33 phases (27%)

### Datasets Affected
- Both **MBPP** and **HumanEval** affected (both store assertions with `assert` keyword)

## Complete Re-run Requirements

### Summary
**24 out of 33 phases (73%) require re-running** after the bug fix.

- **4 phases** have the double assert bug directly ✅ **Code FIXED**
- **20 phases** have cascade dependencies ⚠️ **Must RE-RUN**
- **9 phases** are unaffected ✅ **Can SKIP**

### Re-run Execution Order (By Dependency Tiers)

#### **Tier 1: Direct Bug Fixes** (Must Run First)
These phases had the bug in their code and must be re-run first:

1. **Phase 3.5** - Temperature Robustness (2-6 hours)
2. **Phase 3.6** - Hyperparameter Tuning (2-6 hours)
3. **Phase 7.3** - Instruction-Tuned Baseline (2-6 hours)
4. **Phase 8.2** - Threshold Optimizer (1-2 hours)

**Parallelization**: All 4 can run in parallel
**Estimated time**: 2-6 hours (if parallel), 8-20 hours (if sequential)

---

#### **Tier 2: Immediate Dependencies** (Run After Tier 1)
These phases directly depend on Tier 1 outputs:

5. **Phase 3.8** - AUROC/F1 Evaluation (needs 3.5, 3.6) [1-2 hours]
6. **Phase 4.5** - Coefficient Selection (needs 3.6) [2-4 hours]
7. **Phase 8.1** - Threshold Calculator (needs 3.6) [1 hour]

**Parallelization**: All 3 can run in parallel after Tier 1 completes
**Estimated time**: 2-4 hours (if parallel), 4-7 hours (if sequential)

---

#### **Tier 3: Secondary Dependencies** (Run After Tier 2)
These phases depend on Tier 2 outputs:

8. **Phase 4.6** - Golden Section Refinement (needs 4.5) [2-3 hours]
9. **Phase 4.8** - Steering Effect Analysis (needs 3.5) [3-5 hours]
10. **Phase 8.3** - Selective Steering (needs 3.5, 3.8) [2-3 hours]

**Parallelization**: All 3 can run in parallel after Tier 2 completes
**Estimated time**: 3-5 hours (if parallel), 7-11 hours (if sequential)

---

#### **Tier 4: Tertiary+ Dependencies** (Run After Tier 3)
These phases have deeper transitive dependencies:

11. **Phase 3.10** - Temperature AUROC (needs 3.5, 3.8) [1-2 hours]
12. **Phase 3.11** - Temperature Visualization (needs 3.10) [<1 hour]
13. **Phase 3.12** - Difficulty AUROC (needs 3.5, 3.8) [1-2 hours]
14. **Phase 4.7** - Coefficient Visualization (needs 4.5, 4.6) [<1 hour]
15. **Phase 4.12** - Zero-Disc Steering (needs 3.5, 4.10) [2-3 hours]
16. **Phase 4.14** - Significance Testing (needs 4.8, 4.12) [1 hour]
17. **Phase 5.3** - Weight Orthogonalization (needs 3.5) [3-5 hours]
18. **Phase 5.6** - Zero-Disc Orthogonalization (needs 3.5, 4.10) [2-3 hours]
19. **Phase 5.9** - Orthogonalization Significance (needs 4.8, 5.3, 5.6) [1 hour]
20. **Phase 6.3** - Attention Analysis (needs 3.5, 4.8) [1-2 hours]
21. **Phase 7.6** - Instruct Steering (needs 7.3) [2-4 hours]
22. **Phase 7.9** - Universality Analysis (needs 3.5, 4.8, 7.3, 7.6) [<1 hour]
23. **Phase 7.12** - Instruct AUROC/F1 (needs 7.3) [1-2 hours]

**Parallelization**: Some can run in parallel (e.g., 3.10/3.12/5.3/7.6)
**Estimated time**: 5-8 hours (if parallelized), 17-29 hours (if sequential)

---

### Total Estimated Re-run Time

**Best case (maximum parallelization)**:
- Tier 1: 6 hours (parallel)
- Tier 2: 4 hours (parallel)
- Tier 3: 5 hours (parallel)
- Tier 4: 8 hours (partial parallel)
- **Total: ~23 hours (1 day)**

**Worst case (all sequential)**:
- Tier 1: 20 hours
- Tier 2: 7 hours
- Tier 3: 11 hours
- Tier 4: 29 hours
- **Total: ~67 hours (3 days)**

**Realistic (mixed parallel/sequential)**:
- With 2-4 parallel screens: **~30-40 hours (1.5-2 days)**

## Data Flow and Cascade Effect

### Dependency Chain
```
Phase 0.1/0.2 (Data Preparation)
    ↓ (test_list already contains 'assert')
Phase 3.5 ❌ (Built malformed prompts)
Phase 3.6 ❌ (Built malformed prompts, SAVED to parquet)
    ↓ (Phase 3.6 saves prompts to dataset_hyperparams_temp_0_0.parquet)
Phase 4.5 ❌ (Loaded malformed prompts from Phase 3.6)
Phase 4.6 ❌ (Loaded malformed prompts from Phase 3.6)
Phase 4.8 ❌ (Loaded malformed prompts from Phase 3.6)
Phase 4.10 ❌ (Loaded malformed prompts from Phase 3.6)
Phase 4.12 ❌ (Loaded malformed prompts from Phase 3.6)
Phase 4.14 ❌ (Loaded malformed prompts from Phase 3.6)
```

### Verified Data Sources by Phase

| Phase | Prompt Source | Status | Evidence |
|-------|--------------|--------|----------|
| **Phase 1** | Builds own prompts | ✅ Correct | Uses `'\n'.join(task['test_list'])` - no assert prepending |
| **Phase 3.5** | Builds own prompts | ❌ Buggy (Fixed) | Added `assert` to items with `assert` → double assert |
| **Phase 3.6** | Builds own prompts | ❌ Buggy (Fixed) | Added `assert` to items with `assert` → saved to parquet |
| **Phase 4.5** | Loads from Phase 3.6 | ❌ Cascade | `baseline_file = Path(phase3_6_output).parent / "dataset_hyperparams_temp_0_0.parquet"` |
| **Phase 4.6** | Loads from Phase 3.6 | ❌ Cascade | Loads Phase 4.5 results (which used Phase 3.6 prompts) |
| **Phase 4.8** | Loads from Phase 3.6 | ❌ Cascade | `baseline_file = phase3_6_dir / "dataset_hyperparams_temp_0_0.parquet"` |
| **Phase 4.10** | Loads from Phase 3.6 | ❌ Cascade | Uses same baseline as Phase 4.8 |
| **Phase 4.12** | Loads from Phase 3.6 | ❌ Cascade | Uses same baseline as Phase 4.8 |
| **Phase 4.14** | Loads from Phase 3.6 | ❌ Cascade | Uses Phase 4.8/4.12 results |
| **Phase 5.x** | Loads from Phase 1 | ✅ Unaffected | Uses Phase 1 data, not Phase 3.6 |
| **Phase 7.3** | Builds own prompts | ❌ Buggy (Fixed) | Added `assert` to items with `assert` → double assert |
| **Phase 8.2+** | Builds own prompts | ✅ Correct | Had conditional check from the start |
| **Phase 8.3** | Loads from Phase 1 | ✅ Unaffected | Uses Phase 1 data, not Phase 3.6 |

### Example: Phase 3.6 Saved Malformed Prompt
```python
# From data/phase3_6/dataset_hyperparams_temp_0_0.parquet
df = pd.read_parquet("data/phase3_6/dataset_hyperparams_temp_0_0.parquet")
print(df.iloc[0]['prompt'])

# Output (MALFORMED):
'''Write a python function to find binomial co-efficient.

assert assert binomial_Coeff(5,2) == 10
assert assert binomial_Coeff(4,3) == 4
assert assert binomial_Coeff(3,2) == 3

# Solution:'''
```

### Example: Phase 4.5 Loading Malformed Prompt
```python
# From phase4_5_model_steering/steering_coefficient_selector.py:117-121
baseline_file = Path(phase3_6_output).parent / "dataset_hyperparams_temp_0_0.parquet"
self.baseline_data = pd.read_parquet(baseline_file)

# Later in code:
prompt = row['prompt']  # ❌ This is the malformed prompt from Phase 3.6
```

### Consequences

#### Direct Impact (Phases 3.5, 3.6, 7.3)
1. **Invalid Prompts**: Syntax errors in all generated prompts
2. **Low Pass Rates**:
   - MBPP: Lower than expected due to malformed prompts
   - HumanEval: 17.7% instead of expected ~22% (confirmed in Phase 3.5 run)
3. **Unreliable Baseline Metrics**: Temperature robustness and hyperparameter tuning results invalid

#### Cascade Impact (All Phase 4.x)
1. **Invalid Steering Experiments**: All Phase 4.x steering interventions used malformed prompts
2. **Unreliable Causal Claims**: Steering coefficients selected based on malformed prompt performance
3. **Invalid Statistical Tests**: Phase 4.14 significance tests based on flawed data
4. **Compromised Controls**: Phase 4.10/4.12 random feature controls also affected
5. **Complete Pipeline Invalidation**: Entire MBPP + Gemma-2-2B experimental pipeline (Phases 3.5 → 4.14) must be re-run

#### Data Invalidation Scope
- **Phase 3.5 outputs**: All temperature robustness datasets
- **Phase 3.6 outputs**: All hyperparameter tuning datasets (including saved prompts)
- **Phase 4.5 outputs**: All steering coefficient selections
- **Phase 4.6 outputs**: All coefficient refinements
- **Phase 4.8 outputs**: All steering effect analyses
- **Phase 4.10 outputs**: All random feature control results
- **Phase 4.12 outputs**: All random steering analyses
- **Phase 4.14 outputs**: All statistical significance tests
- **Phase 7.3 outputs**: All instruction-tuned baseline results

## The Fix

### Correct Implementation
```python
# Fixed code (applied to Phases 3.5, 3.6, 7.3):
test_cases_str = "\n".join([
    test.strip() if test.strip().startswith('assert ') else f"assert {test.strip()}"
    for test in row['test_list']
])
```

This checks if `assert` is already present before adding it.

### Phases Fixed (2025-11-21)
- ✅ Phase 3.5: `phase3_5_temperature_robustness/temperature_runner.py:470`
- ✅ Phase 3.6: `phase3_6/hyperparameter_runner.py:237`
- ✅ Phase 7.3: `phase7_3_instruct_baseline/instruct_baseline_runner.py:259`
- ✅ Phase 8.2: `phase8_2_percentile_threshold/threshold_optimizer.py` (also had bug)

### Verification
All other prompt-building phases audited and confirmed correct:
- ✅ Phase 4.12: No prompt building
- ✅ Phase 7.6: No prompt building
- ✅ Phase 8.3: No prompt building

## Action Items

### Immediate ✅ (Completed 2025-11-21)
- [x] Fix Phase 3.5, 3.6, 7.3 code
- [x] Audit all other phases (4.12, 7.6, 8.3)
- [x] Document in this file

### Short-term (Priority: HIGH)
- [ ] **Re-run Phase 3.5** (MBPP + Gemma-2-2B, temp=0.0) to establish correct baseline
  - Expected improvement: Pass rate increase due to valid prompts
  - Estimated time: ~2-6 hours for 487 validation problems

- [ ] **Re-run Phase 3.6** (Hyperparameter Tuning)
  - Depends on: Phase 3.5 completion
  - Purpose: Find optimal hyperparameters with valid prompts

- [ ] **Re-run Phase 7.3** (Instruction-Tuned Baseline)
  - Can run in parallel with Phase 3.5
  - Purpose: Establish instruction-tuned model baseline

### Medium-term (Priority: MEDIUM)
- [ ] **Re-run complete MBPP + Gemma-2-2B pipeline (24 phases)**

  **Tier 1** (Run first, can parallelize all 4):
    1. Phase 3.5 (Temperature Robustness)
    2. Phase 3.6 (Hyperparameter Tuning)
    3. Phase 7.3 (Instruction-Tuned Baseline)
    4. Phase 8.2 (Threshold Optimizer)

  **Tier 2** (Run after Tier 1, can parallelize all 3):
    5. Phase 3.8 (AUROC/F1 Evaluation)
    6. Phase 4.5 (Coefficient Selection)
    7. Phase 8.1 (Threshold Calculator)

  **Tier 3** (Run after Tier 2, can parallelize all 3):
    8. Phase 4.6 (Golden Section Refinement)
    9. Phase 4.8 (Steering Effect Analysis)
    10. Phase 8.3 (Selective Steering)

  **Tier 4** (Run after Tier 3, some parallelizable):
    11. Phase 3.10 (Temperature AUROC)
    12. Phase 3.11 (Temperature Visualization)
    13. Phase 3.12 (Difficulty AUROC)
    14. Phase 4.7 (Coefficient Visualization)
    15. Phase 4.12 (Zero-Disc Steering)
    16. Phase 4.14 (Significance Testing)
    17. Phase 5.3 (Weight Orthogonalization)
    18. Phase 5.6 (Zero-Disc Orthogonalization)
    19. Phase 5.9 (Orthogonalization Significance)
    20. Phase 6.3 (Attention Analysis)
    21. Phase 7.6 (Instruct Steering)
    22. Phase 7.9 (Universality Analysis)
    23. Phase 7.12 (Instruct AUROC/F1)
    24. *(Phase 4.10 doesn't need re-run)*

  - **Unaffected Phases** (can skip): Phase 0, 0.1, 0.2, 0.3, 1, 2.2, 2.5, 2.10, 2.15, 4.10
  - **Estimated total time**: 23-67 hours (1-3 days depending on parallelization)

- [ ] **Validate HumanEval experiments**
  - Re-run Phase 3.5 with HumanEval + corrected prompts
  - Verify pass rate improves from 17.7% to ~22%
  - Re-run Phase 3.6 with HumanEval (generates prompts for Phase 4.x HumanEval experiments)
  - Re-run Phase 4.x with HumanEval if needed

### Long-term (Priority: LOW)
- [ ] Update paper/thesis with corrected results
- [ ] Document lessons learned in development notes
- [ ] Add automated tests to prevent regression

## Summary of Affected Pipeline

### Complete Impact Overview
This bug affects **24 phases** across the experimental pipeline (73% of all phases):
- **4 phases** directly built malformed prompts (3.5, 3.6, 7.3, 8.2) ✅ **CODE FIXED**
- **20 phases** have cascade dependencies on buggy phases ⚠️ **REQUIRE RE-RUN**
- **9 phases** are unaffected (0, 0.1, 0.2, 0.3, 1, 2.2, 2.5, 2.10, 4.10) ✅ **CAN SKIP**

### Why Phase 3.6 is Critical
Phase 3.6 is the **single point of failure** in the pipeline:
1. **Built malformed prompts** using buggy code
2. **Saved these prompts** to `dataset_hyperparams_temp_0_0.parquet`
3. **All Phase 4.x experiments** load this file as their baseline

**Fix Strategy**: Re-running Phase 3.6 with corrected code will automatically fix all downstream Phase 4.x experiments when they're re-run.

### Unaffected Phases (Safe to Keep)
- **Phase 0, 0.1, 0.2, 0.3**: Data preparation, no test execution
- **Phase 1**: Built prompts correctly from the start
- **Phase 2.x**: SAE analysis, only uses activations
- **Phase 4.10**: Statistical feature selection, no test execution

### Re-run Priority by Tier

**Tier 1** (Priority: CRITICAL - Run First):
- Phase 3.5, 3.6, 7.3, 8.2 (4 phases with direct bugs)

**Tier 2** (Priority: HIGH - Run After Tier 1):
- Phase 3.8, 4.5, 8.1 (3 phases depending on Tier 1)

**Tier 3** (Priority: MEDIUM - Run After Tier 2):
- Phase 4.6, 4.8, 8.3 (3 phases depending on Tier 2)

**Tier 4** (Priority: LOW - Run After Tier 3):
- Phase 3.10, 3.11, 3.12, 4.7, 4.12, 4.14, 5.3, 5.6, 5.9, 6.3, 7.6, 7.9, 7.12 (14 phases with deep dependencies)

**Total phases to re-run: 24**

## Prevention Measures

### Recommended Tests
```python
def test_prompt_format_no_double_assert():
    """Ensure prompts don't have double assert."""
    test_list = ['assert foo() == 1', 'assert bar() == 2']
    prompt = build_prompt(test_list=test_list, ...)

    # Check no line has "assert assert"
    assert "assert assert" not in prompt, "Double assert detected!"

def test_test_list_format():
    """Verify test_list format in Phase 0.1/0.2."""
    df = pd.read_parquet("data/phase0_1/validation_mbpp.parquet")

    # All test_list items should start with 'assert'
    for test_list in df['test_list']:
        for test in test_list:
            assert test.strip().startswith('assert '), f"Missing assert: {test}"
```

### Code Review Checklist
When modifying prompt building code:
- [ ] Check if test_list already contains `assert` keyword
- [ ] Add conditional check before prepending `assert`
- [ ] Test with actual Phase 0.1/0.2 data
- [ ] Verify generated prompts are syntactically valid Python

## Timeline

- **2025-11-21 08:00 UTC**: Bug discovered during HumanEval pass rate investigation
- **2025-11-21 09:00 UTC**: Root cause identified, confirmed affecting MBPP too
- **2025-11-21 09:30 UTC**: Fix applied to Phases 3.5, 3.6, 7.3
- **2025-11-21 09:45 UTC**: All phases audited, documentation created
- **Next**: Begin re-running experiments with corrected code

## References
- Original MBPP pass rates: Expected ~60-70% for Gemma-2-2B
- HumanEval system card: Reports ~22% pass@1 for Gemma-2-2B
- Observed rates before fix: Much lower due to invalid prompts

## Notes
This bug was present since the initial implementation and affected all experiments run with Phases 3.5, 3.6, and 7.3. All published results from these phases must be regenerated with the corrected code.
