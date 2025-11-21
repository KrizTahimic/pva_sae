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

### Cascade Affected Experiments (Used Malformed Prompts from Phase 3.6)
- ❌ **Phase 4.5** (Steering Coefficient Selection) - Loaded malformed prompts from Phase 3.6
- ❌ **Phase 4.6** (Coefficient Refinement) - Loaded malformed prompts from Phase 3.6
- ❌ **Phase 4.8** (Steering Effect Analysis) - Loaded malformed prompts from Phase 3.6
- ❌ **Phase 4.10** (Random Feature Control) - Loaded malformed prompts from Phase 3.6
- ❌ **Phase 4.12** (Random Steering Analysis) - Loaded malformed prompts from Phase 3.6
- ❌ **Phase 4.14** (Statistical Significance) - Loaded malformed prompts from Phase 3.6

**Note**: Phase 4.x phases don't build prompts themselves but load pre-generated prompts saved in Phase 3.6 output files (`dataset_hyperparams_temp_0_0.parquet`). Since Phase 3.6 generated malformed prompts, all Phase 4.x experiments used these malformed prompts.

### Unaffected Phases
- ✅ **Phase 1** (Baseline Generation) - Correct format: `'\n'.join(task['test_list'])`
- ✅ **Phase 8.2+** (Threshold Optimization) - Already had conditional check
- ✅ **Phase 5.x** (Weight Orthogonalization) - Uses Phase 1 data, not Phase 3.6
- ✅ **Phase 8.3** (Selective Steering) - Uses Phase 1 data, not Phase 3.6

### Datasets Affected
- Both **MBPP** and **HumanEval** affected (both store assertions with `assert` keyword)

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
- [ ] **Re-run complete MBPP + Gemma-2-2B pipeline**
  - **Critical Path**: Phase 3.6 must be re-run first (generates prompts for Phase 4.x)
  - **Phase 3.6 Dependencies**: Requires Phase 3.5 completion
  - **Full Pipeline Order**:
    1. Phase 3.5 (Temperature Robustness) - Establishes baseline
    2. Phase 3.6 (Hyperparameter Tuning) - Generates corrected prompts
    3. Phase 3.8 (AUROC/F1 Evaluation) - Can run after 3.5/3.6
    4. Phase 4.5 (Steering Coefficient Selection) - Requires Phase 3.6 prompts
    5. Phase 4.6 (Coefficient Refinement) - Requires Phase 4.5
    6. Phase 4.8 (Steering Effect Analysis) - Requires Phase 3.6 prompts
    7. Phase 4.10 (Random Feature Control) - Requires Phase 3.6 prompts
    8. Phase 4.12 (Random Steering Analysis) - Requires Phase 3.6 prompts
    9. Phase 4.14 (Statistical Significance) - Requires Phase 4.8/4.12
  - **Unaffected Phases** (can skip): Phase 1, 2.x, 5.x, 8.3
  - **Estimated total time**: Several days (Phase 3.6 + all Phase 4.x)

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
This bug affects **9 phases** across the experimental pipeline:
- **3 phases** directly built malformed prompts (3.5, 3.6, 7.3) ✅ **FIXED**
- **6 phases** loaded malformed prompts from Phase 3.6 (4.5, 4.6, 4.8, 4.10, 4.12, 4.14) ⚠️ **REQUIRE RE-RUN**

### Why Phase 3.6 is Critical
Phase 3.6 is the **single point of failure** in the pipeline:
1. **Built malformed prompts** using buggy code
2. **Saved these prompts** to `dataset_hyperparams_temp_0_0.parquet`
3. **All Phase 4.x experiments** load this file as their baseline

**Fix Strategy**: Re-running Phase 3.6 with corrected code will automatically fix all downstream Phase 4.x experiments when they're re-run.

### Unaffected Phases (Safe to Keep)
- **Phase 1**: Built prompts correctly from the start
- **Phase 2.x**: SAE analysis, no prompt building
- **Phase 5.x**: Uses Phase 1 data (correct prompts)
- **Phase 8.2+**: Had conditional check from the start
- **Phase 8.3**: Uses Phase 1 data (correct prompts)

### Re-run Priority
1. **HIGH**: Phase 3.5, 3.6, 7.3 (baseline re-establishment)
2. **HIGH**: Phase 3.8 (validation metrics depend on 3.5/3.6)
3. **MEDIUM**: Phase 4.5 → 4.6 → 4.8 → 4.10 → 4.12 → 4.14 (steering experiments)

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
