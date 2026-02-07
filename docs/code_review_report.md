# Code Review Report
**Date**: 2026-02-07
**Scope**: Full project (final pre-production review)
**Reviewers**: 5 specialist agents + 1 verification agent
**Findings**: 6 confirmed / 15 reviewed (60% false positive rate)

## CRITICAL

*No critical findings.*

## HIGH

*No high-severity findings.*

## MEDIUM

### M1: Prompt construction inconsistency between phases
- **Location**: `phase3_5_temperature_robustness/temperature_runner.py:376` vs `phase4_8_steering_analysis/steering_effect_analyzer.py:426`
- **Root Cause**: Phase 3.5 and 3.6 rebuild prompts via `PromptBuilder.build_prompt()` from raw fields, while Phase 4.8 uses pre-built `row['prompt']` from parquet
- **Impact**: If PromptBuilder logic changes between runs, different phases would use different prompt formats
- **Mitigating factor**: Phase 3.5 loads from analysis split which doesn't have a pre-built `prompt` column, so rebuilding is correct for its data source. PromptBuilder has been stable.
- **Test Impact**: None — by-design for different data sources

## LOW

### L1: No test for partial GPU failure + checkpoint recovery
- **Location**: `common/parallel_runner.py:382-387`
- **Root Cause**: No integration test covers GPU failure mid-run with checkpoint-based recovery
- **Impact**: Code handles this defensively (raises RuntimeError, relies on checkpointing), but the specific recovery path is untested
- **Test Impact**: Would need multiprocessing mocks, low priority

### L2: Missing `baseline_passed` column validation in steering_metrics.py
- **Location**: `common/steering_metrics.py:72, 118`
- **Root Cause**: `modified_col` is validated, but `baseline_passed` is accessed directly without existence check
- **Impact**: Would throw `KeyError` instead of descriptive `ValueError` if column missing. Internal functions with well-defined pipeline inputs.
- **Test Impact**: Minor — add column existence check

### L3: Hardcoded paths in utility scripts
- **Location**: `scripts/reevaluate_phase3_6.py:52-56` and other `scripts/reevaluate_*.py`
- **Root Cause**: One-off maintenance scripts use hardcoded paths instead of `get_phase_output_dir()`
- **Impact**: Scripts won't work with non-default configs, but these are maintenance scripts that have already been run
- **Test Impact**: None

### L4: Color constants not used in 3 visualization files
- **Location**: `phase7_12_instruct_auroc_f1/instruct_auroc_f1_evaluator.py`, `phase6_3_attention_analysis/attention_analyzer.py`, `phase7_9_universality_analysis/universality_analysis.py`
- **Root Cause**: Hardcoded color strings instead of `COLOR_CORRECTION`, `COLOR_CORRUPTION`, `COLOR_PRESERVATION` from config
- **Impact**: Purely cosmetic
- **Test Impact**: None

### L5: Redundant full sort for top-K selection in Phase 2.5
- **Location**: `phase2_5_separation_score_analysis/sae_analyzer.py:223-231`
- **Root Cause**: `sorted(416k_items)[:20]` instead of `heapq.nlargest(20, items)`
- **Impact**: ~0.4 seconds wasted in a phase that runs for minutes/hours
- **Test Impact**: None

## Notable False Positives (verified as correct)

| Claimed Bug | Why It's Correct |
|-------------|------------------|
| `results['baseline_passed']` without `== True` | Standard pandas idiom for boolean columns |
| `results[modified_col]` without comparison | Boolean Series works as mask in `&` operations |
| `binomtest(p=0)` in Phase 4.14 | Already fixed — guard at line 120-122 uses `max(1/n_trials, 1e-10)` |
| Probe direction negation | Intentional — `incorrect_direction` detects incorrectness for threshold decision |
| No tests for `direction_utils.py` | Tests exist at `tests/test_critical/test_normalization.py` (30+ tests) |
| Phase discovery swallows exceptions | Intentional two-phase fallback (try 4.9, fall back to 4.6) |
| DataFrame `.copy()` wastes 300-500MB | DataFrames are ~100-300 rows, copies are kilobytes |
| Attention hooks per-task wastes 20-40min | Hook registration is microseconds, not milliseconds |
| Weight orthogonalization model not reloaded | Fresh model explicitly loaded at line 478-485 for correct experiment |

## Agent Accuracy

| Agent | Findings | Confirmed | False Positive Rate |
|-------|----------|-----------|---------------------|
| Bug Hunter | 4 | 0 | 100% |
| Test Coverage | 2 | 1 | 50% |
| Error Handling | 2 | 1 | 50% |
| Consistency | 3 | 3 | 0% |
| Performance | 4 | 1 | 75% |
| **Total** | **15** | **6** | **60%** |

## Production Readiness

**VERDICT: READY FOR PRODUCTION**

No critical or high-severity bugs confirmed. All previously known bug patterns (double-wrapped prompts, degenerate binomtest, cross-GPU merge issues, error handler success assumptions) have been fixed. The 6 confirmed findings are LOW to MEDIUM severity issues that do not affect research correctness or GPU runtime safety.
