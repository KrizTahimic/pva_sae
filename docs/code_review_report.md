# Code Review Report
**Date**: 2026-02-07
**Scope**: Full project
**Findings**: 11 confirmed / 35 reviewed (63% false positive rate)

## Methodology

5 parallel specialist agents (Bug Hunter, Test Coverage Auditor, Error Handling Reviewer, Consistency Checker, Performance Reviewer) independently scanned the full codebase. Their 35 findings were then verified by a skeptical verification agent that read the actual source code at every cited location.

---

## MEDIUM

### M1: SAE Reloaded Per Candidate in Phase 3.8
- **Location**: `phase3_8_auroc_f1_evaluation/auroc_f1_evaluator.py:1088-1097`
- **Root Cause**: `evaluate_single_latent()` calls `load_split_activations()` which loads SAE via `load_sae_for_config()`. When multiple candidates share the same layer, the SAE is loaded, used, deleted, and reloaded.
- **Impact**: For 5-10 candidates with shared layers, redundant SAE loads add minutes per candidate. Moderate impact on Phase 3.8 runtime.
- **Suggested Fix**: Add layer-keyed SAE cache in the candidate evaluation loop.
- **Test Impact**: Add test verifying SAE cache reuse across candidates sharing layers.

### M2: No Unit Tests for model_loader.py
- **Location**: `common/model_loader.py:12-97`
- **Root Cause**: Module wraps HuggingFace model loading with dtype/device selection logic but has no test coverage.
- **Impact**: Device auto-detection and dtype selection logic could silently pick wrong dtype. Low probability but affects all phases.
- **Suggested Fix**: Add `tests/test_common/test_model_loader.py` with mocked `torch.cuda` to test dtype/device selection branches.
- **Test Impact**: New test file needed.

### M3: Parallel Runner Subprocess Paths Undertested
- **Location**: `common/parallel_runner.py:205-252`
- **Root Cause**: Existing tests cover merge logic and task distribution thoroughly, but `_run_gpu_worker` subprocess path and `run_phase_parallel` orchestration are tested only at source-inspection level.
- **Impact**: Worker exception handling, timeout behavior, and CUDA_VISIBLE_DEVICES setup are not exercised in tests.
- **Suggested Fix**: Add integration-style tests with mocked `ProcessPoolExecutor` that inject exceptions and timeouts.
- **Test Impact**: Extend `tests/test_common/test_parallel_runner.py`.

## LOW

### L1: Binomial Test Effect Size vs P-Value Computed Against Different Rates
- **Location**: `phase4_14_statistical_significance/significance_tester.py:122-140`, `phase5_9_orthogonalization_significance/orthogonalization_significance_tester.py:122-140`
- **Root Cause**: When `baseline_rate=0.0`, `effective_rate` is computed for the binomial test, but `effect_size` is computed against the original `baseline_rate`. Both values are returned transparently.
- **Impact**: Minor reporting inconsistency. Only affects edge case where baseline_rate=0.0.
- **Suggested Fix**: Add a comment documenting this design choice.
- **Test Impact**: None needed.

### L2: No Unit Tests for memory_utils.py
- **Location**: `common/memory_utils.py:19-94`
- **Root Cause**: Utility wrapping psutil/torch memory operations, untested.
- **Impact**: Low - code is simple and effects are side-effect-only.
- **Suggested Fix**: Optional `tests/test_common/test_memory_utils.py`.
- **Test Impact**: New test file (optional).

### L3: No Unit Tests for logging.py
- **Location**: `common/logging.py:59-115`
- **Root Cause**: LoggingManager and get_logger untested.
- **Impact**: Low - logging infrastructure, standard patterns.
- **Suggested Fix**: Optional test for RotatingFileHandler and custom formatter.
- **Test Impact**: New test file (optional).

### L4: Phase Discovery Doesn't Validate Manifest File Existence on Disk
- **Location**: `common/phase_discovery.py:341-361`
- **Root Cause**: Manifest loaded but files referenced in manifest not checked for existence.
- **Impact**: Low - "fail later" vs "fail early" pattern. Error still surfaces.
- **Suggested Fix**: Optional validation of output file existence after loading manifest.
- **Test Impact**: None needed.

### L5: Phase 1 Cross-Run Resume Heuristic is Coarse
- **Location**: `common/parallel_runner.py:1612-1621`
- **Root Cause**: Checks if *any* activation files exist and assumes all done. Doesn't count vs total.
- **Impact**: Low - scenario is unlikely in normal operation.
- **Suggested Fix**: Add count check: verify activation file count matches expected total.
- **Test Impact**: Add test for partial activation scenario.

### L6: Phase 3.5 vs Phase 3.6 Column Mismatch (generation_idx)
- **Location**: `phase3_5_temperature_robustness/temperature_runner.py:141` vs `phase3_6_hyperparameter_baseline/hyperparameter_runner.py:209-220`
- **Root Cause**: Phase 3.5 includes `generation_idx` column, Phase 3.6 does not.
- **Impact**: Low - outputs are never merged together.
- **Suggested Fix**: Document the difference.
- **Test Impact**: None needed.

### L7: GPU Cache Cleanup Every Record in Phase 4.8
- **Location**: `phase4_8_steering_analysis/steering_effect_analyzer.py:616-622`
- **Root Cause**: `torch.cuda.empty_cache()` called after every problem.
- **Impact**: Low - overhead is milliseconds vs seconds for generation.
- **Suggested Fix**: Could batch to every 10-50 problems, but current approach is safe.
- **Test Impact**: None needed.

## NEEDS CONTEXT

### NC1: Missing `prompt` Column Assertion in load_baseline_data
- **Location**: `common/steering_setup.py:484-504`
- **Issue**: No assertion that loaded parquet has `prompt` column. All upstream phases produce it, but no guard exists.

---

## Agent Accuracy

| Agent | Findings | Confirmed | False Positive Rate |
|-------|----------|-----------|---------------------|
| Bug Hunter | 7 | 2 (1 dup) | 71% |
| Test Coverage | 6 | 4 | 33% |
| Error Handling | 7 | 2 | 71% |
| Consistency | 4 | 1 | 50% |
| Performance | 11 | 2 | 73% |
| **Total** | **35** | **11** | **63%** |

## Key Takeaway

**No CRITICAL bugs were confirmed.** The codebase is in good shape for production runs. The most impactful confirmed issues are performance optimizations and test coverage gaps, not correctness bugs. The codebase has strong fundamentals: atomic file writes, conservative error handlers, good checkpointing, consistent terminology, and centralized direction normalization.
