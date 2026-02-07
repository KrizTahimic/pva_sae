# Code Review Report

**Date**: 2026-02-07
**Scope**: Full project
**Findings**: 10 confirmed / 35 reviewed (71% false positive rate)

---

## CRITICAL

### C1: Direction Source Detection Bug in 7 Steering Phases
- **Location**: `phase4_8_steering_analysis/steering_effect_analyzer.py:72`, `phase4_5_coefficient_grid_search/steering_coefficient_selector.py:68`, `phase4_6_golden_section_refinement/golden_section_refiner.py:75`, `phase4_9_latent_selection/latent_selector.py:40`, `phase5_3_weight_orthogonalization/weight_orthogonalizer.py:66`, `phase7_6_instruct_steering/instruct_steering_analyzer.py:86`, `phase4_7_coefficient_visualization/coefficient_plotter.py:495`
- **Root Cause**: All 7 phases check `self.direction_source == 'probe_mass_mean'` to detect probe mode, but the codebase supports TWO probe modes: `probe_mass_mean` and `probe_logreg`. Phases 8.1/8.2/8.3 correctly use `in ('probe_logreg', 'probe_mass_mean')`.
- **Impact**: Running with `--direction-source probe_logreg` causes these phases to silently fall into SAE code path, loading wrong directions and potentially crashing or producing incorrect results.
- **Suggested Fix**: Change to `self.use_probe = self.direction_source in ('probe_logreg', 'probe_mass_mean')` in all 7 files.
- **Test Impact**: Add regression test verifying probe mode detection for both probe types.

---

## HIGH

### H1: Phase 6.3 Hardcodes Probe Method Ignoring Direction Source
- **Location**: `phase6_3_attention_analysis/attention_analyzer.py:77`
- **Root Cause**: Line 42 correctly detects both probe types (`self.use_probe = self.direction_source in ('probe_logreg', 'probe_mass_mean')`), but line 77 hardcodes `method="mass_mean"` regardless of which probe was requested.
- **Impact**: `--direction-source probe_logreg` enters probe mode but loads mass_mean directions. May use different layer than intended for analysis.
- **Suggested Fix**: Extract method from `direction_source` or pass through explicitly.
- **Test Impact**: Add test verifying correct method is loaded for each direction source.

---

## MEDIUM

### M1: Prediction Phases Silently Fall to SAE Mode for probe_mass_mean
- **Location**: `phase3_10_temperature_auroc_f1/temperature_evaluator.py:44`, `phase7_12_instruct_auroc_f1/instruct_auroc_f1_evaluator.py:475`
- **Root Cause**: These check `== 'probe_logreg'` only (correct for prediction), but passing `--direction-source probe_mass_mean` silently falls to SAE mode instead of raising an error.
- **Impact**: User gets SAE results when they expected probe results, with no warning.
- **Suggested Fix**: Add explicit validation: `if direction_source == 'probe_mass_mean': raise ValueError("Use probe_logreg for prediction phases")`
- **Test Impact**: Add test for invalid direction source rejection.

### M2: SAE Loader Missing Negative Layer Index Validation
- **Location**: `common/sae_loader.py:283`
- **Root Cause**: Only validates `layer_idx >= n_layers` but not `layer_idx < 0`. Negative indices pass validation.
- **Impact**: Low practical impact (layer indices from config/discovery), but a defensive gap.
- **Suggested Fix**: Add `if layer_idx < 0:` check alongside existing upper bound check.
- **Test Impact**: Add test for negative layer index rejection.

### M3: normalize_direction() Silently Passes NaN Input
- **Location**: `common/direction_utils.py:39`
- **Root Cause**: `NaN < NORM_EPSILON` evaluates to `False`, so NaN directions bypass the zero-check and return NaN silently.
- **Impact**: Corrupted probe/SAE data would propagate NaN through steering pipeline undetected.
- **Suggested Fix**: Add `if torch.isnan(norm): raise ValueError(f"Cannot normalize {name}: contains NaN")`
- **Test Impact**: Add test for NaN input handling.

### M4: pile_filter_utils.py Completely Untested
- **Location**: `common/pile_filter_utils.py` (114 lines, 0 test coverage)
- **Root Cause**: No test file exists. Feature filtering logic untested.
- **Impact**: Bugs in feature filtering would silently use wrong features for steering.
- **Suggested Fix**: Create `tests/test_common/test_pile_filter_utils.py`.
- **Test Impact**: New test file needed.

### M5: initialization.py Deterministic Seeding Untested
- **Location**: `common/initialization.py` (57 lines, 0 test coverage)
- **Root Cause**: `setup_deterministic_generation()` has no tests.
- **Impact**: Non-reproducibility across GPU runs would go undetected.
- **Suggested Fix**: Create `tests/test_common/test_initialization.py`.
- **Test Impact**: New test file needed.

---

## LOW

### L1: Silent Error Conversion in Phase 5.3
- **Location**: `phase5_3_weight_orthogonalization/weight_orthogonalizer.py:301-308`
- **Root Cause**: Infrastructure errors recorded as `orthogonalized_correct=False`, conflating with genuine experiment failures. The `error` field IS preserved.
- **Impact**: Error tasks inflate failure count in metrics. Follows project convention per MEMORY.md.
- **Suggested Fix**: Track errors separately or exclude from metrics calculation.
- **Test Impact**: Modify metrics calculation to filter `error` field.

### L2: Checkpoint Frequency Documentation Mismatch
- **Location**: `common/config.py:19` vs CLAUDE.md
- **Root Cause**: Code uses `CHECKPOINT_FREQUENCY_DEFAULT = 10`, CLAUDE.md says "Checkpoints every 50 records."
- **Impact**: Documentation is misleading.
- **Suggested Fix**: Update CLAUDE.md to reflect actual value of 10.
- **Test Impact**: None.

---

## Agent Accuracy

| Agent | Findings Checked | Confirmed | False Positive Rate |
|-------|-----------------|-----------|---------------------|
| Bug Hunter (A) | 7 | 3 | ~50% |
| Test Coverage (B) | 6 | 2 | ~33% |
| Error Handling (C) | 7 | 2 | ~57% |
| Consistency (D) | 3 | 2 | ~0% |
| Performance (E) | 6 | 1 | ~83% |
| **Total** | **29** | **10** | **~66%** |

**Note**: Agent D (Consistency Checker) had the lowest false positive rate. Agent E (Performance) had the highest — most performance claims were disproven by reading the actual code (e.g., SAEs are cached, DataFrame copies are necessary, eager attention is required for attention extraction).
