# Implementation Plan
**Generated from code review**: 2026-02-07

## Immediate Fixes

### Fix 1: Add validation for wrong probe type in ALL direction-source phases
- **Steering phases** (reject `probe_logreg`): `phase4_5:68`, `phase4_6:75`, `phase4_7:495`, `phase4_8:72`, `phase4_9:40`, `phase5_3:66`, `phase7_6:86`
- **Prediction phases** (reject `probe_mass_mean`): `phase3_8:125`, `phase3_10:44`, `phase7_12:475`
- **Change**: After `self.use_probe = ...` check, add explicit `ValueError` when the wrong probe type is passed. Steering phases error on `probe_logreg`, prediction phases error on `probe_mass_mean`. Currently both silently fall to SAE mode.
- **Test**: Add regression test verifying wrong probe type is rejected in both steering and prediction phases.

### Fix 2: Add negative layer index validation to SAE loader
- **File**: `common/sae_loader.py:283`
- **Change**: Add `if layer_idx < 0:` check alongside existing `if layer_idx >= n_layers:` check.
- **Test**: Add test for negative layer index rejection.

### Fix 3: Add NaN check to normalize_direction()
- **File**: `common/direction_utils.py:39`
- **Change**: Add `if torch.isnan(norm):` check before the existing `if norm < NORM_EPSILON:` check.
- **Test**: Add test for NaN input handling.

### Fix 4: Add tests for pile_filter_utils.py
- **File**: `common/pile_filter_utils.py` (114 lines, 0 coverage)
- **Change**: Create `tests/test_common/test_pile_filter_utils.py` with unit tests for `load_pile_frequencies()` and `apply_pile_filter()`.
- **Test**: New test file.

### Fix 5: Add tests for initialization.py
- **File**: `common/initialization.py` (57 lines, 0 coverage)
- **Change**: Create `tests/test_common/test_initialization.py` verifying `setup_deterministic_generation()` seeds all three random sources.
- **Test**: New test file.

### Fix 6: Update CLAUDE.md checkpoint documentation
- **File**: `CLAUDE.md`
- **Change**: Change "Checkpoints every 50 records" to "Checkpoints every 10 records" to match `CHECKPOINT_FREQUENCY_DEFAULT = 10`.
- **Test**: None.

## Backlog
(none)

## Skipped
| Finding | Reason |
|---------|--------|
| H1: Phase 6.3 hardcodes mass_mean | Intentionally correct — Phase 6.3 only operates on steering/mass_mean data |
| L1: Phase 5.3 error conversion | Follows project convention per MEMORY.md |
