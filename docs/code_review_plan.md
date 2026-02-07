# Implementation Plan
**Generated from code review**: 2026-02-07

## Immediate Fixes

### Fix 1: Unsafe Dictionary Key Access in Phase 4.12
- **File**: `phase4_12_zero_disc_steering/zero_disc_steering_generator.py:135-140`
- **Change**: Add validation that both `'correct'` and `'incorrect'` keys exist in `selection` before accessing them. Raise clear error directing user to re-run Phase 4.9 with `experiment_mode='all'`.
- **Test**: Add test for Phase 4.12 initialization with missing keys.

### Fix 2: Parquet Checkpoint Metadata/Data Mismatch
- **File**: `common/iterative_parallel_runner.py:586-612`
- **Change**: When parquet load fails but metadata exists, raise an error (not just warning) so users know checkpoint data is inconsistent. Clear the stale task IDs from metadata to prevent silent skipping.
- **Test**: Add test for corrupted-parquet-with-valid-metadata scenario.

### Fix 3: Model Loaded Before Checkpoint Check in Phase 1
- **File**: `phase1_latent_selection_dataset/runner.py:301-374`
- **Change**: Add a lightweight pre-check before `self.setup()` that determines if all tasks are already processed. If so, skip model loading entirely.
- **Test**: None needed (optimization).

### Fix 4: Silent Exception Swallowing in Import Loading
- **File**: `common/dataset_utils.py:434-451`
- **Change**: Replace bare `except Exception: pass` with `except Exception as e: logger.debug(f"Failed to load imports: {e}")` and return None.
- **Test**: None needed.

### Fix 5: Subprocess Timeout Enforcement in Workers
- **File**: `common/retry_utils.py:148-167`
- **Change**: Investigate using `multiprocessing.Process` with watchdog for subprocess workers, or leverage subprocess_executor's hard timeout more aggressively. May be limited by CUDA constraints.
- **Test**: Add test for timeout behavior in subprocess context.

### Fix 6: Rename `get_phase_output_dir()` in Registry
- **File**: `common/phase_registry.py:584`
- **Change**: Rename `get_phase_output_dir()` to `get_phase_base_dir()`. Update the single import in `common/phase_discovery.py` that uses it (currently aliased as `registry_get_dir`).
- **Test**: None needed.

### Fix 7: Terminology 'feature' → 'latent' in Output Keys
- **Files**: `phase4_12_zero_disc_steering/zero_disc_steering_generator.py:612`, `phase8_3_selective_steering/selective_steering_analyzer.py:1063`
- **Change**: Replace `'feature'` keys with `'latent'` in output dictionaries. Rename variables in Phase 4.12 loop: `feature_idx` → `latent_idx`, `all_features` → `all_latents`, `feature` → `latent`.
- **Test**: None needed.

### Fix 8: Hardcoded Color Strings → Config Constants
- **Files**: `phase4_14_statistical_significance/significance_tester.py:492-529`, `phase6_3_attention_analysis/attention_analyzer.py`, `phase3_12_difficulty_auroc_f1/difficulty_evaluator.py`, `phase5_6_zero_disc_orthogonalization/zero_disc_weight_orthogonalizer.py`
- **Change**: Replace hardcoded `'green'`, `'red'`, `'gold'` with imports of `COLOR_CORRECTION`, `COLOR_CORRUPTION`, `COLOR_PRESERVATION` from `common/config.py`.
- **Test**: None needed.

## Skipped
| Finding | Reason |
|---------|--------|
| M3: Multi-GPU parquet merge crashes on partial failures | User prefers loud errors over silent partial results. Per-GPU data is preserved on disk anyway. |
