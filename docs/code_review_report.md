# Code Review Report
**Date**: 2026-02-07
**Scope**: Full project
**Findings**: 13 confirmed / 43 reviewed (51% false positive rate)

## MEDIUM

### M1: Unsafe Dictionary Key Access in Phase 4.12
- **Location**: `phase4_12_zero_disc_steering/zero_disc_steering_generator.py:135-140`
- **Root Cause**: Phase 4.12 unconditionally accesses `selection['incorrect']`, but Phase 4.9 only populates this key when `experiment_mode` includes 'corruption'. Running Phase 4.9 in `correction`-only mode leaves `selection['incorrect']` missing.
- **Impact**: `KeyError` crash when Phase 4.12 loads Phase 4.9 output from correction-only runs. Entire phase fails, losing any computed results.
- **Suggested Fix**: Add validation before access:
  ```python
  for key in ('correct', 'incorrect'):
      if key not in selection:
          raise ValueError(f"Phase 4.9 output missing '{key}' latent. Re-run Phase 4.9 with experiment_mode='all'.")
  ```
- **Test Impact**: Add test in `tests/test_phases/` for Phase 4.12 initialization with missing keys.

### M2: Parquet Checkpoint Metadata/Data Mismatch
- **Location**: `common/iterative_parallel_runner.py:586-612`
- **Root Cause**: When a parquet checkpoint file is corrupted, the exception handler logs a warning and continues, but the metadata's `processed_task_ids` still includes the lost tasks. On restart, those tasks are skipped (considered "done") despite having no results.
- **Impact**: Silent data loss — tasks appear completed but their results are missing from the final output. Undetectable without manual row-count verification.
- **Suggested Fix**: On parquet load failure, clear the metadata's `existing_task_ids` for affected tasks, or raise an error forcing manual recovery.
- **Test Impact**: Add test for corrupted-parquet-with-valid-metadata scenario.

### M3: Multi-GPU Parquet Merge Crashes on Partial Failures
- **Location**: `common/parallel_runner.py:862-868`
- **Root Cause**: The parquet merge function raises `RuntimeError` on any single corrupted GPU file, aborting the entire merge. Compare to the JSON merge function (lines 92-108) which gracefully skips corrupted files and continues.
- **Impact**: One corrupted GPU output file (out of 4-8) voids ALL other GPUs' valid results. With 4 GPUs, one failure loses 75% of good data.
- **Suggested Fix**: Follow the JSON merge pattern — log corrupted files, skip them, continue with valid data:
  ```python
  for f in gpu_files:
      try:
          dfs.append(pd.read_parquet(f))
      except Exception as e:
          logger.error(f"Corrupted GPU parquet {f.name}: {e} — skipping")
          corrupted.append(f.name)
  ```
- **Test Impact**: Add test for partial GPU file corruption in parquet merge.

### M4: Model Loaded Before Checkpoint Check in Phase 1
- **Location**: `phase1_latent_selection_dataset/runner.py:301-374`
- **Root Cause**: `self.setup()` (line 301) loads the model, tokenizer, and hooks before checking if all tasks are already processed (line 370). If all tasks are checkpointed from a previous run, the model was loaded for nothing.
- **Impact**: Wastes 30-60 seconds loading a multi-GB model unnecessarily. Minor for single runs, annoying during development/debugging.
- **Suggested Fix**: Move the checkpoint completeness check before `self.setup()`, or add a lightweight pre-check method.
- **Test Impact**: None needed (optimization, not correctness).

## LOW

### L1: Silent Exception Swallowing in Import Loading
- **Location**: `common/dataset_utils.py:434-451`
- **Root Cause**: Bare `except Exception: pass` silently swallows ALL exceptions when loading import code — corrupted JSON, permission errors, missing keys all produce the same `None` result with no logging.
- **Impact**: Debugging difficulty when import-dependent code fails with `NameError` instead of showing which import file was problematic.
- **Suggested Fix**: Log the exception before returning None:
  ```python
  except Exception as e:
      logger.debug(f"Failed to load imports: {e}")
      return None
  ```
- **Test Impact**: None needed.

### L2: Subprocess Timeout Not Enforced in Workers
- **Location**: `common/retry_utils.py:148-167`
- **Root Cause**: `signal.SIGALRM` doesn't work in subprocess workers (Python limitation). The code falls back to time-tracking that only logs after completion — infinite loops are never killed.
- **Impact**: A hanging code generation in a parallel worker wastes GPU time until the hard process timeout kicks in. Known architectural limitation documented in comments.
- **Suggested Fix**: Consider using `multiprocessing.Process` with a separate watchdog, or rely on the subprocess_executor's hard timeout. Low priority since CUDA operations can't be interrupted anyway.
- **Test Impact**: None needed.

### L3: Dual `get_phase_output_dir()` Definitions
- **Location**: `common/phase_registry.py:584` and `common/phase_discovery.py:47`
- **Root Cause**: Two functions with the same name but different signatures exist in different modules. The registry version returns base directory; the discovery version adds model/dataset suffixes.
- **Impact**: Import confusion risk. Currently mitigated because `phase_discovery.py` renames the registry import as `registry_get_dir`.
- **Suggested Fix**: Rename `phase_registry.get_phase_output_dir()` to `get_phase_base_dir()`.
- **Test Impact**: None needed.

### L4: Terminology 'feature' vs 'latent' in Output Keys
- **Location**: `phase4_12_zero_disc_steering/zero_disc_steering_generator.py:612` and `phase8_3_selective_steering/selective_steering_analyzer.py:1063`
- **Root Cause**: Output dictionaries use `'feature'` key instead of `'latent'`, violating project terminology standards in CLAUDE.md.
- **Impact**: Inconsistent JSON output keys. Any downstream consumer expecting `'latent'` would fail.
- **Suggested Fix**: Replace `'feature'` keys with `'latent'` in output dicts. Also rename variables: `feature_idx` → `latent_idx`, `all_features` → `all_latents` in Phase 4.12 loop (line 538).
- **Test Impact**: None needed.

### L5: Hardcoded Color Strings Instead of Config Constants
- **Location**: `phase4_14_statistical_significance/significance_tester.py:492-529`, `phase6_3_attention_analysis/attention_analyzer.py`, `phase3_12_difficulty_auroc_f1/difficulty_evaluator.py`, `phase5_6_zero_disc_orthogonalization/zero_disc_weight_orthogonalizer.py`
- **Root Cause**: Multiple visualization files use hardcoded color strings (`'green'`, `'red'`, `'gold'`) instead of importing `COLOR_CORRECTION`, `COLOR_CORRUPTION`, `COLOR_PRESERVATION` from `common/config.py`.
- **Impact**: If color scheme changes, these files won't update. Cosmetic inconsistency only.
- **Suggested Fix**: Replace hardcoded strings with config imports.
- **Test Impact**: None needed.

## Agent Accuracy
| Agent | Findings | Confirmed | False Positive Rate |
|-------|----------|-----------|---------------------|
| Bug Hunter (A) | 5 | 1 | 80% |
| Test Coverage (B) | 14 | N/A (coverage gaps, not bugs) | N/A |
| Error Handling (C) | 15 | 4 | 53% |
| Consistency (D) | 8 | 4 | 38% |
| Performance (E) | 15 | 4 | 47% |
| **Total** | **43** | **13** | **51%** |

## Test Coverage Gaps (from Agent B — not individually verified)

The following critical modules lack adequate test coverage:

| Module | Risk | Tests? | Key Gap |
|--------|------|--------|---------|
| `common/subprocess_executor.py` | HIGH | None | Exception classification, timeout, semaphore |
| `common/weight_utils.py` | HIGH | Partial | Orthogonalization math verification |
| `common/iterative_parallel_runner.py` | HIGH | Partial | Early stop exception handling, merge failures |
| `common/parallel_runner.py` | HIGH | Partial | Phase 1/2.2/3.6/5.3/7.x merge functions |
| `common/dataset_utils.py` | HIGH | Partial | Code evaluation, error type distribution |
| `common/model_loader.py` | MEDIUM | None | Device detection, dtype selection |
| `common/checkpoint_manager.py` | MEDIUM | Partial | Parquet sidecar, multi-checkpoint dedup |
| `common/selective_steering.py` | MEDIUM | None | State management, should_steer flag |
