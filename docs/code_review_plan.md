# Implementation Plan
**Generated from code review**: 2026-02-07

## Immediate Fixes

### Fix 1: Add min_gpu_success_ratio threshold to iterative parallel runner
- **File**: `common/iterative_parallel_runner.py:489-521`
- **Change**: In `_collect_results()`, after collecting results from all GPUs, check if the number of successful GPUs meets a minimum ratio (e.g., `n_successful / n_total >= 0.75`). If not, return None (total failure) instead of proceeding with partial data. Add a `min_gpu_success_ratio` parameter to `IterativeParallelRunner.__init__()` with default 0.75.
- **Test**: Add test in `tests/test_common/test_iterative_parallel_runner.py` verifying: (1) 4/4 GPUs succeed -> proceeds normally; (2) 3/4 GPUs succeed -> proceeds (meets 0.75 threshold); (3) 1/4 GPUs succeed -> returns None (below threshold); (4) 0/4 GPUs -> returns None.

### Fix 2: Fail fast on corrupted parquet during value result merge
- **File**: `common/iterative_parallel_runner.py:683-688`
- **Change**: Replace the broad `except Exception` with tracking of expected vs actual GPU count. After the glob loop, if `len(dfs) < expected_n_gpus`, log an ERROR (not warning) with the missing GPU files and raise RuntimeError instead of silently continuing. Add `n_gpus` parameter to `_merge_value_results()` for expected count validation.
- **Test**: Add test that creates N GPU parquet files, corrupts one, and verifies the merge raises an error rather than silently skipping.

### Fix 3: Match multi-candidate merge by (layer, latent_idx) key
- **File**: `common/parallel_runner.py:1131-1142`
- **Change**: Instead of merging candidates by positional index, build a dict keyed by `(candidate['layer'], candidate['latent_idx'])` from each GPU's results. Merge by matching keys. This makes the merge order-independent and robust to any future changes in candidate list ordering.
- **Test**: Add test in `tests/test_common/test_parallel_runner.py` verifying that candidates from multiple GPUs with different orderings are correctly merged by key.

### Fix 4: Cache pre-normalized directions at initialization
- **File**: `phase4_5_coefficient_grid_search/steering_coefficient_selector.py:196-202`
- **Change**: In `_load_dependencies()` (SAE mode), pre-compute and cache normalized directions for all candidates into `self._direction_cache = {}` keyed by `(layer, latent_idx)`. Change `_get_latent_direction()` to look up the cache first: `cache_key = (latent['layer'], latent['latent_idx']); if cache_key in self._direction_cache: return self._direction_cache[cache_key]`. Apply same pattern to `phase4_8_steering_analysis/steering_effect_analyzer.py`.
- **Test**: Verify that calling `_get_latent_direction()` twice with same latent returns same tensor (identity check via `is` operator).

## Backlog
(none)

## Skipped
(none - all findings marked for immediate fix)
