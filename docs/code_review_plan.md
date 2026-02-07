# Implementation Plan
**Generated from code review**: 2026-02-07 (Final pre-production review #2)

## Immediate Fixes (ALL COMPLETED)

### Fix 1: M1 — Conservative `.get()` defaults in result classification
- **Files**: `phase4_5_coefficient_grid_search/steering_coefficient_selector.py:1132-1145`, `phase4_6_golden_section_refinement/golden_section_refiner.py:1567-1580`, `phase4_16_difficulty_steering/difficulty_steering_analyzer.py:152-169`
- **Change**: Replaced `.get('field', default)` with explicit `'field' not in r` guard + direct `r['field']` access. Records missing required fields are now skipped instead of relying on potentially misleading defaults.
- **Test**: Updated `tests/test_integration/test_parallel_equivalence.py` and `tests/verify_parallel_output.py` to match.
- **Status**: DONE

### Fix 2: M2 — Fatal error on missing GPU files during merge
- **File**: `common/parallel_runner.py:1647-1652`
- **Change**: Changed `logger.warning()` to `raise RuntimeError()` when fewer GPU result files exist than expected. Prevents partial data being treated as complete.
- **Test**: New test `TestMergeParallelResultsParquetPath::test_missing_gpu_files_raises_error`
- **Status**: DONE

### Fix 3: M3 — Integration tests for general parquet merge router
- **File**: `tests/test_common/test_parallel_runner.py`
- **Change**: Added 4 new tests for `_merge_parallel_results()`:
  - `test_missing_gpu_files_raises_error` — verifies RuntimeError on partial files
  - `test_all_gpu_files_merge_correctly` — verifies correct total row count
  - `test_old_merged_files_cleaned_up` — verifies stale merged files removed
  - `test_deduplication_by_task_id` — verifies overlapping task_ids deduplicated
- **Status**: DONE (all 883 tests pass)

### Fix 4: M4 — Fail-fast dependency pre-check before model loading
- **File**: `phase4_8_steering_analysis/steering_effect_analyzer.py:88-115`
- **Change**: Added `_validate_dependencies_exist()` method called before `load_model_and_tokenizer()`. Checks Phase 3.6 baseline data, Phase 2.6 probes (probe mode), and Phase 2.5 SAE latents (SAE mode) exist before committing VRAM.
- **Status**: DONE

## Backlog

*None — all findings addressed.*

## Skipped

| Finding | Reason |
|---------|--------|
| L1: `load_json()` no error handling | Standard Python utility pattern — callers handle errors where appropriate |
| L2: Probe discovery returns `[]` for missing+corrupted | Corruption is logged as warning; function has clear "probes unavailable" semantics |
| L3: No parquet schema validation | Version checking provides sufficient guard; same-pipeline writes |
| L4: Checkpoint loading growth | Bounded by problem count (~200 records), negligible vs model weights |
