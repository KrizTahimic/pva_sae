# Implementation Plan
**Generated from code review**: 2026-02-07 (Final pre-production review)

## Immediate Fixes (ALL COMPLETED)

### Fix 1: L2 — Add `baseline_passed` column validation in steering_metrics.py
- **File**: `common/steering_metrics.py:70-72, 120-122`
- **Change**: Added explicit column existence check before accessing `baseline_passed`, raising descriptive `ValueError` instead of opaque `KeyError`
- **Test**: Updated `test_steering_metrics.py` to expect `ValueError` with match pattern
- **Status**: DONE

### Fix 2: L2 — Explicit boolean comparisons in steering_metrics.py
- **File**: `common/steering_metrics.py:76, 122, 175`
- **Change**: Changed `results['baseline_passed']` to `results['baseline_passed'] == True` and `results[modified_col]` to `results[modified_col] == True` for consistency
- **Test**: All 879 tests pass
- **Status**: DONE

### Fix 3: L5 — Use heapq.nsmallest for top-K selection in Phase 2.5
- **File**: `phase2_5_separation_score_analysis/sae_analyzer.py:223-231`
- **Change**: Replaced `sorted(...)[:k]` with `heapq.nsmallest(k, ...)` for O(n log k) efficiency
- **Test**: Produces identical output, ~10x faster for 416k items
- **Status**: DONE

### Fix 4: L4 — Use color constants in Phase 7.12
- **File**: `phase7_12_instruct_auroc_f1/instruct_auroc_f1_evaluator.py:227-263`
- **Change**: Replaced hardcoded `#2ecc71`/`#e74c3c` with `COLOR_CORRECT_PREDICTING`/`COLOR_INCORRECT_PREDICTING`
- **Test**: Visualization-only change
- **Status**: DONE

### Fix 5: L4 — Use color constants in Phase 7.9
- **File**: `phase7_9_universality_analysis/universality_analysis.py:216,229,242`
- **Change**: Replaced hardcoded hex colors with `COLOR_CORRECTION`, `COLOR_CORRUPTION`, `COLOR_PRESERVATION`, `COLOR_PRESERVATION_LIGHT`, `COLOR_CORRECT_DARK`, `COLOR_INCORRECT_DARK`
- **Test**: Visualization-only change
- **Status**: DONE

### Fix 6: L3 — Use phase discovery in reevaluate_phase3_6.py
- **File**: `scripts/reevaluate_phase3_6.py:52-63`
- **Change**: Replaced hardcoded `MODEL_DIRS` with `get_phase_output_dir()` lookups via `_get_model_dirs()` function
- **Test**: Script is maintenance-only, already executed
- **Status**: DONE

## Backlog

### B1: L1 — Add integration test for partial GPU failure recovery
- **File**: `common/parallel_runner.py:382-387`
- **Change**: Add test that mocks multiprocessing GPU failure and verifies checkpoint-based recovery
- **Priority**: LOW — defensive error path already works, just untested

### B2: M1 — Document prompt construction convention per data source
- **File**: `CLAUDE.md` or phase docstrings
- **Change**: Document that phases loading from analysis split (without `prompt` column) correctly rebuild prompts, while phases loading from Phase 1 output use `row['prompt']`
- **Priority**: LOW — not a bug, just needs documentation

## Skipped

| Finding | Reason |
|---------|--------|
| Phase 6.3 color constants | Colors are for prompt sections (problem/tests/solution), not correction/corruption — no matching constant exists |
| Other reevaluate scripts (3.5, 3.5_multi, 1_llama, 4_8) | Already run, serve their purpose, LOW priority |
