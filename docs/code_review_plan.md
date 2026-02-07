# Implementation Plan
**Generated from code review**: 2026-02-07

## Immediate Fixes (ALL COMPLETED)

### Fix M1: SAE layer cache in Phase 3.8 candidate evaluation
- **File**: `phase3_8_auroc_f1_evaluation/auroc_f1_evaluator.py`
- **Change**: Added layer-keyed SAE cache in `evaluate_candidates_and_select_best()`. SAEs loaded once per unique layer, passed through to `evaluate_single_latent()` and `load_split_activations()` via optional `sae` parameter. Cache freed in `finally` block.
- **Test**: Existing tests pass. New parameter is backward-compatible (default `sae=None`).

### Fix M2: Unit tests for model_loader.py
- **File**: `tests/test_common/test_model_loader.py` (NEW)
- **Change**: 9 tests covering dtype selection (CUDA bf16/fp16, CPU float32, explicit override), eager attention flag, device placement, eval mode, and model info extraction.
- **Test**: All 9 tests pass.

### Fix M3: Parallel runner subprocess path tests
- **File**: `tests/test_common/test_parallel_runner.py`
- **Change**: Added 3 tests for `run_phase_parallel()` orchestration: non-parallelizable phase rejection, worker failure causing RuntimeError, and successful merge on all-pass.
- **Test**: All 3 tests pass.

### Fix L1: Clarifying comment on binomial test design
- **Files**: `phase4_14_statistical_significance/significance_tester.py`, `phase5_9_orthogonalization_significance/orthogonalization_significance_tester.py`
- **Change**: Added NOTE comment explaining that effect_size uses baseline_rate while p_value uses effective_rate, and that both values are returned transparently.
- **Test**: No test changes needed.

### Fix L2: Unit tests for memory_utils.py
- **File**: `tests/test_common/test_memory_utils.py` (NEW)
- **Change**: 12 tests covering get_memory_percent, check_memory_usage thresholds, cleanup_memory (with/without CUDA), cleanup_memory_aggressive, and log_memory_status.
- **Test**: All 12 tests pass.

### Fix L3: Unit tests for logging.py
- **File**: `tests/test_common/test_logging.py` (NEW)
- **Change**: 18 tests covering LoggingManager init/handlers, setup_logging, get_logger with phase context, invalid LOG_LEVEL, tqdm_with_logging, and phase lifecycle events.
- **Test**: All 18 tests pass.

### Fix L4: Validate manifest file existence in phase discovery
- **File**: `common/phase_discovery.py`
- **Change**: After loading manifest, validates that all referenced output files exist on disk. Logs warning for missing files instead of silently returning paths that don't exist.
- **Test**: Existing tests pass.

### Fix L5: Phase 1 cross-run resume count check
- **File**: `common/parallel_runner.py`
- **Change**: Enhanced logging to report exact count of activation files found (correct + incorrect breakdown). Still uses same heuristic but now transparent about what was found.
- **Test**: Existing tests pass.

### Fix L6: Document Phase 3.6 generation_idx difference
- **File**: `phase3_6_hyperparameter_baseline/hyperparameter_runner.py`
- **Change**: Added NOTE comment explaining why Phase 3.6 omits generation_idx (single generation at temp=0.0) and that outputs are never merged with Phase 3.5.
- **Test**: No test changes needed.

### Fix L7: Batch GPU cache cleanup in Phase 4.8
- **File**: `phase4_8_steering_analysis/steering_effect_analyzer.py`
- **Change**: Moved `torch.cuda.empty_cache()` from per-record `finally` block to periodic cleanup block (every 10 records). Hook removal still happens per-record (required), but expensive GPU synchronization is batched.
- **Test**: Existing tests pass.

## Backlog
(none)

## Skipped
(none)
