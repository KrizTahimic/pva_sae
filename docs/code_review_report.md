# Code Review Report
**Date**: 2026-02-07
**Scope**: Full project (final pre-production review)
**Reviewers**: 5 specialist agents + 1 verification agent
**Findings**: 7 confirmed / 15 reviewed (53% false positive rate)

## CRITICAL

*No critical findings.*

## HIGH

*No high-severity findings.*

## MEDIUM

### M1: Inconsistent `.get()` defaults in preservation classification
- **Location**: `phase4_5_coefficient_grid_search/steering_coefficient_selector.py:1144`, `phase4_6_golden_section_refinement/golden_section_refiner.py:1579`
- **Root Cause**: `r.get('steered_correct', True)` in the preservation check means missing `steered_correct` counts as preserved. Other checks (correction, corruption) conservatively exclude missing data — this one doesn't.
- **Impact**: If `steered_correct` field is ever missing, preservation count inflates. In practice, fields should always be present since they're written by the same pipeline.
- **Suggested Fix**: Use `r.get('steered_correct', False)` for preservation (matching the conservative pattern). Phase 4.16 already does this correctly at line 168.
- **Test Impact**: Update `tests/test_integration/test_parallel_equivalence.py` and `tests/verify_parallel_output.py` to match.

### M2: Partial GPU results merged with warning only
- **Location**: `common/parallel_runner.py:1647-1670`
- **Root Cause**: When fewer GPU result files exist than expected, the merge logs a warning but continues. General parquet merge path for phases 1, 3.5, 3.6, 7.3, etc.
- **Impact**: If a GPU produces a corrupted/missing parquet file, metrics are calculated on partial data. Mitigated by upstream `run_phase_parallel` which raises `RuntimeError` on full worker failures.
- **Suggested Fix**: Add stricter check: raise error if `len(gpu_result_files) < n_gpus` unless explicitly allowed.
- **Test Impact**: Add integration test for `_merge_parallel_results()` with missing GPU files.

### M3: General parquet merge router lacks edge case tests
- **Location**: `common/parallel_runner.py:1554-1745`
- **Root Cause**: `_merge_parallel_results()` has tests for specific phase merges (4.5, 4.8, 8.3) but not for the general parquet path's edge cases (partial failures, old file cleanup at line 1683).
- **Impact**: Edge cases in merge logic could go undetected.
- **Suggested Fix**: Add tests for partial GPU files and old merged file cleanup.
- **Test Impact**: New test in `tests/test_common/test_parallel_runner.py`.

### M4: Phase dependencies loaded after model
- **Location**: `phase4_8_steering_analysis/steering_effect_analyzer.py:91-100`
- **Root Cause**: Model (~5-18GB VRAM) loaded at line 91, then `_load_dependencies()` at line 100. If deps fail, VRAM was wasted.
- **Impact**: On dependency failure, user waits for model load + error, then must fix and reload. Wastes minutes.
- **Suggested Fix**: Add lightweight file-existence pre-check before model loading.
- **Test Impact**: None needed.

## LOW

### L1: `load_json()` has no error handling
- **Location**: `common/utils.py:303-306`
- **Root Cause**: Utility function lets exceptions propagate. Standard Python pattern.
- **Impact**: JSONDecodeError on corrupted files produces confusing traceback instead of "Run phase X first" message.

### L2: Probe discovery returns `[]` for both missing and corrupted files
- **Location**: `common/phase_discovery.py:447-474`
- **Root Cause**: `_discover_probe_best_layers()` returns `[]` for file-not-found and exceptions. Corruption IS logged as warning (line 473).
- **Impact**: If probe file corrupted, phases silently proceed without probes. Phase 2.6 is optional, so defensible.

### L3: No parquet schema validation in checkpoint loading
- **Location**: `common/checkpoint_manager.py:379-391`
- **Root Cause**: Loaded DataFrames not validated for expected columns. Mitigated by version checking (lines 370-377).
- **Impact**: Corrupted checkpoints could cause downstream KeyErrors instead of clean messages.

## Notable False Positives (verified as correct)

| Claimed Bug | Why It's Correct |
|-------------|------------------|
| Early stopping `history[:-1]` excludes best (A2) | `current` is already appended to `history` at call site — `[:-1]` correctly excludes it |
| `torch.no_grad()` missing in extraction (E3) | Already present at `activation_hooks.py:92` |
| SAE GPU memory leak via `.cpu()` (E1) | `.cpu()` correctly frees GPU tensors; directions already cached separately |
| `latent_type` should be `latent_category` (D1) | CLAUDE.md explicitly says use `latent_type` |
| Direction dtype test missing (B3) | Test exists at `test_steering_setup.py:85-113` |
| LLAMA+HumanEval path untested (B5) | Test exists at `test_phase_discovery.py:56-62` |
| Phase manifest JSON not caught (C5) | `JSONDecodeError` propagates correctly — crash is appropriate for corrupt manifests |

## Agent Accuracy

| Agent | Findings | Confirmed | False Positive Rate |
|-------|----------|-----------|---------------------|
| Bug Hunter (A) | 2 | 1 | 50% |
| Test Coverage (B) | 3 | 1 | 67% |
| Error Handling (C) | 5 | 4 | 20% |
| Consistency (D) | 1 | 0 | 100% |
| Performance (E) | 3 | 1 | 67% |
| **Total** | **15** | **7** | **53%** |

## Areas That Passed Review
- Direction normalization (all paths use `normalize_direction()`)
- Path discovery (no hardcoded paths in production code)
- Color scheme consistency (all phases use config constants)
- Probe/SAE direction loading contracts
- Checkpoint atomic writes (temp file + rename pattern)
- Steering metrics terminology
- Dual-direction architecture (Phase 8.2/8.3)
- Config key consistency
- API contracts between phases (data columns, formats)

## Production Readiness

**VERDICT: READY FOR PRODUCTION**

No critical or high-severity bugs confirmed. The 7 confirmed findings are MEDIUM to LOW severity — defensive improvements, not correctness blockers. All previously known bug patterns (double-wrapped prompts, degenerate binomtest, cross-GPU merge issues, error handler success assumptions) have been fixed.
