# Code Review Report
**Date**: 2026-02-07
**Scope**: Full project
**Findings**: 4 confirmed / 15 reviewed (73% false positive rate)

## Methodology

5 parallel specialist agents (Bug Hunter, Test Coverage Auditor, Error Handling Reviewer, Consistency Checker, Performance Reviewer) independently scanned the full codebase. Their 15 highest-severity findings were then verified by a skeptical verification agent that read the actual source code at every cited location.

---

## HIGH

### H1: Partial GPU failures silently accepted in iterative parallel runner
- **Location**: `common/iterative_parallel_runner.py:489-521`
- **Root Cause**: `_collect_results()` continues with partial GPU results when some GPUs fail or time out. If 3 of 4 GPUs succeed, metrics are computed from 75% of the data without escalating the error.
- **Impact**: Early stopping decisions in iterative parallel phases (4.5, 4.6, 8.2) could be based on biased/incomplete subsets. The errors ARE logged but processing continues, producing plausible but potentially incorrect metrics.
- **Suggested Fix**: Add a `min_gpu_success_ratio` threshold (e.g., 0.75). If fewer GPUs succeed than the threshold, treat as total failure and return None. Log the exact problem IDs that were lost.
- **Test Impact**: Add test in `tests/test_common/test_iterative_parallel_runner.py` for partial failure scenario — verify that metrics are either correct or the run is aborted.

## MEDIUM

### M1: Corrupted parquet files skipped silently during value result merge
- **Location**: `common/iterative_parallel_runner.py:683-688`
- **Root Cause**: When merging per-value GPU results from parquet checkpoints, corrupted files are caught by a broad `except Exception` and skipped with only a warning log. The merge continues with incomplete data.
- **Impact**: If a GPU's checkpoint parquet is corrupted (e.g., partial write during crash), the iterative runner's per-value merge produces metrics from fewer GPUs. This affects early stopping decisions mid-run. Not a final output issue since final results use a separate merge path.
- **Suggested Fix**: Either raise an error when a parquet file fails to load (fail fast), or track the expected vs actual GPU count and warn prominently if any are missing.
- **Test Impact**: Add test that verifies behavior when one GPU's parquet is unreadable.

## LOW

### L1: Multi-candidate merge uses positional indexing
- **Location**: `common/parallel_runner.py:1131-1142`
- **Root Cause**: When merging multi-candidate steering results across GPUs, candidates are matched by list index position rather than by a unique identifier (e.g., `layer_latent_idx`).
- **Impact**: Theoretical only — in practice all GPUs load candidates from the same deterministic JSON file, so ordering is identical. Would only be a problem if candidate lists diverged between GPUs, which cannot happen given current data flow.
- **Suggested Fix**: Match candidates by `(layer, latent_idx)` key instead of positional index for robustness.
- **Test Impact**: None required — current behavior is correct for actual usage.

### L2: Direction normalization called redundantly per candidate
- **Location**: `phase4_5_coefficient_grid_search/steering_coefficient_selector.py:196-202`
- **Root Cause**: `_get_latent_direction()` normalizes and converts dtype on every call, including when the same direction is reused across coefficient evaluations.
- **Impact**: Negligible — normalizing a 2304-dim vector takes ~10 microseconds. Even 10,000 calls total ~0.1 seconds. The bottleneck is `model.generate()`.
- **Suggested Fix**: Cache pre-normalized directions at initialization. Low priority.
- **Test Impact**: None required.

---

## Agent Accuracy

| Agent | Findings Reviewed | Confirmed | False Positive Rate |
|-------|:-:|:-:|:-:|
| Bug Hunter (A) | 3 | 1 | 67% |
| Test Coverage (B) | 1 | 0 | 100% |
| Error Handling (C) | 5 | 2 | 60% |
| Consistency (D) | 1 | 0 | 100% |
| Performance (E) | 5 | 1 | 80% |
| **Total** | **15** | **4** | **73%** |

### Key False Positive Patterns
1. **PyTorch `nn.Module.cpu()` misunderstanding** (3 findings) — Agents didn't realize `.cpu()` is in-place for `nn.Module`
2. **Agent fabricated/misread code** (2 findings) — C2 cited code that doesn't match actual source; D9 read `[0]` where code says `sorted(...)[-1]`
3. **Conflated separate functions** (1 finding) — B4 confused `_dedup_by_task_id` (keeps first) with `_merge_phase8_3_results` (keeps last)

### Notable Non-Issues Verified as Correct
- Preservation rate calculation correctly returns NaN for zero correct problems
- Direction normalization raises ValueError on NaN/zero-norm (not silent)
- Corrupted checkpoint metadata raises RuntimeError with recovery instructions
- Merged file selection is consistent across codebase (both use `sorted()[-1]`)
- `.copy()` in `split_by_correctness` is correct pandas practice
- `.detach().clone().cpu()` is the standard safe activation extraction pattern
