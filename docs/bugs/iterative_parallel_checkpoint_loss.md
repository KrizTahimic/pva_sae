# Bug: Iterative Parallel Runner Loses All Work on Single GPU Failure

## Summary

When running iterative-parallel phases (3.5, 4.5, 4.6, 8.2), if **any single GPU** fails or times out, **all work from all GPUs is lost** for that iteration. Checkpoints are only saved after ALL GPUs complete an iteration.

## Affected Phases

| Phase | Description | Iteration Type |
|-------|-------------|----------------|
| 3.5 | Temperature robustness | Temperatures |
| 4.5 | Coefficient grid search | Coefficients |
| 4.6 | Golden section refinement | Refinement points |
| 8.2 | Threshold optimizer | Percentiles |

## Root Cause

In `common/iterative_parallel_runner.py`, checkpoints are saved **per-iteration**, not **per-GPU**:

```python
# Lines 226-228
# Save checkpoint after each iteration
if self.checkpoint_dir:
    self._save_checkpoint(value, merged)  # Only saves AFTER successful merge
```

The merge requires all GPUs to complete:
```python
# Lines 208-224 (simplified)
for gpu_id in range(self.n_gpus):
    result = result_queue.get(timeout=timeout_per_value)  # Blocks until ALL complete
    # ... merge results
```

If any GPU times out, the merge loop exits with error, and `_save_checkpoint()` is never called.

## Observed Behavior

From Phase 3.5 run on 2026-01-25:
- Workers 0, 1, 2 completed temperature=0.0 (~10-12 mins each)
- Worker 3 timed out after 20 mins
- **Result**: All 602 activation files from GPUs 0,1,2 exist, but NO checkpoint saved
- **On restart**: Entire phase runs from scratch, redoing all work

## Impact

- Lost 30+ GPU-minutes of work due to one slow GPU
- No way to resume from partial completion
- Must re-run entire iteration even if 3/4 GPUs succeeded

## Comparison: Data-Parallel Phases (Working Correctly)

Data-parallel phases (1, 2.2, 3.6, 4.8, etc.) handle this better:
- Each GPU saves its own checkpoints independently
- If GPU 3 fails, GPUs 0,1,2 checkpoints are preserved
- On restart, only GPU 3's work is redone

## Proposed Fix

### Option A: Save Per-GPU Results Immediately (Recommended)

Save each GPU's results as soon as they complete, before waiting for all GPUs:

```python
# After receiving result from each GPU
self._save_gpu_checkpoint(gpu_id, value, gpu_result)

# On restart, load partial results and only re-run failed GPUs
```

### Option B: Graceful Degradation

If some GPUs timeout, still save results from completed GPUs and mark iteration as "partial":

```python
if completed_gpus >= min_required_gpus:
    merged = self._merge_results(completed_results)
    self._save_checkpoint(value, merged, partial=True)
```

### Option C: Longer Timeouts with Progress Detection

Instead of fixed timeout, detect if GPU is making progress (new log output) and only timeout if truly stuck.

## Workaround (Current)

For now, users must:
1. Run with longer timeouts (but this risks truly stuck workers)
2. Accept that any failure means full restart
3. Use `--start N --end M` to test on smaller subsets first

## Files to Modify

- `common/iterative_parallel_runner.py` - Core fix
- `phase3_5_temperature_robustness/temperature_runner.py` - Phase-specific handling
- `phase4_5_coefficient_selection/` - Similar changes
- `phase4_6_golden_section/` - Similar changes
- `phase8_2_percentile_optimizer/` - Similar changes

## Priority

**High** - This bug causes significant wasted GPU time on long-running phases.

## Discovered

2026-01-25 during Phase 3.5 parallel run where GPU 3 timed out after 20 minutes.
