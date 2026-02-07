# Implementation Plan
**Generated from code review**: 2026-02-07 (Final pre-production review)

## Immediate Fixes

### Fix 1: Graceful degradation in fallback parquet merge (M1)
- **File**: `common/parallel_runner.py:1643-1648`
- **Change**: Wrap `pd.read_parquet()` in try/except, log warning, skip corrupted files
- **Test**: Verify merge succeeds when one GPU file is corrupted

### Fix 2: Graceful degradation in Phase 8.3 merge (M2)
- **File**: `common/parallel_runner.py:864-868`
- **Change**: Log error and skip corrupted file instead of re-raising RuntimeError
- **Test**: Verify partial merge when one GPU file is bad

### Fix 3: Add try/except to unguarded json.load() in merge paths (L2)
- **File**: `common/parallel_runner.py:634` and `common/parallel_runner.py:1349`
- **Change**: Wrap in try/except with error logging
- **Test**: Minor -- verify graceful handling of corrupted JSON

### Fix 4: Add missing_ok=True to cleanup unlink calls (L4)
- **File**: `common/parallel_runner.py:121, 673, 682`
- **Change**: Change `f.unlink()` to `f.unlink(missing_ok=True)`
- **Test**: None needed

### Fix 5: Remove legacy feature_idx fallback (L5)
- **File**: `experiments/linear_probe_sanity_check/run_sanity_check.py:373`
- **Change**: Replace `best_latent_info.get('latent_idx', best_latent_info.get('feature_idx'))` with `best_latent_info['latent_idx']`
- **Test**: None needed

### Fix 6: Hook registration optimization (L6)
- **File**: `phase4_8_steering_analysis/steering_effect_analyzer.py:567-617`
- **Change**: Move hook registration outside the per-problem loop where possible
- **Test**: Verify steering results unchanged

### Fix 7: Reduce GPU cache cleanup frequency (L7/E7)
- **File**: `phase4_8_steering_analysis/steering_effect_analyzer.py:621-627`
- **Change**: Increase cleanup interval from every 10 to every 50 items
- **Test**: Verify no OOM on production workloads

### Fix 8: Replace .iterrows() with .itertuples() (L8)
- **File**: `phase4_8_steering_analysis/steering_effect_analyzer.py:567, 671, 774`
- **Change**: Use `.itertuples()` or direct indexing
- **Test**: Verify identical outputs

### Fix 9: Remove redundant .clone() in activation hooks (L9)
- **File**: `common/activation_hooks.py:68`
- **Change**: Change `.detach().clone().cpu()` to `.detach().cpu()`
- **Test**: Verify activations unchanged

## Skipped

| Finding | Reason |
|---------|--------|
| L1 (binomial null hypothesis) | Statistical interpretation issue, not code bug. Primary metrics (rates) unaffected. |
| L3 (metadata corruption asymmetry) | Intentional design -- metadata is authority on processed tasks. |
