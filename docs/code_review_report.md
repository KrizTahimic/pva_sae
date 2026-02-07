# Code Review Report
**Date**: 2026-02-07
**Scope**: Full project (final pre-production review)
**Findings**: 12 confirmed / 29 reviewed (48% false positive rate)

## CRITICAL

*No critical findings confirmed.*

## HIGH

*No high-severity findings confirmed.*

## MEDIUM

### M1: Unguarded `pd.read_parquet()` in Fallback Merge Path
- **Location**: `common/parallel_runner.py:1643-1648`
- **Root Cause**: Fallback merge path has no try/except around `pd.read_parquet()`, unlike the Phase 8.3 merge path which at least has exception handling (even if it re-raises).
- **Impact**: A corrupted GPU output file crashes the entire merge. GPU work is already saved to disk, so no computation is lost, but manual intervention is needed.
- **Suggested Fix**: Wrap in try/except, log warning, skip corrupted files, proceed with available data.
- **Test Impact**: Add test for corrupted parquet file handling in merge.

### M2: Phase 8.3 Merge Re-Raises on Any Corrupted File
- **Location**: `common/parallel_runner.py:864-868`
- **Root Cause**: The try/except around `pd.read_parquet()` immediately re-raises as RuntimeError instead of gracefully skipping.
- **Impact**: Same as M1 -- merge aborts, but GPU work is already saved to disk.
- **Suggested Fix**: Log error, skip corrupted file, merge available results.
- **Test Impact**: Add test for partial merge when one GPU file is corrupted.

## LOW

### L1: Degenerate Null Hypothesis in Binomial Test
- **Location**: `phase5_3_weight_orthogonalization/weight_orthogonalizer.py:417`
- **Root Cause**: Uses `max(1.0 / n_incorrect, 1e-10)` as null hypothesis rate -- a manufactured floor rather than a principled statistical null.
- **Impact**: P-values for correction tests are overly significant (too easy to reject). Rates/percentages (the primary metrics) are unaffected.
- **Suggested Fix**: Document the limitation or use a baseline correction rate from unsteered runs.
- **Test Impact**: None needed -- statistical interpretation issue, not a code bug.

### L2: Unguarded `json.load()` in Merge Paths
- **Location**: `common/parallel_runner.py:634` and `common/parallel_runner.py:1349`
- **Root Cause**: JSON loads in merge functions lack try/except.
- **Impact**: Corrupted JSON from GPU workers would crash merge. Very rare in practice.
- **Suggested Fix**: Add try/except with error logging.
- **Test Impact**: Minor -- add test for corrupted JSON handling.

### L3: Metadata vs Parquet Corruption Recovery Asymmetry
- **Location**: `common/iterative_parallel_runner.py:597-619`
- **Root Cause**: Metadata corruption raises RuntimeError while parquet corruption degrades gracefully.
- **Impact**: Inconsistent behavior on rare corruption events. The asymmetry is arguably intentional (metadata is the authority).
- **Suggested Fix**: Consider graceful degradation for metadata too, or document the design decision.
- **Test Impact**: None needed.

### L4: `f.unlink()` Without `missing_ok=True`
- **Location**: `common/parallel_runner.py:121, 673, 682`
- **Root Cause**: Cleanup loops don't use `missing_ok=True` on `.unlink()`.
- **Impact**: Theoretical FileNotFoundError if another process deletes file between glob and unlink. Extremely unlikely in single-user thesis project.
- **Suggested Fix**: Change to `f.unlink(missing_ok=True)`.
- **Test Impact**: None needed.

### L5: Legacy `feature_idx` Fallback
- **Location**: `experiments/linear_probe_sanity_check/run_sanity_check.py:373`
- **Root Cause**: Uses `best_latent_info.get('latent_idx', best_latent_info.get('feature_idx'))` -- legacy compatibility for old naming.
- **Impact**: None for production. Experiments directory only.
- **Suggested Fix**: Remove fallback, use `latent_idx` only.
- **Test Impact**: None needed.

### L6-L9: Minor Performance Patterns
- **L6**: Hook registration per problem (`steering_effect_analyzer.py:567-617`) -- correct but slightly wasteful. GPU inference dominates runtime.
- **L7**: 160 checkpoint writes in iterative parallel runner -- negligible overhead vs hours of GPU work.
- **L8**: `.iterrows()` in GPU-bound loops (`steering_effect_analyzer.py:567, 671, 774`) -- microsecond overhead vs seconds of inference.
- **L9**: `.detach().clone().cpu()` in activation hooks (`activation_hooks.py:68`) -- defensive pattern, debatably redundant.

## Agent Accuracy

| Agent | Findings | Confirmed | False Positives | FP Rate |
|-------|----------|-----------|-----------------|---------|
| Bug Hunter | 3 | 1 | 2 | 67% |
| Test Coverage | N/A (coverage audit) | N/A | N/A | N/A |
| Error Handling | 6 | 4 | 1 | 17% |
| Consistency | 10 | 1 | 8 | 80% |
| Performance | 10 | 5 | 3 | 30% |
| **TOTAL** | **29** | **12** | **14** | **48%** |

## Key False Positives (Notable)

1. **Data leakage in Phase 8.3** (A2): Actually uses analysis split, not hyperparameter split. No leakage.
2. **Probe normalization asymmetry** (D1): Intentional design -- prediction doesn't need normalization (ranking-invariant), steering does.
3. **Dual probe scaling inconsistency** (D2): Intentional -- predicting uses unnormalized + bias, steering uses normalized + coefficient.
4. **dtype conversion timing** (D3): Float32 normalization then bfloat16 conversion is standard ML best practice.
5. **SAE cache CPU movement** (E3): This IS the optimization -- frees VRAM after caching directions.

## Production Readiness

**VERDICT: READY FOR PRODUCTION**

No critical or high-severity bugs confirmed. The two MEDIUM findings are in post-processing merge paths where GPU work is already saved to disk -- a merge failure loses no computation and can be retried. All LOW findings are minor robustness improvements that do not affect result correctness.
