# Code Review Implementation Plan
**Date:** 2026-02-07 | **Branch:** icml | **Findings:** 13 confirmed, 12 to fix

---

## Fix Order (grouped by file to minimize context switches)

### Group 1: `common/parallel_runner.py` (H1, H2)

**H1 — Add try-except around json.load() in merge (line 89-91)**
```python
# Before:
for f in gpu_files:
    with open(f) as fh:
        results.append(json.load(fh))

# After:
for f in gpu_files:
    try:
        with open(f) as fh:
            results.append(json.load(fh))
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load GPU result file {f.name}: {e}")
        raise RuntimeError(f"Corrupted GPU result file {f.name}: {e}. "
                          f"Other GPU files may still be valid in {f.parent}")
```

**H2 — Add try-except around pd.read_parquet() in merge (line 821)**
```python
# Before:
dfs = [pd.read_parquet(f) for f in gpu_files]

# After:
dfs = []
for f in gpu_files:
    try:
        dfs.append(pd.read_parquet(f))
    except Exception as e:
        logger.error(f"Failed to read GPU parquet file {f.name}: {e}")
        raise RuntimeError(f"Corrupted GPU parquet file {f.name}: {e}")
```

### Group 2: `common/iterative_parallel_runner.py` (H3, H4, M1)

**H3 — Re-raise on metadata corruption (lines 574-580)**
```python
# Before:
except Exception as e:
    logger.warning(f"Failed to load existing metadata: {e}")

# After:
except Exception as e:
    raise RuntimeError(
        f"Corrupted checkpoint metadata at {meta_file}: {e}\n"
        f"To recover, delete the metadata file and restart:\n"
        f"  rm {meta_file}"
    )
```

**H4 — Narrow except clause for orchestrator state (line 737)**
```python
# Before:
except Exception:
    data = {'completed_values': [], 'results': {}}

# After:
except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
    logger.warning(f"Could not load orchestrator state ({e}), starting fresh")
    data = {'completed_values': [], 'results': {}}
```

**M1 — Log shutdown exceptions (lines 529-532)**
```python
# Before:
except Exception:
    pass

# After:
except Exception as e:
    logger.debug(f"Failed to send shutdown to worker queue: {e}")
```

### Group 3: `phase4_8_steering_analysis/steering_effect_analyzer.py` (NC1, NC2, L1)

**NC1 — Exclude generation errors from corruption metrics (lines 793-803)**
```python
# Before:
except Exception as e:
    logger.warning(f"Error evaluating preservation for {row['task_id']}: {e}")
    total += 1
    detailed_results.append({
        ...
        'steered_correct': False,
        'flipped': True,
        'steered_code': None,
    })

# After:
except Exception as e:
    logger.warning(f"Generation error for {row['task_id']}: {e}")
    # Don't count generation errors in metrics - they are infrastructure failures,
    # not meaningful signal about steering effectiveness
    detailed_results.append({
        ...
        'steered_correct': None,  # Indeterminate
        'flipped': False,  # Not a real flip
        'generation_error': True,
        'steered_code': None,
    })
    # Note: total NOT incremented - excluded from rate calculation
```

**NC2 — Add SAE cache cleanup between candidates**
After each candidate evaluation, add:
```python
# Clear SAE cache to free VRAM between candidates
for layer_idx, sae in self.sae_cache.items():
    sae.cpu()
self.sae_cache.clear()
torch.cuda.empty_cache()
```

**L1 — Replace iterrows() with itertuples() (line 539, 643, 746)**
```python
# Before:
for enum_idx, (_, row) in enumerate(tqdm_with_logging(problems_to_process.iterrows(), ...)):

# After:
for enum_idx, row in enumerate(tqdm_with_logging(problems_to_process.itertuples(), ...)):
    # Access via row.task_id instead of row['task_id']
```
Note: This requires updating all `row['col']` → `row.col` access patterns in the loop body. Verify all column accesses.

### Group 4: `common/iterative_parallel_runner.py` (L1 continued)

**L1b — Replace iterrows() in checkpoint merge (lines 677-686)**
```python
# Before:
for _, row in df.iterrows():
    task_id = row.get('task_id')

# After: Vectorized approach
all_dfs = []
for parquet_file in value_dir.glob("gpu_*_results.parquet"):
    try:
        all_dfs.append(pd.read_parquet(parquet_file))
    except Exception as e:
        logger.warning(f"Failed to load checkpoint {parquet_file}: {e}")
if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    if 'task_id' in merged.columns:
        merged = merged.drop_duplicates(subset=['task_id'], keep='last')
    all_results = merged.to_dict('records')
```

### Group 5: `common/sae_loader.py` (L3)

**L3 — Add bounds check on decoder access (line 40-42)**
```python
# Before:
def get_decoder_weight(self, latent_idx: int) -> torch.Tensor:
    return self.W_dec[latent_idx, :]

# After:
def get_decoder_weight(self, latent_idx: int) -> torch.Tensor:
    if latent_idx < 0 or latent_idx >= self.W_dec.shape[0]:
        raise IndexError(
            f"latent_idx {latent_idx} out of range [0, {self.W_dec.shape[0]-1}] "
            f"for SAE with {self.W_dec.shape[0]} latents"
        )
    return self.W_dec[latent_idx, :]
```

### Group 6: Consistency — Colors (H5)

**H5 — Replace hardcoded colors with config constants**

Files to update (add `from common.config import COLOR_CORRECTION, COLOR_CORRUPTION, COLOR_PRESERVATION`):
- `phase4_8_steering_analysis/steering_effect_analyzer.py`
- `phase7_6_instruct_steering/instruct_steering_analyzer.py`
- Any other files found via: `grep -r "color='green'\|color='red'\|color='gold'" phase*/`

Replace: `color='green'` → `color=COLOR_CORRECTION`, `color='red'` → `color=COLOR_CORRUPTION`, `color='gold'` → `color=COLOR_PRESERVATION`

### Group 7: Consistency — Terminology (H6, M2)

**H6 — Rename feature_type → latent_type in phase3_10**
- File: `phase3_10_temperature_auroc_f1/temperature_evaluator.py`
- Lines: 304, 305, 307, 311, 323, 332, 342, 350, 366
- Simple find-replace: `feature_type` → `latent_type`

**M2 — Fix docstring terminology**
- File: `phase8_3_selective_steering/selective_steering_analyzer.py:5`
- Replace "feature activation" → "latent activation"
- Search other files: `grep -r "feature activation\|feature type" phase*/` for additional occurrences

---

## Estimated Effort
- **Groups 1-2 (error handling):** ~30 min — straightforward try-except additions
- **Group 3 (phase4_8 changes):** ~45 min — NC1 requires careful metric logic changes
- **Group 4 (iterative_parallel):** ~15 min — vectorize merge
- **Group 5 (sae_loader):** ~5 min
- **Groups 6-7 (consistency):** ~30 min — multi-file but mechanical

## Testing
- Run existing tests: `pytest tests/ -x`
- For H1/H2: Add test for corrupted JSON/parquet handling
- For NC1: Verify preservation metrics exclude generation errors
- For L1: Verify itertuples() access pattern works with all column names
