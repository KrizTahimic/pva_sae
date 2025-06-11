# Phase 1.1: Dataset Splitting

This module splits the Phase 1.0 dataset into balanced subsets for SAE analysis, hyperparameter tuning, and validation using stratified randomized interleaving.

## Quick Start

```bash
# Basic usage with auto-discovery
python3 run.py phase 1.1

# With quality report
python3 run.py phase 1.1 --generate-report

# Specific input dataset
python3 run.py phase 1.1 --input data/phase1_0/my_dataset.parquet
```

## What It Does

**Problem Solved**: The original interleaving approach concentrated low-complexity tasks in early splits and high-complexity tasks in later splits, creating biased datasets.

**Solution**: Stratified randomized interleaving:
1. Divides data into complexity strata (bins)
2. Randomly shuffles within each stratum  
3. Applies interleaved pattern across all strata
4. Ensures each split gets tasks from all complexity levels

## Output Files

```
data/phase1_1/
├── sae_indices.json          # Indices for SAE analysis (50%)
├── hyperparams_indices.json  # Indices for hyperparameter tuning (10%)
├── validation_indices.json   # Indices for validation (40%)
├── split_metadata.json       # Statistics and ratios
├── timestamp.txt             # Creation time
└── split_quality_report.html # Quality analysis (if --generate-report)
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--random-seed` | `42` | Random seed for reproducibility |
| `--n-strata` | `10` | Number of complexity strata |
| `--split-output-dir` | `data/phase1_1` | Output directory |
| `--generate-report` | `false` | Generate HTML quality report |

**Note:** Split ratios are fixed at 50% SAE analysis, 10% hyperparameter tuning, 40% validation based on research requirements.

## Quality Validation

The module automatically validates:
- **Ratio accuracy**: Within 2% of target ratios
- **Complexity distribution**: Similar across all splits (KS test)
- **Correctness balance**: Similar correct/incorrect rates

## Requirements

- Dataset with `task_id` and `complexity_score` columns
- Minimum 10 samples (larger datasets work better)
- Phase 1.0 must be completed first

## Example Output

```
Split 'sae': 487 samples (50.0%), complexity mean=5.23
Split 'hyperparams': 97 samples (10.0%), complexity mean=5.18  
Split 'validation': 390 samples (40.0%), complexity mean=5.25

✅ Split quality: PASS
```

## Integration with Other Phases

- **Input**: Auto-discovers latest Phase 1.0 dataset
- **Output**: Used by Phase 2 for SAE analysis on balanced data
- **Validation**: Quality metrics feed into Phase 3 validation