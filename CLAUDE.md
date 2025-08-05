# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PVA-SAE (Python Value Attribution Sparse Autoencoder) is a thesis research project investigating program validity awareness in language models. The project analyzes how language models internally represent code correctness by:

1. Using Google's Gemma 2 (2B parameters) to generate Python solutions for MBPP problems
2. Classifying solutions as correct (pass@1) or incorrect based on test execution
3. Applying Sparse Autoencoders (SAEs) from GemmaScope to identify latent directions
4. Validating findings through statistical analysis (AUROC, F1) and model steering experiments

The methodology follows these phases:
- Phase 0: Difficulty analysis of MBPP problems
- Phase 0.1: Problem splitting (50% SAE analysis, 10% hyperparameter tuning, 40% validation)
- Phase 1.0: Dataset building with base generation (temperature=0.0)
- Phase 2.2: Pile activation caching for general language feature baseline
- Phase 2.5: SAE activation analysis with pile filtering using separation scores (split-aware)
- Phase 3: Validation through both statistical measures and causal intervention via model steering
- Phase 3.5: Temperature robustness testing on validation split
- Phase 3.6: Hyperparameter tuning set processing with baseline generation
- Phase 3.8: AUROC and F1 evaluation for PVA features
- Phase 3.10: Temperature-based AUROC analysis across different temperatures
- Phase 3.12: Difficulty-based AUROC analysis across problem difficulty levels
- Phase 4.5: Steering coefficient selection for model interventions
- Phase 4.8: Steering effect analysis with correction/corruption experiments

Each phase outputs to its own directory (data/phase0/, data/phase0_1/, data/phase1_0/, etc.) and automatically discovers inputs from previous phases.

### Data Output Structure
```
data/
├── phase0/           # Difficulty mappings
├── phase1_0/         # Generated code + activations (dataset building)
│   ├── dataset_*.parquet
│   ├── dataset_*_metadata.json
│   └── activations/
│       ├── correct/
│       │   └── {task_id}_layer_{n}.npz
│       └── incorrect/
│           └── {task_id}_layer_{n}.npz
├── phase0_1/         # Problem splitting
│   ├── sae_mbpp.parquet         # SAE split with full MBPP data
│   ├── hyperparams_mbpp.parquet # Hyperparameter split with full MBPP data
│   ├── validation_mbpp.parquet  # Validation split with full MBPP data
│   └── split_metadata.json
├── phase2_2/         # Pile activation caching
│   ├── pile_activations.npz
│   └── metadata.json
├── phase2_5/         # SAE analysis results with pile filtering
│   ├── sae_analysis_results.json
│   ├── top_20_features.json
│   ├── layer_{n}_features.json  # Per-layer analysis
│   └── best_layer.json          # Best PVA layer for Phase 3.5
├── phase3/           # Validation results
├── phase3_5/         # Temperature robustness (validation split only)
│   ├── dataset_temp_0_0.parquet  # 5 samples per task at temp 0.0
│   ├── dataset_temp_0_3.parquet  # 5 samples per task at temp 0.3
│   ├── dataset_temp_0_6.parquet  # 5 samples per task at temp 0.6
│   ├── dataset_temp_0_9.parquet  # 5 samples per task at temp 0.9
│   ├── dataset_temp_1_2.parquet  # 5 samples per task at temp 1.2
│   ├── metadata.json
│   └── activations/
│       └── task_activations/     # Single layer activations per task
│           └── {task_id}_layer_{best_layer}.npz
├── phase3_6/         # Hyperparameter tuning set processing
│   ├── dataset_hyperparams_temp_0_0.parquet
│   ├── metadata.json
│   └── activations/
│       └── task_activations/
│           └── {task_id}_layer_{best_layer}.npz
├── phase3_8/         # AUROC and F1 evaluation
│   ├── evaluation_results.json
│   ├── evaluation_summary.txt
│   ├── comparative_metrics.png
│   ├── confusion_matrix_correct.png
│   ├── confusion_matrix_incorrect.png
│   ├── f1_threshold_plot_correct.png
│   └── f1_threshold_plot_incorrect.png
├── phase3_10/        # Temperature-based AUROC analysis
│   ├── temperature_analysis_results.json
│   └── temperature_trends.png
├── phase3_12/        # Difficulty-based AUROC analysis
│   ├── difficulty_analysis_results.json
│   ├── difficulty_summary.txt
│   ├── difficulty_distribution.png
│   ├── metrics_comparison_by_difficulty.png
│   └── roc_curves_by_difficulty_{correct/incorrect}.png
├── phase4_5/         # Steering coefficient selection
│   ├── coefficient_analysis.json
│   ├── selected_coefficients.json
│   ├── phase_4_5_summary.json
│   ├── selected_problems_correct_steering.parquet
│   └── selected_problems_incorrect_steering.parquet
└── phase4_8/         # Steering effect analysis
    ├── steering_effect_analysis.json
    ├── steering_effect_analysis.png
    ├── phase_4_8_summary.json
    ├── selected_correct_problems.parquet
    ├── selected_incorrect_problems.parquet
    └── examples/
        ├── corrected_examples.json
        └── corrupted_examples.json
```

## Simplified Modules

Use common_simplified/ modules instead of common/ for new implementations:
- model_loader.py instead of ModelManager
- activation_hooks.py instead of activation_extraction
- Direct numpy operations instead of ActivationData wrapper

## Key Project Constraints
- **Scale**: 974 MBPP problems, 2000 tokens max, 2B parameter model
- **Resources**: 100GB CPU RAM max, 30GB GPU memory per GPU, up to 3 GPUs
- **Checkpointing**: Save every 50 records, autosave every 100 with 3 versions retained
- **Activations**: Extract during Phase 1 generation, save to disk for Phase 2 analysis

## Memories
- Always use `python3` (not python)
- Use `python3 run.py test-gpu` to test GPU setup
- Use `python3 multi_gpu_launcher.py --phase 1 --start 0 --end 488` for multi-GPU generation
- Multi-GPU uses index-based work splitting, no batching
- Phase 0.1 is CPU-only, splits problems based on difficulty from Phase 0
- Phase 2.2 is CPU-only, caches pile activations for general language feature baseline
- Phase 2.5 is CPU-only, uses saved activations from Phase 1 with optional pile filtering
- Phase 3.5 MUST run after Phase 2.5 because it needs the best_layer output
- Phase 3.5 extracts activations from only ONE layer (identified by Phase 2.5), not all layers like Phase 1
- Phase 3.6 processes hyperparameter tuning split, requires Phase 2.5 for best_layer
- Phase 3.8 evaluates AUROC/F1 metrics, requires Phase 0.1 (splits) and Phase 3.5 or Phase 3.6 data
- Phase 3.10 analyzes AUROC across temperatures, requires Phase 3.8 and Phase 3.5
- Phase 3.12 analyzes AUROC across difficulty levels, requires Phase 3.8 and Phase 0.1
- Phase 4.5 selects steering coefficients, requires Phase 2.5 (features) and Phase 3.6 (baseline)
- Phase 4.8 analyzes steering effects, requires Phase 2.5 (features) and Phase 3.5 (validation data)
- Checkpoint recovery: Auto-discovers latest checkpoints by timestamp
- Memory management: Extract and save activations during Phase 1, load from disk in Phase 2
- Use the full python path directly  ~/miniconda3/envs/pva_sae/bin/python run.py phase 4.5

## Project-Specific Patterns

### Data Pipeline
- **Auto-discovery**: Each phase finds latest outputs from previous phases by timestamp
- **Checkpointing**: Save every 50 records, autosave every 100, keep 3 versions
- **Recovery**: Automatically resume from latest checkpoint on restart

### Multi-GPU Workflow
- **Distribution**: Index-based work splitting via multi_gpu_launcher.py (not Ray)
- **Work allocation**: Each GPU gets contiguous dataset slice (e.g., 0-162, 163-325, 326-488 for 3 GPUs)
- **Process coordination**: Uses subprocess, not distributed computing framework
- **Memory management**: Extract and save activations during Phase 1, load from disk in Phase 2.5


## Key Commands

### Phase-Based Workflow
```bash
# Phase 0: Generate difficulty mapping (outputs to data/phase0/)
python3 run.py phase 0

# Phase 1.0: Build dataset (single GPU) - uses SAE split (489 problems)
python3 run.py phase 1

# Phase 1.0: Build dataset (multi-GPU) - uses SAE split (489 problems)
python3 multi_gpu_launcher.py --phase 1 --start 0 --end 488 --model google/gemma-2-2b

# Phase 0.1: Split problems by difficulty
python3 run.py phase 0.1

# Phase 2.2: Cache pile activations for filtering (single GPU, testing)
python3 run.py phase 2.2 --run-count 100

# Phase 2.2: Cache pile activations (production, 10k samples)
python3 run.py phase 2.2

# Phase 2.5: SAE analysis with pile filtering (default enabled)
python3 run.py phase 2.5

# Phase 2.5: SAE analysis without pile filtering
python3 run.py phase 2.5 --no-pile-filter

# Phase 3: Validation (auto-discovers from phase1 & phase2.5, outputs to data/phase3/)
python3 run.py phase 3

# Phase 3.5: Temperature robustness for validation split (single GPU)
# Uses best layer from Phase 2.5 (hardcoded in config as temperature_test_layer)
python3 run.py phase 3.5

# Phase 3.5: Temperature robustness (single GPU, specific range)
python3 run.py phase 3.5 --start 0 --end 50

# Phase 3.5: Temperature robustness (multi-GPU, all validation data)
python3 multi_gpu_launcher.py --phase 3.5 --model google/gemma-2-2b

# Phase 3.5: Temperature robustness (multi-GPU, specific range)
python3 multi_gpu_launcher.py --phase 3.5 --start 0 --end 100 --model google/gemma-2-2b

# Phase 3.6: Hyperparameter tuning set processing
python3 run.py phase 3.6

# Phase 3.8: AUROC and F1 evaluation (auto-discovers from phase 0.1 & phase 3.5)
python3 run.py phase 3.8

# Phase 3.10: Temperature-based AUROC analysis
python3 run.py phase 3.10

# Phase 3.12: Difficulty-based AUROC analysis
python3 run.py phase 3.12

# Phase 4.5: Steering coefficient selection
python3 run.py phase 4.5

# Phase 4.8: Steering effect analysis
python3 run.py phase 4.8
```

### Data Cleanup
```bash
# Clean all project data with confirmation prompts
python3 clean_data.py

# Force cleanup without prompts (for scripts)
python3 clean_data.py --force

# Preview what would be deleted
python3 clean_data.py --dry-run
```

### Manual Override
```bash
# Override auto-discovery with specific files
python3 run.py phase 1 --input data/phase0/specific_mapping.parquet
python3 run.py phase 2.5 --input data/phase1_0/specific_dataset.parquet
python3 run.py phase 3 --input data/phase2_5/specific_results.json
```

## Code Quality Guidelines

#### Core Principles
- **KISS (Keep It Simple)**: Choose the simplest solution that meets requirements
- **YAGNI (You Ain't Gonna Need It)**: Don't add functionality until actually needed
- **No Backward Compatibility**: Prioritize clean code over maintaining old interfaces
- Delete obsolete code
- Fail fast and early. Avoid Fallbacks.
- **DRY (Don't Repeat Yourself)**: Extract repeated code into reusable functions
- Avoid over-engineering for hypothetical futures

### Problem-Solving Approach
- **Root cause analysis**: Avoid bandaid fixes and really fix the root of the problem
- **Systematic debugging**: Use proper debugging techniques rather than quick patches

### Documentation
- Comment the "why," not the "what"
- Use docstrings for functions and classes
- Keep comments current with code changes

### Naming & Structure
- **Variables**: `user_count`, `total_price` (snake_case)
- **Functions**: `get_user_data()`, `validate_email()` (descriptive names)
- **Constants**: `MAX_RETRIES`, `DEFAULT_TIMEOUT` (UPPER_CASE)
- **Classes**: `UserManager`, `DatabaseConnection` (PascalCase)

### Error Handling & Logging
- Use try-catch blocks with specific exceptions
- Log errors with context for debugging
- Fail fast with meaningful error messages

### Python Best Practices
- **List comprehensions**: `[x*2 for x in nums if x > 0]`
- **Context managers**: `with open('file.txt') as f:`
- **Type hints**: `def process_data(items: List[str]) -> Dict[str, int]:`

### Logging Standards
- **Levels**: DEBUG (diagnostics) → INFO (events) → WARNING (recoverable) → ERROR (failures) → CRITICAL (system issues)
- **Structure with context**: `logger.error(f"Failed to process {item_id}: {str(e)}")`


