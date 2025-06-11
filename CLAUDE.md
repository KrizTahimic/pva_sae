# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PVA-SAE (Python Value Attribution Sparse Autoencoder) is a thesis research project investigating program validity awareness in language models. The project analyzes how language models internally represent code correctness by:

1. Using Google's Gemma 2 (2B parameters) to generate Python solutions for MBPP problems
2. Classifying solutions as correct (pass@1) or incorrect based on test execution
3. Applying Sparse Autoencoders (SAEs) from GemmaScope to identify latent directions
4. Validating findings through statistical analysis (AUROC, F1) and model steering experiments

The methodology follows four phases:
- Phase 0: Difficulty analysis of MBPP problems
- Phase 1: Dataset building (50% SAE analysis, 10% hyperparameter tuning, 40% validation)
- Phase 2: SAE activation analysis using separation scores
- Phase 3: Validation through both statistical measures and causal intervention via model steering

Each phase outputs to its own directory (data/phase0/, data/phase1_0/, data/phase1_1/, etc.) and automatically discovers inputs from previous phases.

### Data Output Structure
```
data/
├── phase0/           # Difficulty mappings
├── phase1_0/         # Generated code + activations (dataset building)
├── phase1_1/         # Data splitting
│   ├── dataset_*.parquet
│   ├── dataset_*_metadata.json
│   └── activations/
│       ├── correct/
│       │   └── {task_id}_layer_{n}.npz
│       └── incorrect/
│           └── {task_id}_layer_{n}.npz
├── phase2/           # SAE analysis results
└── phase3/           # Validation results
```

## Key Project Constraints
- **Scale**: 974 MBPP problems, 2000 tokens max, 2B parameter model
- **Resources**: 100GB CPU RAM max, 30GB GPU memory per GPU, up to 3 GPUs
- **Checkpointing**: Save every 50 records, autosave every 100 with 3 versions retained
- **Activations**: Extract during Phase 1 generation, save to disk for Phase 2 analysis

## Library Documentation Resources

When working with SAELens or TransformerLens, access their official documentation:

### SAELens
- **GitHub**: https://github.com/jbloomAus/SAELens
- **Documentation**: https://jbloomaus.github.io/SAELens/
- **Purpose**: Training and analyzing sparse autoencoders on language models
- **Key Features**: Pre-trained SAE downloads, custom SAE training, SAE-Vis integration

### TransformerLens  
- **GitHub**: https://github.com/TransformerLensOrg/TransformerLens
- **Documentation**: https://transformerlensorg.github.io/TransformerLens/
- **Purpose**: Mechanistic interpretability of GPT-style language models
- **Key Features**: 50+ model support, activation caching, intervention capabilities

## Memories
- Always use `python3` (not python)
- Use `python3 run.py test-gpu` to test GPU setup
- Use `python3 multi_gpu_launcher.py --phase 1 --num-gpus 3` for multi-GPU generation
- Multi-GPU uses index-based work splitting, not batching
- Phase 2 is CPU-only, uses saved activations from Phase 1
- Checkpoint recovery: Auto-discovers latest checkpoints by timestamp
- Memory management: Extract and save activations during Phase 1, load from disk in Phase 2

## Project-Specific Patterns

### Data Pipeline
- **Auto-discovery**: Each phase finds latest outputs from previous phases by timestamp
- **Checkpointing**: Save every 50 records, autosave every 100, keep 3 versions
- **Recovery**: Automatically resume from latest checkpoint on restart

### Multi-GPU Workflow
- **Distribution**: Index-based work splitting via multi_gpu_launcher.py (not Ray)
- **Work allocation**: Each GPU gets contiguous dataset slice (e.g., 0-324, 325-649, 650-973)
- **Process coordination**: Uses subprocess, not distributed computing framework
- **Memory management**: Extract and save activations during Phase 1, load from disk in Phase 2

### Steering Experiments
- **Coefficient range**: Test multiple values from -1.0 to 1.0
- **Layer analysis**: Configurable per experiment (examples use layer 20)
- **Components**: Focus on residual stream (resid_pre, resid_mid, resid_post)

## Key Commands

### Environment Setup
```bash
# Activate conda environment
conda activate pva_sae

# Install dependencies (if not already installed)
pip install -r requirements.txt

# For CUDA support, also install:
pip install accelerate
```

### Phase-Based Workflow
```bash
# Phase 0: Generate difficulty mapping (outputs to data/phase0/)
python3 run.py phase 0

# Phase 1: Build dataset (single GPU)
python3 run.py phase 1 --model google/gemma-2-2b

# Phase 1: Build dataset (multi-GPU)
python3 multi_gpu_launcher.py --phase 1 --num-gpus 3 --model google/gemma-2-2b

# Phase 2: SAE analysis (auto-discovers from phase1, outputs to data/phase2/)
python3 run.py phase 2

# Phase 3: Validation (auto-discovers from phase1 & phase2, outputs to data/phase3/)
python3 run.py phase 3
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

### Auto-Discovery and Manual Override
```bash
# Override auto-discovery with specific files
python3 run.py phase 1 --difficulty-mapping data/phase0/specific_mapping.parquet
python3 run.py phase 2 --dataset data/phase1/specific_dataset.parquet

# Disable auto-discovery
python3 run.py phase 1 --no-auto-discover
```

## Code Quality Guidelines

### 1. **Naming & Structure**
- **Variables**: `user_count`, `total_price` (snake_case)
- **Functions**: `get_user_data()`, `validate_email()` (descriptive names)
- **Constants**: `MAX_RETRIES`, `DEFAULT_TIMEOUT` (UPPER_CASE)
- **Classes**: `UserManager`, `DatabaseConnection` (PascalCase)
- **Keep functions small**: Ideally under 20-30 lines

### 2. **Error Handling & Logging**
- Use try-catch blocks with specific exceptions
- Log errors with context for debugging
- Fail fast with meaningful error messages

```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    return default_value
```

### 3. **Python Best Practices**
- **List comprehensions**: `[x*2 for x in nums if x > 0]`
- **Context managers**: `with open('file.txt') as f:`
- **Type hints**: `def process_data(items: List[str]) -> Dict[str, int]:`
- **DRY principle**: Extract repeated code into functions

### 4. **Logging Standards**
- **Levels**: DEBUG (diagnostics) → INFO (events) → WARNING (recoverable) → ERROR (failures) → CRITICAL (system issues)
- **Structure with context**: `logger.error(f"Failed to process {item_id}: {str(e)}")`

### 5. **Documentation**
- Comment the "why," not the "what"
- Use docstrings for functions and classes
- Keep comments current with code changes