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
- Phase 1.2: Temperature variation generation for validation split robustness
- Phase 2: SAE activation analysis using separation scores (split-aware)
- Phase 3: Validation through both statistical measures and causal intervention via model steering

Each phase outputs to its own directory (data/phase0/, data/phase0_1/, data/phase1_0/, data/phase1_2/, etc.) and automatically discovers inputs from previous phases.

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
├── phase1_2/         # Temperature variations (validation split only)
│   ├── dataset_temp_0_3.parquet  # 5 samples per task at temp 0.3
│   ├── dataset_temp_0_6.parquet  # 5 samples per task at temp 0.6
│   ├── dataset_temp_0_9.parquet  # 5 samples per task at temp 0.9
│   ├── dataset_temp_1_2.parquet  # 5 samples per task at temp 1.2
│   ├── metadata.json
│   └── activations/
│       ├── temp_0_3/
│       │   ├── correct/
│       │   │   └── {task_id}_sample{0-4}_layer_{n}.npz
│       │   └── incorrect/
│       ├── temp_0_6/
│       └── temp_0_9/
│           └── temp_1_2/
├── phase2/           # SAE analysis results (split-aware)
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

## Framework Usage Patterns

### Dual HuggingFace/TransformerLens Approach
This project uses both frameworks for different purposes:

1. **HuggingFace (Primary)** - General model loading and activation extraction
   - Models loaded via `AutoModelForCausalLM.from_pretrained()`
   - Custom activation extraction using PyTorch hooks
   - Broad model support and compatibility
   - Used for dataset generation and primary inference

2. **TransformerLens (Secondary)** - Mechanistic interpretability experiments
   - Models loaded via `HookedTransformer.from_pretrained_no_processing()`
   - Built-in `ActivationCache` for storing activations
   - Specialized interpretability features (patching, steering)
   - Used primarily for analysis and intervention experiments

### Activation Extraction Patterns
- **HuggingFace path**: Custom forward/pre-forward hooks attached to model modules
- **TransformerLens path**: Built-in activation caching functionality
- **Storage**: Activations cached to memory-mapped files for efficiency
- **SAE integration**: Compatible with both frameworks

## Memories
- Always use `python3` (not python)
- Use `python3 run.py test-gpu` to test GPU setup
- Use `python3 multi_gpu_launcher.py --phase 1 --start 0 --end 973` for multi-GPU generation
- Multi-GPU uses index-based work splitting, not batching
- Phase 0.1 is CPU-only, splits problems based on difficulty from Phase 0
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

# Phase 1.0: Build dataset (single GPU)
python3 run.py phase 1 --model google/gemma-2-2b

# Phase 1.0: Build dataset (multi-GPU)
python3 multi_gpu_launcher.py --phase 1 --start 0 --end 973 --model google/gemma-2-2b

# Phase 0.1: Split problems by difficulty
python3 run.py phase 0.1

# Phase 1.2: Generate temperature variations for validation split (single GPU)
python3 run.py phase 1.2 --model google/gemma-2-2b

# Phase 1.2: Generate temperature variations (multi-GPU)
python3 multi_gpu_launcher.py --phase 1.2 --model google/gemma-2-2b

# Phase 2: SAE analysis with split selection
python3 run.py phase 2 --split sae        # Analyze SAE split (default)
python3 run.py phase 2 --split validation # Analyze validation split (with temperature aggregation)

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
python3 run.py phase 1 --input data/phase0/specific_mapping.parquet
python3 run.py phase 2 --input data/phase1_0/specific_dataset.parquet
python3 run.py phase 3 --input data/phase2/specific_results.json
```

## Code Quality Guidelines

#### Core Principles
- **KISS (Keep It Simple)**: Choose the simplest solution that meets requirements
- **YAGNI (You Ain't Gonna Need It)**: Don't add functionality until actually needed
- **No Backward Compatibility**: Prioritize clean code over maintaining old interfaces
- Fail fast and early. Avoid Fallbacks.
- **Minimize Scope**: Declare variables in smallest scope possible, avoid global state
- **DRY (Don't Repeat Yourself)**: Extract repeated code into reusable functions

#### Implementation Guidelines- Prefer readability over cleverness
- Avoid over-engineering for hypothetical futures
- **Single Responsibility**: One clear purpose per function/class

### Problem-Solving Approach
- **Root cause analysis**: Avoid bandaid fixes and really fix the root of the problem
- **Systematic debugging**: Use proper debugging techniques rather than quick patches
- **No Backward Compatibility**: Prioritize clean code over maintaining old interfaces

### Naming & Structure
- **Variables**: `user_count`, `total_price` (snake_case)
- **Functions**: `get_user_data()`, `validate_email()` (descriptive names)
- **Constants**: `MAX_RETRIES`, `DEFAULT_TIMEOUT` (UPPER_CASE)
- **Classes**: `UserManager`, `DatabaseConnection` (PascalCase)
### Error Handling & Logging
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

### Python Best Practices
- **List comprehensions**: `[x*2 for x in nums if x > 0]`
- **Context managers**: `with open('file.txt') as f:`
- **Type hints**: `def process_data(items: List[str]) -> Dict[str, int]:`


### Logging Standards
- **Levels**: DEBUG (diagnostics) → INFO (events) → WARNING (recoverable) → ERROR (failures) → CRITICAL (system issues)
- **Structure with context**: `logger.error(f"Failed to process {item_id}: {str(e)}")`

### Documentation
- Comment the "why," not the "what"
- Use docstrings for functions and classes
- Keep comments current with code changes

