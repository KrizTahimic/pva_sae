# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 CRITICAL: Always Ask Permission Before Editing Code

**NEVER edit, write, or modify any code files without explicit user permission, especially during planning phases.**

When in planning mode:
- ✅ DO: Read files, search code, analyze architecture, create documentation/notes
- ✅ DO: Propose changes, explain what needs to be done, create step-by-step plans
- ❌ DON'T: Edit code files, create new code files, run tests, make commits
- ❌ DON'T: Assume you should implement just because planning is complete

**Always ask the user explicitly**: "Should I proceed with implementation?" or "Ready to start coding?"

---

## ⚠️ CRITICAL: Environment Setup

**ALWAYS activate the conda environment before running ANY commands:**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pva_sae
```

Without this, all Python commands will fail with `ModuleNotFoundError`. All commands in this document assume you're in the activated `pva_sae` environment.

### Example Working Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pva_sae && python3 run.py phase 8.3 --start 0 --end 4
```

## ⚠️ CRITICAL: Running Long Processes (Screen)

**This project runs on a remote GCP instance via SSH. If you disconnect, processes will stop unless you use `screen`.**

**🚨 IMPORTANT FOR CLAUDE CODE: Do NOT execute screen commands directly. Only provide instructions/recommendations for the user to run screen sessions manually in their terminal. Running screen via Claude Code makes it difficult for users to monitor and follow long-running tasks.**

### Essential Screen Commands

```bash
# Start a new screen session (do this BEFORE running any phase)
screen -S pva_phase

# Now activate conda and run your phase
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pva_sae
python3 run.py phase 8.3 --start 0 --end 4

# Detach from screen (keeps process running, safe to close laptop)
# Press: Ctrl+A, then press D

# List all screen sessions
screen -ls

# Reattach to your session later
screen -r pva_phase8

# If only one session exists, just use:
screen -r

# Kill a screen session (from outside screen)
screen -X -S pva_phase quit

# Kill current session (from inside screen)
exit
```

### Best Practices

1. **Always use screen for long-running phases** - Most phases take 30 minutes to several hours
2. **Name your sessions** - Use descriptive names: `screen -S phase1_generation`
3. **One phase per session** - Don't run multiple phases in the same screen session
4. **Check before starting** - Use `screen -ls` to see if you already have a session running
5. **Checkpoint awareness** - Phases auto-checkpoint every 50 records, so you can safely kill and restart if needed
6. **User executes screen manually** - Claude Code should ONLY provide screen instructions, never execute screen commands directly

### Common Workflow

```bash
# Start your work session
screen -S phase8_selective_steering
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pva_sae
python3 run.py phase 8.3
# Ctrl+A, D to detach
# Close laptop, go home

# Later, check progress
screen -r phase8_selective_steering
# View output, check if complete
# Ctrl+A, D to detach again if still running
```

## Project Overview

PVA-SAE (Python Value Attribution using Sparse Autoencoders) is a research project investigating how language models internally represent program correctness. The project uses Google's Gemma 2 model (2B parameters), GemmaScope SAEs, and the MBPP dataset to:

1. Generate code solutions and classify them as correct/incorrect
2. Identify latent directions (SAE features) that encode correctness
3. Validate findings through statistical analysis (AUROC, F1 scores)
4. Perform causal interventions via model steering

## Architecture Decisions: Why This Codebase Is Structured This Way

### Inspiration vs. Implementation

This project is inspired by [sae_entities](../sae_entities/) (Ferrando et al., 2024 - "Do I Know This Entity?"), which uses ~11K lines of code in a notebook-style codebase. Our implementation is ~30K lines across 94 files. This is intentional, not over-engineering.

**Reference:** `../sae_entities/` - Compare their approach for context.

### Resource-Constrained Design

The key difference: **they have abundant compute, we don't.**

| Constraint | sae_entities (Research Lab) | pva_sae (Thesis Project) |
|------------|----------------------------|--------------------------|
| If run fails | Re-run, no big deal | Lost hours of GPU time |
| Activation computation | Recompute if needed | Must cache - too expensive |
| Checkpointing | Nice to have | **Essential** - SSH disconnects, timeouts |
| Code reuse | Generate fresh each time | Must reuse - generation takes hours |

### Why 30+ Phases?

Each phase is a **checkpoint boundary**. If Phase 4.5 fails after 3 hours, you don't lose Phases 1-4. Benefits:

1. **Granular reruns** - Fix a bug in Phase 4.8, rerun only that phase
2. **Checkpointing every 50 records** - Resume interrupted runs automatically
3. **Activation caching** (Phase 2.2) - Compute Pile activations once, reuse forever
4. **Auto-discovery** - Phases find outputs from previous phases automatically

### Infrastructure That Exists Because of Constraints

| Infrastructure | Why It Exists |
|---------------|---------------|
| `--start N --end M` flags | Test on subset before committing to 6-hour run |
| Checkpoint files | Resume after SSH disconnect or timeout |
| `common/utils.py` auto-discovery | Don't manually track paths across 30+ phases |
| Config dataclass | Switch model/dataset without editing 30 files |
| Phase-specific output dirs | Don't overwrite Gemma results when running LLAMA |

### What This Means for Development

- **Don't consolidate phases** unless you have unlimited compute to re-run everything
- **Keep checkpointing** - it saves hours of GPU time
- **Preserve activation caching** - Phase 2.2 outputs are expensive to regenerate
- **When adding features** (LLAMA, HumanEval, CoT), add new phases rather than modifying existing ones

---

## Core Architecture

### PCDGE Pattern

The fundamental pattern used throughout all phases:

- **Prompt**: Build prompt from MBPP problem (`common/prompt_utils.py`)
- **Capture**: Extract activations via PyTorch hooks during generation
- **Decompose**: Apply SAE decomposition using GemmaScope's JumpReLU SAE
- **Generate**: LLM generates Python code solution
- **Evaluate**: Execute tests to classify as correct (pass@1) or incorrect

This pattern is implemented across phases but with different goals (baseline generation, temperature robustness, steering analysis, etc.).

### Phase-Based Execution

The project is organized into sequential phases, each with:
- Dedicated directory: `phase{N}_{description}/`
- Output directory: `data/phase{N}/`
- Auto-discovery: Later phases automatically find outputs from earlier phases
- Main entry: `python3 run.py phase {N}`

Phase categories:
- **Phases 0-0.1**: Data preparation (difficulty analysis, problem splitting)
- **Phases 1-2.10**: Feature discovery (dataset generation, SAE analysis, feature selection)
- **Phases 3.5-3.12**: Statistical validation (temperature robustness, AUROC/F1 evaluation)
- **Phases 4.5-4.14**: Causal validation (steering coefficient selection, effect analysis, significance testing)
- **Phases 5.3-5.9**: Weight orthogonalization (permanent model modifications)
- **Phase 6.3**: Attention pattern analysis
- **Phases 7.3-7.12**: Model comparison (instruction-tuned vs base model)
- **Phase 8.3**: Selective steering

## Command Usage

### Basic Structure

```bash
python3 run.py phase {PHASE_NUMBER} [OPTIONS]
```

### Common Commands

```bash
# Data preparation
python3 run.py phase 0           # Difficulty analysis (974 MBPP problems)
python3 run.py phase 0.1         # Split into SAE/hyperparams/validation sets

# Feature discovery
python3 run.py phase 1           # Generate dataset with activations (single GPU)
python3 run.py phase 2.2         # Cache Pile activations (baseline)
python3 run.py phase 2.5         # SAE analysis with pile filtering

# Statistical validation
python3 run.py phase 3.5         # Temperature robustness testing
python3 run.py phase 3.8         # AUROC and F1 evaluation

# Causal validation
python3 run.py phase 4.5         # Steering coefficient selection
python3 run.py phase 4.8         # Steering effect analysis
python3 run.py phase 4.14        # Statistical significance testing

# Model comparison
python3 run.py phase 7.3         # Instruction-tuned baseline
python3 run.py phase 7.12        # Instruction-tuned evaluation
```

### Important Options

```bash
# Dataset range (for testing or processing subsets)
--start N --end M                # Process indices N to M

# Input override (skip auto-discovery)
--input PATH                     # Use specific input file

# Visualization regeneration (skip computation)
--viz-only                       # Regenerate plots from saved data (seconds vs hours)

# Experiment modes (phases 4.5, 4.6, 4.8)
--correction-only                # Only correction experiments
--corruption-only                # Only corruption experiments
--preservation-only              # Only preservation (phase 4.8)

# Model selection
--model google/gemma-2-2b        # Base model (default)
--model google/gemma-2-2b-it     # Instruction-tuned model
```

### Visualization Regeneration (--viz-only)

Many phases produce visualizations (plots, charts). To iterate on visualizations without rerunning expensive computations:

```bash
# Full run (hours) - generates data + visualizations
python3 run.py phase 4.8

# Regenerate visualizations only (seconds) - uses saved JSON
python3 run.py phase 4.8 --viz-only
```

Supported phases: 2.15, 3.8, 3.10, 3.11, 3.12, 4.7, 4.8, 4.14, 4.16, 5.3, 5.6, 5.9, 6.3, 7.6, 7.9, 7.12

**Requirement**: The phase must have been run normally at least once to generate the data JSON file.

## Configuration System

### Centralized Config

All configuration is in `common/config.py` using a dataclass with namespaced fields:
- `model_*`: Model settings
- `dataset_*`: Dataset settings
- `activation_*`: Activation extraction
- `sae_*`: SAE analysis settings
- `phase{N}_*`: Phase-specific output directories

### Config Precedence

CLI args > environment variables > config file > defaults

### Key Settings

```python
# Model
DEFAULT_MODEL_NAME = "google/gemma-2-2b"
MAX_NEW_TOKENS = 800

# Activations
activation_layers = list(range(1, 26, 1))  # All 25 layers
activation_position = -1  # Last token
activation_hook_type = "resid_post"

# SAE (GemmaScope)
sae_repo_id = "google/gemma-scope-2b-pt-res"
sae_width = "16k"
sae_latent_threshold = 0.02

# Pile filtering
pile_filter_enabled = True  # Filter out general language features
pile_threshold = 0.02
pile_samples = 10000
```

### Switching Datasets and Models (Pure Manual Config)

**IMPORTANT**: To switch between datasets (MBPP/HumanEval) or models (Gemma/LLAMA), simply edit `common/config.py` directly. No CLI arguments needed.

#### How to Switch Datasets

Edit the `dataset_name` field in `common/config.py`:

```python
# common/config.py
@dataclass
class Config:
    # === DATASET SETTINGS ===
    # Options: "mbpp" (Muennighoff/mbpp) or "humaneval"
    dataset_name: str = "humaneval"  # ← Change this to switch datasets
```

#### How to Switch Models

Edit the `model_name` field in `common/config.py`:

```python
# common/config.py
@dataclass
class Config:
    # === MODEL SETTINGS ===
    model_name: str = "google/gemma-2-2b"  # ← Change this to switch models
    # Options:
    #   - "google/gemma-2-2b" (default)
    #   - "meta-llama/Llama-3.1-8B" (LLAMA)
```

#### Example Workflows

**Experiment 1: Gemma + HumanEval** (6-8 phases)
```python
# Edit config.py once:
dataset_name: str = "humaneval"
model_name: str = "google/gemma-2-2b"
```

Then run phases without extra arguments:
```bash
python3 run.py phase 3.5
python3 run.py phase 3.8
python3 run.py phase 4.8
python3 run.py phase 5.3
python3 run.py phase 6.3
python3 run.py phase 8.3
```

**Experiment 2: LLAMA + MBPP** (10+ phases)
```python
# Edit config.py once:
dataset_name: str = "mbpp"
model_name: str = "meta-llama/Llama-3.1-8B"
```

Then run:
```bash
python3 run.py phase 1
python3 run.py phase 2.2
python3 run.py phase 2.5
# ... etc
```

**Why This Approach?**
- ✅ **Simpler commands** - No need to type `--dataset humaneval --model llama` for every phase
- ✅ **Less error-prone** - No risk of forgetting CLI flags mid-experiment
- ✅ **Clear current state** - Just check config.py to see what experiment is running
- ✅ **Sequential workflow** - Perfect for running one experiment at a time (6-10 phases)

## Key Files and Their Roles

### Entry Points

- `run.py`: Main CLI entry point for all phases
- `common/config.py`: Centralized configuration
- `common/utils.py`: Auto-discovery, memory utilities, device detection

### Shared Utilities

- `common/prompt_utils.py`: MBPP prompt building
- `common/gpu_utils.py`: GPU memory management, cleanup
- `common/logging.py`: Phase-aware logging
- `common/steering_metrics.py`: Correction/corruption rate calculation
- `common/weight_utils.py`: Weight orthogonalization utilities

### Critical Phase Implementations

- `phase1_simplified/runner.py`: PCDGE implementation for dataset generation
- `phase2_5_simplified/sae_analyzer.py`: SAE feature analysis with separation scores
- `phase2_10_t_statistic_latent_selector/t_statistic_selector.py`: Welch's t-test feature selection
- `phase3_8/auroc_f1_evaluator.py`: AUROC/F1 metric calculation
- `phase4_8_steering_analysis/steering_effect_analyzer.py`: Steering intervention analysis

## Data Directory Structure

```
data/
├── phase0/           # Difficulty mappings (cyclomatic complexity)
├── phase0_1/         # Split datasets (sae_mbpp.parquet, hyperparams_mbpp.parquet, validation_mbpp.parquet)
├── phase1_0/         # Generated code + activations
├── phase2_2/         # Pile activation baseline
├── phase2_5/         # SAE analysis results (top_20_features.json per layer)
├── phase3_5/         # Temperature robustness data
├── phase3_8/         # AUROC/F1 evaluation metrics
├── phase4_8/         # Steering effect analysis
└── phase7_12/        # Instruction-tuned model comparisons
```

## Important Architectural Details

### Activation Extraction

Activations are captured at the **last prompt token** (position -1) from the **residual stream** (`resid_post`) using PyTorch forward hooks before code generation begins. This is the point where the model has processed the entire problem specification but hasn't started generating the solution.

### SAE Feature Analysis

GemmaScope SAEs use JumpReLU activation (not standard ReLU). Features are evaluated using:
- **Separation Score**: `mean(correct_activations) - mean(incorrect_activations)`
- **T-Statistic**: Welch's t-test (unequal variance) for statistical significance
- **Pile Filtering**: Features activating >2% on general text (Pile-10k) are excluded

### Steering Mechanism

Model steering adds a direction to activations via forward hooks:
```python
steered_activation = original_activation + coefficient * sae_decoder_direction
```

Coefficient selection uses adaptive coarse-to-fine search (Phase 4.5) then golden section refinement (Phase 4.6).

### Statistical Controls

The project uses rigorous controls:
- **Random features** (Phase 4.10, 4.12): Features with zero separation score
- **Binomial tests** (Phases 4.14, 5.9): Validate steering effects vs. chance
- **Temperature variation** (Phases 3.5, 3.10): Robustness across [0.0, 0.3, 0.6, 0.9, 1.2]
- **Difficulty stratification** (Phase 3.12): Performance across complexity levels

## Common Development Workflows

### Running a Complete Pipeline

```bash
# 1. Prepare data
python3 run.py phase 0      # ~5 min
python3 run.py phase 0.1    # ~1 min

# 2. Generate baseline dataset (single GPU, or split across GPUs)
python3 run.py phase 1      # ~2-6 hours for 487 problems

# 3. Analyze features
python3 run.py phase 2.2    # Cache pile activations
python3 run.py phase 2.5    # SAE analysis

# 4. Validate
python3 run.py phase 3.5    # Temperature robustness
python3 run.py phase 3.8    # AUROC/F1

# 5. Causal intervention
python3 run.py phase 4.5    # Find coefficients
python3 run.py phase 4.8    # Test steering effects
```

### Resuming from Checkpoints

All generation phases (1, 3.5, 4.8, etc.) create checkpoints every 50 records. If interrupted, simply re-run the same command - it will auto-resume from the latest checkpoint.

### Testing on Small Samples

Use `--start` and `--end` to test on a subset:
```bash
python3 run.py phase 1 --start 0 --end 10  # Test first 10 problems
```

### Debugging Configuration

View the final configuration without running:
```bash
python3 run.py phase 3.8 --show-config
```

### GPU Management

```bash
# Test GPU detection
python3 run.py test-gpu

# Clean GPU memory
python3 run.py cleanup-gpu

# System status
python3 run.py status
```

## 🚨 CRITICAL: Model/Dataset-Aware Paths

**LEARNED LESSON: We almost overwrote Gemma+MBPP results by running LLAMA+MBPP!**

### Problem 1: Output Directories

Many phases use `config.get_phase_output_dir()` which does NOT add model/dataset suffixes. This causes different experiments to overwrite each other.

**Always use the utils function, not the config method:**
```python
# ❌ WRONG - outputs to data/phase2_2/ for ALL models
output_dir = Path(config.get_phase_output_dir("2.2"))

# ✅ CORRECT - outputs to data/phase2_2_llama/ for LLAMA
from common.utils import get_phase_output_dir
output_dir = Path(get_phase_output_dir("2.2", config))
```

### Problem 2: Input File Discovery

Many phases have hardcoded input paths that only find Gemma+MBPP results. They need to dynamically find inputs based on current model/dataset.

**Always use get_phase_output_dir for input discovery too:**
```python
# ❌ WRONG - always looks in Gemma directory
input_dir = Path(config.phase1_output_dir)

# ✅ CORRECT - looks in model/dataset-specific directory
input_dir = Path(get_phase_output_dir("1", config))
```

### Rules to Follow

1. **NEVER** use `config.get_phase_output_dir()` - it doesn't add suffixes
2. **NEVER** use hardcoded `config.phase{N}_output_dir` for input paths
3. **ALWAYS** use `get_phase_output_dir(phase, config)` from `common/utils.py`
4. **ALWAYS** check each phase for both output AND input path handling before running

---

## Important Notes

### Hardware Requirements

- GPU: 24GB+ VRAM (for 2B parameter model + SAE)
- CPU RAM: 100GB+ (for activation processing)
- Disk: ~50GB (model weights + datasets)
- Supports: CUDA (NVIDIA), MPS (Apple Silicon), CPU

### Auto-Discovery System

Later phases automatically find outputs from earlier phases by searching for the most recent timestamped file in the expected directory. This can be overridden with `--input` if needed.

### Reproducibility

- All random seeds are fixed (default: 42)
- Checkpointing enables resuming interrupted runs
- Problem splits are stratified by difficulty for balanced evaluation
- Temperature=0.0 for deterministic baseline generation

### GemmaScope Integration

GemmaScope SAEs are loaded from HuggingFace:
- Repo: `google/gemma-scope-2b-pt-res`
- Width: 16k features
- Each layer has different average sparsity (see `common/config.py::GEMMA_2B_SPARSITY`)
- Layers 0-25 available for Gemma-2B

## When Making Changes

### Adding a New Phase

1. Create `phase{N}_{name}/` directory
2. Add output directory to `common/config.py`
3. Implement runner class following existing patterns
4. Add phase handler to `run.py::main()`
5. Update auto-discovery logic if needed

### Modifying SAE Analysis

Key file: `phase2_5_simplified/sae_analyzer.py`
- Separation score calculation: Line ~200
- Feature filtering logic: Line ~150
- Top-k selection: Line ~250

### Changing Steering Behavior

Key files:
- Coefficient selection: `phase4_5_model_steering/steering_coefficient_selector.py`
- Hook implementation: Look for `create_steering_hook` functions
- Metrics: `common/steering_metrics.py`

### GPU Memory Issues

If encountering OOM errors:
1. Reduce batch sizes in config
2. Enable `activation_cleanup_after_batch = True`
3. Use `activation_clear_cache_between_layers = True`
4. Run phases sequentially instead of in parallel
5. Use `python3 run.py cleanup-gpu --aggressive`

---

## Code Style Conventions

Follow these conventions when writing or modifying code in this project.

### No Backward Compatibility

When refactoring, make clean breaks - don't add legacy fallbacks or backward compatibility code:

```python
# ❌ AVOID - Legacy fallback clutters code
def discover_outputs(phase):
    if manifest.exists():
        return parse_manifest(manifest)
    # Legacy fallback
    return legacy_discover(phase)

# ✅ GOOD - Clean break, clear error
def discover_outputs(phase):
    if not manifest.exists():
        raise FileNotFoundError(f"Run phase {phase} first.")
    return parse_manifest(manifest)
```

**Rationale:** This is research code with one user. Backward compatibility adds complexity without benefit. When refactoring, update all call sites rather than maintaining two code paths.

### Tensor Operations (einops)

Use `einops.rearrange` for complex reshapes - makes tensor shapes self-documenting:

```python
# ✅ GOOD - Shape transformation is explicit
from einops import rearrange
steering = rearrange(decoder_direction, 'd -> 1 1 d') * coefficient

# ❌ AVOID - Shape not obvious without tracing
steering = decoder_direction.unsqueeze(0).unsqueeze(0) * coefficient
```

**When NOT to use einops** (keep simple):
- Simple matmul: `x @ self.W_enc` - `@` operator is clearer
- Basic squeeze: `activation.squeeze(0)` - obvious enough
- Transpose for loading: `weights['encoder.weight'].T` - standard pattern

Add shape comments where einops isn't used:
```python
# Shape: [batch, seq_len, d_model]
residual = input[0]
```

### List Comprehensions (Pythonic Style)

Prefer comprehensions over verbose loops:

```python
# ✅ GOOD - List comprehension
features = [{'idx': i, 'score': scores[i].item()} for i in range(n)]

# ❌ AVOID - Verbose loop
features = []
for i in range(n):
    features.append({'idx': i, 'score': scores[i].item()})

# ✅ GOOD - Dict unpacking for adding keys
all_features = [{**feat, 'layer': layer_idx} for feat in features]

# ❌ AVOID - Copy and modify
for feat in features:
    new_feat = feat.copy()
    new_feat['layer'] = layer_idx
    all_features.append(new_feat)

# ✅ GOOD - Counter for counting
from collections import Counter
layer_counts = Counter(feat['layer'] for feat in features)

# ❌ AVOID - Manual dict tracking
layer_counts = {}
for feat in features:
    layer = feat['layer']
    layer_counts[layer] = layer_counts.get(layer, 0) + 1
```

### Type Hints (Python 3.9+)

Use modern type hint syntax:

```python
# ✅ GOOD - Built-in generics (Python 3.9+)
def process(items: list[dict[str, Any]], config: Config) -> tuple[list, int]:

# ❌ AVOID - Old typing imports
from typing import List, Dict, Tuple
def process(items: List[Dict[str, Any]], config: Config) -> Tuple[List, int]:
```

Keep `Optional`, `Union`, `Callable` from typing module (still needed).

### Variable Naming

```python
# ✅ GOOD - Descriptive names
lower_bound = bounds['lower']
upper_bound = bounds['upper']
temperature_indices = np.arange(len(temperatures))

# ❌ AVOID - Single letters outside comprehensions
a = bounds['lower']
b = bounds['upper']
x = np.arange(len(temperatures))
```

**Terminology for correctness states:**
- `test_passed` = original test result (from Phase 1)
- `baseline_passed` = generation without any intervention
- `steered_passed` = generation with steering hook
- `orthogonalized_passed` = generation with weight orthogonalization

### Function Structure

- Use **early returns** to reduce nesting
- Split functions >50 lines into smaller helpers
- Follow **single responsibility principle**

```python
# ✅ GOOD - Early returns, flat structure
def calculate_rate(results):
    if isinstance(results, pd.DataFrame):
        return _rate_from_dataframe(results)
    if isinstance(results, list):
        return _rate_from_list(results)
    raise TypeError(f"Expected list or DataFrame, got {type(results)}")

# ❌ AVOID - Deep nesting
def calculate_rate(results):
    if isinstance(results, pd.DataFrame):
        if not results.empty:
            if 'column' in results.columns:
                # ... deeply nested logic
```
