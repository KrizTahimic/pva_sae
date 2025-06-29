# PVA-SAE: Python Value Attribution using Sparse Autoencoders

This repository contains the implementation for a thesis project investigating program validity awareness in language models using Sparse Autoencoders (SAEs).

## Overview

This research analyzes how language models internally represent the concept of code correctness. By using Google's Gemma 2 model (2B parameters) and the MBPP (Mostly Basic Programming Problems) dataset, we:

1. Generate Python code solutions using a base language model
2. Classify solutions as correct (pass@1) or incorrect
3. Analyze model representations using Sparse Autoencoders to identify latent directions
4. Validate findings through statistical analysis and model steering

## Phase Execution Order

The project follows a specific phase execution order where each phase depends on outputs from previous phases:

### Phase 0: Difficulty Analysis
- Analyzes complexity of all 974 MBPP problems using cyclomatic complexity
- Creates difficulty mapping for consistent data splits across experiments
- Enables reproducible interleaved sampling based on problem difficulty

### Phase 0.1: Problem Splitting
- Splits MBPP problems into three sets based on difficulty scores from Phase 0
- Stratified randomized interleaving ensures equal difficulty distribution
- Creates: 50% SAE analysis, 10% hyperparameter tuning, 40% validation splits
- Outputs complete parquet files for each split with all MBPP data and complexity scores

### Phase 1: Dataset Building
- Uses the SAE split (50% of MBPP) from Phase 0.1 for dataset generation
- Standardized prompt template: problem description + test cases + code initiator
- Classification: correct (passes all 3 tests) vs incorrect
- Generates solutions for 489 problems in the SAE split only
- **Extracts activations from ALL layers [0, 6, 8, 15, 17] during generation**

### Phase 2: SAE Analysis
- Utilizes pre-trained SAEs from GemmaScope with JumpReLU architecture
- Analyzes residual stream at final token position using saved activations from Phase 1
- Computes separation scores to identify distinguishing latent dimensions
- **Identifies the best PVA layer** (e.g., layer 8) for Phase 3.5

### Phase 3: Validation
- **Statistical Analysis**: 
  - AUROC: Measures discrimination ability across all thresholds
  - F1 Score: Harmonic mean of precision and recall (optimized on hyperparameter set)
- **Model Steering**: 
  - Manipulates identified latent directions to test causal influence
  - Correction Rate: Proportion of incorrect→correct after steering
  - Corruption Rate: Proportion of correct→incorrect after steering

### Phase 3.5: Temperature Robustness Testing
- **MUST run after Phase 2** as it uses the best PVA layer identified
- Tests robustness on validation split at temperatures [0.3, 0.6, 0.9, 1.2]
- Generates 5 samples per temperature for each validation task
- **Extracts activations from ONLY the best layer** (not all layers)
- Activations come from prompt processing, not generation (temperature only affects sampling)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pva_sae.git
cd pva_sae

# Install dependencies
pip install -r requirements.txt

# For CUDA support
pip install accelerate
```

## Usage

### Quick Start

Run individual phases using the unified script with automatic data flow:

```bash
# Phase 0: Generate difficulty mapping (outputs to data/phase0/)
python3 run.py phase 0

# Phase 0.1: Split problems by difficulty (auto-discovers from phase0, outputs to data/phase0_1/)
python3 run.py phase 0.1

# Phase 1: Build dataset (uses SAE split from phase0.1, outputs to data/phase1_0/)
python3 run.py phase 1 --model google/gemma-2-2b

# Phase 2: Analyze with SAEs (auto-discovers from phase1, outputs to data/phase2/)
python3 run.py phase 2

# Phase 3: Run validation (auto-discovers from phase1 & phase2, outputs to data/phase3/)
python3 run.py phase 3

# Phase 3.5: Temperature robustness testing (requires phase 2 to identify best layer)
# Tests validation split at multiple temperatures, extracting only from best PVA layer
python3 run.py phase 3.5 --model google/gemma-2-2b
```

The pipeline automatically discovers outputs from previous phases, creating a seamless workflow.

### Simplified Architecture

The project has been refactored to follow KISS (Keep It Simple, Stupid) and YAGNI (You Aren't Gonna Need It) principles:

- **Direct Model Loading**: Replaced complex ModelManager with simple HuggingFace transformers calls
- **Simple Generation**: Removed RobustGenerator's 267 lines of retry logic in favor of direct generation
- **Clean Activation Handling**: Eliminated ActivationData wrapper class for direct numpy arrays
- **60% Less Code**: Refactored from ~1600 lines to ~650 lines while maintaining all functionality

### Phase-Specific Examples
```bash
# Quick test with small dataset (first 10 problems from SAE split)
python3 run.py phase 1 --model google/gemma-2-2b --start 0 --end 10

# Run phase 0 without saving (dry run)
python3 run.py phase 0 --dry-run

# Override auto-discovery with specific files
python3 run.py phase 1 --input data/phase0/specific_mapping.parquet
python3 run.py phase 2 --input data/phase1/specific_dataset.parquet
```

## Project Structure

```
pva_sae/
├── common/                         # Shared utilities and configurations
├── common_simplified/              # Simplified modules following KISS principle
├── phase0_difficulty_analysis/     # Phase 0: MBPP complexity preprocessing
├── phase0_1_problem_splitting/     # Phase 0.1: Problem splitting
├── phase1_simplified/              # Phase 1: Simplified dataset generation
├── phase2_simplified/              # Phase 2: Simplified SAE analysis
├── phase3_validation/              # Phase 3: Validation (not yet implemented)
├── phase3_5_temperature_robustness/# Phase 3.5: Temperature robustness testing
├── data/                           # Phase-based data directory
│   ├── phase0/                    # Difficulty mappings
│   ├── phase1_0/                  # Generated datasets
│   ├── phase0_1/                  # Split parquet files
│   ├── phase1_2/                  # Temperature variations
│   ├── phase2/                    # SAE analysis results
│   ├── phase3/                    # Validation results
│   └── logs/                      # Execution logs
├── run.py                         # Main entry point
└── requirements.txt               # Dependencies
```

## Checkpoint Recovery System

The system includes robust checkpoint recovery for long-running dataset builds:

### Automatic Recovery
- **Checkpoints**: Saved every 50 records (configurable) to `checkpoints/` directory
- **Auto-Resume**: Automatically detects and offers to resume from latest checkpoint
- **Emergency Saves**: Graceful shutdown on Ctrl+C or system signals

### Recovery After Crashes
```bash
# System automatically detects checkpoints on restart
python3 run.py phase 1 --model google/gemma-2-2b

# Prompts: "Found checkpoint with 150 processed records. Resume? (y/n)"
```

### Manual Recovery
```bash
# List available checkpoints
ls data/datasets/checkpoints/

# Note: Manual checkpoint resumption requires direct Python API usage
# as the simplified run.py focuses on clean phase execution
```

### What Gets Preserved
- **Progress**: Exact record indices processed
- **Results**: All generated code and test results
- **Statistics**: Success rates, timing data, model configuration
- **State**: Can resume from any interruption point

### Production Features
```bash
# Production build with full SAE split (489 problems)
python3 run.py phase 1 --model google/gemma-2-2b --start 0 --end 488 --cleanup

# Enable progress streaming and verbose output
python3 run.py phase 1 --model google/gemma-2-2b --stream --verbose
```

## Multi-GPU Usage

The project supports parallel processing across multiple GPUs for Phase 1 and Phase 1.2:

### Phase 1: Dataset Building
```bash
# Auto-detect all available GPUs (processes 489 SAE split problems)
python3 multi_gpu_launcher.py --phase 1 --start 0 --end 488 --model google/gemma-2-2b

# Use specific GPUs (e.g., GPUs 0, 1, and 2)
python3 multi_gpu_launcher.py --phase 1 --gpus 0,1,2 --start 0 --end 488 --model google/gemma-2-2b
```

### Phase 1.2: Temperature Variations
```bash
# Generate temperature variations in parallel (auto-distributes validation set)
python3 multi_gpu_launcher.py --phase 1.2 --model google/gemma-2-2b

# Use specific GPUs
python3 multi_gpu_launcher.py --phase 1.2 --gpus 0,1 --model google/gemma-2-2b
```

**Multi-GPU Features:**
- Automatic workload distribution (index-based splitting)
- Independent process per GPU with separate logging
- Graceful interruption handling (Ctrl+C stops all processes)
- Progress monitoring across all GPUs
- Automatic result merging for Phase 1

## Phase 1.2: Temperature Generation Testing

Phase 1.2 generates multiple samples at different temperatures for the validation split. For development and testing, you can use special arguments to process a subset:

### Test Mode Arguments
- `--samples N`: Process only the first N validation samples (default: all)
- `--test-temps T1 T2 ...`: Override temperature values (default: 0.3 0.6 0.9 1.2)
- `--test-samples-per-temp N`: Override samples per temperature (default: 5)

### Example Commands
```bash
# Minimal test: 3 samples, 2 temps, 2 samples each = 12 total generations
python3 run.py phase 1.2 --samples 3 --test-temps 0.3 0.6 --test-samples-per-temp 2

# Medium test: 10 samples with default settings = 200 total generations
python3 run.py phase 1.2 --samples 10

# Test with specific model
python3 run.py phase 1.2 --samples 5 --model google/gemma-2-2b

# Production run: all validation samples at all temperatures
python3 run.py phase 1.2 --model google/gemma-2-2b
```

### Test Output Structure
Test mode outputs to `data/test_phase1_2/` with the same structure as production:
```
data/test_phase1_2/
├── output/
│   ├── dataset_temp_0_3.parquet
│   ├── dataset_temp_0_6.parquet
│   ├── metadata.json
│   └── activations/
│       ├── temp_0_3/
│       │   ├── correct/
│       │   └── incorrect/
│       └── temp_0_6/
```

This allows you to:
- Quickly verify the pipeline works before full runs
- Test specific temperature combinations
- Debug activation extraction without processing hundreds of samples
- Keep test data separate from production data

## Output Files and Naming Convention

The project uses a descriptive naming convention to make files easy to identify and organize:

### File Naming Format
All output files follow this pattern:
```
<prefix>_<model>_<range>_<suffix>_<timestamp>.<extension>
```

**Components:**
- **prefix**: File type (dataset, results, checkpoint, autosave, etc.)
- **model**: Sanitized model name (e.g., gemma-2-2b, gemma-2-2b)
- **range**: Index range processed (e.g., 0-973, 325-649)
- **suffix**: Additional descriptors (merged, final, progress150, etc.)
- **timestamp**: Human-readable format (YYYY-MM-DD_HH-MM-SS)

### Dataset Files (`data/datasets/`)
```bash
# Main datasets (Parquet format)
dataset_gemma-2-2b_0-973_2024-01-06_14-30-45.parquet

# Merged datasets from parallel processing
dataset_gemma-2-2b_0-973_merged_2024-01-06_15-45-30.parquet

# Results in JSON format
results_gemma-2-2b_0-973_2024-01-06_14-30-45.json

# Metadata files (accompany each dataset)
dataset_gemma-2-2b_0-973_2024-01-06_14-30-45_metadata.json
```

### Checkpoint Files (`data/datasets/checkpoints/`)
```bash
# Regular checkpoints (every 50 records)
checkpoint_gemma-2-2b_0-149_2024-01-06_14-35-20.json

# Final checkpoints
final_gemma-2-2b_0-973_2024-01-06_15-30-45.json
```

### Autosave Files (`data/datasets/`)
```bash
# Periodic autosaves during processing
autosave_gemma-2-2b_0-973_progress150_2024-01-06_14-35-45.parquet
autosave_gemma-2-2b_0-973_progress300_2024-01-06_14-50-15.parquet
```

### Log Files (`data/logs/`)
```bash
# Main application logs
pva_sae_phase1_2024-01-06_14-30-45.log
pva_sae_multi_gpu_launcher_2024-01-06_14-30-45.log

# Multi-GPU processing logs (one per GPU)
gpu_0_gemma-2-2b_0-324_2024-01-06_14-30-45.log
gpu_1_gemma-2-2b_325-649_2024-01-06_14-30-45.log
gpu_2_gemma-2-2b_650-973_2024-01-06_14-30-45.log
```

### Reading and Organizing Files

**Finding related files:**
```bash
# List all datasets for a specific model
ls data/datasets/dataset_gemma-2-2b_*.parquet

# List files from a specific time period
ls data/datasets/*_2024-01-06_*.parquet

# Find checkpoints for a specific range
ls data/datasets/checkpoints/checkpoint_*_0-973_*.json
```

**Merging parallel processing results:**
```bash
# Merge all recent dataset files
python3 merge_datasets.py --pattern "dataset_gemma-2-2b_*.parquet"

# Merge with custom output name
python3 merge_datasets.py --output final_dataset_full_mbpp.parquet
```

## Hardware Requirements

- Supports CUDA (NVIDIA GPUs), MPS (Apple Silicon), or CPU
- Recommended: GPU with at least 24GB VRAM for Gemma 2 9B model
- Disk space: ~50GB for model weights and datasets

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{tahimic2025identifying,
  title={Identifying and Steering Program Validity Awareness Latent Directions in LLMs: A Sparse Autoencoder Analysis of Code Hallucinations},
  author={Tahimic, Kriz Royce},
  year={2025},
  school={De La Salle University}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.