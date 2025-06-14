# PVA-SAE: Python Value Attribution using Sparse Autoencoders

This repository contains the implementation for a thesis project investigating program validity awareness in language models using Sparse Autoencoders (SAEs).

## Overview

This research analyzes how language models internally represent the concept of code correctness. By using Google's Gemma 2 model (2B parameters) and the MBPP (Mostly Basic Programming Problems) dataset, we:

1. Generate Python code solutions using a base language model
2. Classify solutions as correct (pass@1) or incorrect
3. Analyze model representations using Sparse Autoencoders to identify latent directions
4. Validate findings through statistical analysis and model steering

## Methodology

### Phase 0: Difficulty Analysis
- Analyzes complexity of all 974 MBPP problems using cyclomatic complexity
- Creates difficulty mapping for consistent data splits across experiments
- Enables reproducible interleaved sampling based on problem difficulty

### Phase 0.1: Problem Splitting
- Splits MBPP problems into three sets based on difficulty scores from Phase 0
- Stratified randomized interleaving ensures equal difficulty distribution
- Creates: 50% SAE analysis, 10% hyperparameter tuning, 40% validation splits
- Outputs task IDs for each split, enabling consistent experiments

### Phase 1: Dataset Building
- Uses MBPP dataset with pre-computed difficulty mappings from Phase 0
- Standardized prompt template: problem description + test cases + code initiator
- Classification: correct (passes all 3 tests) vs incorrect
- Generates solutions for all 974 problems (splits applied in later phases)

### Phase 2: SAE Analysis
- Utilizes pre-trained SAEs from GemmaScope with JumpReLU architecture
- Analyzes residual stream at final token position
- Computes separation scores to identify distinguishing latent dimensions
- Filters out general language patterns (>2% activation on Pile dataset)

### Phase 3: Validation
- **Statistical Analysis**: 
  - AUROC: Measures discrimination ability across all thresholds
  - F1 Score: Harmonic mean of precision and recall (optimized on hyperparameter set)
- **Robustness Analysis**:
  - Temperature Variation: Tests across temperatures (0.0, 0.3, 0.6, 0.9, 1.2)
  - Difficulty Variation: Evaluates performance across difficulty levels
- **Model Steering**: 
  - Manipulates identified latent directions to test causal influence
  - Correction Rate: Proportion of incorrect→correct after steering
  - Corruption Rate: Proportion of correct→incorrect after steering
  - Binomial Testing: Statistical significance with baseline control steering

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

# Phase 1: Build dataset (auto-discovers from phase0, outputs to data/phase1_0/)
python3 run.py phase 1 --model google/gemma-2-2b

# Phase 1.2: Generate temperature variations (uses phase0.1 splits, outputs to data/phase1_2/)
# Generates 5 samples each at temperatures [0.3, 0.6, 0.9, 1.2] for validation set only
python3 run.py phase 1.2 --model google/gemma-2-2b

# Phase 2: Analyze with SAEs (auto-discovers from phase1, outputs to data/phase2/)
# Must specify which split to analyze: sae (50%), hyperparams (10%), or validation (40%)
python3 run.py phase 2 --split sae

# Phase 3: Run validation (auto-discovers from phase1 & phase2, outputs to data/phase3/)
python3 run.py phase 3
```

The pipeline automatically discovers outputs from previous phases, creating a seamless workflow.

### Phase-Specific Examples

#### Phase 1.2 Testing
```bash
# Test with just 3 validation samples at 2 temperatures
python3 run.py phase 1.2 --samples 3 --test-temps 0.3 0.6 --test-samples-per-temp 2

# Quick test with 5 samples using default temperatures
python3 run.py phase 1.2 --samples 5

# Test specific model with limited samples
python3 run.py phase 1.2 --samples 10 --model google/gemma-2-2b

python3 run.py phase 1.2 --samples 2 --test-samples-per-temp 2
```

Test mode outputs to `data/test_phase1_2/` to keep production data clean.

#### Other Phase Examples
```bash
# Quick test with small dataset
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
├── phase0_difficulty_analysis/     # Phase 0: MBPP complexity preprocessing
├── phase1_0_dataset_building/      # Phase 1.0: Dataset generation
├── phase0_1_problem_splitting/     # Phase 0.1: Problem splitting
├── phase1_2_temperature_generation/# Phase 1.2: Temperature variations
├── phase2_sae_analysis/            # Phase 2: SAE analysis
├── phase3_validation/              # Phase 3: Validation
├── data/                           # Phase-based data directory
│   ├── phase0/                    # Difficulty mappings
│   ├── phase1_0/                  # Generated datasets
│   ├── phase0_1/                  # Split task IDs
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
# Production build with full MBPP dataset
python3 run.py phase 1 --model google/gemma-2-2b --start 0 --end 973 --cleanup

# Enable progress streaming and verbose output
python3 run.py phase 1 --model google/gemma-2-2b --stream --verbose
```

## Multi-GPU Usage

The project supports parallel processing across multiple GPUs for Phase 1 and Phase 1.2:

### Phase 1: Dataset Building
```bash
# Auto-detect all available GPUs
python3 multi_gpu_launcher.py --phase 1 --start 0 --end 973 --model google/gemma-2-2b

# Use specific GPUs (e.g., GPUs 0, 1, and 2)
python3 multi_gpu_launcher.py --phase 1 --gpus 0,1,2 --start 0 --end 973 --model google/gemma-2-2b
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