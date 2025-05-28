# PVA-SAE: Python Value Attribution using Sparse Autoencoders

This repository contains the implementation for a thesis project investigating program validity awareness in language models using Sparse Autoencoders (SAEs).

## Overview

This research analyzes how language models internally represent the concept of code correctness. By using Google's Gemma 2 model (9B parameters) and the MBPP (Mostly Basic Programming Problems) dataset, we:

1. Generate Python code solutions using a base language model
2. Classify solutions as correct (pass@1) or incorrect
3. Analyze model representations using Sparse Autoencoders to identify latent directions
4. Validate findings through statistical analysis and model steering

## Methodology

### 1. Dataset Building
- Uses MBPP dataset (974 programming problems)
- Standardized prompt template: problem description + test cases + code initiator
- Classification: correct (passes all 3 tests) vs incorrect
- Dataset split: 50% SAE analysis, 10% hyperparameter tuning, 40% validation

### 2. SAE Analysis
- Utilizes pre-trained SAEs from GemmaScope with JumpReLU architecture
- Analyzes residual stream at final token position
- Computes separation scores to identify distinguishing latent dimensions
- Filters out general language patterns (>2% activation on Pile dataset)

### 3. Validation
- **Statistical Analysis**: 
  - AUROC: Measures discrimination ability across all thresholds
  - F1 Score: Harmonic mean of precision and recall (optimized on hyperparameter set)
- **Robustness Analysis**:
  - Temperature Variation: Tests across temperatures (0, 0.5, 1.0, 1.5, 2.0)
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

Run individual phases using the unified script:

```bash
# Phase 0: Analyze difficulty for all MBPP problems
python3 run.py --phase 0

# Phase 1: Build dataset with Gemma 2 model
python3 run.py --phase 1 --model google/gemma-2-9b

# Phase 2: Analyze with Sparse Autoencoders
python3 run.py --phase 2 --dataset data/datasets/latest_dataset.parquet

# Phase 3: Run validation experiments
python3 run.py --phase 3 --dataset data/datasets/latest_dataset.parquet
```

### Phase-Specific Examples

```bash
# Quick test with small dataset
python3 run.py --phase 1 --model google/gemma-2-2b --start 0 --end 10

# Load existing difficulty mapping
python3 run.py --phase 0 --load-existing data/datasets/difficulty_mapping_latest.json

# Custom SAE analysis parameters
python3 run.py --phase 2 --dataset my_dataset.parquet --latent-threshold 0.05

# Validation with custom temperature range
python3 run.py --phase 3 --dataset my_dataset.parquet --temperatures 0.0 1.0 2.0
```

## Project Structure

```
pva_sae/
├── common/                        # Shared utilities and configurations
├── phase0_difficulty_analysis/    # Phase 0: MBPP complexity preprocessing
├── phase1_dataset_building/       # Phase 1: Dataset generation
├── phase2_sae_analysis/           # Phase 2: SAE analysis
├── phase3_validation/             # Phase 3: Validation
├── orchestration/                 # Pipeline coordination
├── data/                          # Consolidated data directory
│   ├── datasets/                 # Generated datasets
│   └── logs/                     # Execution logs
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
python3 run.py --phase 1 --model google/gemma-2-9b

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
python3 run.py --phase 1 --model google/gemma-2-9b --start 0 --end 973 --cleanup

# Enable progress streaming and verbose output
python3 run.py --phase 1 --model google/gemma-2-9b --stream --verbose
```

## Output Files

- **Logs**: Timestamped logs in `data/logs/`
- **Datasets**: JSON and Parquet files in `data/datasets/`
- **Checkpoints**: Recovery files in `data/datasets/checkpoints/`
- **Autosaves**: Periodic backups during long runs
- **Metadata**: Accompanying metadata files with statistics

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