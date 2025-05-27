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

```python
from phase1_dataset_building import EnhancedMBPPTester

# Initialize tester with Gemma 2 model
tester = EnhancedMBPPTester(model_name="google/gemma-2-9b")

# Build dataset with automatic cleanup
results = tester.build_dataset_mvp_with_cleanup(
    start_idx=0, 
    end_idx=100,  # Process first 100 MBPP problems
    save_format="both"  # Save as JSON and Parquet
)
```

### Dataset Building Only

```python
from phase1_dataset_building import EnhancedDatasetManager, ModelManager, DatasetBuilder

# Setup components
dataset_manager = EnhancedDatasetManager()
dataset_manager.load_dataset()

model_manager = ModelManager("google/gemma-2-9b")
model_manager.load_model()

# Build dataset
builder = DatasetBuilder(model_manager, dataset_manager)
results = builder.build_dataset(start_idx=0, end_idx=100)
builder.save_dataset(format="parquet")
```

## Project Structure

```
pva_sae/
├── common/                        # Shared utilities and configurations
├── phase1_dataset_building/       # Phase 1: Dataset generation
├── phase2_sae_analysis/           # Phase 2: SAE analysis
├── phase3_validation/             # Phase 3: Validation
├── orchestration/                 # Pipeline coordination
├── data/                          # Consolidated data directory
│   ├── datasets/                 # Generated datasets
│   └── logs/                     # Execution logs
├── scripts/                       # Entry point scripts
└── requirements.txt          # Dependencies
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
python scripts/run_production_build.py --model google/gemma-2-9b

# Prompts: "Found checkpoint with 150 processed records. Resume? (y/n)"
```

### Manual Recovery
```bash
# List available checkpoints
ls data/datasets/checkpoints/

# Resume from specific checkpoint
python scripts/run_production_build.py --resume checkpoints/checkpoint_0_973_20250527_180144.json
```

### What Gets Preserved
- **Progress**: Exact record indices processed
- **Results**: All generated code and test results
- **Statistics**: Success rates, timing data, model configuration
- **State**: Can resume from any interruption point

### Production Features
```python
from phase1_dataset_building import ProductionMBPPTester

# Production build with full hardening
tester = ProductionMBPPTester(model_name="google/gemma-2-9b")
dataset_path = tester.build_dataset_production(
    start_idx=0, 
    end_idx=973,  # Full MBPP dataset
    resume_from_checkpoint=None  # Auto-detect latest
)
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
@thesis{pva_sae2025,
  title={Program Validity Awareness in Language Models using Sparse Autoencoders},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.