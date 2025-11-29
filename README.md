# Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders

This repository contains the implementation for investigating program validity awareness in language models using Sparse Autoencoders (SAEs).

## Overview

This research analyzes how language models internally represent the concept of code correctness. Using Google's Gemma 2 model (2B parameters) and the MBPP (Mostly Basic Programming Problems) dataset, we:

1. Generate Python code solutions using a base language model
2. Classify solutions as correct (pass@1) or incorrect based on test execution
3. Apply Sparse Autoencoders from GemmaScope to identify latent directions that encode correctness
4. Validate findings through statistical analysis (AUROC, F1) and causal intervention via model steering

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd pva_sae

# Install dependencies
pip install -r requirements.txt

# For CUDA support
pip install accelerate
```

## Project Structure

```
pva_sae/
├── common/                         # Shared utilities and configurations
├── common_simplified/              # Simplified modules following KISS principle
├── phase0_difficulty_analysis/     # MBPP complexity preprocessing
├── phase0_1_problem_splitting/     # Problem splitting into train/val/test
├── phase1_simplified/              # Dataset generation
├── phase2_2_pile_baseline/        # Pile activation baseline
├── phase2_5_sae_analysis/         # SAE feature analysis
├── phase3_5_temperature_robustness/# Temperature robustness testing
├── phase3_6_hyperparameter_tuning/ # Hyperparameter optimization
├── phase3_8_evaluation/           # AUROC and F1 evaluation
├── phase3_10_temperature_trends/  # Temperature-based analysis
├── phase3_12_difficulty_analysis/ # Difficulty-based analysis
├── phase4_5_adaptive_steering/    # Adaptive coefficient selection
├── phase4_6_coefficient_refinement/# Binary search refinement
├── phase4_8_steering_analysis/    # Steering effect analysis
├── phase4_10_random_features/     # Random baseline features
├── phase4_12_statistical_control/ # Statistical validation
├── phase7_3_instruct_baseline/    # Instruction-tuned model baseline
├── phase7_6_instruct_steering/    # Instruction-tuned steering
├── phase7_12_instruct_evaluation/ # Instruction-tuned evaluation
├── data/                          # Phase-based data outputs
├── run.py                         # Main entry point
└── requirements.txt               # Dependencies
```

## Methodology

### Phase Overview

The project follows a systematic experimental pipeline:

#### Data Preparation
- **Phase 0**: Difficulty analysis of 974 MBPP problems using cyclomatic complexity
- **Phase 0.1**: Problem splitting (50% SAE analysis, 10% hyperparameter tuning, 40% validation)

#### Feature Discovery
- **Phase 1.0**: Dataset building with code generation at temperature=0.0
- **Phase 2.2**: Pile activation caching for general language feature baseline
- **Phase 2.5**: SAE analysis to identify discriminative features using separation scores

#### Statistical Validation
- **Phase 3.5**: Temperature robustness testing across [0.0, 0.3, 0.6, 0.9, 1.2]
- **Phase 3.6**: Hyperparameter tuning set processing
- **Phase 3.8**: AUROC and F1 evaluation for identified features
- **Phase 3.10**: Temperature-based AUROC analysis
- **Phase 3.12**: Difficulty-based AUROC analysis

#### Causal Validation
- **Phase 4.5**: Adaptive steering coefficient selection using coarse-to-fine search
- **Phase 4.6**: Binary search refinement for optimal coefficients
- **Phase 4.8**: Steering effect analysis (correction and corruption rates)
- **Phase 4.10**: Random feature selection for baseline control
- **Phase 4.12**: Statistical significance testing with random steering

#### Model Comparison
- **Phase 7.3**: Instruction-tuned model baseline generation
- **Phase 7.6**: Instruction-tuned model steering analysis
- **Phase 7.12**: Comparative AUROC/F1 evaluation

## Usage

### Quick Start

Run phases sequentially using the unified script:

```bash
# Data preparation
python3 run.py phase 0     # Difficulty analysis
python3 run.py phase 0.1   # Problem splitting

# Feature discovery
python3 run.py phase 1     # Dataset generation
python3 run.py phase 2.2   # Pile baseline
python3 run.py phase 2.5   # SAE analysis

# Validation
python3 run.py phase 3.5   # Temperature robustness
python3 run.py phase 3.8   # AUROC/F1 evaluation
python3 run.py phase 4.8   # Steering analysis

# Model comparison
python3 run.py phase 7.3   # Instruction-tuned baseline
python3 run.py phase 7.12  # Comparative evaluation
```

### Multi-GPU Support

For parallel processing across multiple GPUs:

```bash
# Dataset generation (Phase 1)
python3 multi_gpu_launcher.py --phase 1 --start 0 --end 488 --model google/gemma-2-2b

# Temperature robustness (Phase 3.5)
python3 multi_gpu_launcher.py --phase 3.5 --model google/gemma-2-2b
```

### Key Parameters

- Dataset generation uses temperature=0.0 for deterministic outputs
- SAE analysis focuses on residual stream at final token position
- Steering coefficients selected adaptively via coarse-to-fine search
- Statistical validation uses bootstrap resampling for confidence intervals

## Data Output Structure

Each phase outputs to its own directory with structured results:

```
data/
├── phase0/           # Difficulty mappings
├── phase0_1/         # Split datasets (SAE/hyperparams/validation)
├── phase1_0/         # Generated code and activations
├── phase2_2/         # Pile activation baseline
├── phase2_5/         # SAE analysis results and top features
├── phase3_5/         # Temperature robustness data
├── phase3_8/         # AUROC/F1 evaluation metrics
├── phase4_8/         # Steering effect analysis
└── phase7_12/        # Instruction-tuned model comparisons
```

## Key Findings

The analysis identifies latent directions in language models that:
1. Discriminate between correct and incorrect code with high AUROC (>0.7)
2. Demonstrate causal influence through steering interventions
3. Show robustness across different generation temperatures
4. Transfer between base and instruction-tuned model variants

## Hardware Requirements

- GPU with at least 24GB VRAM (for 2B parameter model)
- 100GB CPU RAM for activation processing
- ~50GB disk space for model weights and datasets
- Supports CUDA (NVIDIA), MPS (Apple Silicon), or CPU

## Reproducibility

- All random seeds are fixed for deterministic results
- Checkpointing system enables resuming interrupted runs
- Auto-discovery pipeline ensures consistent data flow between phases
- Problem splits stratified by difficulty for balanced evaluation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
