# PVA-SAE: Python Value Attribution using Sparse Autoencoders

This repository contains the implementation for investigating program validity awareness in language models using Sparse Autoencoders (SAEs).

## Overview

This research analyzes how language models internally represent the concept of code correctness. Using Google's Gemma 2 model (2B parameters) with the MBPP (Mostly Basic Programming Problems) and HumanEval datasets, we:

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

## Supported Configurations

### Datasets
| Dataset | Status | Description |
|---------|--------|-------------|
| MBPP | Supported | Mostly Basic Programming Problems (974 tasks) |
| HumanEval | Supported | OpenAI's hand-written Python problems (164 tasks) |

### Models
| Model | SAE | Status |
|-------|-----|--------|
| `google/gemma-2-2b` | GemmaScope 16k | Supported |
| `google/gemma-2-2b-it` | GemmaScope 16k | Supported |
| `meta-llama/Llama-3.1-8B` | LlamaScope 8x | Planned |
| `meta-llama/Llama-3.1-8B-Instruct` | LlamaScope 8x | Planned |

To switch configurations, edit `common/config.py`:
```python
dataset_name: str = "humaneval"  # Options: "mbpp", "humaneval"
model_name: str = "google/gemma-2-2b"  # See table above
```

## Project Structure

```
pva_sae/
├── common/                         # Shared utilities and configurations
├── common_simplified/              # Simplified modules following KISS principle
├── phase0_difficulty_analysis/     # MBPP complexity preprocessing
├── phase0_1_problem_splitting/     # Problem splitting into train/val/test
├── phase0_2_humaneval_preprocessing/ # HumanEval to MBPP format conversion
├── phase0_3_humaneval_imports/     # HumanEval import scanning
├── phase1_simplified/              # Dataset generation
├── phase2_2_pile_caching/          # Pile activation baseline
├── phase2_5_simplified/            # SAE feature analysis
├── phase2_10_t_statistic_latent_selector/ # T-statistic feature selection
├── phase2_15_layerwise_visualization/ # Layer-wise analysis visualization
├── phase3_5_temperature_robustness/# Temperature robustness testing
├── phase3_6/                       # Hyperparameter tuning set processing
├── phase3_8/                       # AUROC and F1 evaluation
├── phase3_10_temperature_auroc_f1/ # Temperature-based AUROC analysis
├── phase3_11_temperature_trends_updated/ # Temperature trends visualization
├── phase3_12_difficulty_auroc_f1/ # Difficulty-based AUROC analysis
├── phase4_5_model_steering/        # Adaptive coefficient selection
├── phase4_6_binary_refinement/     # Golden section search refinement
├── phase4_7_coefficient_visualization/ # Coefficient optimization plots
├── phase4_8_steering_analysis/     # Steering effect analysis
├── phase4_10_zero_discrimination/  # Zero-discrimination feature selection
├── phase4_12_zero_disc_steering/   # Zero-discrimination steering (control)
├── phase4_14_statistical_significance/ # Binomial significance testing
├── phase5_3_weight_orthogonalization/ # Permanent weight modifications
├── phase5_6_zero_disc_orthogonalization/ # Control experiment
├── phase5_9_orthogonalization_significance/ # Statistical validation
├── phase6_3_attention_analysis/    # Attention pattern analysis
├── phase7_3_instruct_baseline/     # Instruction-tuned model baseline
├── phase7_6_instruct_steering/     # Instruction-tuned steering
├── phase7_9_universality_analysis/ # Cross-model universality analysis
├── phase7_12/                      # Instruction-tuned evaluation
├── phase8_1_threshold_calculator/  # Percentile threshold calculation
├── phase8_2_threshold_optimizer/   # Threshold optimization
├── phase8_3_selective_steering/    # Selective steering analysis
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
- **Phase 0.2**: HumanEval to MBPP format conversion
- **Phase 0.3**: HumanEval import statement scanning

#### Feature Discovery
- **Phase 1.0**: Dataset building with code generation at temperature=0.0
- **Phase 2.2**: Pile activation caching for general language feature baseline
- **Phase 2.5**: SAE analysis to identify discriminative features using separation scores
- **Phase 2.10**: T-statistic latent selection using Welch's t-test
- **Phase 2.15**: Layer-wise analysis visualization (heatmaps)

#### Statistical Validation
- **Phase 3.5**: Temperature robustness testing across [0.0, 0.3, 0.6, 0.9, 1.2]
- **Phase 3.6**: Hyperparameter tuning set processing
- **Phase 3.8**: AUROC and F1 evaluation for identified features
- **Phase 3.10**: Temperature-based AUROC analysis
- **Phase 3.11**: Temperature trends visualization update
- **Phase 3.12**: Difficulty-based AUROC analysis

#### Causal Validation
- **Phase 4.5**: Adaptive steering coefficient selection using coarse-to-fine search
- **Phase 4.6**: Golden section search refinement for optimal coefficients
- **Phase 4.7**: Coefficient optimization visualization
- **Phase 4.8**: Steering effect analysis (correction and corruption rates)
- **Phase 4.10**: Zero-discrimination feature selection for baseline control
- **Phase 4.12**: Zero-discrimination steering generation (control experiment)
- **Phase 4.14**: Binomial significance testing (targeted vs control steering)

#### Weight Orthogonalization
- **Phase 5.3**: Permanent weight modifications to remove PVA feature directions
- **Phase 5.6**: Zero-discrimination control experiment (validates specificity)
- **Phase 5.9**: Statistical significance testing via triangulation analysis

#### Mechanistic Analysis
- **Phase 6.3**: Attention pattern analysis comparing baseline vs steered generation

#### Model Comparison
- **Phase 7.3**: Instruction-tuned model baseline generation
- **Phase 7.6**: Instruction-tuned model steering analysis
- **Phase 7.9**: Cross-model universality analysis
- **Phase 7.12**: Comparative AUROC/F1 evaluation

#### Selective Steering
- **Phase 8.1**: Percentile threshold calculation from hyperparameter set
- **Phase 8.2**: Threshold optimization via grid search (maximize net benefit)
- **Phase 8.3**: Selective steering analysis (intervene only when threshold exceeded)

## Usage

### Quick Start

Run phases sequentially using the unified script:

```bash
# Data preparation (MBPP)
python3 run.py phase 0     # Difficulty analysis
python3 run.py phase 0.1   # Problem splitting

# Data preparation (HumanEval)
python3 run.py phase 0.2   # HumanEval preprocessing
python3 run.py phase 0.3   # HumanEval imports

# Feature discovery
python3 run.py phase 1     # Dataset generation
python3 run.py phase 2.2   # Pile baseline
python3 run.py phase 2.5   # SAE analysis
python3 run.py phase 2.10  # T-statistic selection

# Statistical validation
python3 run.py phase 3.5   # Temperature robustness
python3 run.py phase 3.6   # Hyperparameter set
python3 run.py phase 3.8   # AUROC/F1 evaluation

# Causal validation
python3 run.py phase 4.5   # Coefficient selection
python3 run.py phase 4.8   # Steering analysis
python3 run.py phase 4.14  # Significance testing

# Weight orthogonalization
python3 run.py phase 5.3   # Weight orthogonalization
python3 run.py phase 5.9   # Significance testing

# Mechanistic analysis
python3 run.py phase 6.3   # Attention pattern analysis

# Model comparison
python3 run.py phase 7.3   # Instruction-tuned baseline
python3 run.py phase 7.12  # Comparative evaluation

# Selective steering
python3 run.py phase 8.1   # Threshold calculation
python3 run.py phase 8.2   # Threshold optimization
python3 run.py phase 8.3   # Selective steering
```


### Key Parameters

- Dataset generation uses temperature=0.0 for deterministic outputs
- SAE analysis focuses on residual stream at final token position
- Steering coefficients selected adaptively via coarse-to-fine search
- Statistical validation uses bootstrap resampling for confidence intervals
- Switch datasets/models by editing `dataset_name` and `model_name` in `common/config.py`

## Data Output Structure

Each phase outputs to its own directory with structured results:

```
data/
├── phase0/           # Difficulty mappings
├── phase0_1/         # Split datasets (SAE/hyperparams/validation)
├── phase0_2/         # HumanEval preprocessed data
├── phase0_3/         # HumanEval import mappings
├── phase1_0/         # Generated code and activations
├── phase2_2/         # Pile activation baseline
├── phase2_5/         # SAE analysis results and top features
├── phase2_10/        # T-statistic selected features
├── phase3_5/         # Temperature robustness data
├── phase3_6/         # Hyperparameter tuning set results
├── phase3_8/         # AUROC/F1 evaluation metrics
├── phase4_5/         # Steering coefficient selection
├── phase4_8/         # Steering effect analysis
├── phase4_14/        # Statistical significance results
├── phase5_3/         # Weight orthogonalization results
├── phase5_9/         # Orthogonalization significance tests
├── phase6_3/         # Attention pattern analysis
├── phase7_12/        # Instruction-tuned model comparisons
├── phase8_1/         # Percentile thresholds
├── phase8_2/         # Threshold optimization results
└── phase8_3/         # Selective steering results
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

## Roadmap

### Completed
- Gemma-2-2B analysis with MBPP dataset
- HumanEval dataset support for selected phases
- Instruction-tuned model (gemma-2-2b-it) comparison

### In Progress
- LLAMA-3.1-8B support with LlamaScope SAEs (`llama_scope_lxr_8x`)
- Multi-model comparison analysis across Gemma and LLAMA families

### Planned
- LLAMA-3.1-8B-Instruct evaluation
- Cross-model mechanistic analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.