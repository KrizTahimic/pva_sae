# SAE-Code-Correctness: Sparse Autoencoder Analysis of Code Correctness

This repository contains the implementation for investigating program validity awareness in language models using Sparse Autoencoders (SAEs).

## Overview

This research analyzes how language models internally represent the concept of code correctness. Using Google's Gemma 2 model (2B parameters) with the MBPP and HumanEval datasets, we:

1. Generate Python code solutions using a base language model
2. Classify solutions as correct (pass@1) or incorrect based on test execution
3. Apply Sparse Autoencoders from GemmaScope to identify latent directions that encode correctness
4. Validate findings through statistical analysis (AUROC, F1) and causal intervention via model steering

## Installation

```bash
git clone [repository-url]
cd sae-code-correctness
pip install -r requirements.txt
pip install accelerate  # For CUDA support
```

## Data

Experiment data is hosted on HuggingFace: [kriztahimic/sae-code-correctness-data](https://huggingface.co/datasets/kriztahimic/sae-code-correctness-data)

```bash
# Download all data (~1.7 GB)
pip install huggingface_hub
huggingface-cli download kriztahimic/sae-code-correctness-data --local-dir ./data --repo-type dataset

# Or download specific phase only
huggingface-cli download kriztahimic/sae-code-correctness-data \
    --local-dir ./data --include "phase4_8/*" --repo-type dataset
```

## Supported Configurations

### Datasets
| Dataset | Description |
|---------|-------------|
| MBPP | Mostly Basic Programming Problems (974 tasks) |
| HumanEval | OpenAI's hand-written Python problems (164 tasks) |

### Models
| Model | SAE | Status |
|-------|-----|--------|
| `google/gemma-2-2b` | GemmaScope 16k | Supported |
| `google/gemma-2-2b-it` | GemmaScope 16k | Supported |
| `google/gemma-2-9b` | GemmaScope 16k | Supported |
| `meta-llama/Llama-3.1-8B` | LlamaScope 8x | Infrastructure Ready |

To switch configurations, edit `common/config.py`:
```python
dataset_name: str = "humaneval"  # Options: "mbpp", "humaneval"
model_name: str = "google/gemma-2-2b"
```

## Project Structure

```
sae-code-correctness/
├── common/                           # Shared utilities
├── phase0_difficulty_analysis/       # MBPP complexity preprocessing
├── phase0_1_problem_splitting/       # Problem splitting
├── phase1_latent_selection_dataset/  # Dataset generation
├── phase2_2_pile_caching/            # Pile activation baseline
├── phase2_5_separation_score_analysis/ # SAE latent analysis
├── phase2_10_t_statistic_latent_selector/ # T-statistic selection
├── phase3_5_temperature_robustness/  # Temperature testing
├── phase3_8/                         # AUROC/F1 evaluation
├── phase4_5_model_steering/          # Coefficient selection
├── phase4_8_steering_analysis/       # Steering effect analysis
├── phase5_3_weight_orthogonalization/ # Weight modifications
├── phase7_3_instruct_baseline/       # Instruction-tuned baseline
├── phase7_12/                        # Instruction-tuned evaluation
├── phase8_3_selective_steering/      # Selective steering
├── data/                             # Phase outputs
├── run.py                            # Main entry point
└── requirements.txt
```

## Quick Start

Run phases sequentially:

```bash
# Data preparation
python3 run.py phase 0       # Difficulty analysis
python3 run.py phase 0.1     # Problem splitting

# Feature discovery
python3 run.py phase 1       # Dataset generation
python3 run.py phase 2.2     # Pile baseline
python3 run.py phase 2.5     # SAE analysis

# Validation
python3 run.py phase 3.5     # Temperature robustness
python3 run.py phase 3.8     # AUROC/F1

# Causal intervention
python3 run.py phase 4.5     # Coefficient selection
python3 run.py phase 4.8     # Steering analysis

# Model comparison
python3 run.py phase 7.3     # Instruction-tuned baseline
python3 run.py phase 7.12    # Comparative evaluation

# Selective steering
python3 run.py phase 8.3     # Threshold-based steering
```

### Key Options

```bash
--start N --end M    # Process subset (for testing)
--viz-only           # Regenerate plots only (seconds vs hours)
```

## Methodology

The project follows a systematic experimental pipeline:

| Stage | Phases | Purpose |
|-------|--------|---------|
| Data Preparation | 0, 0.1-0.3 | Difficulty analysis, problem splitting, HumanEval conversion |
| Feature Discovery | 1, 2.2, 2.5, 2.10 | Dataset generation, Pile baseline, SAE analysis |
| Statistical Validation | 3.5-3.12 | Temperature robustness, AUROC/F1, difficulty analysis |
| Causal Validation | 4.5-4.14 | Steering coefficients, effect analysis, significance testing |
| Weight Orthogonalization | 5.3-5.9 | Permanent weight modifications, control experiments |
| Model Comparison | 7.3-7.12 | Instruction-tuned baseline, cross-model analysis |
| Selective Steering | 8.1-8.3 | Threshold optimization, selective intervention |

## Data Output Structure

```
data/
├── phase0/       # Difficulty mappings
├── phase0_1/     # Split datasets (selection 50%, tuning 10%, analysis 40%)
├── phase1_0/     # Generated code + activations
├── phase2_5/     # SAE analysis results
├── phase3_8/     # AUROC/F1 metrics
├── phase4_8/     # Steering analysis
└── phase7_12/    # Instruction-tuned comparisons
```

All data available on [HuggingFace](https://huggingface.co/datasets/kriztahimic/sae-code-correctness-data).

## Key Findings

The analysis identifies latent directions in language models that:
1. Discriminate between correct and incorrect code with high AUROC (>0.7)
2. Demonstrate causal influence through steering interventions
3. Show robustness across different generation temperatures
4. Transfer between base and instruction-tuned model variants

## Hardware Requirements

- GPU: 24GB+ VRAM (for 2B model)
- CPU RAM: 100GB+ (activation processing)
- Disk: ~50GB (model weights + datasets)
- Supports: CUDA, MPS (Apple Silicon), CPU

## Reproducibility

- All random seeds fixed (default: 42)
- Checkpointing enables resuming interrupted runs
- Auto-discovery ensures consistent data flow
- Problem splits stratified by difficulty

## Roadmap

### Completed
- Gemma-2-2B analysis with MBPP
- HumanEval dataset support
- Instruction-tuned model comparison
- Gemma-2-9B support
- LlamaScope SAE loader

### Ready to Run
- LLAMA-3.1-8B experiments

### Planned
- Cross-model mechanistic analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.
