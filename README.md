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
  - Difficulty Variation: Evaluates performance across APPS difficulty levels
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
from interp.data_processing import EnhancedMBPPTester

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
from interp.data_processing import EnhancedDatasetManager, ModelManager, DatasetBuilder

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
├── interp/                    # Main implementation
│   ├── data_processing.py     # Core data processing pipeline
│   ├── dp_v2.py              # Enhanced data processing
│   ├── sae_analysis.py       # SAE analysis (to be implemented)
│   ├── steering.py           # Model steering experiments
│   ├── robustness_analysis.py # Robustness testing
│   └── statistical_analysis.py # Statistical validation
├── mbpp_logs/                # Execution logs
├── interp/mbpp_datasets/     # Generated datasets
└── requirements.txt          # Dependencies
```

## Output Files

- **Logs**: Timestamped logs in `mbpp_logs/`
- **Datasets**: JSON and Parquet files in `interp/mbpp_datasets/`
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