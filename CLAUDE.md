# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PVA-SAE (Python Value Attribution Sparse Autoencoder) is a thesis research project investigating program validity awareness in language models. The project analyzes how language models internally represent code correctness by:

1. Using Google's Gemma 2 (9B parameters) to generate Python solutions for MBPP problems
2. Classifying solutions as correct (pass@1) or incorrect based on test execution
3. Applying Sparse Autoencoders (SAEs) from GemmaScope to identify latent directions
4. Validating findings through statistical analysis (AUROC, F1) and model steering experiments

The methodology follows three phases: dataset building (50% SAE analysis, 10% hyperparameter tuning, 40% validation), SAE activation analysis using separation scores, and validation through both statistical measures and causal intervention via model steering.

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

### Running the Data Processing Pipeline
```bash
# Run the main data processing script
python3 interp/data_processing.py

# Run the enhanced data processing version
python3 interp/dp_v2.py

# Run production hardened version (recommended for full dataset)
python3 interp/run_production_build.py --test-run  # Test with 10 records
python3 interp/run_production_build.py --model google/gemma-2-9b  # Full production run
```

### Testing and Development
```bash
# Build dataset for thesis (Note: default model is gemma-2-2b, thesis uses gemma-2-9b)
python3 -c "
from interp.data_processing import EnhancedMBPPTester
tester = EnhancedMBPPTester(model_name='google/gemma-2-9b')
tester.build_dataset_mvp_with_cleanup(start_idx=0, end_idx=100)
"

# Quick test with smaller model
python3 -c "
from interp.data_processing import EnhancedMBPPTester
tester = EnhancedMBPPTester()  # Uses default gemma-2-2b
tester.build_dataset_mvp_with_cleanup(start_idx=0, end_idx=2)
"

# Check logs
ls -la mbpp_logs/

# View generated datasets
ls -la interp/mbpp_datasets/
```

## Architecture Overview

### Core Components

1. **Data Processing Pipeline** (`interp/data_processing.py`, `interp/dp_v2.py`)
   - `ModelManager`: Handles model loading and code generation
   - `EnhancedDatasetManager`: Manages MBPP dataset and prompt templates
   - `TestExecutor`: Executes generated code against test cases
   - `DatasetBuilder`: Orchestrates the generation and classification pipeline
   - `EnhancedMBPPTester`: Main entry point for dataset building

2. **Analysis Modules** (planned/in development)
   - `sae_analysis.py`: Sparse autoencoder analysis
   - `steering.py`: Model steering experiments
   - `robustness_analysis.py`: Robustness testing
   - `statistical_analysis.py`: Statistical analysis

3. **Production Hardening** (`interp/data_processing_hardened.py`)
   - `HardeningConfig`: Configuration for production parameters
   - `CheckpointManager`: Saves/loads progress for resume capability
   - `ProgressTracker`: Enhanced progress monitoring with ETA
   - `ResourceMonitor`: Memory and GPU monitoring
   - `HardenedDatasetBuilder`: Production-ready dataset builder
   - `ProductionMBPPTester`: Orchestrates hardened pipeline

### Key Design Patterns

- **Manager Pattern**: Separate managers for models, datasets, logging, and directories
- **Builder Pattern**: `DatasetBuilder` for complex dataset construction
- **Result Objects**: `TestResult`, `GenerationResult` for structured data
- **Template Builder**: Standardized prompt construction from MBPP records

### Data Flow

1. MBPP record → PromptTemplateBuilder → standardized prompt
2. Prompt → ModelManager → generated Python code
3. Generated code + test cases → TestExecutor → test results
4. Test results → classification (correct/incorrect)
5. Results → JSON/Parquet files with metadata

### File Organization

- `mbpp_logs/`: Timestamped execution logs
- `interp/mbpp_datasets/`: Generated datasets (JSON/Parquet)
- Automatic cleanup keeps only latest 2-3 files of each type

### Hardware Support

The system automatically detects and uses:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU fallback

## Important Constants

- Default model: `google/gemma-2-2b` (thesis uses `google/gemma-2-9b`)
- Max new tokens: 2000
- Dataset formats: JSON and Parquet with pandas
- Classification criterion: pass@1 (must pass all 3 test cases on first attempt)
- Temperature: 0 (deterministic generation)

## Thesis-Specific Information

### Dataset Splits
- SAE Analysis: 50% of generated dataset
- Hyperparameter Tuning: 10% (for F1 threshold and steering coefficient)
- Validation: 40%

### SAE Analysis Details
- Uses pre-trained SAEs from GemmaScope with JumpReLU architecture
- Analyzes residual stream at final token position
- Filters latents with >2% activation on Pile dataset
- Identifies latents with highest separation scores

### Validation Metrics
- **Statistical Analysis**:
  - AUROC: Measures discrimination ability across all thresholds
  - F1 Score: Harmonic mean of precision and recall (threshold optimized on hyperparameter set)
- **Robustness Analysis**:
  - Temperature Variation: Tests model at temperatures 0, 0.5, 1.0, 1.5, 2.0 (5 samples each)
  - Difficulty Variation: Stratified analysis across difficulty levels
- **Model Steering**:
  - Correction Rate: Proportion of incorrect→correct after steering
  - Corruption Rate: Proportion of correct→incorrect after steering
  - Binomial Testing: Compares to baseline random steering (p < 0.05)