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

### Running the Pipeline

#### Full Three-Phase Pipeline
```bash
# Run complete pipeline
python3 scripts/run_full_pipeline.py --model google/gemma-2-9b

# Run with custom configuration
python3 scripts/run_full_pipeline.py --config configs/experiment.json

# Dry run to see configuration
python3 scripts/run_full_pipeline.py --dry-run
```

#### Phase 1: Dataset Building Only
```bash
# Run phase 1 with default settings
python3 scripts/run_phase1.py --model google/gemma-2-9b

# Quick test with smaller model
python3 scripts/run_phase1.py --model google/gemma-2-2b --start 0 --end 10

# Production build with hardening
python3 interp/run_production_build.py --test-run  # Test with 10 records
python3 interp/run_production_build.py --model google/gemma-2-9b  # Full production run
```

### Testing and Development
```bash
# Build dataset for thesis (Note: default model is gemma-2-2b, thesis uses gemma-2-9b)
python3 -c "
from interp.phase1_dataset_building import EnhancedMBPPTester
tester = EnhancedMBPPTester(model_name='google/gemma-2-9b')
tester.build_dataset_mvp_with_cleanup(start_idx=0, end_idx=100)
"

# Quick test with smaller model
python3 -c "
from interp.phase1_dataset_building import EnhancedMBPPTester
tester = EnhancedMBPPTester()  # Uses default gemma-2-2b
tester.build_dataset_mvp_with_cleanup(start_idx=0, end_idx=2)
"

# Check logs
ls -la interp/data/logs/

# View generated datasets
ls -la interp/data/datasets/
```

## New Architecture Overview (Refactored)

### Project Structure

```
pva_sae/
├── interp/
│   ├── common/               # Shared utilities and configurations
│   │   ├── utils.py         # Device detection, cleanup, helpers
│   │   ├── config.py        # All configuration classes
│   │   ├── models.py        # Model management
│   │   └── logging.py       # Logging utilities
│   │
│   ├── phase1_dataset_building/   # Phase 1: Dataset generation
│   │   ├── dataset_manager.py    # MBPP dataset and prompts
│   │   ├── test_executor.py      # Code execution and testing
│   │   ├── dataset_builder.py    # Dataset building logic
│   │   └── mbpp_tester.py        # Main orchestrators
│   │
│   ├── phase2_sae_analysis/       # Phase 2: SAE analysis
│   │   └── sae_analyzer.py       # (To be implemented)
│   │
│   ├── phase3_validation/         # Phase 3: Validation
│   │   ├── statistical_validator.py  # AUROC/F1 analysis
│   │   ├── robustness_tester.py     # Temperature testing
│   │   └── model_steerer.py         # Steering experiments
│   │
│   ├── orchestration/         # Pipeline coordination
│   │   └── pipeline.py       # Three-phase orchestrator
│   │
│   └── data/                 # Consolidated data directory
│       ├── datasets/         # Generated datasets
│       └── logs/            # Execution logs
│
└── scripts/                  # Entry point scripts
    ├── run_full_pipeline.py  # Complete pipeline
    └── run_phase1.py        # Phase 1 only
```

### Core Components

1. **Common Utilities** (`interp/common/`)
   - Shared configurations, utilities, and model management
   - No code duplication across modules
   - Centralized logging and experiment tracking

2. **Phase 1: Dataset Building** (`interp/phase1_dataset_building/`)
   - `DatasetManager`: Loads and manages MBPP dataset
   - `PromptTemplateBuilder`: Creates standardized prompts
   - `TestExecutor`: Runs generated code against test cases
   - `DatasetBuilder`: Coordinates generation and classification
   - `HardenedDatasetBuilder`: Production-grade with checkpointing
   - `ProductionMBPPTester`: Main entry point for production runs

3. **Phase 2: SAE Analysis** (`interp/phase2_sae_analysis/`)
   - SAE activation analysis (to be implemented)
   - Latent direction identification
   - Separation score calculation

4. **Phase 3: Validation** (`interp/phase3_validation/`)
   - Statistical validation (AUROC, F1)
   - Robustness testing across temperatures
   - Model steering experiments
   - Binomial testing for significance

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

- `interp/data/logs/`: Timestamped execution logs
- `interp/data/datasets/`: Generated datasets (JSON/Parquet)
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