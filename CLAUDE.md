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

### Running Individual Phases

#### Phase 0: Difficulty Analysis
```bash
# Run difficulty analysis for all MBPP problems
python3 run.py --phase 0

# Load and validate existing difficulty mapping
python3 run.py --phase 0 --load-existing data/datasets/difficulty_mapping_20241201_120000.json

# Run without saving (analysis only)
python3 run.py --phase 0 --no-save --verbose
```

#### Phase 1: Dataset Building
```bash
# Run dataset building with default settings
python3 run.py --phase 1 --model google/gemma-2-9b

# Quick test with smaller model and range
python3 run.py --phase 1 --model google/gemma-2-2b --start 0 --end 10

# Stream output and cleanup before building
python3 run.py --phase 1 --model google/gemma-2-9b --stream --cleanup
```

#### Phase 2: SAE Analysis
```bash
# Run SAE analysis on generated dataset
python3 run.py --phase 2 --dataset data/datasets/latest_dataset.parquet

# Use custom SAE model and threshold
python3 run.py --phase 2 --dataset data/datasets/latest_dataset.parquet --sae-model path/to/sae --latent-threshold 0.05
```

#### Phase 3: Validation
```bash
# Run validation with default settings
python3 run.py --phase 3 --dataset data/datasets/latest_dataset.parquet

# Custom temperature and steering coefficient ranges
python3 run.py --phase 3 --dataset data/datasets/latest_dataset.parquet --temperatures 0.0 1.0 2.0 --steering-coeffs -2.0 0.0 2.0
```

### Testing and Development
```bash
# Quick tests with small datasets
python3 run.py --phase 0 --verbose  # Test difficulty analysis
python3 run.py --phase 1 --model google/gemma-2-2b --start 0 --end 2  # Test dataset building

# Test dataset splitting
python3 -c "
from orchestration.pipeline import DatasetSplitter
splitter = DatasetSplitter('data/datasets/latest_dataset.parquet')
sae_data, tuning_data, validation_data = splitter.split_dataset()
print(f'Split sizes: SAE={len(sae_data)}, Tuning={len(tuning_data)}, Validation={len(validation_data)}')
"

# Check logs and datasets
ls -la data/logs/
ls -la data/datasets/
```

## New Architecture Overview (Refactored)

### Project Structure

```
pva_sae/
├── common/                        # Shared utilities and configurations
│   ├── utils.py                  # Device detection, cleanup, helpers
│   ├── config.py                 # All configuration classes
│   ├── models.py                 # Model management
│   └── logging.py                # Logging utilities
│
├── phase1_dataset_building/       # Phase 1: Dataset generation
│   ├── dataset_manager.py        # MBPP dataset and prompts
│   ├── test_executor.py          # Code execution and testing
│   ├── dataset_builder.py        # Dataset building logic
│   └── mbpp_tester.py            # Main orchestrators
│
├── phase2_sae_analysis/           # Phase 2: SAE analysis
│   └── sae_analyzer.py           # (To be implemented)
│
├── phase3_validation/             # Phase 3: Validation
│   ├── statistical_validator.py  # AUROC/F1 analysis
│   ├── robustness_tester.py     # Temperature testing
│   └── model_steerer.py          # Steering experiments
│
├── orchestration/                 # Pipeline coordination
│   └── pipeline.py               # Three-phase orchestrator
│
├── data/                          # Consolidated data directory
│   ├── datasets/                 # Generated datasets
│   └── logs/                     # Execution logs
│
├── scripts/                       # Entry point scripts
│   ├── run_full_pipeline.py      # Complete pipeline
│   ├── run_phase1.py             # Phase 1 only
│   └── run_production_build.py   # Production dataset building
│
└── tests/                         # Test files
```

### Core Components

1. **Common Utilities** (`common/`)
   - Shared configurations, utilities, and model management
   - No code duplication across modules
   - Centralized logging and experiment tracking

2. **Phase 1: Dataset Building** (`phase1_dataset_building/`)
   - `DatasetManager`: Loads and manages MBPP dataset
   - `PromptTemplateBuilder`: Creates standardized prompts
   - `TestExecutor`: Runs generated code against test cases
   - `DatasetBuilder`: Coordinates generation and classification
   - `HardenedDatasetBuilder`: Production-grade with checkpointing
   - `ProductionMBPPTester`: Main entry point for production runs

3. **Phase 2: SAE Analysis** (`phase2_sae_analysis/`)
   - SAE activation analysis (to be implemented)
   - Latent direction identification
   - Separation score calculation

4. **Phase 3: Validation** (`phase3_validation/`)
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

- `data/logs/`: Timestamped execution logs
- `data/datasets/`: Generated datasets (JSON/Parquet)
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

**Implementation**: Uses interleaved sampling to ensure uniform complexity distribution across all splits. No buckets or categories - pure quantitative approach that maintains exact ratio targets while preserving complexity ordering.

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

## Coding Practices

The codebase follows these simple, maintainable practices suitable for research-quality code:

### Simplicity First
- **Clear over clever**: Prefer explicit, readable code over complex optimizations
- **Single responsibility**: Each class/function has one clear purpose
- **No premature optimization**: Focus on correctness and clarity first

### Code Organization
- **Manager pattern**: Separate managers handle models, datasets, logging, directories
- **Result objects**: Structured data with `TestResult`, `GenerationResult` classes
- **No code duplication**: Shared utilities in `common/` package
- **Clear naming**: Function and variable names describe their purpose

### Error Handling
- **Fail fast**: Validate inputs early and provide clear error messages
- **Graceful degradation**: Continue processing when possible, log failures clearly
- **Timeout handling**: All model calls and code execution have reasonable timeouts

### Testing & Validation
- **Executable examples**: All code examples in documentation should work
- **Incremental testing**: Start with small datasets (2-10 samples) before full runs
- **Automatic validation**: Generated code is actually executed and tested

### Data Management
- **Timestamped outputs**: All files include generation timestamps
- **Automatic cleanup**: Keep only latest 2-3 versions of each file type
- **Multiple formats**: Save both JSON (human-readable) and Parquet (efficient)
- **Metadata tracking**: Configuration and execution details saved with datasets
- **Interleaved splitting**: Deterministic, bucket-free dataset splits maintaining complexity balance

### Dependencies
- **Minimal viable**: Only include dependencies that are actually needed
- **Standard libraries**: Prefer Python standard library when sufficient
- **Hardware agnostic**: Auto-detect CUDA/MPS/CPU, graceful fallbacks

### Documentation
- **Code as documentation**: Clear naming reduces need for comments
- **Command examples**: All commands in CLAUDE.md are copy-pastable and tested
- **Architecture overview**: High-level structure documented with examples