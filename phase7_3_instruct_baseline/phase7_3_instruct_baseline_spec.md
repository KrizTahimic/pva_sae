# Phase 7.3: Instruction-Tuned Model Baseline Generation Specification

## Executive Summary

1. **Purpose**: Generate baseline data for instruction-tuned Gemma model to test PVA feature universality
2. **Model**: google/gemma-2-2b-it (instruction-tuned variant) instead of base model
3. **Dataset**: Validation split from Phase 0.1 (388 MBPP problems)
4. **Method**: Code generation at temperature 0.0 with activation extraction from best layers
5. **Extraction**: Best layers only (as identified in Phase 2.5/2.10)
6. **Output**: Baseline dataset and activations for Phase 7.6 steering analysis
7. **Hypothesis**: Test if PVA features discovered in base model are present in instruction-tuned variant

## Pipeline Sequence

```
1. Load Validation Split from Phase 0.1
   └─> Read validation_mbpp.parquet → Extract 388 problems → Prepare for processing

2. Discover Best Layers from Phase 2.5/2.10
   └─> Read top_20_features.json → Extract best correct & incorrect layers → Setup activation extraction

3. Generate Code Solutions with Instruction-Tuned Model
   a. Load model (google/gemma-2-2b-it) and setup activation hooks for best layers
   b. Process each validation problem with deterministic generation
   c. Extract activations from best layers during generation
   d. Evaluate generated code against test cases

4. Save Baseline Data for Phase 7.6
   └─> Save dataset → Save activation files → Create metadata → Enable Phase 7.6 steering analysis
```

## Phase Relationship

### Dependencies
- **Phase 0.1**: Provides validation split (validation_mbpp.parquet with 388 problems)
- **Phase 2.5/2.10**: Provides best layers and feature indices for PVA features

### Enables
- **Phase 7.6**: Requires instruction-tuned baseline data for steering effect analysis
- **Cross-Model Comparison**: Compare with Phase 3.5 (base model validation baseline)

### Key Differences from Related Phases

#### vs Phase 3.6 (Hyperparameter Baseline)
- **Model**: gemma-2-2b-it vs gemma-2-2b
- **Dataset**: Validation split (388) vs hyperparameter split (97 problems)
- **Purpose**: Universality testing vs threshold optimization

#### vs Phase 3.5 (Temperature Robustness)
- **Model**: gemma-2-2b-it vs gemma-2-2b
- **Temperature**: Single (0.0) vs multiple temperatures
- **Sample Size**: 388 problems, 1 sample each vs multiple samples per problem

## Core Implementation

### Base Architecture
```python
# Adapt Phase 3.6's HyperparameterDataRunner for instruction-tuned model
class InstructBaslineRunner:
    """Generate baseline for instruction-tuned model on validation set."""
    
    def __init__(self, config: Config):
        # Load instruction-tuned model instead of base model
        self.model_name = "google/gemma-2-2b-it"  # Key difference
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_name,
            device=self.device
        )
        # Discover best layers from Phase 2.5/2.10
        self.best_layers = self._discover_best_layers()
        self.extraction_layers = list(set([
            self.best_layers['correct'], 
            self.best_layers['incorrect']
        ]))
        
    def _load_validation_data(self):
        """Load validation split instead of hyperparameter split"""
        validation_file = Path(self.config.phase0_1_output_dir) / "validation_mbpp.parquet"
        # Load 388 validation problems
        
    def run(self):
        """Process validation split at temperature 0.0"""
        # Generate code solutions with activation extraction
        # Save dataset and activations for Phase 7.6
```

### Key Adaptations from Phase 3.6
1. **Model Change**: Use gemma-2-2b-it throughout
2. **Data Source**: validation_mbpp.parquet instead of hyperparams_mbpp.parquet
3. **Dataset Size**: 388 problems instead of 97
4. **Purpose**: Universality testing instead of threshold optimization

### Development Arguments
```bash
# Process first 10 problems for development/testing
python3 run.py phase 7.3 --end 9

# Process specific range
python3 run.py phase 7.3 --start 10 --end 19

# Process all validation problems (production)
python3 run.py phase 7.3
```

## Technical Specifications

### Input Requirements
1. **Phase 0.1 Output**: `data/phase0_1/validation_mbpp.parquet`
   - Contains 388 validation problems (40% of MBPP)
   - Includes task_id, text, test_list, cyclomatic_complexity columns

2. **Phase 2.5/2.10 Output**: `data/phase2_10/top_20_features.json` or `data/phase2_5/top_20_features.json`
   - Contains best features with layer numbers and feature indices
   - Used to determine which layers to extract activations from

3. **Model Access**: google/gemma-2-2b-it
   - Instruction-tuned variant of Gemma 2B
   - Requires GPU memory for inference and activation extraction

### Processing Configuration
```python
# Core settings
model_name = "google/gemma-2-2b-it"  # Instruction-tuned model
temperature = 0.0  # Deterministic generation only
samples_per_problem = 1  # Single generation per problem
extraction_layers = [best_correct_layer, best_incorrect_layer]  # From Phase 2.5/2.10
max_new_tokens = 2000  # Standard generation limit
```

### Output Structure
```
data/phase7_3/
├── dataset_instruct_temp_0_0.parquet    # Generated solutions and test results
├── metadata.json                         # Processing metadata and best layers info
└── activations/
    └── task_activations/                 # Activation files for Phase 7.6
        ├── task_001_layer_15.npz         # Best correct layer activations
        ├── task_001_layer_17.npz         # Best incorrect layer activations (if different)
        ├── task_002_layer_15.npz
        └── ...
```

### Memory and Performance
- **GPU Memory**: ~10-15GB for model + activation extraction
- **Processing Time**: ~2-3 minutes per problem (388 total = ~12-19 hours)
- **Storage**: ~2GB for activations + ~200MB for parquet data
- **Development Mode**: `--end 9` processes 10 problems in ~20-30 minutes

## Understanding the Data Flow

### From Phase 0.1 to Phase 7.3
```python
# Phase 0.1 creates validation split
validation_data = pd.read_parquet('data/phase0_1/validation_mbpp.parquet')
print(f"Validation problems: {len(validation_data)}")  # Should be 388

# Phase 7.3 processes these problems with instruction-tuned model
for _, problem in validation_data.iterrows():
    # Generate code at temperature 0.0 with gemma-2-2b-it
    # Extract activations from best layers
    # Save for Phase 7.6 steering analysis
```

### From Phase 2.5/2.10 to Phase 7.3
```python
# Phase 2.5/2.10 provides best layer information
top_features = json.load(open('data/phase2_10/top_20_features.json'))
best_layers = {
    'correct': top_features['correct'][0]['layer'],
    'incorrect': top_features['incorrect'][0]['layer']
}

# Phase 7.3 uses this to extract from optimal layers
extraction_layers = list(set([
    best_layers['correct'],      # e.g., layer 15
    best_layers['incorrect']     # e.g., layer 17
]))
```

### From Phase 7.3 to Phase 7.6
```python
# Phase 7.3 creates baseline data for instruction-tuned model
# Phase 7.6 loads this data for steering effect analysis
instruct_baseline = pd.read_parquet('data/phase7_3/dataset_instruct_temp_0_0.parquet')
# Applies steering to test if PVA features work in instruction-tuned model
```

## Implementation Approach

### Strategy: Maximum Reuse
1. **Base Class**: Copy Phase 3.6's HyperparameterDataRunner
2. **Modify**: Change model to gemma-2-2b-it
3. **Adapt**: Change data source from hyperparameter to validation split
4. **Keep**: All other logic (activation extraction, evaluation, saving)

### Code Reuse from Phase 3.6
```python
# Reuse these components directly:
- Best layers discovery and deduplication logic
- Activation extraction setup and hooks
- Code generation with activation capture
- Code evaluation against test cases
- Activation saving to .npz files
- Metadata creation and saving
- Checkpoint and recovery system
- Memory management

# Modify these components:
- Model name: "google/gemma-2-2b-it" instead of "google/gemma-2-2b"
- Data loading: validation_mbpp.parquet vs hyperparams_mbpp.parquet
- Output paths: phase7_3 instead of phase3_6
```

### Development Workflow
```bash
# 1. Test with small sample
python3 run.py phase 7.3 --end 4  # Process 5 problems

# 2. Verify outputs
ls data/phase7_3/activations/task_activations/
cat data/phase7_3/metadata.json

# 3. Test Phase 7.6 compatibility
python3 run.py phase 7.6  # Should work with Phase 7.3 data

# 4. Full validation processing
python3 run.py phase 7.3  # Process all 388 problems
```

## Key Implementation Details

### Model-Specific Considerations
```python
def __init__(self, config: Config):
    """Initialize with instruction-tuned model."""
    self.config = config
    self.device = detect_device()
    
    # CRITICAL: Use instruction-tuned model
    self.model_name = "google/gemma-2-2b-it"
    logger.info(f"Loading INSTRUCTION-TUNED model {self.model_name}")
    
    self.model, self.tokenizer = load_model_and_tokenizer(
        self.model_name,
        device=self.device
    )
    
    # Rest remains the same as Phase 3.6
    self.best_layers = self._discover_best_layers()
    self._setup_activation_extraction()
```

### Prompt Handling for Instruction-Tuned Model
```python
def generate_with_activations(self, prompt: str, task_id: str):
    """
    Generate code with instruction-tuned model.
    
    Note: Instruction-tuned models may expect different prompt formats.
    The tokenizer should handle this automatically, but we log for verification.
    """
    logger.debug(f"Generating with instruction-tuned model for task {task_id}")
    
    # Rest of generation logic remains the same
    # The model will use its instruction-following capabilities
```

### Validation Data Loading
```python
def _load_validation_data(self) -> pd.DataFrame:
    """Load validation split from Phase 0.1."""
    validation_file = Path(self.config.phase0_1_output_dir) / "validation_mbpp.parquet"
    
    if not validation_file.exists():
        raise FileNotFoundError(
            f"Validation data not found at {validation_file}. "
            "Please run Phase 0.1 first."
        )
    
    data = pd.read_parquet(validation_file)
    logger.info(f"Loaded {len(data)} validation problems for instruction-tuned baseline")
    
    return data
```

## Implementation Checklist

### Setup Phase
- [ ] Create `phase7_3_instruct_baseline/` directory structure
- [ ] Create `instruct_baseline_runner.py` based on Phase 3.6's HyperparameterDataRunner
- [ ] Add Phase 7.3 to `run.py` argument parser
- [ ] Add Phase 7.3 configuration to `common/config.py`
- [ ] Update `common/utils.py` PHASE_CONFIGS for Phase 7.3

### Core Implementation
- [ ] Implement model loading with gemma-2-2b-it
- [ ] Implement `_discover_best_layers()` method (copy from Phase 3.6)
- [ ] Implement `_load_validation_data()` method  
- [ ] Implement activation extraction for best layers
- [ ] Add support for `--start` and `--end` arguments

### Data Processing
- [ ] Load validation split (388 problems) from Phase 0.1
- [ ] Generate code solutions at temperature 0.0 with instruction-tuned model
- [ ] Save activation files compatible with Phase 7.6 expectations
- [ ] Create metadata.json with model info and processing details

### Integration and Testing
- [ ] Test with small sample: `python3 run.py phase 7.3 --end 4`
- [ ] Verify activation files are created correctly
- [ ] Compare baseline performance with Phase 3.5 (base model)
- [ ] Run full validation processing: `python3 run.py phase 7.3`

### Validation
- [ ] Ensure output structure matches Phase 7.6 expectations
- [ ] Verify activation file format matches Phase 3.6 pattern
- [ ] Check metadata.json contains model name and version info
- [ ] Compare pass rates between instruction-tuned and base models

## Expected Outcomes

### Performance Comparison
We expect the instruction-tuned model to potentially show:
1. **Higher baseline pass rate** due to instruction-following capabilities
2. **Different activation patterns** but same layer importance
3. **Similar PVA feature presence** if universality hypothesis holds

### Success Criteria
- Successfully generate baseline for all 388 validation problems
- Extract activations from best layers identified in base model
- Enable Phase 7.6 steering analysis
- Provide data for cross-model comparison

## Why Phase 7.3 is Essential

### Testing Universality
```
Base Model (Phase 3.5) → PVA Features (Phase 2.5) → Steering (Phase 4.8)
                ↓                    ↓                        ↓
Instruction Model (Phase 7.3) → Same Features? → Steering Works? (Phase 7.6)
```

### Research Questions
1. Do PVA features discovered in base model exist in instruction-tuned variant?
2. Are the same layers important for program validity in both models?
3. Can we steer instruction-tuned models using base model features?

### Without Phase 7.3
- Cannot test if PVA features are universal across model variants
- Cannot validate steering on instruction-tuned models
- Limited understanding of feature transferability

### With Phase 7.3
- Enables complete universality testing pipeline
- Allows cross-model steering comparison
- Validates broader applicability of PVA-SAE approach

## Reusable Utilities from Common Modules

Based on Phase 3.6's implementation, Phase 7.3 will reuse:

### From `common_simplified/`:
```python
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.activation_hooks import ActivationExtractor
from common_simplified.helpers import (
    extract_code,
    evaluate_code,
    save_json,
    format_time,
    load_json
)
```

### From `common/`:
```python
from common.prompt_utils import PromptBuilder
from common.config import Config
from common.logging import get_logger
from common.utils import (
    detect_device,
    discover_latest_phase_output,
    ensure_directory_exists
)
from common.retry_utils import retry_with_timeout, create_exclusion_summary
```

### Phase 7.3 Specific Configuration:
```python
# Add to common/config.py
phase7_3_output_dir: str = "data/phase7_3"
phase7_3_model_name: str = "google/gemma-2-2b-it"
```

## Notes

### Multi-GPU Support
Phase 7.3 can potentially use multi-GPU launcher for faster processing:
```bash
python3 multi_gpu_launcher.py --phase 7.3 --model google/gemma-2-2b-it
```

### Memory Management
- Checkpoint every 50 tasks (same as Phase 3.6)
- Monitor RAM usage and warn at 85%
- Force garbage collection after checkpoints

### Error Handling
- Retry failed tasks with timeout protection
- Exclude consistently failing tasks
- Save exclusion summary for analysis

This specification provides a complete blueprint for implementing Phase 7.3 as the foundation for testing PVA feature universality in instruction-tuned models.