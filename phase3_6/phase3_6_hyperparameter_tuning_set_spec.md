# Phase 3.6: Hyperparameter Tuning Set Processing for PVA-SAE

## Executive Summary

1. **Purpose**: Generate activation data for hyperparameter split to enable threshold optimization in Phase 3.8
2. **Task**: Code generation at temperature 0.0 with best layers activation extraction for hyperparameter tuning problems
3. **Dataset**: 97 MBPP problems from hyperparameter split (10% of total dataset)
4. **Implementation**: Simplified version of Phase 3.5 focused on hyperparameter split processing
5. **Why Needed**: Phase 3.8 requires hyperparameter split activation data for F1-optimal threshold selection
6. **Development Support**: `--start` and `--end` arguments for processing subsets during development
7. **Key Output**: Activation files for best layers (correct and incorrect) to enable Phase 3.8 evaluation
8. **Temperature**: Single temperature (0.0) for deterministic code generation

## Pipeline Sequence

```
1. Load Hyperparameter Split from Phase 0.1
   └─> Read hyperparams_mbpp.parquet → Extract 97 problems → Prepare for processing

2. Discover Best Layers from Phase 3.5 Metadata
   └─> Read Phase 3.5 metadata.json → Extract best correct & incorrect layers → Setup activation extraction

3. Generate Code Solutions at Temperature 0.0
   a. Load model (google/gemma-2-2b) and setup activation hooks for best layers
   b. Process each hyperparameter problem with deterministic generation
   c. Extract activations from best layers during generation
   d. Evaluate generated code against test cases

4. Save Activation Data for Phase 3.8
   └─> Save activation files → Create metadata → Enable Phase 3.8 threshold optimization
```

## Phase Relationship

### Dependencies
- **Phase 0.1**: Provides hyperparameter split (hyperparams_mbpp.parquet with 97 problems)
- **Phase 3.5**: Provides best layers metadata (correct and incorrect feature layers)

### Enables
- **Phase 3.8**: Requires hyperparameter split activation data for F1-optimal threshold optimization

### Key Differences from Related Phases

#### vs Phase 1 (Dataset Building)
- **Scope**: Hyperparameter split (97) vs SAE split (489 problems)
- **Layers**: Best layers only vs all layers (0, 6, 8, 15, 17)
- **Purpose**: Threshold optimization data vs main SAE analysis dataset

#### vs Phase 3.5 (Temperature Robustness)
- **Data Source**: Hyperparameter split vs validation split
- **Temperature**: Single (0.0) vs multiple (0.3, 0.6, 0.9, 1.2)
- **Sample Size**: 97 vs 388 problems
- **Purpose**: Threshold optimization vs temperature robustness testing

#### vs Phase 3.8 (Evaluation)
- **Role**: Data generation vs data evaluation
- **Output**: Activation files vs AUROC/F1 metrics
- **Processing**: Code generation vs metric calculation

## Core Implementation

### Base Architecture
```python
# Reuse Phase 3.5's TemperatureRobustnessRunner as foundation
class HyperparameterDataRunner:
    """Simplified runner for hyperparameter split processing."""
    
    def __init__(self, config: Config):
        # Load model and discover best layers from Phase 3.5
        self.best_layers = self._discover_best_layers_from_phase3_5()
        self.extraction_layers = list(set([
            self.best_layers['correct'], 
            self.best_layers['incorrect']
        ]))
        
    def _discover_best_layers_from_phase3_5(self):
        """Read best layers from Phase 3.5 metadata.json"""
        # Auto-discover Phase 3.5 output and read metadata
        # Extract both correct and incorrect best layers
        pass
        
    def run(self):
        """Process hyperparameter split at temperature 0.0"""
        # Load hyperparameter data from Phase 0.1
        # Generate code solutions with activation extraction
        # Save activation files for Phase 3.8
        pass
```

### Key Simplifications from Phase 3.5
1. **Single Temperature**: Only temperature 0.0 (deterministic generation)
2. **Hyperparameter Split**: 97 problems instead of 388 validation problems
3. **Best Layers Only**: Extract from discovered best layers, not all layers
4. **No Temperature Variation**: Single generation per problem, not multiple samples

### Development Arguments
```bash
# Process first 10 problems for development/testing
python3 run.py phase 3.6 --end 9

# Process specific range
python3 run.py phase 3.6 --start 10 --end 19

# Process all hyperparameter problems (production)
python3 run.py phase 3.6
```

## Technical Specifications

### Input Requirements
1. **Phase 0.1 Output**: `data/phase0_1/hyperparams_mbpp.parquet`
   - Contains 97 hyperparameter tuning problems (10% of MBPP)
   - Includes task_id, text, test_list, cyclomatic_complexity columns

2. **Phase 3.5 Output**: `data/phase3_5/metadata.json`
   - Contains best_layers with correct and incorrect layer numbers
   - Contains best feature indices (correct_feature_idx, incorrect_feature_idx)

3. **Model Access**: google/gemma-2-2b
   - Same model used in Phase 1 and Phase 3.5
   - Requires GPU memory for inference and activation extraction

### Processing Configuration
```python
# Core settings
temperature = 0.0  # Deterministic generation only
samples_per_problem = 1  # Single generation per problem
extraction_layers = [best_correct_layer, best_incorrect_layer]  # From Phase 3.5
max_new_tokens = 2000  # Standard generation limit
```

### Output Structure
```
data/phase3_6/
├── dataset_hyperparams_temp_0_0.parquet    # Generated solutions and test results
├── metadata.json                           # Processing metadata and best layers info
└── activations/
    └── task_activations/                    # Activation files for Phase 3.8
        ├── task_001_layer_15.npz           # Best correct layer activations
        ├── task_001_layer_17.npz           # Best incorrect layer activations (if different)
        ├── task_002_layer_15.npz
        └── ...
```

### Memory and Performance
- **GPU Memory**: ~10-15GB for model + activation extraction
- **Processing Time**: ~2-3 minutes per problem (97 total = ~3-5 hours)
- **Storage**: ~500MB for activations + ~50MB for parquet data
- **Development Mode**: `--end 9` processes 10 problems in ~20-30 minutes

## Understanding the Data Flow

### From Phase 0.1 to Phase 3.6
```python
# Phase 0.1 creates hyperparameter split
hyperparams_data = pd.read_parquet('data/phase0_1/hyperparams_mbpp.parquet')
print(f"Hyperparameter problems: {len(hyperparams_data)}")  # Should be 97

# Phase 3.6 processes these problems
for _, problem in hyperparams_data.iterrows():
    # Generate code at temperature 0.0
    # Extract activations from best layers
    # Save for Phase 3.8 threshold optimization
```

### From Phase 3.5 to Phase 3.6
```python
# Phase 3.5 provides best layer information
phase3_5_metadata = json.load(open('data/phase3_5/metadata.json'))
best_layers = phase3_5_metadata['best_layers']

# Phase 3.6 uses this to extract from optimal layers
extraction_layers = [
    best_layers['correct'],      # e.g., layer 15
    best_layers['incorrect']     # e.g., layer 17
]
```

### From Phase 3.6 to Phase 3.8
```python
# Phase 3.6 creates activation data for hyperparameter split
# Phase 3.8 loads this data for threshold optimization
hyperparams_activations = load_split_activations(
    'hyperparams',                    # Split name
    best_layers['correct'],           # Layer number
    best_layers['correct_feature_idx'], # Feature index
    'correct'                         # Feature type
)
```

## Implementation Approach

### Strategy: Simplify and Reuse
1. **Base Class**: Copy Phase 3.5's TemperatureRobustnessRunner
2. **Simplify**: Remove temperature variation logic
3. **Adapt**: Change data source from validation to hyperparameter split
4. **Optimize**: Extract only from best layers (not all layers)

### Code Reuse from Phase 3.5
```python
# Reuse these components directly:
- Model loading and device detection
- Best layers discovery and deduplication logic (handles same/different layers elegantly)
- Activation extraction setup and hooks
- Code generation with activation capture
- Activation saving to .npz files
- Metadata creation and saving

# Modify these components:
- Data loading: hyperparams_mbpp.parquet vs validation_mbpp.parquet
- Temperature processing: single 0.0 vs multiple temperatures
- Layer selection: best layers vs all layers (but reuse Phase 3.5's deduplication approach)
```

### Development Workflow
```bash
# 1. Test with small sample
python3 run.py phase 3.6 --end 4  # Process 5 problems

# 2. Verify outputs
ls data/phase3_6/activations/task_activations/
cat data/phase3_6/metadata.json

# 3. Test Phase 3.8 compatibility
python3 run.py phase 3.8  # Should now work with hyperparameter data

# 4. Full hyperparameter processing
python3 run.py phase 3.6  # Process all 97 problems
```

## Key Implementation Details

### Best Layers Discovery
```python
def _discover_best_layers_from_phase3_5(self) -> Dict[str, int]:
    """
    Read best layers from Phase 3.5 metadata.
    
    Note: Phase 3.5 already handles the case where both best layers might be the same
    layer elegantly using set() for deduplication. We copy this exact approach.
    
    Returns:
        Dict with 'correct' and 'incorrect' layer numbers
    """
    # Auto-discover Phase 3.5 output directory
    phase3_5_output = discover_latest_phase_output("3.5")
    if not phase3_5_output:
        raise FileNotFoundError("Phase 3.5 metadata not found. Run Phase 3.5 first.")
    
    # Load metadata and extract best layers
    metadata_file = Path(phase3_5_output).parent / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    best_layers = metadata['best_layers']
    logger.info(f"Using best layers - Correct: {best_layers['correct']}, Incorrect: {best_layers['incorrect']}")
    
    return best_layers

def _setup_activation_extraction(self):
    """
    Setup activation extraction layers, handling same/different layer cases.
    
    This copies Phase 3.5's elegant approach for handling coincidental same layers.
    """
    # Determine unique layers to extract from (same logic as Phase 3.5)
    unique_layers = list(set([self.best_layers['correct'], self.best_layers['incorrect']]))
    self.extraction_layers = unique_layers
    
    if len(unique_layers) == 1:
        logger.info(f"Both correct and incorrect features use the same layer: {unique_layers[0]}")
    else:
        logger.info(f"Using different layers - Correct: {self.best_layers['correct']}, Incorrect: {self.best_layers['incorrect']}")
    
    # Initialize activation extractor for unique layers only
    self.activation_extractor = ActivationExtractor(
        self.model,
        layers=self.extraction_layers  # Extract from unique layers only
    )
```

### Hyperparameter Data Loading
```python
def _load_hyperparameter_data(self) -> pd.DataFrame:
    """Load hyperparameter split from Phase 0.1."""
    hyperparams_file = Path(self.config.phase0_1_output_dir) / "hyperparams_mbpp.parquet"
    
    if not hyperparams_file.exists():
        raise FileNotFoundError(
            f"Hyperparameter data not found at {hyperparams_file}. "
            "Please run Phase 0.1 first."
        )
    
    data = pd.read_parquet(hyperparams_file)
    logger.info(f"Loaded {len(data)} hyperparameter problems")
    
    return data
```

### Activation Extraction Simplified
```python
def generate_with_activations(self, prompt: str, task_id: str) -> Tuple[str, bool]:
    """Generate code and extract activations from best layers only."""
    # Setup hooks for best layers
    self.activation_extractor.setup_hooks()
    
    try:
        # Generate at temperature 0.0
        generated_text = self.generate_at_temperature(prompt, 0.0)
        generated_code = extract_code(generated_text, prompt)
        
        # Evaluate test cases
        test_passed = evaluate_code(generated_code, row['test_list'])
        
        # Get activations from best layers
        activations = self.activation_extractor.get_activations()
        
        # Save activations for this task
        self._save_task_activations(task_id, activations)
        
        return generated_code, test_passed
        
    finally:
        self.activation_extractor.remove_hooks()
```

## Implementation Checklist

### Setup Phase
- [ ] Create `phase3_6/` directory structure
- [ ] Create `hyperparameter_runner.py` based on Phase 3.5's TemperatureRobustnessRunner
- [ ] Add Phase 3.6 to `run.py` argument parser (choices=[0, 0.1, 1, 2.2, 2.5, 3, 3.5, 3.6, 3.8])
- [ ] Add Phase 3.6 validation to `common/config.py`
- [ ] Update `common/utils.py` PHASE_CONFIGS for Phase 3.6

### Core Implementation
- [ ] Implement `_discover_best_layers_from_phase3_5()` method
- [ ] Implement `_setup_activation_extraction()` method (copy Phase 3.5's elegant same/different layer handling)
- [ ] Implement `_load_hyperparameter_data()` method  
- [ ] Simplify activation extraction to best layers only
- [ ] Remove temperature variation logic (single temperature 0.0)
- [ ] Add support for `--start` and `--end` arguments from config

### Data Processing
- [ ] Load hyperparameter split (97 problems) from Phase 0.1
- [ ] Generate code solutions at temperature 0.0 with activation extraction
- [ ] Save activation files compatible with Phase 3.8 expectations
- [ ] Create metadata.json with best layers and processing info

### Integration and Testing
- [ ] Test with small sample: `python3 run.py phase 3.6 --end 4`
- [ ] Verify activation files are created correctly
- [ ] Test Phase 3.8 compatibility with generated data
- [ ] Run full hyperparameter processing: `python3 run.py phase 3.6`

### Validation
- [ ] Ensure output structure matches Phase 3.8 expectations
- [ ] Verify activation file format (task_id_layer_N.npz with 'arr_0')
- [ ] Check metadata.json contains required best_layers information
- [ ] Confirm dataset parquet has test_passed column for labels

## Data Dependencies

### Input Requirements
Phase 3.6 requires the following completed phases:

1. **Phase 0.1**: Problem Splitting
   - `data/phase0_1/hyperparams_mbpp.parquet` - 97 hyperparameter tuning problems
   - Contains: task_id, text, test_list, cyclomatic_complexity columns

2. **Phase 3.5**: Temperature Robustness Testing  
   - `data/phase3_5/metadata.json` - Contains best layers for correct and incorrect features
   - Required fields: best_layers.correct, best_layers.incorrect, best_layers.correct_feature_idx, best_layers.incorrect_feature_idx

3. **Model Access**: google/gemma-2-2b from HuggingFace
   - Same model used across all phases for consistency

### Output Guarantees
Phase 3.6 produces outputs that enable:

1. **Phase 3.8**: AUROC and F1 Evaluation
   - Activation files for hyperparameter split threshold optimization
   - Compatible data format with Phase 3.8 expectations
   - Best layers activation data for both correct and incorrect features

## Why Phase 3.6 is Essential

### The Missing Link
```
Phase 0.1 → Phase 3.6 → Phase 3.8
  (splits)   (hyperparams data)   (threshold optimization)

Phase 2.5 → Phase 3.6 → Phase 3.8  
(best layers)  (activation extraction)  (evaluation)

Phase 3.5 → Phase 3.8
(validation data)  (final evaluation)
```

### Without Phase 3.6
- Phase 3.8 cannot find F1-optimal thresholds (no hyperparameter split activation data)
- Threshold optimization fails: "Loaded 0 samples for hyperparams split"
- Evaluation pipeline is incomplete

### With Phase 3.6
- Phase 3.8 gets hyperparameter split activation data for threshold optimization
- F1-optimal thresholds can be calculated on hyperparameter split (97 problems)
- Final evaluation can proceed on validation split (388 problems)
- Complete bidirectional PVA feature evaluation becomes possible

## Reusable Utilities from Common Modules

Based on Phase 3.5's imports and following KISS/DRY principles, Phase 3.6 can reuse:

### From `common_simplified/` (Core Processing):
```python
# Model loading and management
from common_simplified.model_loader import load_model_and_tokenizer

# Activation extraction during generation  
from common_simplified.activation_hooks import ActivationExtractor

# Code evaluation and data utilities
from common_simplified.helpers import (
    extract_code,           # Extract code from generated text
    evaluate_code,          # Test generated code against test cases
    load_mbpp_from_phase0_1, # Load hyperparameter split data
    save_json,              # Save metadata
    format_time             # Human-readable timing
)
```

### From `common/` (Infrastructure):
```python
# Configuration and constants
from common.config import Config
from common import MAX_NEW_TOKENS, DEFAULT_MODEL_NAME

# Logging with phase-aware context
from common.logging import get_logger

# Prompt building for code generation
from common.prompt_utils import PromptBuilder

# Device detection and utilities
from common.utils import (
    detect_device,                  # Auto-detect CUDA/MPS/CPU
    discover_latest_phase_output,   # Find Phase 3.5 metadata
    ensure_directory_exists,        # Create output directories
    get_timestamp,                  # File naming timestamps
    torch_memory_cleanup,           # Memory management context manager
    torch_no_grad_and_cleanup      # Combined no_grad + cleanup
)

# GPU management (inherited from Phase 3.5 usage)
from common.gpu_utils import cleanup_gpu_memory
```

### What Phase 3.5 Uses (Our Template):
Based on Phase 3.5's actual imports:
```python
# Core model and activation handling
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.activation_hooks import ActivationExtractor  
from common_simplified.helpers import evaluate_code, extract_code

# Infrastructure
from common.prompt_utils import PromptBuilder
from common.config import Config
from common.logging import get_logger
from common.utils import detect_device, discover_latest_phase_output
```

### Proven Patterns from Phase 3.5:
1. **Model Loading**: `load_model_and_tokenizer(model_name, device)`
2. **Activation Hooks**: `ActivationExtractor(model, layers).setup_hooks()`
3. **Code Generation**: Temperature-based generation with evaluation
4. **Memory Management**: Cleanup after processing batches
5. **Progress Tracking**: tqdm with detailed logging
6. **Auto-discovery**: Find Phase outputs automatically

### Phase 3.6 Specific Additions:
```python
# Additional utilities we need to implement
def _discover_best_layers_from_phase3_5() -> Dict[str, int]:
    """Discover best layers from Phase 3.5 metadata (copy exact pattern)"""
    
def _load_hyperparameter_data() -> pd.DataFrame:
    """Load hyperparameter split from Phase 0.1 (copy Phase 3.5 validation loading pattern)"""
    
def _save_task_activations(task_id: str, activations: Dict[int, torch.Tensor]):
    """Save activations per task (copy Phase 3.5 exact approach)"""
```

### KISS Principle Application:
- **Don't Reinvent**: Copy Phase 3.5's working patterns exactly
- **Reuse Proven Code**: Use same utilities that work in Phase 3.5
- **Simplify**: Remove only temperature variation, keep all other logic
- **DRY**: Leverage all existing common utilities instead of reimplementing

## Development Notes

### For Testing and Development
- Use `--end 9` to process only 10 problems (~20-30 minutes)
- Verify activation files are created before running full dataset
- Check Phase 3.8 compatibility with small sample before production run
- Monitor GPU memory usage during development

### For Production
- Process all 97 hyperparameter problems (~3-5 hours)
- Ensure sufficient storage space (~500MB for activations)
- Run Phase 3.8 immediately after to verify complete pipeline
- Consider multi-GPU support if processing time becomes a bottleneck

This specification provides a complete blueprint for implementing Phase 3.6 as the essential bridge between Phase 0.1/2.5/3.5 and Phase 3.8, enabling complete bidirectional PVA feature evaluation.