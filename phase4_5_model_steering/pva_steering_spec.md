# Phase 4.5: Steering Coefficient Selection Specification

## Executive Summary

1. **Purpose**: Find optimal steering coefficients for PVA features through empirical grid search
2. **Method**: Add SAE decoder directions to residual stream activations during generation
3. **Steering Coefficients**: Grid search [1, 3, 10, 30, 100, 300, 1000] with manual evaluation
4. **Application**: Continuous steering throughout generation process
5. **Evaluation**: Flip rate (pass/fail changes) and generation divergence metrics
6. **Dataset**: MBPP hyperparameter tuning split from Phase 3.6 baseline data (10 problems for search)
7. **Baseline**: Pre-generated results from Phase 3.6 (temperature 0.0, no steering)
8. **Output**: Selected coefficients saved for Phase 4.8 comprehensive analysis

## Pipeline Sequence

```
1. Load dependencies and features
   └─> Load best features from Phase 2.5 → Load baseline data from Phase 3.6 → Initialize SAEs

2. Prepare MBPP prompts and baseline
   └─> Load pre-generated baseline from Phase 3.6 (includes prompts and results) → Select problems

3. Set up steering hooks
   └─> Target residual stream (pre-hook) → Add direction * coefficient → Apply during generation

4. Generate steered outputs
   └─> Steered generation → Execute tests → Compare against Phase 3.6 baseline

5. Evaluate steering effect
   └─> Calculate flip rates → Measure generation divergence → Plot coefficient effects → Save selected coefficients
```

## Understanding Model Steering in PVA Context

### Steering Mechanism
The system modifies model activations by adding SAE decoder directions to the residual stream:
- **Direction**: Decoder vector from SAE corresponding to PVA features (correct/incorrect-preferring)
- **Coefficient**: Scalar multiplier controlling steering strength
- **Position**: Applied continuously to all positions (no specific targeting needed)
- **Timing**: Applied throughout the entire generation process
- **Processing**: Single prompt at a time (no batching) for consistent results

### Steering Targets
- **Incorrect → Correct**: Make model generate more correct code (using correct-preferring features)
- **Correct → Incorrect**: Make model generate more buggy code (using incorrect-preferring features)

## Core Implementation

### Required Imports
```python
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable
from tqdm import tqdm

from common.prompt_utils import PromptBuilder
from common.logging import get_logger
from common.utils import discover_latest_phase_output
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase4_5.steering_evaluator")
```

### 1. Steering Hook Implementation

```python
def create_steering_hook(sae_decoder_direction: torch.Tensor, 
                        coefficient: float) -> Callable:
    """
    Create a hook that adds SAE decoder direction to residual stream.
    
    Args:
        sae_decoder_direction: Decoder vector from SAE [d_model]
        coefficient: Scalar multiplier for steering strength
    
    Returns:
        Hook function for forward_pre_hook
    """
    def hook_fn(module, input):
        # input[0] is residual stream: [1, seq_len, d_model]
        residual = input[0]
        
        # Add steering vector scaled by coefficient to all positions
        steering = sae_decoder_direction.unsqueeze(0).unsqueeze(0) * coefficient
        residual = residual + steering
        
        return (residual,) + input[1:]
    
    return hook_fn
```

### 2. Class-Based Implementation Structure

```python
class SteeringCoefficientSelector:
    """Select optimal steering coefficients through grid search."""
    
    def __init__(self, config: Config):
        """Initialize with configuration, load dependencies."""
        # Load model, PVA features from Phase 2.5, baseline from Phase 3.6
        
    def _load_dependencies(self) -> None:
        """Load features from Phase 2.5 and baseline data from Phase 3.6."""
        # Similar to Phase 3.5's _discover_best_layers()
        
    def _create_steering_hook(self, decoder_direction: torch.Tensor, 
                            coefficient: float) -> Callable:
        """Create steering hook for residual stream modification."""
        # Returns hook function that adds decoder_direction * coefficient
        
    def evaluate_coefficient(self, coefficient: float, 
                           problems_subset: List[Dict]) -> Dict:
        """Evaluate a single coefficient on subset of problems."""
        # For each problem: apply steering, generate, evaluate
        # Return flip rate and divergence metrics
        
    def run(self) -> Dict:
        """Run coefficient grid search and save results."""
        # Main loop over coefficients [1, 3, 10, 30, 100, 300, 1000]
        # Save selected coefficients for Phase 4.8
```

### 3. Evaluation Metrics

```python
def calculate_flip_rate(results: List[Dict]) -> float:
    """
    Calculate percentage of problems where steering changed pass/fail outcome.
    """
    flipped = sum(1 for r in results if r['flipped'])
    return (flipped / len(results)) * 100

def calculate_generation_divergence(results: List[Dict]) -> Dict:
    """
    Measure how different steered generations are from baseline.
    Uses SequenceMatcher to calculate similarity at token and character levels.
    
    Returns:
        Dict with mean similarity metrics
    """
    # Calculate token-level and character-level similarities
    # Return mean_token_similarity, mean_char_similarity, mean_length_ratio
```

## Implementation References

### Reuse from Existing Modules

**From `common_simplified/`:**
- `model_loader.load_model_and_tokenizer()` - Model initialization
- `helpers.evaluate_code()` - Test generated code against test cases
- `helpers.extract_code()` - Extract code from model output
- `helpers.save_json()`, `helpers.load_json()` - Save/load results

**From `common/`:**
- `prompt_utils.PromptBuilder` - Build MBPP prompts
- `logging.get_logger()` - Module logging
- `utils.discover_latest_phase_output()` - Find previous phase outputs
- `config.Config` - Configuration management (includes phase4_5_coefficients and phase4_5_problems_per_coeff)

**From `phase2_5_simplified/`:**
- `sae_analyzer.load_gemma_scope_sae()` - Load SAE models
- Pattern for loading top features from JSON

**From `phase3_5/`:**
- Class structure pattern with `__init__` and `run()` methods
- Pattern for discovering best layers/features from previous phases
- Single-prompt processing approach

**From `phase3_6/`:**
- Pattern for loading baseline parquet data
- Structure for saving results and metadata

### Key Implementation Notes

1. **Steering Hook**: Apply to residual stream using `register_forward_pre_hook()` at the layer identified as best in Phase 2.5 (similar to Phase 3.5's approach)
2. **Continuous Steering**: Apply to all positions (no position mask needed)
3. **Single Prompt**: Process one problem at a time (no batching)
4. **Memory Management**: Clear GPU cache periodically
5. **Deterministic Generation**: Use temperature=0.0, do_sample=False

## Experimental Configuration

### Dataset Setup
```python
# Load baseline results from Phase 3.6 (includes all MBPP data and pre-generated results)
phase3_6_output = discover_latest_phase_output('3.6')
baseline_data = pd.read_parquet(Path(phase3_6_output).parent / 'dataset_hyperparams_temp_0_0.parquet')

# baseline_data already contains:
# - task_id: MBPP task identifier
# - text: Problem description  
# - test_list: Test cases
# - prompt: Pre-formatted prompt from Phase 3.6
# - generated_code: Baseline generated code at temperature 0.0
# - test_passed: Whether baseline code passed tests
# - cyclomatic_complexity: Difficulty metric

# Example prompt already in data:
# """Problem description..."""
# def function_name(args):
```

### Steering Parameters

#### Coefficient Selection Methodology
The coefficient values follow a **logarithmic progression** with approximately 3x increases between consecutive values. This systematic approach:

- **Covers multiple orders of magnitude**: From subtle steering (1) to extreme effects (1000)
- **Maintains methodological rigor**: Each step represents a meaningful scaling factor
- **Enables efficient exploration**: Balances comprehensive coverage with computational efficiency
- **Follows hyperparameter optimization best practices**: Log-scale search is standard for wide parameter ranges

The progression `[1, 3, 10, 30, 100, 300, 1000]` systematically explores:
- **1-10**: Minimal to moderate steering (likely optimal range)
- **30-100**: Strong steering (quality vs. effect tradeoffs)  
- **300-1000**: Extreme steering (saturation and failure mode testing)

#### Configuration Management
These coefficient values and the number of problems per coefficient (10) will be stored in `common/config.py` following the project's centralized configuration pattern:

```python
# === STEERING COEFFICIENT SELECTION (Phase 4.5) ===
phase4_5_coefficients: List[float] = field(default_factory=lambda: [1, 3, 10, 30, 100, 300, 1000])
phase4_5_problems_per_coeff: int = 10
phase4_5_output_dir: str = "data/phase4_5"
```

This approach allows for easy modification without code changes and supports environment variable overrides like other phase configurations.

#### Parameter Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `coefficients` | [1, 3, 10, 30, 100, 300, 1000] | Grid search values (logarithmic progression) |
| `processing` | Sequential | Single prompt at a time (no batching) |
| `n_examples_per_coeff` | 10 | Examples for coefficient selection (70 total evaluations) |
| `n_final_eval` | 100 | Problems for final evaluation |
| `max_new_tokens` | 500 | Maximum tokens to generate |

### Evaluation Conditions
1. **Baseline**: Pre-generated results from Phase 3.6 (temperature 0.0, no steering)
2. **Correct-steering**: Steer with the single best correct-preferring feature from Phase 2.5
3. **Incorrect-steering**: Steer with the single best incorrect-preferring feature from Phase 2.5

Note: We use exactly two features total - the single best feature for each direction (correct/incorrect) as identified in Phase 2.5's top features output.

## Implementation Workflow

### Step 1: Initialize SteeringCoefficientSelector
1. Load configuration and initialize model
2. Discover and load PVA features from Phase 2.5 output
3. Load baseline results from Phase 3.6 parquet file
4. Initialize SAEs for both correct and incorrect features

### Step 2: Run Coefficient Grid Search
1. For each coefficient in [1, 3, 10, 30, 100, 300, 1000]:
   - Select 10 problems from Phase 3.6 baseline data using stratified selection to ensure difficulty diversity
   - For each problem:
     - Get baseline results from Phase 3.6 data
     - Apply steering hook with current coefficient
     - Generate code with steering
     - Compare against baseline (flip rate)
   - Calculate metrics (flip rate, generation divergence)
   - Save examples for manual inspection

Note: Problem selection uses stratified sampling based on cyclomatic complexity to ensure we test across different difficulty levels.

### Step 3: Manual Coefficient Selection
1. Review flip rates and divergence metrics for each coefficient
2. Examine generated code examples for quality
3. Select optimal coefficients balancing:
   - Meaningful flip rate (20-60%)
   - Code quality preservation
   - Avoid syntax corruption

### Step 4: Save Results
1. Save selected coefficients to JSON:
   - Coefficient values for correct/incorrect features
   - Selection rationale
   - Timestamp and phase info
2. Create phase summary with all analysis results
3. Generate visualization plots (optional)

## Manual Coefficient Selection Guidelines

### What to Look For

Given the logarithmic coefficient progression, expect different behaviors across ranges:

1. **Flip Rate Patterns by Coefficient Range**
   - **Coefficients 1-10**: Low flip rates (5-25%), subtle effects
   - **Coefficients 30-100**: Moderate flip rates (20-60%), likely optimal range
   - **Coefficients 300-1000**: High flip rates (50-90%+), potential code breaking
   - **Ideal Selection**: Balance flip rate with code quality preservation

2. **Code Quality Indicators**
   - **Low coefficients (1-10)**: Expect minimal structural changes
   - **Medium coefficients (30-100)**: Semantic changes, preserved syntax
   - **High coefficients (300-1000)**: Risk of syntax errors, structural corruption
   - Verify function signatures remain intact across all ranges

3. **Generation Coherence by Range**
   - **1-10**: High similarity (>0.8), subtle modifications
   - **30-100**: Moderate similarity (0.4-0.8), meaningful changes
   - **300-1000**: Low similarity (<0.5), major alterations or failures
   - Track where steering effects saturate vs. continue scaling

### Example Analysis Output
```
Coefficient: 30
Flip Rate: 42.0%
Examples of flips:
  - Task 001: Pass→Fail (logic error introduced)
  - Task 005: Fail→Pass (edge case handled)
  - Task 012: Pass→Fail (syntax error in loop)

Generation Quality:
  - Average token similarity: 0.62
  - Average character similarity: 0.71
  - Most changes: variable names, operators, control flow

Coefficient: 300
Flip Rate: 78.0%
Examples of flips:
  - Task 003: Pass→Fail (syntax corruption)
  - Task 007: Fail→Pass (accidental fix)
  - Task 014: Pass→Fail (function signature broken)

Generation Quality:
  - Average token similarity: 0.31
  - Average character similarity: 0.45
  - Most changes: structural corruption, indentation errors
```

## Key Insights for Coefficient Selection

1. **Feature Effectiveness**: Look for meaningful flip rates without breaking code structure
2. **Asymmetric Coefficients**: Different features may need different steering strengths
3. **Quality vs. Effect**: Balance between steering effect and maintaining code coherence
4. **Manual Review**: Human judgment needed to select optimal coefficients

## Output Files

- `selected_coefficients.json`: Chosen coefficients for Phase 4.8
- `coefficient_examples/`: Example generations for each coefficient
- `coefficient_analysis_plots.png`: Visualization of coefficient effects
- `phase_4_5_summary.json`: Complete summary of this phase

## Implementation Checklist

### Setup
- [ ] Load Phase 2.5 outputs for best PVA features
- [ ] Load baseline results from Phase 3.6 (includes all data and pre-generated results)
- [ ] Initialize SAEs and extract decoder directions

### Coefficient Search
- [ ] Implement steering hooks for generation
- [ ] Run grid search on 10 problems per coefficient (7 coefficients × 10 = 70 total)
- [ ] Calculate flip rates and divergence metrics
- [ ] Save examples for manual inspection
- [ ] Select optimal coefficients

### Output
- [ ] Save selected coefficients to JSON file
- [ ] Create phase summary with rationale
- [ ] Document coefficient selection process
- [ ] Prepare outputs for Phase 4.8

### Analysis
- [ ] Compare steering effects (flip rates, divergence)
- [ ] Analyze failure modes and success patterns
- [ ] Document generation quality observations
- [ ] Create final summary report

## Implementation Decisions

This section consolidates key implementation decisions for clarity:

1. **Layer Targeting**: Steering hooks are applied at the specific layer identified as best in Phase 2.5 for each feature type. This follows the same pattern as Phase 3.5, which discovers and uses the best layer from previous phase outputs.

2. **Problem Selection**: The 10 problems per coefficient are selected using stratified sampling based on cyclomatic complexity. This ensures we test steering effects across different difficulty levels, providing a more comprehensive evaluation.

3. **Feature Selection**: We use exactly two features - the single best correct-preferring feature and the single best incorrect-preferring feature from Phase 2.5's top features output. This focused approach allows for cleaner interpretation of steering effects.

## CLI Support

This phase integrates with the project's run.py system and supports:

```bash
python3 run.py phase 4.5
```

The implementation will follow existing project patterns and integrate with the centralized configuration and logging systems in `common/config.py` and `common/utils.py`.

## Notes

- **Efficiency Gain**: Uses pre-generated baseline from Phase 3.6, eliminating need for baseline generation
- **Data Consistency**: Hyperparameter split aligns with Phase 3.6 processing for perfect data alignment
- **Generation Time**: Steering during generation is slower than classification, but baseline reuse speeds up process
- **Deterministic Generation**: Use temperature=0.0 for reproducibility (matches Phase 3.6)
- **Memory Management**: Clear GPU cache between runs
- **Error Handling**: Some steered code may fail to parse - track these cases
- **Continuous Steering**: Apply throughout generation, not just at start
- **Single Prompt Processing**: Process one problem at a time for memory efficiency and consistency with other phases