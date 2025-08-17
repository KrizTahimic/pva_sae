# Phase 5.3: Weight Orthogonalization Analysis Specification

## Executive Summary

1. **Purpose**: Permanently modify neural network weights to remove Program Validity Awareness (PVA) information
2. **Method**: Project out SAE decoder directions from Gemma-2 weight matrices
3. **Target Weights**: Embedding, attention output projections, MLP down projections
4. **Application**: Study PVA feature necessity through permanent weight modification
5. **Comparison**: Alternative to Phase 4.8's temporary steering hooks
6. **Dataset**: Phase 3.5 validation data for consistency with steering experiments
7. **Statistical Validation**: Binomial tests to verify orthogonalization effects
8. **Input**: Phase 2.5 PVA features, Phase 3.5 baseline data, Gemma-2 2B model
9. **Output**: Statistical analysis comparing permanent vs temporary interventions

## Mathematical Foundation

### Core Formula
The orthogonalization removes the projection of weight matrix **W** onto direction **d**:

```
W_orthogonalized = W - ((W @ d) / ||d||²) × d^T
```

Where:
- **W**: Weight matrix to orthogonalize
- **d**: Direction vector (typically SAE decoder direction)
- **@**: Matrix multiplication
- **||d||²**: Squared L2 norm of direction

### Geometric Interpretation
- Removes the component of each weight vector that aligns with the target direction
- Preserves all orthogonal components unchanged
- Results in weights that cannot represent information along the removed direction

## Pipeline Sequence

```
1. Load dependencies
   └─> Phase 2.5 PVA features → Phase 3.5 baseline data → Initialize model

2. Extract decoder directions
   └─> Best correct feature → Best incorrect feature → Load SAE decoders

3. Apply orthogonalization
   └─> Backup original weights → Orthogonalize for correct → Test and restore → Orthogonalize for incorrect

4. Evaluate effects
   └─> Generate with modified weights → Calculate metrics → Statistical validation

5. Compare with steering
   └─> Load Phase 4.8 results → Generate comparison → Visualize differences
```

## Core Implementation

### 1. Generic Orthogonalization Function

**File**: `common/weight_utils.py` (new file)

```python
import torch
import einops
from typing import Tuple
from torch import Tensor, FloatTensor

def get_orthogonalized_matrix(matrix: FloatTensor, vec: FloatTensor) -> FloatTensor:
    """
    Remove projection of matrix rows onto direction vector.
    
    Args:
        matrix: Weight matrix to orthogonalize [..., d_model]
        vec: Direction to project out [d_model]
    
    Returns:
        Orthogonalized matrix with same shape as input
    """
    vec = vec / torch.norm(vec)  # Normalize direction
    vec = vec.to(matrix.device).to(matrix.dtype)  # Match device/dtype
    
    # Compute and subtract projection
    proj = einops.einsum(matrix, vec.unsqueeze(-1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj
```

### 2. Model-Specific Implementation

**File**: `common_simplified/weight_orthogonalization.py` (new file)

```python
import torch
from transformers import AutoModelForCausalLM
from common.weight_utils import get_orthogonalized_matrix
from common.logging import get_logger
from typing import Dict, List

logger = get_logger("weight_orthogonalization")

def orthogonalize_gemma_weights(
    model: AutoModelForCausalLM, 
    direction: torch.Tensor,
    target_weights: List[str] = None
) -> Dict[str, float]:
    """
    Orthogonalize Gemma-2 model weights along PVA direction.
    
    Args:
        model: HuggingFace Gemma model
        direction: PVA feature decoder direction [d_model]
        target_weights: List of weight types to orthogonalize
                       (default: ['embed', 'attn_o', 'mlp_down'])
    
    Returns:
        Dictionary of weight change magnitudes for verification
    """
    if target_weights is None:
        target_weights = ['embed', 'attn_o', 'mlp_down']
    
    changes = {}
    
    # Embedding weights (vocab_size, d_model)
    if 'embed' in target_weights:
        original = model.model.embed_tokens.weight.data.clone()
        model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
            model.model.embed_tokens.weight.data, direction
        )
        change = torch.norm(model.model.embed_tokens.weight.data - original).item()
        changes['embedding'] = change
        logger.info(f"Orthogonalized embeddings, change magnitude: {change:.4f}")
    
    # Process each transformer layer
    for i, block in enumerate(model.model.layers):
        # Attention output projection - transposed storage
        if 'attn_o' in target_weights:
            original = block.self_attn.o_proj.weight.data.clone()
            block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
                block.self_attn.o_proj.weight.data.T, direction
            ).T
            change = torch.norm(block.self_attn.o_proj.weight.data - original).item()
            changes[f'layer_{i}_attn_o'] = change
        
        # MLP down projection - transposed storage
        if 'mlp_down' in target_weights:
            original = block.mlp.down_proj.weight.data.clone()
            block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
                block.mlp.down_proj.weight.data.T, direction
            ).T
            change = torch.norm(block.mlp.down_proj.weight.data - original).item()
            changes[f'layer_{i}_mlp_down'] = change
    
    total_change = sum(changes.values())
    logger.info(f"Total weight change magnitude: {total_change:.4f}")
    
    return changes
```

## Weight Matrix Targets

### Targeted Weights
| Component | Matrix | Shape | Purpose |
|-----------|--------|-------|---------|
| **Embeddings** | `embed_tokens.weight` | `[vocab_size, d_model]` | Token-to-vector mapping |
| **Attention** | `self_attn.o_proj.weight` | `[d_model, d_model]` | Attention output projection |
| **MLP** | `mlp.down_proj.weight` | `[d_intermediate, d_model]` | MLP output projection |

### Why These Weights?
- **Embeddings**: Control initial representation of tokens
- **Attention Output**: Affect how attention patterns influence residual stream
- **MLP Output**: Control how MLP computations contribute to residual stream

### Weights NOT Orthogonalized
- Attention K, Q, V projections
- MLP up projections and gate projections
- Layer norms and unembedding

## Implementation Approach

### Key Design Decisions

1. **Mirror Phase 4.8 Structure**: Follow similar experimental design for direct comparison
2. **Reuse Existing Infrastructure**: Leverage common modules for consistency
3. **Fresh Model Instances**: Reload model for each orthogonalization (weights permanently modified)
4. **Statistical Rigor**: Use same metrics and tests as steering experiments

### Reusable Components

#### From `common/`:
- `from common.prompt_utils import PromptBuilder` - For prompt generation
- `from common.logging import get_logger` - For logging
- `from common.utils import discover_latest_phase_output, ensure_directory_exists, detect_device` - For utilities
- `from common.config import Config` - For configuration
- `from common.steering_metrics import calculate_correction_rate, calculate_corruption_rate` - For metrics

#### From `common_simplified/`:
- `from common_simplified.model_loader import load_model_and_tokenizer` - For model setup
- `from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json` - For code evaluation

#### From Other Phases:
- `from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae` - For loading SAE models
- `from phase4_8_steering_analysis.steering_effect_analyzer import SteeringEffectAnalyzer` - For baseline comparison

#### Additional Required Imports:
- `import torch` - For tensor operations
- `import pandas as pd` - For data manipulation
- `import numpy as np` - For numerical operations
- `from scipy.stats import binomtest` - For statistical testing
- `import matplotlib.pyplot as plt` - For plotting
- `from pathlib import Path` - For path handling
- `from tqdm import tqdm` - For progress bars
- `import json` - For JSON operations
- `import copy` - For model state backup
- `from datetime import datetime` - For timestamps

## Class Structure

Create a `WeightOrthogonalizer` class similar to Phase 4.8's `SteeringEffectAnalyzer`:

```python
class WeightOrthogonalizer:
    """Analyze effects of permanent weight orthogonalization on PVA."""
    
    def __init__(self, config: Config):
        """Initialize with configuration, load dependencies."""
        self.config = config
        self.device = detect_device()
        self.output_dir = Path(config.phase5_3_output_dir)
        ensure_directory_exists(self.output_dir)
        
        # Load dependencies
        self._load_dependencies()
    
    def _load_dependencies(self):
        """Load PVA features and baseline data."""
        # Load Phase 2.5 features
        # Load Phase 3.5 baseline data
        # Initialize SAE models
    
    def _backup_model_state(self, model):
        """Create backup of original weights."""
        return {name: param.clone() for name, param in model.named_parameters()}
    
    def _restore_model_state(self, model, state_dict):
        """Restore model to original weights."""
        for name, param in model.named_parameters():
            param.data = state_dict[name]
    
    def apply_orthogonalization_and_evaluate(self, feature_type='correct'):
        """Apply orthogonalization and measure effects."""
        # Load fresh model
        # Apply orthogonalization
        # Generate and evaluate
        # Calculate metrics
        # Return results
    
    def compare_with_steering(self):
        """Compare orthogonalization with Phase 4.8 steering."""
        # Load Phase 4.8 results
        # Generate comparison metrics
        # Create visualization
    
    def run(self):
        """Main execution pipeline."""
        # Test correct orthogonalization
        # Test incorrect orthogonalization
        # Statistical validation
        # Compare with steering
        # Save results and visualizations
```

## Configuration Entries

Add to `common/config.py`:

```python
# === WEIGHT ORTHOGONALIZATION (Phase 5.3) ===
phase5_3_output_dir: str = "data/phase5_3"
orthogonalization_target_weights: List[str] = field(
    default_factory=lambda: ['embed', 'attn_o', 'mlp_down']
)
```

## Experimental Usage

### PVA Orthogonalization Workflow

```python
# 1. Load PVA features from Phase 2.5
top_features = load_json("data/phase2_5/top_20_features.json")
best_correct_feature = top_features['correct'][0]
best_incorrect_feature = top_features['incorrect'][0]

# 2. Load SAE and extract decoder direction
sae = load_gemma_scope_sae(
    layer=best_correct_feature['layer'],
    width="16k",
    device=device
)
decoder_direction = sae.W_dec[best_correct_feature['feature_idx']]

# 3. Load fresh model for orthogonalization
model, tokenizer = load_model_and_tokenizer(
    "google/gemma-2-2b",
    device=device
)

# 4. Apply orthogonalization
from common_simplified.weight_orthogonalization import orthogonalize_gemma_weights
weight_changes = orthogonalize_gemma_weights(
    model, 
    decoder_direction,
    target_weights=['embed', 'attn_o', 'mlp_down']
)

# 5. Evaluate on validation data
validation_data = pd.read_parquet("data/phase3_5/dataset_temp_0_0.parquet")
for idx, row in validation_data.iterrows():
    # Generate with orthogonalized model
    prompt = row['prompt']
    output = model.generate(...)
    
    # Evaluate correctness
    code = extract_code(output)
    passed = evaluate_code(code, row['tests'])
    
    # Compare with baseline
    # ... analysis ...
```
```

## Key Implementation Details

### 1. Transposed Weight Handling
HuggingFace linear layers store weights transposed:
```python
# Mathematical: output = input @ W^T + bias
# Storage: W.shape = [out_features, in_features]
# Therefore, orthogonalize then transpose back
weight.data = get_orthogonalized_matrix(weight.data.T, direction).T
```

### 2. Direction Normalization
Direction normalized to unit length for numerical stability:
```python
vec = vec / torch.norm(vec)
```

### 3. Device/Dtype Matching
Direction cast to match weight matrix:
```python
vec = vec.to(matrix.device).to(matrix.dtype)
```

### 4. Model Reloading Pattern
Models must be reloaded after orthogonalization (changes are permanent):
```python
# For each experiment:
model, tokenizer = load_model_and_tokenizer(config.model_name)
# Apply orthogonalization
# Run evaluation
# Model cannot be reused - reload for next experiment
```

## Comparison with Steering (Phase 4.8)

| Aspect | Weight Orthogonalization (Phase 5.3) | Activation Steering (Phase 4.8) |
|--------|--------------------------------------|----------------------------------|
| **Permanence** | Permanent weight modification | Temporary hook-based |
| **Scope** | Affects all tokens/positions | Can target specific positions |
| **Reversibility** | Requires model reload | Remove hooks to revert |
| **Strength Control** | Binary (applied or not) | Adjustable coefficient (e.g., 30.0) |
| **Memory Usage** | No runtime overhead | Hook overhead per forward pass |
| **Use Case** | Feature necessity/ablation | Dynamic behavioral control |
| **Implementation** | Modifies model.state_dict() | Registers forward_pre_hook |
| **Batch Processing** | No per-sample variation | Can vary per batch/sample |

## Expected Outcomes

### Success Criteria
- **Weight Modification**: Measurable Frobenius norm changes in target weights
- **Behavioral Change**: >5% change in correction/corruption rates vs baseline
- **Statistical Significance**: p-values < 0.05 for orthogonalization effects
- **Comparison**: Clear differences vs Phase 4.8 steering approach

### Validation
The presence of statistically significant orthogonalization effects validates:
1. PVA features are encoded in model weights, not just activations
2. Permanent weight modification is viable for feature ablation
3. Comparison with steering reveals temporal vs permanent intervention tradeoffs

## Output Files

```
data/phase5_3/
├── orthogonalization_results.json      # Detailed metrics and statistics
├── comparison_with_steering.json       # Phase 4.8 vs 5.3 comparison
├── phase_5_3_summary.json             # Executive summary
├── weight_changes.json                 # Magnitude of weight modifications
├── visualizations/
│   ├── correction_rates_comparison.png # Orthogonalization vs steering
│   ├── weight_change_heatmap.png      # Weight modification magnitudes
│   └── statistical_significance.png    # P-values and confidence intervals
└── examples/
    ├── correct_orthogonalized/        # Examples with correct feature removed
    │   ├── baseline_correct.json      # Originally correct becoming incorrect
    │   └── baseline_incorrect.json    # Originally incorrect (no improvement)
    └── incorrect_orthogonalized/      # Examples with incorrect feature removed
        ├── baseline_correct.json      # Originally correct (preserved)
        └── baseline_incorrect.json    # Originally incorrect becoming correct
```

## Implementation Checklist

### Prerequisites
- [ ] Complete Phase 2.5 (PVA feature identification)
- [ ] Complete Phase 3.5 (validation baseline data)
- [ ] Complete Phase 4.8 (steering comparison baseline)
- [ ] Verify GPU memory for model copies

### Implementation Steps
- [ ] Create `common/weight_utils.py` with orthogonalization function
- [ ] Create `common_simplified/weight_orthogonalization.py` with Gemma implementation
- [ ] Create `phase5_3_weight_orthogonalization/` module structure
- [ ] Implement `WeightOrthogonalizer` class
- [ ] Add Phase 5.3 to `run.py`
- [ ] Update `common/config.py` with Phase 5.3 settings

### Validation
- [ ] Verify weight changes via Frobenius norm
- [ ] Test on subset before full validation set
- [ ] Compare metrics with Phase 4.8 steering
- [ ] Run statistical significance tests
- [ ] Generate example outputs for qualitative analysis

### Best Practices
- [ ] Always reload model between experiments
- [ ] Save weight change magnitudes for analysis
- [ ] Use same generation parameters as baseline
- [ ] Document any deviations from Phase 4.8 methodology

## Notes

- **Model Architecture**: Specifically designed for Gemma-2 2B (project's primary model)
- **Memory Management**: Each orthogonalization requires fresh model load (~8GB)
- **Numerical Precision**: Use bfloat16 on GPU, float32 on CPU
- **Feature Selection**: Uses top PVA features from Phase 2.5 (typically layer 8-17)
- **Experimental Design**: Direct comparison with Phase 4.8 steering for comprehensive analysis
- **Runtime**: Expect ~2-3 hours for full validation set with both feature types
- **Reproducibility**: Set seeds and use temperature=0.0 for deterministic generation