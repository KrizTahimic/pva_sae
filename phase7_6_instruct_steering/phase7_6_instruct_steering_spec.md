# Phase 7.6: Instruction-Tuned Model Steering Analysis Specification

## Executive Summary

1. **Purpose**: Test if PVA-based steering works on instruction-tuned models using refined coefficients
2. **Method**: Apply steering coefficients from Phase 4.6 to instruction-tuned model baseline
3. **Model**: google/gemma-2-2b-it (instruction-tuned variant)
4. **Metrics**: Correction rate (incorrect→correct) and Corruption rate (correct→incorrect)
5. **Dataset**: Phase 7.3 validation baseline (dataset_instruct_temp_0_0.parquet)
6. **Statistical Validation**: Binomial tests to verify steering effects are significant
7. **Comparison**: Analyze differences in steering effectiveness vs base model (Phase 4.8)
8. **Hypothesis**: PVA features discovered in base model can steer instruction-tuned variants

## Pipeline Sequence

```
1. Load dependencies
   └─> Phase 7.3 baseline data → Phase 2.5 PVA features → Phase 4.6 coefficients → Initialize model

2. Split baseline by correctness
   └─> Initially correct problems → Initially incorrect problems

3. Apply steering interventions with refined coefficients
   └─> Correct steering on incorrect baseline → Incorrect steering on correct baseline

4. Calculate metrics and statistics
   └─> Correction rate → Corruption rate → Binomial tests → Cross-model comparison
```

## Phase Relationship

### Dependencies
- **Phase 7.3**: Provides instruction-tuned baseline data and activations
- **Phase 2.5/2.10**: Provides PVA features (same features used for base model)
- **Phase 4.6**: Provides refined steering coefficients from binary search

### Comparison Target
- **Phase 4.8**: Base model steering results for effectiveness comparison

### Key Differences from Phase 4.8

#### Model and Data
- **Model**: gemma-2-2b-it vs gemma-2-2b
- **Baseline Source**: Phase 7.3 vs Phase 3.5
- **Expected Baseline Performance**: Likely higher due to instruction-following

#### Analysis Focus
- **Primary**: Test if steering works on instruction-tuned model
- **Secondary**: Compare steering effectiveness between model variants
- **Research Question**: Do PVA features transfer across model architectures?

## Core Implementation

### Adaptation from Phase 4.8
```python
class InstructSteeringAnalyzer:
    """Analyze steering effects on instruction-tuned model."""
    
    def __init__(self, config: Config):
        # Load instruction-tuned model
        self.model_name = "google/gemma-2-2b-it"
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_name,
            device=self.device
        )
        
        # Load refined coefficients from Phase 4.6
        self._load_refined_coefficients()
        
        # Load PVA features from Phase 2.5 (same as base model)
        self._load_pva_features()
        
        # Load Phase 7.3 baseline instead of Phase 3.5
        self.baseline_data = self._load_instruct_baseline()
    
    def _load_refined_coefficients(self):
        """Load coefficients from Phase 4.6 refinement."""
        # Phase 4.6 provides refined coefficients through binary search
        phase_4_6_dir = Path(self.config.phase4_6_output_dir)
        coefficients_file = phase_4_6_dir / "refined_coefficients.json"
        
        with open(coefficients_file, 'r') as f:
            coefficients = json.load(f)
        
        self.correct_coefficient = coefficients['correct_coefficient']
        self.incorrect_coefficient = coefficients['incorrect_coefficient']
        
    def run(self):
        """Run steering analysis on instruction-tuned model."""
        # Split baseline by correctness
        # Apply steering with refined coefficients
        # Calculate metrics
        # Compare with Phase 4.8 results
```

## Core Metrics (Same as Phase 4.8)

### 1. Correction Rate
Percentage of initially incorrect solutions that become correct when steered:
```python
correction_rate = (incorrect_to_correct_count / total_initially_incorrect) * 100
```

### 2. Corruption Rate
Percentage of initially correct solutions that become incorrect when steered:
```python
corruption_rate = (correct_to_incorrect_count / total_initially_correct) * 100
```

### 3. Cross-Model Comparison
Compare with Phase 4.8 base model results:
```python
correction_rate_difference = instruct_correction_rate - base_correction_rate
corruption_rate_difference = instruct_corruption_rate - base_corruption_rate
```

## Statistical Validation

### Within-Model Tests
- Binomial tests for correction and corruption rates
- Null hypothesis: No steering effect (p = baseline pass rate)
- Alternative hypothesis: Steering changes pass rate

### Cross-Model Analysis
- Compare effect sizes between models
- Test if steering is more/less effective in instruction-tuned model
- Analyze patterns in which problems are affected

## Implementation Details

### Key Components to Reuse

#### From Phase 4.8:
```python
# Core steering logic
from phase4_5_model_steering.steering_coefficient_selector import create_steering_hook

# Statistical testing
from scipy.stats import binomtest

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

#### From Common Modules:
```python
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json
from common.prompt_utils import PromptBuilder
from common.logging import get_logger
from common.utils import discover_latest_phase_output, ensure_directory_exists, detect_device
from common.config import Config
```

### Refined Coefficients from Phase 4.6

Phase 4.6 uses binary search to refine coefficients. We use these refined values:
```python
# Load from Phase 4.6 outputs
refined_coefficients = {
    "correct_coefficient": 46.0,  # Example refined value
    "incorrect_coefficient": 46.0  # Example refined value
}
```

### Dataset Configuration
- **Source**: Phase 7.3 `dataset_instruct_temp_0_0.parquet`
- **Size**: All validation problems (~388)
- **Temperature**: 0.0 (deterministic generation)

## Expected Outcomes

### Success Scenarios

#### Scenario 1: Full Transfer
- Similar correction/corruption rates as base model
- PVA features work universally across model variants
- Validates broad applicability of approach

#### Scenario 2: Partial Transfer
- Reduced but significant steering effects
- Features partially transfer with different effectiveness
- Suggests model-specific tuning needed

#### Scenario 3: Enhanced Transfer
- Higher steering rates in instruction-tuned model
- Instruction-following enhances feature responsiveness
- Opens new research directions

### Failure Scenario
- No significant steering effects
- Features don't transfer to instruction-tuned model
- Indicates model-specific feature discovery needed

## Output Structure

```
data/phase7_6/
├── steering_effect_analysis.json         # Detailed results and metrics
├── cross_model_comparison.json          # Comparison with Phase 4.8
├── steering_effect_visualization.png    # Visualization plots
├── phase_7_6_summary.json              # Phase summary
└── examples/                            # Example steered generations
    ├── corrected_examples.json         # Incorrect→correct examples
    └── corrupted_examples.json         # Correct→incorrect examples
```

## Cross-Model Analysis

### Comparison Metrics
```python
def compare_with_base_model(self, base_results_path: Path):
    """Compare steering effectiveness with base model."""
    # Load Phase 4.8 results
    base_results = load_json(base_results_path / "steering_effect_analysis.json")
    
    comparison = {
        "base_model": {
            "correction_rate": base_results["correction_rate"],
            "corruption_rate": base_results["corruption_rate"]
        },
        "instruct_model": {
            "correction_rate": self.correction_rate,
            "corruption_rate": self.corruption_rate
        },
        "differences": {
            "correction_rate_diff": self.correction_rate - base_results["correction_rate"],
            "corruption_rate_diff": self.corruption_rate - base_results["corruption_rate"]
        }
    }
    
    return comparison
```

### Visualization
Create comparative plots showing:
1. Side-by-side correction/corruption rates
2. Problem-level steering success overlap
3. Difficulty-based effectiveness comparison

## Implementation Workflow

### Step 1: Load Dependencies
```python
# Load instruction-tuned model
model, tokenizer = load_model_and_tokenizer("google/gemma-2-2b-it", device)

# Load refined coefficients from Phase 4.6
coefficients = load_json("data/phase4_6/refined_coefficients.json")

# Load PVA features from Phase 2.5
features = load_json("data/phase2_5/top_20_features.json")

# Load Phase 7.3 baseline
baseline = pd.read_parquet("data/phase7_3/dataset_instruct_temp_0_0.parquet")
```

### Step 2: Split and Steer
```python
# Split by initial correctness
correct_baseline = baseline[baseline['test_passed'] == True]
incorrect_baseline = baseline[baseline['test_passed'] == False]

# Apply steering interventions
correction_results = apply_correct_steering(incorrect_baseline)
corruption_results = apply_incorrect_steering(correct_baseline)
```

### Step 3: Analyze Results
```python
# Calculate metrics
correction_rate = calculate_correction_rate(correction_results)
corruption_rate = calculate_corruption_rate(corruption_results)

# Statistical testing
correction_pvalue = binomtest(...)
corruption_pvalue = binomtest(...)

# Cross-model comparison
comparison = compare_with_base_model("data/phase4_8")
```

## Implementation Checklist

### Setup Phase
- [ ] Create `phase7_6_instruct_steering/` directory structure
- [ ] Create `instruct_steering_analyzer.py` based on Phase 4.8
- [ ] Add Phase 7.6 to `run.py` argument parser
- [ ] Add Phase 7.6 configuration to `common/config.py`

### Core Implementation
- [ ] Load instruction-tuned model (gemma-2-2b-it)
- [ ] Load refined coefficients from Phase 4.6
- [ ] Load Phase 7.3 baseline data
- [ ] Implement steering with refined coefficients
- [ ] Calculate correction and corruption rates

### Analysis Implementation
- [ ] Run binomial statistical tests
- [ ] Compare with Phase 4.8 base model results
- [ ] Generate visualization plots
- [ ] Save example steered generations

### Integration and Testing
- [ ] Test with subset: `python3 run.py phase 7.6 --end 49`
- [ ] Verify steering effects are measured correctly
- [ ] Compare metrics with Phase 4.8
- [ ] Run full analysis: `python3 run.py phase 7.6`

### Validation
- [ ] Ensure statistical tests are significant (p < 0.05)
- [ ] Verify cross-model comparison is meaningful
- [ ] Check example generations for quality
- [ ] Document universality findings

## Research Implications

### If Steering Works
- PVA features are universal across model variants
- SAE-based interpretability transfers to instruction-tuned models
- Steering can be applied to production models

### If Steering Doesn't Work
- Features are model-specific
- Instruction-tuning changes internal representations
- Need separate feature discovery for each model variant

### Mixed Results
- Some features transfer, others don't
- Effectiveness varies by feature type
- Suggests hierarchical feature organization

## CLI Usage

```bash
# Run Phase 7.6 analysis
python3 run.py phase 7.6

# With specific range (for testing)
python3 run.py phase 7.6 --start 0 --end 49
```

## Dependencies Summary

### Required Completed Phases
1. **Phase 2.5/2.10**: PVA feature discovery
2. **Phase 4.6**: Refined steering coefficients
3. **Phase 7.3**: Instruction-tuned model baseline

### Optional for Comparison
1. **Phase 4.8**: Base model steering results

## Key Configuration

Add to `common/config.py`:
```python
# Phase 7.6 Configuration
phase7_6_output_dir: str = "data/phase7_6"
phase7_6_model_name: str = "google/gemma-2-2b-it"
# Use refined coefficients from Phase 4.6
phase7_6_use_refined_coefficients: bool = True
```

## Notes

### Model Differences
- Instruction-tuned models may respond differently to steering
- Baseline performance likely higher due to instruction-following
- Feature importance may shift between model variants

### Computational Considerations
- Similar runtime to Phase 4.8 (~12-19 hours for full dataset)
- GPU memory requirements same as Phase 7.3
- Can use checkpoint system if needed

### Research Value
This phase provides critical evidence for:
1. Universality of PVA features
2. Transferability of interpretability methods
3. Applicability to production models

This specification provides a complete blueprint for testing whether PVA-based steering transfers to instruction-tuned models, validating the universality hypothesis and broader applicability of the approach.