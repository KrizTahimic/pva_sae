# Phase 4.8: Steering Effect Analysis Specification

## Executive Summary

1. **Purpose**: Analyze the effects of model steering on validation data to establish causal validity
2. **Method**: Apply steering coefficients to correct and incorrect baseline problems
3. **Metrics**: Correction rate (incorrect→correct) and Corruption rate (correct→incorrect)
4. **Dataset**: Phase 3.5 validation data (dataset_temp_0_0.parquet)
5. **Statistical Validation**: Binomial tests to verify steering effects are significant
6. **Input**: Phase 3.5 baseline data, Phase 2.5 PVA features, steering coefficients from config
7. **Output**: Statistical analysis of steering effectiveness with visualizations

## Pipeline Sequence

```
1. Load dependencies
   └─> Phase 3.5 baseline data → Phase 2.5 PVA features → Initialize model

2. Split baseline by correctness
   └─> Initially correct problems → Initially incorrect problems

3. Apply steering interventions
   └─> Correct steering on incorrect baseline → Incorrect steering on correct baseline

4. Calculate metrics and statistics
   └─> Correction rate → Corruption rate → Binomial tests → Visualizations
```

## Core Metrics

### 1. Correction Rate
Percentage of initially incorrect solutions that become correct when steered with correct-preferring features:

```python
correction_rate = (incorrect_to_correct_count / total_initially_incorrect) * 100
```

### 2. Corruption Rate
Percentage of initially correct solutions that become incorrect when steered with incorrect-preferring features:

```python
corruption_rate = (correct_to_incorrect_count / total_initially_correct) * 100
```

## Statistical Validation

Use binomial tests to verify steering effects are significantly different from chance (null hypothesis: no effect).

## Implementation Approach

### Key Design Decisions

1. **Mirror Phase 4.5 Structure**: Follow the same pattern of loading baseline data, splitting by correctness, and applying appropriate steering
2. **Reuse Existing Functions**: Import `create_steering_hook` from Phase 4.5
3. **Single Data Source**: Use Phase 3.5 validation data as the sole baseline
4. **Proper Experimental Design**: Test each steering type on the appropriate subset

### Reusable Components

#### From `common/`:
- `from common.prompt_utils import PromptBuilder` - For prompt generation (if needed)
- `from common.logging import get_logger` - For logging
- `from common.utils import discover_latest_phase_output, ensure_directory_exists, detect_device` - For utilities
- `from common.config import Config` - For configuration

#### From `common_simplified/`:
- `from common_simplified.model_loader import load_model_and_tokenizer` - For model setup
- `from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json` - For code evaluation and I/O

#### From Other Phases:
- `from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae` - For loading SAE models
- `from phase4_5_model_steering.steering_coefficient_selector import create_steering_hook` - For steering

#### Additional Required Imports:
- `import torch` - For tensor operations
- `import pandas as pd` - For data manipulation
- `import numpy as np` - For numerical operations
- `from scipy.stats import binomtest` - For statistical testing
- `import matplotlib.pyplot as plt` - For plotting
- `import seaborn as sns` - For heatmaps
- `from pathlib import Path` - For path handling
- `from tqdm import tqdm` - For progress bars
- `import json` - For JSON operations
- `import time` - For timing
- `from datetime import datetime` - For timestamps

### Core Components

1. **Load Dependencies**
   - Phase 3.5 baseline data from `dataset_temp_0_0.parquet`
   - PVA features from Phase 2.5 `top_20_features.json`
   - Steering coefficients from config (phase4_8_correct_coefficient, phase4_8_incorrect_coefficient)
   - Model and tokenizer using `load_model_and_tokenizer`

2. **Split Baseline Data**
   - Split Phase 3.5 data by `test_passed` column
   - Initially correct subset for corruption rate testing
   - Initially incorrect subset for correction rate testing

3. **Apply Steering**
   - Import `create_steering_hook` from `phase4_5_model_steering.steering_coefficient_selector`
   - Apply correct steering to initially incorrect problems
   - Apply incorrect steering to initially correct problems
   - Generate and evaluate steered outputs

4. **Calculate Metrics**
   - Correction rate from correct steering results
   - Corruption rate from incorrect steering results
   - Run binomial statistical tests
   - Create visualizations

5. **Save Results**
   - Detailed results as JSON
   - Visualization plots
   - Summary statistics

## Key Implementation Details

### Dataset
- Source: Phase 3.5 `dataset_temp_0_0.parquet` (validation split)
- Size: All available problems (typically ~400)
- Temperature: 0.0 (deterministic generation)

### Steering Coefficients
- Correct steering: `config.phase4_8_correct_coefficient` (default: 30.0)
- Incorrect steering: `config.phase4_8_incorrect_coefficient` (default: 30.0)

### Statistical Testing
- Binomial test with null hypothesis of no effect
- Significance level: α = 0.05
- Alternative hypothesis: 'greater' (one-tailed test)

## Class Structure

Create a `SteeringEffectAnalyzer` class similar to Phase 4.5's `SteeringCoefficientSelector`:

1. `__init__`: Load dependencies (model, features, baseline data)
2. `_split_baseline_by_correctness`: Separate correct/incorrect problems
3. `_apply_steering`: Generate with steering hook applied
4. `calculate_correction_rate`: Measure incorrect→correct transitions
5. `calculate_corruption_rate`: Measure correct→incorrect transitions
6. `run`: Main execution method with full pipeline

## Expected Outcomes

### Success Criteria
- **Correction Rate**: >10% of incorrect solutions become correct with correct-preferring steering
- **Corruption Rate**: >10% of correct solutions become incorrect with incorrect-preferring steering
- **Statistical Significance**: p-values < 0.05 for both effects

### Validation
The presence of statistically significant steering effects validates that:
1. SAE features capture program validity awareness
2. Model behavior can be causally influenced through these features
3. The identified features are meaningful, not spurious correlations

## Output Files

```
data/phase4_8/
├── steering_effect_analysis.json    # Detailed results and metrics
├── steering_effect_analysis.png     # Visualization plots
├── phase_4_8_summary.json          # Phase summary
└── examples/                       # Example steered generations
    ├── corrected_examples.json     # Incorrect→correct examples
    └── corrupted_examples.json     # Correct→incorrect examples
```

## Notes

- Process all problems from Phase 3.5 baseline (no sampling needed)
- Use the same generation parameters as baseline (temperature=0.0)
- Save example generations for qualitative analysis
- Follow Phase 4.5's pattern for consistency