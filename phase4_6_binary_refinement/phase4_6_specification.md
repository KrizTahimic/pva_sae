# Phase 4.6: Binary Search Coefficient Refinement

## Overview

Phase 4.6 refines the steering coefficients found in Phase 4.5 using binary search to achieve more precise optimal values. This phase takes the coarse-grained results from Phase 4.5's grid search and performs fine-grained optimization within the identified optimal range.

## Motivation

Phase 4.5 uses a grid search with increments of 10 (e.g., 10, 20, 30, ..., 100) and employs early stopping when performance decreases. While efficient, this approach may miss the true optimal coefficient that lies between grid points. Phase 4.6 addresses this limitation by:

1. Using the best coefficient and its neighbors from Phase 4.5 as search bounds
2. Performing binary search within those bounds
3. Finding coefficients with precision up to ±0.5 of the true optimum

## Dependencies

### Required Previous Phases
- **Phase 2.5**: Provides PVA features and SAE models
- **Phase 3.6**: Provides hyperparameter tuning dataset (baseline)
- **Phase 4.5**: Provides initial coefficient estimates and search bounds

### Input Files
```
data/phase2_5/
├── top_20_features.json        # Best PVA features
└── best_layer.json             # Optimal layer for steering

data/phase3_6/
└── dataset_hyperparams_temp_0_0.parquet  # Baseline generations

data/phase4_5/
├── coefficient_analysis.json   # Search history from grid search
└── selected_coefficients.json  # Best coefficients from Phase 4.5
```

## Algorithm

### Binary Search Refinement Process

```python
def binary_search_refinement(steering_type):
    # 1. Load Phase 4.5 results
    phase4_5_results = load_coefficient_analysis()
    best_coeff = phase4_5_results[steering_type]['optimal_coefficient']
    tested_coeffs = phase4_5_results[steering_type]['tested_coefficients']
    
    # 2. Determine search bounds
    # Find coefficients tested before and after the best
    lower_bound = previous_tested_coefficient(best_coeff)
    upper_bound = next_tested_coefficient(best_coeff)
    
    # 3. Binary search
    while upper_bound - lower_bound > tolerance:
        mid = (lower_bound + upper_bound) / 2
        score = evaluate_coefficient(mid)
        
        if score > best_score:
            best_score = score
            best_coefficient = mid
            
        # Update bounds based on gradient
        if score_improving_toward_upper:
            lower_bound = mid
        else:
            upper_bound = mid
    
    return best_coefficient
```

### Evaluation Metrics

#### For Correct Steering
- **Primary Metric**: Correction Rate (incorrect → correct transitions)
- **Dataset**: All initially incorrect problems from Phase 3.6
- **Goal**: Maximize correction rate

#### For Incorrect Steering  
- **Primary Metric**: Composite Score = (Corruption Rate + Code Similarity) / 2
- **Dataset**: All initially correct problems from Phase 3.6
- **Goal**: Maximize corruption while maintaining code structure

## Configuration

### Command Line Arguments
```bash
# Basic usage
python3 run.py phase 4.6

# With custom tolerance
python3 run.py phase 4.6 --tolerance 0.5

# Limited dataset for testing
python3 run.py phase 4.6 --start 0 --end 10
```

### Configuration Parameters (common/config.py)
```python
phase4_6_tolerance: float = 1.0      # Stop when range < 1.0
phase4_6_max_iterations: int = 10    # Maximum binary search iterations
phase4_6_output_dir: str = "data/phase4_6"
```

## Output Structure

```
data/phase4_6/
├── refinement_history.json      # Complete search history
├── refined_coefficients.json    # Final refined coefficients
├── phase_4_6_summary.json       # Summary with all metrics
└── refinement_examples/         # Example generations
    ├── correct_refined_{coeff}/
    │   ├── corrected_examples.json
    │   └── summary.json
    └── incorrect_refined_{coeff}/
        ├── corrupted_examples.json
        └── summary.json
```

### refined_coefficients.json Structure
```json
{
  "correct": {
    "refined_coefficient": 23.5,
    "phase4_5_coefficient": 20.0,
    "improvement": 3.5,
    "layer": 15,
    "feature_index": 13498,
    "best_score": 28.3,
    "search_iterations": 4,
    "search_bounds": {
      "lower": 10,
      "upper": 30,
      "optimal_from_phase4_5": 20
    },
    "metrics": {
      "correction_rate": 28.3
    }
  },
  "incorrect": {
    "refined_coefficient": 42.0,
    "phase4_5_coefficient": 40.0,
    "improvement": 2.0,
    ...
  }
}
```

## Implementation Details

### Search Bound Determination
1. Load Phase 4.5's coefficient_analysis.json
2. Extract the list of tested coefficients
3. Find the position of the optimal coefficient
4. Set bounds as:
   - If optimal is first: [optimal, next_tested]
   - If optimal is last: [previous_tested, optimal]
   - Otherwise: [previous_tested, next_tested]

### Binary Search Strategy
1. Start with Phase 4.5 optimal as baseline
2. Test midpoint between bounds
3. Update bounds based on performance:
   - If midpoint improves: Move bound toward better direction
   - If midpoint worsens: Move bound away from worse direction
4. Stop when range < tolerance or max iterations reached

### Efficiency Optimizations
- Reuse model and SAE loading from Phase 4.5
- Cache baseline evaluations to avoid redundant computation
- Clear GPU memory after each coefficient evaluation
- Use same datasets as Phase 4.5 (no resampling)

## Example Run

```bash
# Run Phase 4.6 after Phase 4.5 completes
python3 run.py phase 4.6

# Output
Starting Phase 4.6: Binary Search Coefficient Refinement
Loading Phase 4.5 results for search bounds...
Correct steering search bounds: [10, 30] (Phase 4.5 optimal: 20)
Incorrect steering search bounds: [30, 50] (Phase 4.5 optimal: 40)

============================================================
Starting binary search refinement for correct steering
============================================================
Initial bounds: [10.0, 30.0]
Phase 4.5 optimal: 20.0

Iteration 1: Testing coefficient 20.00
  Correction rate: 25.0%

Iteration 2: Testing coefficient 25.00
  Correction rate: 27.5%
  New best coefficient: 25.00

Iteration 3: Testing coefficient 22.50
  Correction rate: 26.8%

Iteration 4: Testing coefficient 23.75
  Correction rate: 27.2%

Binary search complete for correct steering
Refined coefficient: 25.00
Best correction rate: 27.5%
Improvement from Phase 4.5: +2.5%

============================================================
PHASE 4.6 RESULTS SUMMARY
============================================================
Correct steering:
  - Phase 4.5 coefficient: 20.00
  - Refined coefficient: 25.00
  - Improvement: +5.00
  - Best score: 27.5%
  - Search iterations: 4

Incorrect steering:
  - Phase 4.5 coefficient: 40.00
  - Refined coefficient: 42.00
  - Improvement: +2.00
  - Best score: 71.2%
  - Search iterations: 3

Phase 4.6 completed in 487.3 seconds
Results saved to: data/phase4_6
```

## Comparison with Phase 4.5

| Aspect | Phase 4.5 (Grid Search) | Phase 4.6 (Binary Refinement) |
|--------|------------------------|------------------------------|
| **Search Method** | Grid with early stopping | Binary search |
| **Coefficient Precision** | ±10 | ±0.5-1.0 |
| **Evaluations** | ~7-10 per steering type | ~3-5 per steering type |
| **Dependencies** | Phase 2.5, 3.6 | Phase 2.5, 3.6, 4.5 |
| **Use Case** | Initial coefficient discovery | Fine-tuning for precision |

## Future Improvements

1. **Adaptive Tolerance**: Adjust tolerance based on score variance
2. **Multi-Point Evaluation**: Test multiple points per iteration for robustness
3. **Gradient Estimation**: Use finite differences to estimate gradient direction
4. **Cross-Validation**: Validate refined coefficients on held-out data
5. **Visualization**: Add plots showing search trajectory and convergence