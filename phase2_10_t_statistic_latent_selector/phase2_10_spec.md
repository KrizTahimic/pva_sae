# Phase 2.10: T-Statistic Latent Selection for PVA-SAE

## Executive Summary

1. **Purpose**: Identify optimal SAE latents for detecting Python code correctness using t-statistics
2. **Method**: Systematic search across layers and features using Welch's t-test for statistically rigorous feature selection
3. **Key Improvement**: Uses t-statistics instead of simple separation scores (Phase 2.5 approach)
4. **Evaluation**: AUROC/F1 evaluation happens separately in Phase 3.8 (Already Implemented)
5. **Dataset**: MBPP test split, separated into correct and incorrect solutions
6. **Output**: Top 20 features for both correct-preferring and incorrect-preferring directions

## Latent Discovery Pipeline

```
1. Feature Scoring Phase
   └─> Compute t-statistics for correct vs incorrect activations → Rank features → Filter with Pile dataset

2. Global Feature Selection  
   └─> Rank all features across all layers → Select top features globally

3. AUROC/F1 Evaluation (Phase 3.8 - Already Implemented)
   └─> Binary classification setup → Threshold optimization → Performance measurement
```

## Phase 1: T-Statistic Feature Scoring

### 1.1 T-Statistic Computation

```python
def compute_t_statistics(self, correct_features: torch.Tensor, incorrect_features: torch.Tensor) -> Dict[str, List[float]]:
    """
    Calculate t-statistics between correct and incorrect code activations.
    
    Uses Welch's t-test which:
    - Handles unequal variances between groups
    - Provides effect size normalized by pooled variance
    - Returns positive values when first group > second group
    
    Returns:
        Dict with 't_stats_correct' (correct > incorrect) and 
        't_stats_incorrect' (incorrect > correct) lists
    """
    from scipy import stats
    
    t_stats_correct = []  # Correct > Incorrect direction
    t_stats_incorrect = []  # Incorrect > Correct direction
    
    n_features = correct_features.shape[1]
    
    for i in range(n_features):
        correct_acts = correct_features[:, i].cpu().numpy()
        incorrect_acts = incorrect_features[:, i].cpu().numpy()
        
        # Handle zero activations
        if (correct_acts == 0).all() and (incorrect_acts == 0).all():
            t_stats_correct.append(0.0)
            t_stats_incorrect.append(0.0)
        else:
            # Welch's t-test (unequal variance)
            t_stat = stats.ttest_ind(
                correct_acts,
                incorrect_acts,
                equal_var=False
            ).statistic
            
            # Positive t-stat means correct > incorrect
            t_stats_correct.append(float(t_stat))
            # Negative of t-stat for incorrect > correct direction
            t_stats_incorrect.append(float(-t_stat))
    
    return {
        't_stats_correct': t_stats_correct,
        't_stats_incorrect': t_stats_incorrect
    }
```

### 1.2 Statistical Significance

While we rank by t-statistic magnitude, features with very low sample sizes or near-zero variance should be filtered:

```python
def filter_unreliable_features(self, t_stats: List[float], n_samples: int) -> List[float]:
    """
    Set t-statistics to 0 for features with insufficient data.
    """
    min_samples = self.config.t_statistic_min_samples  # Default: 10
    
    if n_samples < min_samples:
        # Not enough data for reliable t-test
        return [0.0] * len(t_stats)
    
    return t_stats
```

## Phase 2: Global Feature Selection

### 2.1 Cross-Layer Ranking

Unlike Phase 2.5's separation scores, t-statistics are directly comparable across layers:

```python
def select_top_k_features_globally(self, all_results: Dict, k: int = 20) -> Dict:
    """
    Select top k features globally across all layers using t-statistics.
    
    T-statistics provide normalized effect sizes that are comparable
    across different layers and feature dimensions.
    """
    all_features_correct = []
    all_features_incorrect = []
    
    for layer_idx, layer_results in all_results.items():
        for i, t_stat in enumerate(layer_results['t_stats_correct']):
            all_features_correct.append({
                'layer': layer_idx,
                'feature_idx': i,
                't_statistic': t_stat,
                'f_correct': layer_results['f_correct'][i],
                'f_incorrect': layer_results['f_incorrect'][i]
            })
        
        for i, t_stat in enumerate(layer_results['t_stats_incorrect']):
            all_features_incorrect.append({
                'layer': layer_idx,
                'feature_idx': i,
                't_statistic': t_stat,
                'f_correct': layer_results['f_correct'][i],
                'f_incorrect': layer_results['f_incorrect'][i]
            })
    
    # Sort by t-statistic magnitude
    top_correct = sorted(
        all_features_correct,
        key=lambda x: x['t_statistic'],
        reverse=True
    )[:k]
    
    top_incorrect = sorted(
        all_features_incorrect,
        key=lambda x: x['t_statistic'],
        reverse=True
    )[:k]
    
    return {
        'correct': top_correct,
        'incorrect': top_incorrect
    }
```

### 2.2 Pile Filtering (Optional)

Filter out features that are common in general text:

```python
def apply_pile_filter(self, top_features: Dict, pile_frequencies: Dict) -> Dict:
    """
    Remove features that activate frequently on general text.
    Features with pile frequency > threshold are likely general
    language features rather than code-specific.
    """
    threshold = self.config.pile_threshold  # Default: 0.02
    filtered = {'correct': [], 'incorrect': []}
    
    for category in ['correct', 'incorrect']:
        for feature in top_features[category]:
            layer = feature['layer']
            idx = feature['feature_idx']
            
            if layer in pile_frequencies:
                pile_freq = pile_frequencies[layer][idx]
                if pile_freq < threshold:
                    filtered[category].append(feature)
            else:
                # Keep if no pile data available
                filtered[category].append(feature)
            
            if len(filtered[category]) >= 20:
                break
    
    return filtered
```

## Phase 3: AUROC/F1 Evaluation (Already Implemented)

**Status**: ✅ Already Implemented in Phase 3.8

The selected features from Phase 2.10 are evaluated using:
- `phase3_8/auroc_f1_evaluator.py`: Full AUROC/F1 evaluation pipeline
- Uses hyperparameter split for threshold optimization
- Validates on separate validation split
- Generates confusion matrices and ROC curves

## Key Differences from Phase 2.5

| Aspect | Phase 2.5 (Separation Score) | Phase 2.10 (T-Statistic) |
|--------|------------------------------|-------------------------|
| **Metric** | `f_correct - f_incorrect` | Welch's t-test statistic |
| **Variance Handling** | Ignores variance | Accounts for variance |
| **Effect Size** | Raw difference | Normalized by pooled std |
| **Statistical Rigor** | Basic | Statistically principled |
| **Comparability** | Hard to compare across layers | Directly comparable |
| **Reliability** | All features treated equally | Can filter by sample size |

## Implementation Notes

### Already Implemented Components
- ✅ Phase 1 activation extraction and storage
- ✅ Phase 2.2 pile activation caching
- ✅ Phase 3.8 AUROC/F1 evaluation framework
- ✅ SAE loading utilities (`load_gemma_scope_sae`)
- ✅ Activation loading utilities

### New Components for Phase 2.10
- ⬜ T-statistic computation function
- ⬜ Statistical filtering for reliability
- ⬜ Integration with existing pile filtering
- ⬜ Output format matching Phase 2.5 structure

### File Organization
```
phase2_10_t_statistic_latent_selector/
├── __init__.py
├── phase2_10_spec.md              # This specification
└── t_statistic_selector.py        # Main implementation
```

### Output Structure
```
data/phase2_10/
├── sae_analysis_results.json      # Summary with top features
├── top_20_features.json           # Selected features for Phase 3
├── layer_{n}_features.json        # Per-layer t-statistics
└── best_layer.json               # Best layer identification
```

## Usage

```bash
# Run Phase 2.10 t-statistic selection
python3 run.py phase 2.10

# Run with pile filtering disabled
python3 run.py phase 2.10 --no-pile-filter

# Run Phase 3.8 evaluation using Phase 2.10 features
python3 run.py phase 3.8
```

## Key Insights

1. **Statistical Rigor**: T-statistics provide proper hypothesis testing framework
2. **Variance Awareness**: Features with consistent effects ranked higher
3. **Effect Size Normalization**: Comparable metrics across layers
4. **Pile Filtering**: Removes general language features
5. **Compatibility**: Same output format as Phase 2.5 for seamless integration

## Success Criteria

1. Successfully compute t-statistics for all layer/feature combinations
2. Generate same output format as Phase 2.5 for compatibility
3. Phase 3.8 can autodiscover and use Phase 2.10 outputs
4. Selected features show improved AUROC/F1 compared to Phase 2.5