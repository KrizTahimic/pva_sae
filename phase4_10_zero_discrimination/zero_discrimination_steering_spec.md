# Zero-Discrimination Steering Technical Specification

## Executive Summary

1. **Purpose**: Establish rigorous baseline control for SAE-based model steering experiments
2. **Method**: Select SAE features with zero separation scores between correct/incorrect programs
3. **Rationale**: Prove steering effects come from PVA-aware features, not arbitrary activation modifications
4. **Phases**: 
   - Phase 4.10: Zero-Discrimination PVA Feature Selection
   - Phase 4.12: Baseline Steering Analysis
   - Phase 4.14: Statistical Significance Testing
5. **Key Finding**: Zero-discrimination features produce minimal steering effect, validating targeted approach

## Zero-Discrimination Steering as Statistical Control

### Why Zero-Discrimination Steering is Essential

Zero-discrimination steering serves as a **rigorous baseline control** to demonstrate that:
- Targeted PVA features have specific, meaningful effects on program correctness
- Arbitrary directions added to residual streams don't produce the same steering effects
- The observed steering is due to program validity awareness, not just activation intervention

### Selection Criteria for Zero-Discrimination Features

Zero-discrimination features must satisfy:
1. **Zero separation score**: Features that don't differentiate between correct/incorrect programs
   - `abs(freq_correct - freq_incorrect) ≈ 0`
2. **Activation presence**: Features that do activate (but equally for both program types)
3. **Layer matching**: Selected from the same layers as targeted features for fair comparison
4. **No overlap**: Excluded from top-20 discriminative features to ensure true baseline

---

## Phase 4.10: Zero-Discrimination PVA Feature Selection

### Purpose
Identify and select SAE features with the MOST zero separation scores between correct and incorrect programs. These features serve as the null hypothesis baseline for steering experiments.

### Dependencies
- **Input**: Phase 1 activations (`data/phase1_0/activations/`)
- **Exclusion list**: Phase 2.5 top features (`data/phase2_5/top_20_features.json`)
- **Output**: `data/phase4_10/zero_discrimination_features.json`

### Selection Algorithm

```python
def select_zero_discrimination_features(n_features=10):
    """
    Select features with separation scores closest to zero.
    These are NOT random - they're specifically chosen for having no discrimination.
    """
    # Load all SAE features from Phase 1
    all_features = load_all_sae_features()  # ~409,600 features (25 layers × 16,384)
    
    # Calculate separation scores for ALL features
    for layer in range(1, 26):
        for feature_idx in range(16384):
            freq_correct = calculate_activation_frequency(layer, feature_idx, 'correct')
            freq_incorrect = calculate_activation_frequency(layer, feature_idx, 'incorrect')
            
            separation_score = abs(freq_correct - freq_incorrect)
            
            # Store if close to zero AND activates sufficiently
            if separation_score < 0.001 and (freq_correct + freq_incorrect) > 0.01:
                zero_disc_candidates.append({
                    'layer': layer,
                    'feature': feature_idx,
                    'separation_score': separation_score,
                    'freq_correct': freq_correct,
                    'freq_incorrect': freq_incorrect
                })
    
    # Exclude any features in top-20 discriminative list
    top_features = load_phase2_5_top_features()
    zero_disc_candidates = [f for f in zero_disc_candidates 
                           if f'{f["layer"]}_{f["feature"]}' not in top_features]
    
    # Sort by separation score (ascending - most zero first)
    zero_disc_candidates.sort(key=lambda x: x['separation_score'])
    
    # Select the MOST zero-discrimination features
    return zero_disc_candidates[:n_features]
```

### Output Format

```json
{
  "metadata": {
    "phase": "4.10",
    "description": "Zero-discrimination PVA features for baseline control",
    "selection_criteria": "Minimum absolute separation between correct/incorrect",
    "n_features_selected": 10,
    "total_features_evaluated": 409600,
    "timestamp": "2024-01-20T10:30:00"
  },
  "features": [
    {
      "rank": 1,
      "layer": 13,
      "feature_idx": 2341,
      "feature_id": "L13F2341",
      "separation_score": 0.0001,
      "freq_correct": 0.342,
      "freq_incorrect": 0.3421,
      "decoder_direction": null  // Will be loaded when needed
    },
    // ... more features
  ],
  "excluded_top_features": ["L12F5432", "L10F8923", ...]  // Phase 2.5 features excluded
}
```

---

## Phase 4.12: Zero-Discrimination Steering Generation

### Purpose
Generate steering results using zero-discrimination features as a control condition to compare against targeted PVA steering in Phase 4.14.

### Dependencies
- **Zero-discrimination features**: Phase 4.10 (`data/phase4_10/zero_discrimination_features.json`)
- **Validation data**: Phase 3.5 (`data/phase3_5/dataset_temp_0_0.parquet`)
- **Best layer**: Phase 2.5 (`data/phase2_5/best_layer.json`)
- **Coefficients**: From Phase 4.8 configuration (29 for correct, 287 for incorrect)
- **Output**: `data/phase4_12/zero_disc_steering_results.json`

### Experimental Design

#### Steering Coefficients (from Phase 4.8)
```python
CORRECT_COEFFICIENT = 29      # For correct-preferring features
INCORRECT_COEFFICIENT = 287   # For incorrect-preferring features
```

#### Steering Experiments

**Correction Experiments (incorrect→correct steering):**
- Apply zero-discrimination feature + coefficient 29 to initially incorrect problems
- Measure correction rate (incorrect→correct transitions)

**Corruption Experiments (correct→incorrect steering):**
- Apply zero-discrimination feature + coefficient 287 to initially correct problems
- Measure corruption rate (correct→incorrect transitions)

#### Implementation

```python
class ZeroDiscSteeringAnalyzer:
    def __init__(self):
        self.correct_coeff = 29
        self.incorrect_coeff = 287
        self.best_layer = self.load_best_layer()  # From Phase 2.5
        
    def run_zero_disc_steering(self):
        """
        Apply zero-discrimination steering to validation problems.
        """
        results = {
            'correction_experiments': {},
            'corruption_experiments': {}
        }
        
        # Load zero-discrimination features
        zero_disc_features = self.load_zero_discrimination_features()
        
        # Load validation data from Phase 3.5
        validation_data = self.load_phase3_5_validation_data()
        
        # Split by initial correctness
        incorrect_problems = validation_data[~validation_data['test_passed']]
        correct_problems = validation_data[validation_data['test_passed']]
        
        # CORRECTION EXPERIMENTS (incorrect problems)
        for problem in incorrect_problems:
            # Apply zero-discrimination steering with correct coefficient
            steered_result = self.generate_with_steering(
                problem,
                feature=zero_disc_features['correct'],  # Best zero-disc feature
                coefficient=self.correct_coeff
            )
            
            results['correction_experiments'][problem['task_id']] = {
                'initial_correct': False,
                'steered_correct': steered_result['passed'],
                'generation': steered_result['generation']
            }
        
        # CORRUPTION EXPERIMENTS (correct problems)
        for problem in correct_problems:
            # Apply zero-discrimination steering with incorrect coefficient
            steered_result = self.generate_with_steering(
                problem,
                feature=zero_disc_features['incorrect'],  # Best zero-disc feature
                coefficient=self.incorrect_coeff
            )
            
            results['corruption_experiments'][problem['task_id']] = {
                'initial_correct': True,
                'steered_correct': steered_result['passed'],
                'generation': steered_result['generation']
            }
            
        return results
```

### Key Implementation Points

1. **Same Coefficients**: Use exact same coefficients as Phase 4.8 (29/287)
2. **Same Dataset**: Use Phase 3.5 validation data for consistency
3. **Same Layer**: Apply steering at best_layer from Phase 2.5
4. **Parallel to Phase 4.8**: This phase generates zero-disc results just as Phase 4.8 generates targeted results

### Output Format

```json
{
  "metadata": {
    "phase": "4.12",
    "best_layer": 12,
    "coefficients": {
      "correct": 29,
      "incorrect": 287
    },
    "zero_disc_features_used": {
      "correct": "L13F2341",
      "incorrect": "L8F9234"
    },
    "n_problems_tested": {
      "correction": 200,
      "corruption": 200
    }
  },
  "correction_results": {
    "task_001": {
      "initial_correct": false,
      "steered_correct": false,  // Expected: minimal correction
      "generation": "def solution():..."
    },
    // ... more results
  },
  "corruption_results": {
    "task_100": {
      "initial_correct": true,
      "steered_correct": true,   // Expected: minimal corruption
      "generation": "def solution():..."
    },
    // ... more results
  },
  "summary_metrics": {
    "correction_rate": 0.02,  // Should be near zero
    "corruption_rate": 0.01   // Should be near zero
  }
}
```

---

## Phase 4.14: Statistical Significance Testing

### Purpose
Validate that targeted PVA steering significantly outperforms zero-discrimination steering using binomial tests.

### Dependencies
- **Zero-disc steering**: Phase 4.12 results (`data/phase4_12/zero_disc_steering_results.json`)
- **Targeted steering**: Phase 4.8 results (`data/phase4_8/steering_effect_analysis.json`)
- **Output**: `data/phase4_14/statistical_significance.json`

### Statistical Tests

#### Binomial Test Implementation

```python
from scipy.stats import binomtest

def test_steering_significance(phase4_8_results, phase4_12_results):
    """
    Test if targeted steering is significantly better than zero-discrimination steering.
    Uses binomial test only (no Cohen's d or Bonferroni).
    """
    
    # Extract correction results from Phase 4.8 (targeted)
    targeted_correction_rate = phase4_8_results['correction_rate']
    targeted_n_corrected = phase4_8_results['n_corrected']
    n_incorrect_total = phase4_8_results['n_incorrect_total']
    
    # Extract correction results from Phase 4.12 (zero-disc)
    zero_disc_corrections = sum(1 for r in phase4_12_results['correction_results'].values() 
                               if r['steered_correct'])
    zero_disc_correction_rate = zero_disc_corrections / n_incorrect_total
    
    # Binomial test: Is targeted significantly better than zero-discrimination?
    # Null hypothesis: targeted correction rate = zero-disc correction rate
    correction_pvalue = binomtest(
        targeted_n_corrected,
        n_incorrect_total,
        p=zero_disc_correction_rate,  # Expected rate under null
        alternative='greater'
    ).pvalue
    
    # Similar for corruption experiments
    targeted_corruption_rate = phase4_8_results['corruption_rate']
    targeted_n_corrupted = phase4_8_results['n_corrupted']
    n_correct_total = phase4_8_results['n_correct_total']
    
    zero_disc_corruptions = sum(1 for r in phase4_12_results['corruption_results'].values() 
                               if not r['steered_correct'])
    zero_disc_corruption_rate = zero_disc_corruptions / n_correct_total
    
    corruption_pvalue = binomtest(
        targeted_n_corrupted,
        n_correct_total,
        p=zero_disc_corruption_rate,
        alternative='greater'
    ).pvalue
    
    return {
        'correction_test': {
            'zero_disc_rate': zero_disc_correction_rate,
            'targeted_rate': targeted_correction_rate,
            'p_value': correction_pvalue,
            'significant': correction_pvalue < 0.05
        },
        'corruption_test': {
            'zero_disc_rate': zero_disc_corruption_rate,
            'targeted_rate': targeted_corruption_rate,
            'p_value': corruption_pvalue,
            'significant': corruption_pvalue < 0.05
        }
    }
```

### Expected Results

| Test | Zero-Disc Rate | Targeted Rate | p-value | Significant |
|------|---------------|---------------|---------|-------------|
| Correction | ~0.02 | ~0.45 | <0.001 | Yes |
| Corruption | ~0.01 | ~0.38 | <0.001 | Yes |

### Output Format

```json
{
  "metadata": {
    "phase": "4.14",
    "test_type": "binomial",
    "alternative": "greater",
    "significance_level": 0.05
  },
  "correction_significance": {
    "n_samples": 50,
    "zero_discrimination": {
      "successes": 1,
      "rate": 0.02
    },
    "targeted": {
      "successes": 23,
      "rate": 0.46
    },
    "binomial_test": {
      "p_value": 1.23e-10,
      "significant": true,
      "interpretation": "Targeted steering significantly outperforms zero-discrimination baseline"
    }
  },
  "corruption_significance": {
    // Similar structure
  },
  "conclusion": "PVA features have specific causal effects on program correctness, not explained by arbitrary activation modifications"
}
```

---

## Implementation Checklist

### Phase 4.10: Zero-Discrimination Feature Selection
- [ ] Load all Phase 1 activations
- [ ] Calculate separation scores for all 409,600 features
- [ ] Filter for features with separation_score < 0.001
- [ ] Exclude Phase 2.5 top-20 features
- [ ] Select features with MOST zero separation
- [ ] Save decoder directions for selected features
- [ ] Output to `data/phase4_10/zero_discrimination_features.json`

### Phase 4.12: Zero-Discrimination Steering Generation
- [ ] Load zero-discrimination features from Phase 4.10
- [ ] Load validation data from Phase 3.5
- [ ] Use Phase 4.8 coefficients (29 for correct, 287 for incorrect)
- [ ] Apply zero-discrimination steering to all validation problems
- [ ] Apply steering at best_layer from Phase 2.5
- [ ] Calculate correction and corruption rates
- [ ] Output to `data/phase4_12/zero_disc_steering_results.json`

### Phase 4.14: Statistical Significance Testing
- [ ] Load Phase 4.8 targeted steering results
- [ ] Load Phase 4.12 zero-discrimination results
- [ ] Implement binomial tests (no Cohen's d or Bonferroni)
- [ ] Test targeted vs zero-discrimination for correction
- [ ] Test targeted vs zero-discrimination for corruption
- [ ] Report p-values and significance
- [ ] Output to `data/phase4_14/statistical_significance.json`

## Key Insights

1. **Not Random**: These are carefully selected zero-discrimination features, not random selections
2. **Rigorous Control**: Zero-discrimination baseline proves effects come from PVA content
3. **Fair Comparison**: Same coefficients, positions, and layers - only features differ
4. **Statistical Validation**: Binomial tests confirm targeted steering significantly outperforms baseline
5. **Causal Evidence**: Demonstrates specific features have causal influence on program correctness

## Important Distinctions

- **Zero-Discrimination ≠ Random**: Features are selected for having precisely zero separation
- **Zero-Discrimination ≠ Dead**: Features must activate, just equally for both classes
- **Zero-Discrimination ≠ Pile**: Pile filtering refines top features; this finds non-discriminative features
- **Baseline ≠ No Steering**: Baseline steering applies zero-discrimination features with same coefficients

## Notes

- All features come from same SAE repository (GemmaScope)
- Coefficients fixed from Phase 4.8 optimization
- Single-layer steering at Phase 2.5 best layer
- Validation split ensures no data leakage
- Three-phase design separates concerns cleanly