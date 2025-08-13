# Steering Coefficient Selection Strategy

```python
def adaptive_coefficient_search(steering_type):
    # Use FULL hyperparameter set for ALL evaluations
    if steering_type == 'correct':
        eval_set = all_initially_incorrect_problems  # ~50 problems
    else:
        eval_set = all_initially_correct_problems    # ~50 problems
    
    # Phase 1: Coarse search to find active range
    coarse_points = [1, 10, 100, 1000]
    results = evaluate_full_set(coarse_points, eval_set)
    
    # Identify where effect begins and saturates
    lower_bound = last_point_with_minimal_effect
    upper_bound = first_point_with_saturation
    
    # Phase 2: Binary search within active range
    tolerance = 2  # Stop when range < 2
    while upper_bound - lower_bound > tolerance:
        mid = (lower_bound + upper_bound) / 2
        result = evaluate_full_set(mid, eval_set)
        
        # Composite scoring
        if steering_type == 'correct':
            score = (correction_rate + preservation_rate) / 2 #
        else:
            score = (corruption_rate + similarity) / 2 
        
        # Update bounds based on gradient
        if score_improving_toward_upper:
            lower_bound = mid
        elif score_is_equal:
            return lower_bound
        else:
            upper_bound = mid
    
    return (upper_bound + lower_bound) / 2
```

### Benefits of Full Set Evaluation

1. **Statistical Reliability**: ~50 problems provide stable, reproducible metrics
2. **Complete Coverage**: Captures effects across all difficulty levels and problem types
3. **Direct Comparability**: Same problems for every coefficient enables precise selection
4. **Fine-Grained Discrimination**: Can detect subtle differences between adjacent coefficients

### Implementation Details

1. **Data Splitting** (from Phase 3.6 baseline)
   - Initially Correct: For testing incorrect steering (corruption)
   - Initially Incorrect: For testing correct steering (correction)
   - Each subset contains ~50 problems from hyperparameter tuning split

2. **Evaluation Process**
   - Each coefficient evaluated on ENTIRE relevant subset
   - No sampling or stratification needed
   - Consistent test set ensures fair comparison

3. **Computational Cost**
   - ~50 problems × 7 coefficients × 2 steering types = ~700 evaluations
   - Still manageable on single GPU (~2-3 hours total)
   - Much more reliable than 5 problems × 7 coefficients

## Key Metrics

### Correct Steering (test on incorrect problems)
- **Primary**: Correction Rate (incorrect → correct)
- **Secondary**: Preservation Rate (correct stay correct)
- **Composite Score**: `(correction_rate + (1 - corruption_rate)) / 2`

### Incorrect Steering (test on correct problems)
- **Primary**: Corruption Rate (correct → incorrect)
- **Secondary**: Code Similarity (maintain structure)
- **Composite Score**: `(corruption_rate + similarity) / 2`

## Incorrect Steering Strategy

After finding optimal coefficient C for correct steering:
- Test incorrect steering at [1, 10, 100, 1000]
- Run same binary search process
- May need different magnitude (often smaller for subtle bugs)

## Why Start at 1?

This is **steering** (adding features), not ablation (removing features):
- Coefficients ≥ 1 add the SAE decoder direction
- Coefficients < 1 would subtract the feature (ablation)
- Start at 1 (neutral addition) and search upward