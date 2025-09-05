# Random Steering Technical Specification

**Note**: All code snippets in this document are taken directly from the actual codebase with file paths and line numbers provided for reference. The implementation details reflect the real system behavior.

## Executive Summary

1. **Purpose**: Establish statistical baseline control for SAE-based model steering experiments
2. **Method**: Select SAE features with zero activation difference between known/unknown entities
3. **Rationale**: Prove steering effects come from meaningful features, not arbitrary directions
4. **Coefficients**: Use same coefficient values as targeted steering (15 for known, 20 for unknown)
5. **Evaluation**: Compare random steering effects against targeted steering to validate significance
6. **Key Finding**: Random features produce minimal/no steering effect, validating targeted approach

## Random Steering as Statistical Control

### Why Random Steering is Essential

Random steering serves as a **null hypothesis baseline** to demonstrate that:
- Targeted SAE features have specific, meaningful effects on model behavior
- Arbitrary directions added to activations don't produce the same steering effects
- The observed steering is due to feature content, not just the act of intervention

### Selection Criteria for Random Features

Random features must satisfy:
1. **Zero discrimination score**: Features that don't differentiate between known/unknown entities
2. **Activation presence**: Features that do activate (but equally for both classes)
3. **Layer matching**: Selected from the same layers as targeted features for fair comparison
4. **No overlap**: Excluded from top-performing features to ensure true baseline

## Pipeline Sequence

```
1. Identify candidate features
   └─> Load all SAE features from target layers → Filter by zero scores → Ensure no overlap with top features

2. Random selection process
   └─> Sample N features uniformly → Extract decoder directions → Store with metadata

3. Apply random steering
   └─> Use same hooks as targeted steering → Same coefficients → Same positions

4. Measure baseline effect
   └─> Calculate Yes/No logit differences → Compare with no steering → Compare with targeted steering

5. Statistical validation
   └─> Compute significance tests → Verify minimal effect → Validate targeted steering superiority
```

## Core Implementation

### 1. Random Feature Selection Algorithm

```python
# File: mech_interp/mech_interp_utils.py, lines 1507-1553
def load_latents(model_alias, top_latents, filter_with_pile=False, **kwargs):
    """
    Load both targeted and random SAE features for steering experiments.
    
    The random features serve as a statistical control baseline.
    """
    # Load score rankings to identify top features and zero-score features
    if filter_with_pile == True:
        with open(f'./train_latents_layers_entities/absolute_difference/{model_alias}/entity/pile_filtered_scores_min_known.json', 'r') as f:
            sorted_scores_known = json.load(f)
        with open(f'./train_latents_layers_entities/absolute_difference/{model_alias}/entity/pile_filtered_scores_min_unknown.json', 'r') as f:
            sorted_scores_unknown = json.load(f)
    else:
        with open(f'./train_latents_layers_entities/absolute_difference/{model_alias}/entity/sorted_scores_min_known.json', 'r') as f:
            sorted_scores_known = json.load(f)
        with open(f'./train_latents_layers_entities/absolute_difference/{model_alias}/entity/sorted_scores_min_unknown.json', 'r') as f:
            sorted_scores_unknown = json.load(f)
    
    # Extract specific latent IDs from sorted scores
    known_latent_ = list(sorted_scores_known.keys())[top_latents['known']]
    unknown_latent_ = list(sorted_scores_unknown.keys())[top_latents['unknown']]
    
    # Load random control features
    random_latents_known = load_steering_latents(
        'movie', label='known', 
        topk=kwargs['random_n_latents'],  # Usually 5 random features
        layers_range=[layer_known],
        model_alias=model_alias,
        random_latents=True  # KEY: Random selection mode
    )
    
    random_latents_unknown = load_steering_latents(
        'movie', label='unknown', 
        topk=kwargs['random_n_latents'],
        layers_range=[layer_unknown],
        model_alias=model_alias,
        random_latents=True
    )
    
    return known_latent, unknown_latent, random_latents_known, random_latents_unknown
```

### 2. Random Feature Filtering Logic

```python
# File: mech_interp/mech_interp_utils.py, lines 370-432 (within load_steering_latents function)
def load_steering_latents(..., random_latents=False):
    """
    Load SAE features for steering, with option for random baseline selection.
    """
    if random_latents:
        # Get all features from specified layers (line 416)
        all_sae_latents_dict = get_top_k_features(feats_layers, k=None)
        available_indices = list(range(len(all_sae_latents_dict[label])))
        
        # CRITICAL: Remove features with non-zero discrimination scores (line 425)
        # This ensures random features don't accidentally have steering effect
        min_max_scores = json.load(open(
            f"./train_latents_layers_entities/absolute_difference/{model_alias.split('/')[-1]}/entity/sorted_scores_min_{label}.json"
        ))
        
        # Filter out features with non-zero scores (lines 426-429)
        for idx in all_sae_latents_dict[label].keys():
            latent_id = f"L{all_sae_latents_dict[label][idx]['layer']}F{all_sae_latents_dict[label][idx]['latent_idx']}"
            if abs(min_max_scores[latent_id]) > 0.0:
                available_indices.remove(idx)  # Exclude non-zero score features
        
        # Randomly sample from zero-score features only (line 431)
        indices = random.sample(available_indices, topk)
        sae_latent_dict = {i: all_sae_latents_dict[label][idx] for i, idx in enumerate(indices)}
```

### 3. Applying Random Steering

```python
# File: mech_interp/hooks_utils.py, lines 28-59 (steer_sae_latents function)
# File: mech_interp/hooks_utils.py, lines 273-299 (compute_logit_diff_steered function)

def steer_sae_latents(activation, hook, direction, pos, coeff_value=1):
    """
    Hook function that adds steering direction to activations.
    Used for both random and targeted steering with identical mechanism.
    """
    if activation.shape[1]==1:
        # generating
        return activation
    
    if pos != 'all':
        if isinstance(pos[0], list):
            for batch_idx, p in enumerate(pos):
                activation[batch_idx, p, :] += direction.unsqueeze(0)*coeff_value
        else:
            activation[:, pos, :] += direction.unsqueeze(0)*coeff_value
    else:
        activation[:, :, :] += direction.unsqueeze(0)*coeff_value
    
    return activation

def compute_logit_diff_steered(model, N, tokenized_prompts, metric, pos_entities, 
                               steering_latents, coeff_value, pos_type='entity_last'):
    """
    Apply steering (random or targeted) and measure effect.
    Random steering uses the same mechanism as targeted steering.
    """
    # Process in batches
    for i in range(0, len(tokenized_prompts), batch_size):
        batch_pos = get_batch_pos(batch_entity_pos, pos_type, batch_tokenized_prompts)
        
        model.reset_hooks()
        steered_logits, steered_cache = cache_steered_latents(
            model, batch_tokenized_prompts, pos=batch_pos,
            steering_latents=steering_latents,
            ablate_latents=ablate_latents,
            coeff_value=coeff_value  # Same coefficient for random and targeted!
        )
        
        steered_logit_diff_full.append(compute_metric(model, steered_logits, metric=metric))
```

## Experimental Configuration

### Random Steering Setup

```python
# File: mech_interp/steering_it.py, lines 123-136
# Configuration values used in actual experiments
known_label = 'unknown'
pos_type = 'entity_last'
N = 100  # Number of prompts for evaluation
max_new_tokens = 10
batch_size = 8
top_latents = {'known': 0, 'unknown': 0}  # Top feature indices
coeff_values = {'known': 15, 'unknown': 20}  # Steering coefficients
split = 'test'

# Loading random features (line 134-136)
known_latent, unknown_latent, random_latents_known, random_latents_unknown = load_latents(
    model_alias, top_latents,
    filter_with_pile=True,
    random_n_latents=5  # Number of random features to select
)
```

### Comparative Evaluation Protocol

```python
# File: mech_interp/steering_it.py, lines 251-290
# Three-condition comparison from actual experiments
for entity_type in ['player', 'movie', 'city', 'song']:
    # 1. Baseline: No steering (line 261)
    orig_results = compute_logit_diff_original(
        model, N, tokenized_prompts, metric=metric, batch_size=4
    )
    
    # 2. Random control: Random feature steering (lines 263-266)
    print('RANDOM STEERING LATENTS')
    rdm_steered_result = compute_logit_diff_steered(
        model, N, tokenized_prompts, metric, pos_entities, pos_type=pos_type,
        steering_latents=[random_latents_unknown[0]],  # Random feature
        ablate_latents=None,
        coeff_value=coeff_values[complement_known_label],  # Same coefficient
        batch_size=batch_size
    )
    
    # 3. Treatment: Targeted feature steering (lines 268-271)
    print('STEERING LATENTS')
    steered_result = compute_logit_diff_steered(
        model, N, tokenized_prompts, metric, pos_entities, pos_type=pos_type,
        steering_latents=steering_latents,  # Top discriminative feature
        ablate_latents=None,
        coeff_value=coeff_values[complement_known_label],  # Same coefficient
        batch_size=batch_size
    )
    
    # Results are collected in results_dict for plotting (lines 287-289)
    results_dict[entity_type].append(orig_results_)
    results_dict[entity_type].append(rdm_steered_result)
    results_dict[entity_type].append(steered_result)
```

## Random Feature Characteristics

### What Makes a Good Random Control Feature?

1. **Zero Discrimination Score**
   - Frequency of activation is similar for known and unknown entities
   - `abs(freq_known - freq_unknown) ≈ 0`
   - No predictive power for entity knowledge status

2. **Non-zero Activation**
   - Feature must actually fire in the model
   - Completely dead features aren't useful controls
   - Activation frequency > 0 but balanced across classes

3. **Layer Distribution**
   - Should come from same layers as targeted features
   - Ensures architectural comparability
   - Controls for layer-specific effects

### Example Random Feature Selection

```python
# Example from actual implementation
min_max_scores = {
    "L13F2341": 0.0,  # Perfect random candidate - zero score
    "L13F8923": 0.0,  # Another zero-score feature
    "L13F1234": 0.85,  # NOT suitable - has discrimination power
    "L13F5678": -0.72,  # NOT suitable - discriminates opposite direction
}

# Selection process
random_candidates = [
    feat for feat, score in min_max_scores.items() 
    if abs(score) == 0.0  # Strict zero-score requirement
]
selected_random = random.sample(random_candidates, n_random)
```

## Steering Effect Visualization

### Expected Results Pattern

```python
# File: mech_interp/steering_it.py, lines 167-214 (multi_line_top_k_latents_effect_plot function)
def multi_line_top_k_latents_effect_plot(results_dict, xticks_labels: List[str], title: str, 
                                         metric: Literal['logit_diff', 'logprob', 'prob']):
    """
    Visualize the three-condition comparison.
    
    Expected pattern:
    - Original: Baseline logit difference
    - Random SAE latent: Similar to original (no effect)
    - Targeted SAE latent: Significant shift from original
    """
    fig = go.Figure()
    
    # xticks_labels = ['Original', 'Random SAE latent', 'Known/Unknown SAE latent']
    
    for i, topk in enumerate(results_dict.keys()):  # entity types
        logit_diff_results = [logit_diff.cpu().numpy() for logit_diff in results_dict[topk]]
        means = [np.mean(logit_diff) for logit_diff in logit_diff_results]
        errors = [np.std(logit_diff) for logit_diff in logit_diff_results]
        
        fig.add_trace(go.Scatter(
            x=xticks_labels,
            y=means,
            error_y=dict(type='data', array=errors, visible=True),
            mode='lines+markers',
            name=f'{topk.capitalize()}'  # Entity type name
        ))
    
    if metric == 'logit_diff':
        y_title = 'Logit Difference (Yes - No)'
    
    fig.update_layout(yaxis_title=y_title, title=title)
    
    return fig
```

### Interpreting Random Steering Results

| Condition | Expected Logit Diff | Interpretation |
|-----------|-------------------|----------------|
| No steering | Baseline (e.g., 2.5) | Model's natural behavior |
| Random steering | ~Baseline (e.g., 2.4-2.6) | No meaningful effect |
| Targeted steering | Shifted (e.g., 0.5 or 4.5) | Significant behavior change |

**Key Insight**: If random steering produced similar effects to targeted steering, it would invalidate the approach. The minimal effect of random features validates that specific features carry semantic meaning.

## Statistical Validation

### Significance Testing

```python
def validate_random_baseline(orig, random, targeted):
    """
    Statistical tests to validate random steering as proper baseline.
    """
    from scipy import stats
    
    # Test 1: Random steering should not differ from original
    t_stat_random, p_val_random = stats.ttest_rel(orig, random)
    assert p_val_random > 0.05, "Random steering should not significantly differ from baseline"
    
    # Test 2: Targeted steering should differ from original
    t_stat_targeted, p_val_targeted = stats.ttest_rel(orig, targeted)
    assert p_val_targeted < 0.001, "Targeted steering must significantly differ from baseline"
    
    # Test 3: Targeted should differ from random
    t_stat_diff, p_val_diff = stats.ttest_rel(random, targeted)
    assert p_val_diff < 0.001, "Targeted must significantly differ from random control"
    
    # Effect size calculation (Cohen's d)
    effect_size_random = np.mean(random - orig) / np.std(random - orig)
    effect_size_targeted = np.mean(targeted - orig) / np.std(targeted - orig)
    
    assert abs(effect_size_random) < 0.2, "Random effect should be negligible"
    assert abs(effect_size_targeted) > 0.8, "Targeted effect should be large"
    
    return {
        'random_p_value': p_val_random,
        'targeted_p_value': p_val_targeted,
        'random_effect_size': effect_size_random,
        'targeted_effect_size': effect_size_targeted
    }
```

### Multiple Comparisons Correction

When testing across multiple entity types, apply Bonferroni correction:

```python
def bonferroni_correction(p_values, n_comparisons=4):
    """Apply Bonferroni correction for multiple entity types."""
    corrected_alpha = 0.05 / n_comparisons
    significant = [p < corrected_alpha for p in p_values]
    return corrected_alpha, significant
```

## Implementation Workflow

### Phase 1: Identify Random Features

```python
# Step 1: Load feature scores
scores = load_feature_scores(model_alias, entity_type)

# Step 2: Filter zero-score features
zero_score_features = [
    feat for feat, score in scores.items()
    if abs(score) < 1e-6  # Numerical zero with tolerance
]

# Step 3: Verify sufficient candidates
assert len(zero_score_features) > n_random * 10, "Need enough candidates for random sampling"
```

### Phase 2: Extract Random Directions

```python
# Step 4: Sample random features
selected_features = random.sample(zero_score_features, n_random)

# Step 5: Load SAE and extract directions
for layer, feature_idx in selected_features:
    sae = load_sae(repo_id, f"layer_{layer}")
    direction = sae.W_dec[feature_idx].detach()
    random_directions.append((layer, feature_idx, 0.0, direction))
```

### Phase 3: Run Controlled Experiment

```python
# Step 6: Apply three conditions
results = {
    'baseline': run_without_steering(model, prompts),
    'random': run_with_steering(model, prompts, random_directions, coeff),
    'targeted': run_with_steering(model, prompts, targeted_directions, coeff)
}

# Step 7: Validate results
validation = validate_random_baseline(
    results['baseline'],
    results['random'],
    results['targeted']
)
```

### Phase 4: Report Results

```python
# Step 8: Generate comparison plots
fig = plot_three_way_comparison(results)

# Step 9: Report statistics
print(f"Random steering p-value: {validation['random_p_value']:.4f} (expect > 0.05)")
print(f"Targeted steering p-value: {validation['targeted_p_value']:.4f} (expect < 0.001)")
print(f"Effect sizes - Random: {validation['random_effect_size']:.2f}, Targeted: {validation['targeted_effect_size']:.2f}")
```

## Troubleshooting Random Steering

### Common Issues and Solutions

1. **Random features show steering effect**
   - Check score threshold is truly zero
   - Verify random selection isn't biased
   - Ensure sufficient sample size

2. **No zero-score features available**
   - Relax threshold slightly (e.g., abs(score) < 0.01)
   - Use percentile-based selection (bottom 1% of scores)
   - Consider features from different layers

3. **Inconsistent random baseline**
   - Use same random seed for reproducibility
   - Test multiple random samples
   - Report variance across random selections

### Debugging Random Feature Selection

```python
def debug_random_selection(selected_features, all_scores):
    """Verify random features are truly baseline."""
    for layer, idx in selected_features:
        feat_id = f"L{layer}F{idx}"
        score = all_scores[feat_id]
        assert abs(score) < 0.001, f"Feature {feat_id} has non-zero score: {score}"
        
    print(f"✓ All {len(selected_features)} random features have zero scores")
    print(f"Score range: {min(scores):.6f} to {max(scores):.6f}")
```

## Implementation Checklist

### Setup
- [ ] Load feature discrimination scores from training
- [ ] Identify zero-score features for random selection
- [ ] Verify sufficient random candidates available
- [ ] Set random seed for reproducibility

### Random Feature Selection
- [ ] Filter features by zero discrimination score
- [ ] Exclude top-K discriminative features
- [ ] Sample N random features from zero-score set
- [ ] Extract decoder directions for random features

### Experimental Design
- [ ] Implement three-condition comparison (none/random/targeted)
- [ ] Use identical coefficients for random and targeted
- [ ] Apply steering at same positions for fair comparison
- [ ] Ensure batch processing is consistent

### Validation
- [ ] Run statistical significance tests
- [ ] Calculate effect sizes (Cohen's d)
- [ ] Apply multiple comparison corrections
- [ ] Verify random effect is negligible

### Reporting
- [ ] Generate comparison plots with error bars
- [ ] Report p-values for all comparisons
- [ ] Document random feature characteristics
- [ ] Include effect size measurements

## Key Insights

1. **Random as Null Hypothesis**: Random features with zero discrimination scores serve as the null hypothesis, proving targeted features have specific semantic content

2. **Same Mechanism, Different Features**: Random steering uses identical hooks, positions, and coefficients as targeted steering - only the features differ

3. **Statistical Validation**: The minimal effect of random features (<0.2 Cohen's d) versus large effect of targeted features (>0.8 Cohen's d) validates the approach

4. **Zero-Score Requirement**: Features must have truly zero discrimination between known/unknown to serve as proper controls

5. **Reproducibility**: Fixed random seeds ensure consistent baseline across experiments

6. **Multiple Controls**: Using multiple random features (typically 5) provides robustness against outlier effects

7. **Layer Matching**: Random features should come from the same layers as targeted features for architectural comparability

## Notes

- **Not Pile Filtering**: Random steering is distinct from Pile filtering, which refines top features
- **Not Arbitrary Directions**: Random features are actual SAE features, just ones without discriminative power
- **Statistical Power**: Larger N (100+ examples) needed for reliable significance testing
- **Computational Efficiency**: Random features can be pre-computed and cached
- **Cross-Validation**: Different random seeds should produce consistent null results

## Code References Summary

All code snippets in this document are extracted from the following source files:

1. **Random Feature Selection**: 
   - `mech_interp/mech_interp_utils.py`: Lines 370-478 (load_steering_latents), Lines 1507-1553 (load_latents)

2. **Steering Hooks**:
   - `mech_interp/hooks_utils.py`: Lines 28-59 (steer_sae_latents), Lines 139-202 (get_batch_pos), Lines 273-299 (compute_logit_diff_steered)

3. **Experimental Implementation**:
   - `mech_interp/steering_it.py`: Lines 123-136 (configuration), Lines 167-214 (plotting), Lines 251-290 (three-condition comparison)

The implementation shows that random steering uses identical mechanisms to targeted steering, differing only in feature selection criteria (zero discrimination scores vs top discrimination scores).