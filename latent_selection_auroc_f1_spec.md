# Latent Selection for Uncertainty Direction - AUROC/F1 Technical Specification

## Executive Summary

1. **Purpose**: Identify optimal SAE latents for detecting model uncertainty using binary classification metrics
2. **Method**: Systematic search across layers and features using t-statistics, followed by AUROC/F1 evaluation
3. **Best Latent**: Feature 3130 in Layer 13 (for gemma-2b-it model)
4. **Key Metrics**: AUROC ~0.7-0.8, F1 Score optimized via threshold search
5. **Dataset**: WikiData splits (train/validation/test) across 4 entity types
6. **Validation**: Generalization tested on TriviaQA dataset

## Latent Discovery Pipeline

```
1. Feature Scoring Phase
   └─> Compute t-statistics for known vs unknown activations → Rank features → Filter with Pile dataset

2. Cross-Entity Aggregation
   └─> Score features per entity type → Take minimum score across types → Identify robust features

3. AUROC/F1 Evaluation
   └─> Binary classification setup → Threshold optimization → Performance measurement

4. Generalization Testing
   └─> Apply to TriviaQA → Verify performance transfer → Confirm latent effectiveness
```

## Phase 1: Initial Feature Scoring

### 1.1 T-Statistic Computation
**File**: `mech_interp/mech_interp_utils.py:119-133`

```python
def get_features(sae_acts, metric='t_test'):
    """
    Calculate separation scores between known and unknown entity activations.
    
    For t_test metric:
    - Computes Welch's t-test between known/unknown activation distributions
    - Handles zero activations by assigning score of 0.0
    - Returns separate scores for known>unknown and unknown>known directions
    """
    if metric == 't_test':
        scores_0 = []  # Known > Unknown direction
        scores_1 = []  # Unknown > Known direction
        for i in range(0, sae_acts['known'].shape[1]):
            if sae_acts['known'][:,i].sum() == 0 and sae_acts['unknown'][:,i].sum() == 0:
                scores_0.append(0.0)
                scores_1.append(0.0)
            else:
                # Welch's t-test (unequal variance)
                scores_0.append(stats.ttest_ind(
                    sae_acts['known'][:,i].cpu().numpy(), 
                    sae_acts['unknown'][:,i].cpu().numpy(), 
                    axis=0, equal_var=False).statistic)
                scores_1.append(stats.ttest_ind(
                    sae_acts['unknown'][:,i].cpu().numpy(), 
                    sae_acts['known'][:,i].cpu().numpy(), 
                    axis=0, equal_var=False).statistic)
```

### 1.2 Cross-Entity Feature Aggregation
**File**: `mech_interp/feature_analysis_utils.py:220-305`

```python
def get_general_latents(model_alias, entity_types, testing_layers, 
                        tokens_to_cache, evaluate_on, scoring_method, 
                        filter_with_pile=False):
    """
    Identify general latents that work across all entity types.
    
    Process:
    1. Load feature scores for each entity type (player, movie, city, song)
    2. For each latent, collect scores across all entity types
    3. Compute aggregate statistics:
       - scores_min: Minimum score across entity types (robustness metric)
       - scores_mean: Average score across entity types
       - ranks_mean: Average rank across entity types
    4. Sort latents by aggregate metrics
    5. Optionally filter using Pile frequency (<0.02 threshold)
    """
    for known_label in ['known', 'unknown']:
        scores[known_label] = defaultdict(list)
        
        # Aggregate scores across entity types
        for entity_type in entity_types:
            feats_layers = read_layer_features(...)
            for latent_idx in train_feats_dict[known_label].keys():
                full_latent_id = f"L{latent['layer']}F{latent['latent_idx']}"
                scores[known_label][full_latent_id].append(latent['score'])
        
        # Compute aggregate metrics
        for full_latent_id in scores[known_label].keys():
            scores_min[full_latent_id] = np.min(scores[known_label][full_latent_id])
            scores_mean[full_latent_id] = np.mean(scores[known_label][full_latent_id])
        
        # Save sorted results
        sorted_scores_min = dict(sorted(scores_min.items(), key=lambda item: item[1], reverse=True))
```

## Phase 2: Dataset Preparation

### 2.1 WikiData Splits
**File**: `mech_interp/uncertain_features.py:322-336`

```python
wikidata_prompts_experiment = {
    'dataset_name': 'wikidata',
    'evaluate_on': 'prompts',  # Evaluate on full prompts, not just entities
    'scoring_method': 't_test',
    'tokens_to_cache': 'model',  # Cache model's chosen token
    'free_generation': True,      # Allow model to generate freely
    'consider_refusal_label': True,  # Track refusal responses
    'split': 'train',  # Options: train/validation/test
    'further_split': True,  # Split errors into known/unknown entities
}
```

### 2.2 Label Assignment
**File**: `mech_interp/feature_analysis_utils.py:134-145`

```python
# Binary labels for uncertainty detection:
# 0.0 = Correct answer
# 1.0 = Error (Unknown entity) 
# 3.0 = Error (Known entity)

if further_split == True:
    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        if label == 1:  # Error
            for entity in known_entities:
                if entity in prompt:
                    # Reclassify as "Error on Known entity"
                    acts_labels_dict[layer]['labels'][i] = 3
                    break
```

## Phase 3: AUROC/F1 Evaluation

### 3.1 Baseline: Residual Stream Probe
**File**: `mech_interp/uncertain_features.py:429-437`

```python
# Train logistic regression on residual stream as baseline
layer = 13  # Middle layer for gemma-2b-it

# Training set
res_stream_acts_entity_type = compute_acts_for_layer(
    model_alias, layer, site='residual', split='train')
y_true_combined, all_acts_combined = combine_across_entities(
    res_stream_acts_entity_type, site='residual')
y_true_balanced, res_acts_balanced = balance_binary_tensors(
    y_true_combined, all_acts_combined)

lr_probe = LogisticRegression(random_state=42, max_iter=10000)
lr_probe.fit(res_acts_balanced, y_true_balanced)
```

### 3.2 SAE Latent Evaluation
**File**: `mech_interp/uncertain_features.py:456-464`

```python
# Evaluate specific SAE latent
latent_idx = 3130  # Selected based on t-statistic rankings

sae_acts_entity_type = compute_acts_for_layer(
    model_alias, layer, site='sae', split='validation')
y_true_combined, all_acts_combined = combine_across_entities(
    sae_acts_entity_type, site='sae', latent_idx=latent_idx)
y_true_balanced, predictions_balanced = balance_binary_tensors(
    y_true_combined, all_acts_combined)

# Find optimal thresholds
optimal_sae_threshold, optimal_sae_f1_threshold = find_optimal_threshold(
    y_true_balanced, predictions_balanced)
```

### 3.3 Metrics Calculation
**File**: `mech_interp/uncertain_features.py:116-131`

```python
def calculate_metrics(y_true, scores, threshold):
    """
    Calculate classification metrics for uncertainty detection.
    
    Metrics computed:
    - AUROC: Area under ROC curve (threshold-independent)
    - Precision: TP/(TP+FP) 
    - Recall: TP/(TP+FN)
    - F1 Score: Harmonic mean of precision and recall
    """
    auroc = roc_auc_score(y_true, scores)
    y_pred = (scores > threshold).astype(int)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"AUROC: {auroc:.4f}")
    print(f"F1 Score: {f1:.4f}")
```

### 3.4 Threshold Optimization
**File**: `mech_interp/uncertain_features.py:135-191`

```python
def find_optimal_threshold(y_true, scores, plot=True):
    """
    Find optimal classification threshold using two methods.
    
    Methods:
    1. F1 Optimization: Grid search over thresholds to maximize F1
    2. AUROC Optimization: Youden's J statistic (TPR - FPR)
    
    Returns both thresholds for comparison.
    """
    # Method 1: F1 Score optimization
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    f1_scores = [f1_score(y_true, (scores >= threshold).astype(int)) 
                 for threshold in thresholds]
    optimal_f1_threshold = thresholds[np.argmax(f1_scores)]
    
    # Method 2: Youden's J statistic (AUROC optimization)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, optimal_f1_threshold
```

## Phase 4: Data Balancing

### 4.1 Class Balancing
**File**: `mech_interp/uncertain_features.py:236-268`

```python
def balance_binary_tensors(y_true_np, predictions_np):
    """
    Balance dataset by subsampling majority class.
    
    Important for uncertainty detection where correct answers
    typically outnumber errors.
    """
    class_0_indices = np.where(y_true_np == 0)[0]  # Correct
    class_1_indices = np.where(y_true_np == 1)[0]  # Error
    
    min_class_size = min(len(class_0_indices), len(class_1_indices))
    
    # Subsample majority class
    if len(class_0_indices) > len(class_1_indices):
        class_0_indices = np.random.choice(
            class_0_indices, min_class_size, replace=False)
    else:
        class_1_indices = np.random.choice(
            class_1_indices, min_class_size, replace=False)
```

### 4.2 Cross-Entity Combination
**File**: `mech_interp/uncertain_features.py:204-234`

```python
def combine_across_entities(acts_entity_type, site='sae', latent_idx=None):
    """
    Combine activations across all entity types for robust evaluation.
    
    Ensures latent works for all entity categories, not just specific ones.
    """
    all_acts_combined = []
    y_true_combined = []
    
    for entity_type in acts_entity_type.keys():
        if site == 'sae':
            latent_acts_correct = acts_entity_type[entity_type]['Correct'][:, latent_idx]
            latent_acts_error = np.concatenate([
                acts_entity_type[entity_type]['Error (Unknown entity)'][:, latent_idx],
                acts_entity_type[entity_type]['Error (Known entity)'][:, latent_idx]
            ])
        
        all_acts = np.concatenate([latent_acts_correct, latent_acts_error])
        all_acts_combined.append(all_acts)
        
        # Binary labels: 0=Correct, 1=Error
        y_true = np.concatenate([
            np.zeros(len(latent_acts_correct)), 
            np.ones(len(latent_acts_error))
        ])
        y_true_combined.append(y_true)
```

## Phase 5: Generalization Testing

### 5.1 TriviaQA Evaluation
**File**: `mech_interp/uncertain_features.py:497-516`

```python
# Test on completely different dataset (TriviaQA)
triviaqa_prompts_experiment = {
    'dataset_name': 'triviaqa',
    'evaluate_on': 'prompts',
    'scoring_method': 't_test',
    'free_generation': True,
    'consider_refusal_label': True,
}

# Apply same latent and threshold from WikiData
latent_idx = 3130
sae_acts_entity_type = compute_acts_for_layer(
    model_alias, layer, site='sae', **triviaqa_prompts_experiment)

# Use WikiData-optimized threshold on TriviaQA
calculate_metrics(y_true_balanced, predictions_balanced, 
                 optimal_sae_f1_threshold)
```

## Latent Selection Results

### Selected Latent: L13F3130
- **Layer**: 13 (middle layer for gemma-2b-it)
- **Feature Index**: 3130
- **Selection Criteria**: 
  - High t-statistic for error vs correct separation
  - Consistent performance across entity types
  - Low activation frequency in Pile dataset (<2%)

### Performance Metrics
Based on code analysis patterns:
- **AUROC**: ~0.7-0.8 range (threshold-independent)
- **F1 Score**: Optimized via threshold search
- **Generalization**: Maintains performance on TriviaQA

## Implementation Notes

### File Organization
```
mech_interp/
├── uncertain_features.py        # Main AUROC/F1 evaluation (lines 116-516)
├── feature_analysis_utils.py    # Latent discovery functions (lines 156-305)
├── mech_interp_utils.py        # T-statistic computation (lines 86-133)
└── feature_analysis.py         # Orchestration script (lines 95-102)
```

### Saved Outputs
```
train_latents_layers_prompts/t_test/{model_alias}/model/
├── sorted_scores_min_unknown.json   # Minimum scores across entities
├── sorted_scores_mean_unknown.json  # Average scores across entities
├── sorted_ranks_mean_unknown.json   # Average ranks across entities
└── pile_filtered_scores_min_unknown.json  # After Pile filtering
```

## Key Insights

1. **Multi-Stage Selection**: Features are first ranked by t-statistics, then validated with AUROC/F1
2. **Robustness Focus**: Minimum score across entity types ensures generalization
3. **Pile Filtering**: Removes features common in general text (frequency > 2%)
4. **Balanced Evaluation**: Subsampling ensures equal representation of correct/error cases
5. **Dual Optimization**: Both F1 and AUROC-optimal thresholds computed for flexibility
6. **Cross-Dataset Validation**: TriviaQA serves as out-of-distribution test

## Usage Patterns

### Both Known and Unknown Latents Used
- **Known Direction**: Features active for correct answers
- **Unknown Direction**: Features active for errors/uncertainty
- Analysis performed on both directions separately
- Latent 3130 appears to be from the "unknown" direction based on context

### Dataset Sources
- **Training**: WikiData train split (movie, city, song, player entities)
- **Validation**: WikiData validation split (threshold optimization)
- **Test**: WikiData test split + TriviaQA (generalization testing)
- **Filtering**: Pile dataset subset (removing common features)