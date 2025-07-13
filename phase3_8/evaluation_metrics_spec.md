# Phase 3.8: AUROC and F1 Evaluation for PVA-SAE

## Executive Summary

1. **Purpose**: Evaluate SAE features for program validity awareness in Python code generation
2. **Task**: Two separate binary classifications for bidirectional PVA features
3. **Metrics**: AUROC (Area Under ROC Curve) and F1 Score for each direction
4. **Implementation**: Direct evaluation of SAE feature activations using scikit-learn metrics
5. **Threshold Selection**: F1-optimal threshold via grid search on hyperparameter split (per feature)
6. **Dataset**: Generated Python solutions for MBPP problems
7. **Two Feature Directions**:
   - **Correct-preferring feature**: Labels: 1=Correct, 0=Incorrect (flipped convention)
   - **Incorrect-preferring feature**: Labels: 1=Incorrect, 0=Correct (standard convention)
8. **Data Source**: Phase 3.5 activations from validation split (388 problems)

## Pipeline Sequence

```
1. Load Phase 3.5 metadata and identify best features
   └─> Read metadata.json → Extract best correct & incorrect features → Note their layers & indices

2. Evaluate Correct-Preferring Feature
   a. Load activations and create flipped labels (1=correct, 0=incorrect)
   b. Find optimal F1 threshold on hyperparameter split
   c. Evaluate on validation split with optimal threshold
   d. Generate visualizations and save results

3. Evaluate Incorrect-Preferring Feature
   a. Load activations and create standard labels (1=incorrect, 0=correct)
   b. Find optimal F1 threshold on hyperparameter split
   c. Evaluate on validation split with optimal threshold
   d. Generate visualizations and save results

4. Output combined results to data/phase3_8/
   └─> Save metrics for both features → Save all plots → Generate comparative summary
```

## Phase Relationship

### Why Phase 3.8 Depends on Phase 3.5
Phase 3.8 uses the same activation data as Phase 3.5 because:
- Both phases evaluate the validation split (388 problems)
- Both use the same model and prompts
- Phase 3.5 already extracts activations for the best PVA layer
- Reusing these activations is efficient and ensures consistency

### Key Difference
- **Phase 3.5**: Tests robustness across different temperatures (0.0, 0.3, 0.6, 0.9, 1.2)
- **Phase 3.8**: Computes AUROC and F1 metrics using the temperature 0.0 data from Phase 3.5

## Understanding the Evaluation Task

### Two Binary Classification Problems
PVA-SAE identifies features in both directions, requiring separate evaluations:

#### 1. Correct-Preferring Feature Evaluation
- **Positive Class (1)**: Correct code - solution passes test cases
- **Negative Class (0)**: Incorrect code - solution fails test cases
- **Label Convention**: FLIPPED from standard (1=correct, 0=incorrect)
- **Interpretation**: High activation → code is likely correct

#### 2. Incorrect-Preferring Feature Evaluation
- **Positive Class (1)**: Incorrect code - solution fails test cases
- **Negative Class (0)**: Correct code - solution passes test cases  
- **Label Convention**: STANDARD (1=incorrect, 0=correct)
- **Interpretation**: High activation → code is likely incorrect

### Evaluation Context
**MBPP (Mostly Basic Programming Problems)**: Python programming challenges with test cases for validation

## Understanding Key Variables

### For Correct-Preferring Features

#### `y_true_correct` - Ground Truth Labels (FLIPPED)
- **1** = Correct code (passes all test cases) - positive class
- **0** = Incorrect code (fails test cases) - negative class

Example:
```python
y_true_correct = [1, 1, 0, 1, 0, 0, 1, 0]
# Interpretation: [correct, correct, error, correct, error, error, correct, error]
```

#### `scores_correct` - Correct Feature Activations
- Higher activations indicate higher likelihood of CORRECT code

Example:
```python
scores_correct = [0.89, 0.91, 0.12, 0.76, 0.23, 0.15, 0.85, 0.34]
# High values suggest code is likely correct
```

### For Incorrect-Preferring Features

#### `y_true_incorrect` - Ground Truth Labels (STANDARD)
- **1** = Incorrect code (fails test cases) - positive class
- **0** = Correct code (passes all test cases) - negative class

Example:
```python
y_true_incorrect = [0, 0, 1, 0, 1, 1, 0, 1]
# Interpretation: [correct, correct, error, correct, error, error, correct, error]
```

#### `scores_incorrect` - Incorrect Feature Activations
- Higher activations indicate higher likelihood of INCORRECT code

Example:
```python
scores_incorrect = [0.12, 0.23, 0.89, 0.34, 0.76, 0.91, 0.15, 0.67]
# High values suggest code is likely incorrect
```

### Conceptual Example

```python
# Task: "Write a function to find the sum of all even numbers in a list"

# Generated Code (Incorrect):
def sum_even(lst):
    return sum(lst)  # Bug: sums all numbers, not just even

# For Correct-Preferring Feature:
y_true_correct: 0 (incorrect code → negative class)
score_correct: 0.12 (low activation → feature doesn't fire for incorrect code)

# For Incorrect-Preferring Feature:
y_true_incorrect: 1 (incorrect code → positive class)
score_incorrect: 0.89 (high activation → feature fires for incorrect code)

# Generated Code (Correct):
def sum_even(lst):
    return sum(x for x in lst if x % 2 == 0)

# For Correct-Preferring Feature:
y_true_correct: 1 (correct code → positive class)
score_correct: 0.91 (high activation → feature fires for correct code)

# For Incorrect-Preferring Feature:
y_true_incorrect: 0 (correct code → negative class)
score_incorrect: 0.15 (low activation → feature doesn't fire for correct code)
```

Threshold optimization for each feature:
- **Correct-preferring**: Scores above threshold → Predict as correct (1)
- **Incorrect-preferring**: Scores above threshold → Predict as incorrect (1)

## Core Implementation

### 1. Metric Calculation Function (Generic)
```python
def calculate_metrics(y_true, scores, threshold, feature_type, output_dir):
    """Calculate metrics for either correct or incorrect preferring features."""
    # Calculate AUROC - threshold independent
    auroc = roc_auc_score(y_true, scores)
    
    # Apply threshold for binary predictions
    y_pred = (scores > threshold).astype(int)
    
    # Calculate threshold-dependent metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\nMetrics for {feature_type}-preferring feature:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    
    plot_confusion_matrix(y_true, y_pred, feature_type, output_dir)
    
    return {
        'auroc': auroc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'threshold': threshold
    }
```

**Key aspects:**
- Works for both feature types with appropriate labels
- Returns metrics dict for saving
- Labels confusion matrix by feature type

### 2. F1-Optimal Threshold Finding
```python
def find_optimal_threshold(y_true, scores, feature_type, output_dir):
    """Find optimal threshold for a specific feature type."""
    # Grid search for F1-Optimal Threshold
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    f1_scores = [f1_score(y_true, (scores >= threshold).astype(int)) 
                 for threshold in thresholds]
    
    # Find threshold that maximizes F1 score
    optimal_f1_threshold = thresholds[np.argmax(f1_scores)]
    max_f1_score = max(f1_scores)
    
    # Plot F1 scores against thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores vs Thresholds - {feature_type.capitalize()}-Preferring Feature')
    plt.grid(True)
    plt.axvline(x=optimal_f1_threshold, color='r', linestyle='--', 
               label=f'Optimal F1 Threshold: {optimal_f1_threshold:.3f}')
    plt.axhline(y=max_f1_score, color='g', linestyle='--', 
               label=f'Max F1 Score: {max_f1_score:.3f}')
    plt.legend()
    
    # Save plot
    plt.savefig(output_dir / f'f1_threshold_plot_{feature_type}.png')
    plt.close()
    
    # Evaluate at optimal threshold
    print(f'\nF1 optimal for {feature_type}-preferring feature:')
    metrics = calculate_metrics(y_true, scores, optimal_f1_threshold, feature_type, output_dir)
    
    return optimal_f1_threshold, metrics
```

**F1-Optimal Strategy:**
- Separate optimization for each feature type
- Grid search across 100 threshold values
- Saves feature-specific plots



## Implementation Workflow

### Phase 1: Load Phase 3.5 Metadata and Best Features
```python
# Load Phase 3.5 metadata to get best features
metadata = json.load(open('data/phase3_5/metadata.json'))
best_features = metadata['best_layers']

# Extract feature information
correct_layer = best_features['correct']
correct_feature_idx = best_features['correct_feature_idx']
incorrect_layer = best_features['incorrect']
incorrect_feature_idx = best_features['incorrect_feature_idx']

print(f"Best correct-preferring feature: idx {correct_feature_idx} at layer {correct_layer}")
print(f"Best incorrect-preferring feature: idx {incorrect_feature_idx} at layer {incorrect_layer}")

# Import GemmaScope SAE loader (reuse from Phase 2.5)
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae
from common.utils import detect_device

# Function to load activations for a specific feature
def load_split_activations(split_name, layer_num, feature_idx, feature_type):
    split_data = pd.read_parquet(f'data/phase0_1/{split_name}_mbpp.parquet')
    # Load temperature 0.0 dataset from Phase 3.5
    temp_data = pd.read_parquet('data/phase3_5/dataset_temp_0_0.parquet')
    
    # Detect device and load SAE for encoding (do this once outside the loop)
    device = detect_device()
    sae = load_gemma_scope_sae(layer_num, device)
    logger.info(f"Loaded SAE for layer {layer_num} with 16,384 features on {device}")
    
    activations = []
    labels = []
    missing_tasks = []
    
    for _, row in split_data.iterrows():
        task_id = row['task_id']
        # Load raw activations from Phase 3.5
        act_file = f'data/phase3_5/activations/task_activations/{task_id}_layer_{layer_num}.npz'
        
        if not os.path.exists(act_file):
            missing_tasks.append(task_id)
            continue
            
        act_data = np.load(act_file)
        
        # Get raw activation from Phase 3.5 (stored as 'arr_0')
        # Shape: (1, 2304) - raw residual stream activation
        raw_activation = torch.from_numpy(act_data['arr_0']).to(device)
        
        # Encode through SAE to get features
        # Shape: (1, 16384) - SAE feature activations
        with torch.no_grad():
            sae_features = sae.encode(raw_activation)
        
        # Extract specific feature value
        feature_activation = sae_features[0, feature_idx].item()
        activations.append(feature_activation)
        
        # Get test result from temperature 0.0 dataset
        task_results = temp_data[temp_data['task_id'] == task_id]['test_passed'].values
        if len(task_results) == 0:
            logger.warning(f"No test results found for task {task_id}")
            continue
            
        test_passed = task_results.mean() > 0.5  # Majority vote if multiple samples
        
        # Create label based on feature type
        if feature_type == 'correct':
            # For correct-preferring: 1=correct, 0=incorrect (flipped)
            label = 1 if test_passed else 0
        else:
            # For incorrect-preferring: 1=incorrect, 0=correct (standard)
            label = 0 if test_passed else 1
        
        labels.append(label)
    
    if missing_tasks:
        logger.warning(f"Missing activation files for {len(missing_tasks)} tasks: {missing_tasks[:5]}...")
    
    # Clean up SAE to free memory
    del sae
    torch.cuda.empty_cache()
    
    return np.array(labels), np.array(activations)
```

### Phase 2: Evaluate Correct-Preferring Feature
```python
# Load hyperparameter split for correct feature
y_true_hp_correct, scores_hp_correct = load_split_activations(
    'hyperparams', correct_layer, correct_feature_idx, 'correct'
)

print(f"\nCorrect-preferring feature (hyperparameter split):")
print(f"Total samples: {len(y_true_hp_correct)}")
print(f"Positive class (correct code): {sum(y_true_hp_correct == 1)}")
print(f"Negative class (incorrect code): {sum(y_true_hp_correct == 0)}")

# Find optimal threshold
optimal_threshold_correct, hp_metrics_correct = find_optimal_threshold(
    y_true_hp_correct, 
    scores_hp_correct,
    'correct',
    output_dir
)

# Load validation split
y_true_val_correct, scores_val_correct = load_split_activations(
    'validation', correct_layer, correct_feature_idx, 'correct'
)

print(f"\nCorrect-preferring feature (validation split):")
print(f"Total samples: {len(y_true_val_correct)}")

# Evaluate on validation set
val_metrics_correct = calculate_metrics(
    y_true_val_correct, 
    scores_val_correct, 
    optimal_threshold_correct,
    'correct',
    output_dir
)
```

### Phase 3: Evaluate Incorrect-Preferring Feature
```python
# Load hyperparameter split for incorrect feature
y_true_hp_incorrect, scores_hp_incorrect = load_split_activations(
    'hyperparams', incorrect_layer, incorrect_feature_idx, 'incorrect'
)

print(f"\nIncorrect-preferring feature (hyperparameter split):")
print(f"Total samples: {len(y_true_hp_incorrect)}")
print(f"Positive class (incorrect code): {sum(y_true_hp_incorrect == 1)}")
print(f"Negative class (correct code): {sum(y_true_hp_incorrect == 0)}")

# Find optimal threshold
optimal_threshold_incorrect, hp_metrics_incorrect = find_optimal_threshold(
    y_true_hp_incorrect, 
    scores_hp_incorrect,
    'incorrect',
    output_dir
)

# Load validation split
y_true_val_incorrect, scores_val_incorrect = load_split_activations(
    'validation', incorrect_layer, incorrect_feature_idx, 'incorrect'
)

print(f"\nIncorrect-preferring feature (validation split):")
print(f"Total samples: {len(y_true_val_incorrect)}")

# Evaluate on validation set
val_metrics_incorrect = calculate_metrics(
    y_true_val_incorrect, 
    scores_val_incorrect, 
    optimal_threshold_incorrect,
    'incorrect',
    output_dir
)
```

### Phase 4: Save Combined Results
```python
# Compile results for both features
results = {
    'phase': '3.8',
    'correct_preferring_feature': {
        'feature': {
            'idx': correct_feature_idx,
            'layer': correct_layer
        },
        'threshold_optimization': {
            'split': 'hyperparameter',
            'n_samples': len(y_true_hp_correct),
            'optimal_threshold': float(optimal_threshold_correct),
            'metrics': hp_metrics_correct
        },
        'validation_metrics': {
            'split': 'validation',
            'n_samples': len(y_true_val_correct),
            'metrics': val_metrics_correct
        }
    },
    'incorrect_preferring_feature': {
        'feature': {
            'idx': incorrect_feature_idx,
            'layer': incorrect_layer
        },
        'threshold_optimization': {
            'split': 'hyperparameter',
            'n_samples': len(y_true_hp_incorrect),
            'optimal_threshold': float(optimal_threshold_incorrect),
            'metrics': hp_metrics_incorrect
        },
        'validation_metrics': {
            'split': 'validation',
            'n_samples': len(y_true_val_incorrect),
            'metrics': val_metrics_incorrect
        }
    },
    'creation_timestamp': datetime.now().isoformat()
}

# Save comprehensive results
with open('data/phase3_8/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Generate comparative summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"\nCorrect-Preferring Feature (Layer {correct_layer}, Feature {correct_feature_idx}):")
print(f"  Validation AUROC: {val_metrics_correct['auroc']:.4f}")
print(f"  Validation F1: {val_metrics_correct['f1']:.4f}")
print(f"\nIncorrect-Preferring Feature (Layer {incorrect_layer}, Feature {incorrect_feature_idx}):")
print(f"  Validation AUROC: {val_metrics_incorrect['auroc']:.4f}")
print(f"  Validation F1: {val_metrics_incorrect['f1']:.4f}")
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `threshold_grid_size` | 100 | Number of thresholds tested for F1 optimization |
| `random_seed` | 42 | For reproducible results |
| `plot_visualizations` | True | Generate and save threshold plots and confusion matrices |
| `output_dir` | data/phase3_8/ | Directory for saving results and visualizations |
| `sae_width` | 16384 | Number of features in GemmaScope SAE |
| `raw_activation_dim` | 2304 | Dimension of raw residual stream activations |

## Metric Interpretation

### AUROC (Area Under ROC Curve)
- **Range**: 0 to 1 (0.5 = random classifier)
- **Interpretation**: Probability that model ranks random positive > random negative
- **Advantage**: Threshold-independent, good for imbalanced datasets
- **Usage**: Overall discriminative ability assessment

### F1 Score
- **Formula**: 2 * (precision * recall) / (precision + recall)
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: Harmonic mean of precision and recall
- **Advantage**: Balances false positives and false negatives
- **Usage**: When both precision and recall are important

### F1-Optimal Threshold Selection
- **Method**: Grid search across threshold range
- **Objective**: Maximize F1 score (harmonic mean of precision and recall)
- **Rationale**: Balances precision and recall equally
- **Use case**: When false positives and false negatives are equally costly

## Visualization Components

### Confusion Matrix with Feature Type
```python
def plot_confusion_matrix(y_true, y_pred, feature_type, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Adjust labels based on feature type
    if feature_type == 'correct':
        labels = ['Incorrect', 'Correct']
    else:
        labels = ['Correct', 'Incorrect']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {feature_type.capitalize()}-Preferring Feature')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    plt.savefig(output_dir / f'confusion_matrix_{feature_type}.png')
    plt.close()
```

### Comparative Visualization
```python
def plot_comparative_metrics(results, output_dir):
    """Create side-by-side comparison of both feature performances."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract metrics
    metrics = ['AUROC', 'F1', 'Precision', 'Recall']
    correct_vals = [
        results['correct_preferring_feature']['validation_metrics']['metrics'][m.lower()]
        for m in metrics
    ]
    incorrect_vals = [
        results['incorrect_preferring_feature']['validation_metrics']['metrics'][m.lower()]
        for m in metrics
    ]
    
    # Plot bars
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, correct_vals, width, label='Correct-Preferring')
    ax1.bar(x + width/2, incorrect_vals, width, label='Incorrect-Preferring')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Feature Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim([0, 1])
    
    # Save
    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_metrics.png')
    plt.close()
```

## Implementation Checklist

### Setup Phase
- [ ] Import scikit-learn metrics: `roc_auc_score`, `f1_score`, `precision_score`, `recall_score`
- [ ] Import visualization tools: matplotlib, seaborn for confusion matrix
- [ ] Set random seed for reproducibility
- [ ] Create output directory: `data/phase3_8/`

### Data Loading
- [ ] Load Phase 3.5 metadata.json to get best features for both directions
- [ ] Extract layer and feature indices for correct-preferring feature
- [ ] Extract layer and feature indices for incorrect-preferring feature
- [ ] Load Phase 3.5 temperature 0.0 dataset for test_passed labels

### Correct-Preferring Feature Evaluation
- [ ] Load hyperparameter split activations for correct feature
- [ ] Create flipped labels (1=correct, 0=incorrect)
- [ ] Find F1-optimal threshold on hyperparameter split
- [ ] Load validation split activations for correct feature
- [ ] Evaluate on validation split with optimal threshold
- [ ] Generate and save visualizations (threshold plot, confusion matrix)

### Incorrect-Preferring Feature Evaluation
- [ ] Load hyperparameter split activations for incorrect feature
- [ ] Create standard labels (1=incorrect, 0=correct)
- [ ] Find F1-optimal threshold on hyperparameter split
- [ ] Load validation split activations for incorrect feature
- [ ] Evaluate on validation split with optimal threshold
- [ ] Generate and save visualizations (threshold plot, confusion matrix)

### Output Generation
- [ ] Create evaluation_results.json with metrics for both features
- [ ] Save all threshold optimization plots
- [ ] Save all confusion matrices
- [ ] Generate comparative visualization
- [ ] Create human-readable summary with both feature results

## Data Dependencies

### Input Requirements
Phase 3.8 requires the following completed phases:

1. **Phase 0.1**: Problem splits
   - `data/phase0_1/hyperparams_mbpp.parquet` - For threshold optimization
   - `data/phase0_1/validation_mbpp.parquet` - For final evaluation

2. **Phase 3.5**: Temperature Robustness Testing
   - `data/phase3_5/metadata.json` - Contains best features for both directions
   - `data/phase3_5/activations/task_activations/{task_id}_layer_{n}.npz` - Raw activation files
     - Contains `arr_0` array with shape (1, 2304) - raw residual stream activations
     - NOT pre-encoded SAE features - requires encoding through GemmaScope
   - `data/phase3_5/dataset_temp_0_0.parquet` - Temperature 0.0 results with test_passed labels
   - May need activations from multiple layers if best features are on different layers

3. **GemmaScope SAE Models** (downloaded automatically):
   - Loaded from `google/gemma-scope-2b-pt-res` HuggingFace repository
   - 16,384 features per layer
   - Different sparsity levels per layer (defined in `GEMMA_2B_SPARSITY`)

### Output Structure
```
data/phase3_8/
├── evaluation_results.json              # All metrics for both features
├── f1_threshold_plot_correct.png       # F1 vs threshold for correct-preferring
├── f1_threshold_plot_incorrect.png     # F1 vs threshold for incorrect-preferring
├── confusion_matrix_correct.png        # Confusion matrix for correct-preferring
├── confusion_matrix_incorrect.png      # Confusion matrix for incorrect-preferring
├── comparative_metrics.png             # Side-by-side comparison
└── evaluation_summary.txt              # Human-readable summary
```

## Reusable Utilities from Common Modules

### From `common_simplified/helpers.py`:
- `load_mbpp_from_phase0_1(split_name, phase0_1_dir)` - Load MBPP split data
- `save_json(data, filepath)` - Save results to JSON

### From `common/utils.py`:
- `detect_device()` - Detect available device (CUDA > MPS > CPU)
- `ensure_directory_exists(directory)` - Create output directories
- `discover_latest_phase_output(phase)` - Auto-discover Phase 3.5 outputs

### From `phase2_5_simplified/sae_analyzer.py`:
- `load_gemma_scope_sae(layer_idx, device)` - Load GemmaScope SAE for encoding
- `JumpReLUSAE` - SAE model class with encode() method

### From `common/config.py`:
- `GEMMA_2B_SPARSITY` - Sparsity levels for each layer (needed for SAE loading)

### To Be Implemented in Phase 3.8:
- Metric calculations (AUROC, F1, precision, recall) using scikit-learn
- Visualization functions for confusion matrix and threshold plots
- Threshold optimization logic
- Main evaluation pipeline that handles SAE encoding

## Notes

- **Data Dependencies**: Phase 3.8 requires completed Phase 3.5 (contains both activations and best feature info)
- **SAE Encoding Required**: Phase 3.5 saves raw model activations; Phase 3.8 must encode them through GemmaScope SAE
- **Two Separate Evaluations**: Evaluates correct-preferring and incorrect-preferring features independently
- **Label Conventions**: 
  - Correct-preferring: 1=correct, 0=incorrect (flipped)
  - Incorrect-preferring: 1=incorrect, 0=correct (standard)
- **Threshold Selection**: Separate F1-optimal thresholds for each feature type
- **No Balancing**: Uses natural distribution of correct/incorrect samples
- **Memory Management**: Load SAE once per split, clean up after encoding to free GPU memory
- **Direct Evaluation**: No intermediate classifiers; raw SAE activations serve as scores

## Development Note

For proper testing of Phase 3.8, ensure Phase 3.5 has been run with:
- At least 10-20 tasks from the validation split
- A mix of correct and incorrect results (both classes needed for metrics)
- Temperature 0.0 only is sufficient for Phase 3.8 evaluation