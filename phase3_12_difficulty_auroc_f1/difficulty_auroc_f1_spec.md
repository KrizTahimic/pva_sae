# Phase 3.12: Difficulty-Based AUROC Analysis for PVA-SAE

## Executive Summary

1. **Purpose**: Analyze how PVA feature effectiveness varies across problem difficulty levels in Python code generation
2. **Task**: Stratify validation dataset by cyclomatic complexity and evaluate AUROC performance per difficulty group
3. **Metrics**: AUROC (Area Under ROC Curve) and F1 Score for each difficulty level for both feature directions
4. **Implementation**: Leverage Phase 3.8 infrastructure with difficulty-based data grouping
5. **Difficulty Grouping**: Three-tier classification based on cyclomatic complexity scores
6. **Dataset**: Validation split from Phase 0.1 (388 problems) with cyclomatic complexity annotations
7. **Feature Analysis**: Evaluate both correct-preferring and incorrect-preferring features per difficulty
8. **F1 Threshold Strategy**: Use Phase 3.8's global F1-optimal threshold for all difficulty groups
9. **Data Source**: Phase 3.5 activations and Phase 3.8 best features from validation split

## Pipeline Sequence

```
1. Load Phase 3.8 results and identify best features + thresholds
   └─> Read evaluation_results.json → Extract best correct & incorrect features → Note layers & indices → Extract pre-computed F1-optimal thresholds

2. Load validation dataset and group by difficulty
   a. Load Phase 0.1 validation split with cyclomatic complexity
   b. Group into Easy (complexity=1), Medium (complexity=2-3), Hard (complexity>=4)
   c. Report group sizes and distributions

3. Evaluate Correct-Preferring Feature Across Difficulty Groups
   a. For each difficulty group:
      - Load activations and create flipped labels (1=correct, 0=incorrect)
      - Calculate AUROC (threshold-independent)
      - Calculate F1 using Phase 3.8's global threshold
      - Generate ROC curve and save visualization
   b. Compare AUROC and F1 across difficulty levels
   c. Generate comparative analysis

4. Evaluate Incorrect-Preferring Feature Across Difficulty Groups
   a. For each difficulty group:
      - Load activations and create standard labels (1=incorrect, 0=correct)
      - Calculate AUROC (threshold-independent)
      - Calculate F1 using Phase 3.8's global threshold
      - Generate ROC curve and save visualization
   b. Compare AUROC and F1 across difficulty levels
   c. Generate comparative analysis

5. Output comprehensive difficulty analysis to data/phase3_12/
   └─> Save per-group results → Save comparative visualizations → Generate insights summary
```

## Phase Relationship

### Why Phase 3.12 Depends on Phase 3.8
Phase 3.12 extends Phase 3.8 by adding difficulty stratification:
- Both phases evaluate the same validation split (388 problems)
- Phase 3.12 reuses Phase 3.8's best feature identification
- Phase 3.12 leverages Phase 3.8's AUROC calculation infrastructure
- Phase 3.12 uses the same SAE loading and activation processing logic

### Why Phase 3.12 Depends on Phase 0.1
- Phase 0.1 provides the validation split with cyclomatic complexity annotations
- Cyclomatic complexity is pre-calculated and stored in the dataset
- Phase 0.1 ensures consistent problem splitting across all validation phases

### Why Phase 3.12 Depends on Phase 3.5
- Phase 3.5 provides the raw activations for the best PVA layer
- Phase 3.5 provides the temperature 0.0 test results for ground truth labels
- Phase 3.5 metadata contains the optimal layer selection for activation extraction

### Key Difference from Phase 3.8
- **Phase 3.8**: Evaluates features on entire validation set
- **Phase 3.12**: Evaluates features separately on Easy/Medium/Hard subsets to understand complexity effects

## Understanding the Evaluation Task

### Difficulty-Based AUROC Analysis
Phase 3.12 answers the research question: "Do PVA features work better on simple or complex programming problems?"

#### Difficulty Grouping Strategy
Based on cyclomatic complexity distribution in validation set:
- **Easy**: Complexity = 1 (106 tasks, 27.3%)
- **Medium**: Complexity = 2-3 (181 tasks, 46.6%)
- **Hard**: Complexity >= 4 (101 tasks, 26.0%)

#### Two Feature Directions Per Difficulty Level

##### 1. Correct-Preferring Feature Analysis
- **Evaluation**: AUROC for correct vs incorrect classification per difficulty
- **Labels**: 1=Correct, 0=Incorrect (flipped convention)
- **Research Question**: Does correct-preferring feature effectiveness vary with problem complexity?

##### 2. Incorrect-Preferring Feature Analysis
- **Evaluation**: AUROC for incorrect vs correct classification per difficulty
- **Labels**: 1=Incorrect, 0=Correct (standard convention)
- **Research Question**: Does incorrect-preferring feature effectiveness vary with problem complexity?

### Expected Insights
- **Complexity Sensitivity**: Which feature type is more affected by problem complexity?
- **Difficulty Threshold**: Is there a complexity level where feature effectiveness drops?
- **Feature Complementarity**: Do correct/incorrect features show different complexity patterns?

## Understanding Key Variables

### Difficulty Group Structure
```python
# Example difficulty grouping
difficulty_groups = {
    'easy': {
        'complexity_range': [1, 1],
        'task_ids': ['mbpp_1', 'mbpp_5', ...],  # 106 tasks
        'description': 'Simple problems with basic logic'
    },
    'medium': {
        'complexity_range': [2, 3],
        'task_ids': ['mbpp_2', 'mbpp_8', ...],  # 181 tasks
        'description': 'Moderate complexity with conditional logic'
    },
    'hard': {
        'complexity_range': [4, 16],
        'task_ids': ['mbpp_12', 'mbpp_23', ...],  # 101 tasks
        'description': 'Complex problems with nested logic'
    }
}
```

### Per-Group Evaluation Variables

#### For Correct-Preferring Features (per difficulty group)
```python
# Easy group example
y_true_correct_easy = [1, 1, 0, 1, 0, 1, 1, 0, ...]  # 106 labels
scores_correct_easy = [0.89, 0.91, 0.12, 0.76, ...]  # 106 activations
auroc_correct_easy = roc_auc_score(y_true_correct_easy, scores_correct_easy)
```

#### For Incorrect-Preferring Features (per difficulty group)
```python
# Hard group example
y_true_incorrect_hard = [0, 0, 1, 0, 1, 0, 0, 1, ...]  # 101 labels
scores_incorrect_hard = [0.15, 0.23, 0.89, 0.34, ...]  # 101 activations
auroc_incorrect_hard = roc_auc_score(y_true_incorrect_hard, scores_incorrect_hard)
```

### Comparative Analysis Variables
```python
# AUROC comparison across difficulty levels
auroc_comparison = {
    'correct_preferring': {
        'easy': 0.85,
        'medium': 0.78,
        'hard': 0.72
    },
    'incorrect_preferring': {
        'easy': 0.81,
        'medium': 0.83,
        'hard': 0.79
    }
}
```

## Core Implementation

### 1. Difficulty Grouping Function
```python
def group_by_difficulty(validation_data):
    """Group validation tasks by cyclomatic complexity into Easy/Medium/Hard."""
    
    # Define difficulty thresholds
    difficulty_groups = {
        'easy': validation_data[validation_data['cyclomatic_complexity'] == 1],
        'medium': validation_data[validation_data['cyclomatic_complexity'].between(2, 3)],
        'hard': validation_data[validation_data['cyclomatic_complexity'] >= 4]
    }
    
    # Log group sizes
    for group_name, group_data in difficulty_groups.items():
        print(f"{group_name.capitalize()} group: {len(group_data)} tasks "
              f"(complexity range: {group_data['cyclomatic_complexity'].min()}-"
              f"{group_data['cyclomatic_complexity'].max()})")
    
    return difficulty_groups
```

### 2. Per-Group AUROC and F1 Calculation (Reused from Phase 3.8)
```python
def calculate_difficulty_metrics(difficulty_groups, best_features, global_threshold, feature_type, output_dir, sae, device, temp_data):
    """Calculate AUROC and F1 for each difficulty group for a specific feature type.
    
    Args:
        difficulty_groups: Dict of difficulty groups
        best_features: Best feature information
        global_threshold: F1-optimal threshold from Phase 3.8
        feature_type: 'correct' or 'incorrect'
        output_dir: Output directory for plots
        sae: Pre-loaded SAE model
        device: Device for computation
        temp_data: Pre-loaded temperature 0.0 dataset
    """
    
    results = {}
    
    for group_name, group_data in difficulty_groups.items():
        print(f"\nEvaluating {feature_type}-preferring feature on {group_name} group:")
        
        # Get feature info
        if feature_type == 'correct':
            layer = best_features['correct']
            feature_idx = best_features['correct_feature_idx']
        else:
            layer = best_features['incorrect']
            feature_idx = best_features['incorrect_feature_idx']
        
        # Load activations for this group (reuse Phase 3.8 logic)
        y_true, scores = load_group_activations(group_data, layer, feature_idx, feature_type, sae, device, temp_data)
        
        # Calculate AUROC (threshold-independent)
        auroc = roc_auc_score(y_true, scores)
        
        # Calculate F1 using global threshold from Phase 3.8
        y_pred = (scores > global_threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        
        # Save ROC curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {feature_type.capitalize()}-Preferring Feature ({group_name.capitalize()} Group)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'roc_curve_{feature_type}_{group_name}.png')
        plt.close()
        
        results[group_name] = {
            'auroc': auroc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'n_samples': len(y_true),
            'n_positive': sum(y_true),
            'n_negative': len(y_true) - sum(y_true),
            'complexity_range': [
                group_data['cyclomatic_complexity'].min(),
                group_data['cyclomatic_complexity'].max()
            ]
        }
        
        print(f"  AUROC: {auroc:.4f}")
        print(f"  F1: {f1:.4f} (using global threshold: {global_threshold:.4f})")
        print(f"  Samples: {len(y_true)} (pos: {sum(y_true)}, neg: {len(y_true) - sum(y_true)})")
    
    return results
```

### 3. Activation Loading for Difficulty Groups (Adapted from Phase 3.8)
```python
def load_group_activations(group_data, layer_num, feature_idx, feature_type, sae, device, temp_data):
    """Load activations for a specific difficulty group.
    
    Args:
        group_data: DataFrame with tasks for this difficulty group
        layer_num: Layer number for SAE
        feature_idx: Feature index to extract
        feature_type: 'correct' or 'incorrect'
        sae: Pre-loaded SAE model
        device: Device for computation
        temp_data: Pre-loaded temperature 0.0 dataset
    """
    
    activations = []
    labels = []
    missing_tasks = []
    
    for _, row in group_data.iterrows():
        task_id = row['task_id']
        
        # Load raw activations from Phase 3.5
        act_file = f'data/phase3_5/activations/task_activations/{task_id}_layer_{layer_num}.npz'
        
        if not os.path.exists(act_file):
            missing_tasks.append(task_id)
            continue
            
        # Load and encode through SAE (same as Phase 3.8)
        act_data = np.load(act_file)
        raw_activation = torch.from_numpy(act_data['arr_0']).to(device)
        
        with torch.no_grad():
            sae_features = sae.encode(raw_activation)
        
        # Extract specific feature value
        feature_activation = sae_features[0, feature_idx].item()
        activations.append(feature_activation)
        
        # Get test result and create label
        task_results = temp_data[temp_data['task_id'] == task_id]['test_passed'].values
        if len(task_results) == 0:
            continue
            
        test_passed = task_results.mean() > 0.5
        
        # Create label based on feature type
        if feature_type == 'correct':
            label = 1 if test_passed else 0  # Flipped for correct-preferring
        else:
            label = 0 if test_passed else 1  # Standard for incorrect-preferring
        
        labels.append(label)
    
    if missing_tasks:
        logger.warning(f"Missing activation files for {len(missing_tasks)} tasks")
    
    # Check for edge cases
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        logger.warning(f"WARNING: {feature_type}-preferring feature has imbalanced classes - "
                      f"positive: {n_positive}, negative: {n_negative}")
    elif n_positive < 5 or n_negative < 5:
        logger.warning(f"WARNING: {feature_type}-preferring feature has very few samples in one class - "
                      f"positive: {n_positive}, negative: {n_negative}")
    
    return np.array(labels), np.array(activations)
```

## Implementation Workflow

### Phase 1: Load Dependencies and Setup
```python
# Load Phase 3.8 results to get best features and thresholds
phase3_8_results = json.load(open('data/phase3_8/evaluation_results.json'))
best_features = {
    'correct': phase3_8_results['correct_preferring_feature']['feature']['layer'],
    'correct_feature_idx': phase3_8_results['correct_preferring_feature']['feature']['idx'],
    'incorrect': phase3_8_results['incorrect_preferring_feature']['feature']['layer'],
    'incorrect_feature_idx': phase3_8_results['incorrect_preferring_feature']['feature']['idx']
}

# Extract global F1-optimal thresholds from Phase 3.8
global_thresholds = {
    'correct': phase3_8_results['correct_preferring_feature']['threshold_optimization']['optimal_threshold'],
    'incorrect': phase3_8_results['incorrect_preferring_feature']['threshold_optimization']['optimal_threshold']
}

print(f"Best correct-preferring feature: idx {best_features['correct_feature_idx']} at layer {best_features['correct']} (threshold: {global_thresholds['correct']:.4f})")
print(f"Best incorrect-preferring feature: idx {best_features['incorrect_feature_idx']} at layer {best_features['incorrect']} (threshold: {global_thresholds['incorrect']:.4f})")

# Load validation dataset with cyclomatic complexity
validation_data = pd.read_parquet('data/phase0_1/validation_mbpp.parquet')
print(f"Validation dataset loaded: {len(validation_data)} tasks")
print(f"Cyclomatic complexity range: {validation_data['cyclomatic_complexity'].min()}-{validation_data['cyclomatic_complexity'].max()}")

# Create output directory
output_dir = Path('data/phase3_12')
output_dir.mkdir(parents=True, exist_ok=True)
```

### Phase 2: Group by Difficulty
```python
# Group validation tasks by difficulty
difficulty_groups = group_by_difficulty(validation_data)

# Generate difficulty distribution visualization
plt.figure(figsize=(10, 6))
group_sizes = [len(group) for group in difficulty_groups.values()]
group_names = list(difficulty_groups.keys())

plt.bar(group_names, group_sizes, color=['lightgreen', 'orange', 'lightcoral'])
plt.xlabel('Difficulty Group')
plt.ylabel('Number of Tasks')
plt.title('Task Distribution by Difficulty Level')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(group_sizes):
    plt.text(i, v + 2, str(v), ha='center', va='bottom')

plt.savefig(output_dir / 'difficulty_distribution.png')
plt.close()
```

### Phase 3: Evaluate Correct-Preferring Feature
```python
print("\n" + "="*60)
print("EVALUATING CORRECT-PREFERRING FEATURE ACROSS DIFFICULTY LEVELS")
print("="*60)

# Load temperature 0.0 dataset once
temp_data = pd.read_parquet('data/phase3_5/dataset_temp_0_0.parquet')

# Detect device once
device = detect_device()

# Load SAE for correct-preferring feature
correct_layer = best_features['correct']
sae_correct = load_gemma_scope_sae(correct_layer, device)
logger.info(f"Loaded SAE for layer {correct_layer} on {device}")

correct_results = calculate_difficulty_metrics(
    difficulty_groups, 
    best_features, 
    global_thresholds['correct'],
    'correct', 
    output_dir,
    sae_correct,
    device,
    temp_data
)

# Clean up SAE after use
del sae_correct
torch.cuda.empty_cache()

# Generate comparative visualization for correct-preferring
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
difficulties = list(correct_results.keys())
aurocs = [correct_results[d]['auroc'] for d in difficulties]
f1s = [correct_results[d]['f1'] for d in difficulties]

# AUROC plot
ax1.plot(difficulties, aurocs, marker='o', linewidth=2, markersize=8, 
         label='Correct-Preferring Feature', color='blue')
ax1.set_xlabel('Difficulty Level')
ax1.set_ylabel('AUROC')
ax1.set_title('AUROC Performance by Difficulty Level - Correct-Preferring Feature')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)
ax1.legend()

# Add value labels
for i, auroc in enumerate(aurocs):
    ax1.text(i, auroc + 0.02, f'{auroc:.3f}', ha='center', va='bottom')

# F1 plot
ax2.plot(difficulties, f1s, marker='o', linewidth=2, markersize=8, 
         label='Correct-Preferring Feature', color='green')
ax2.set_xlabel('Difficulty Level')
ax2.set_ylabel('F1 Score')
ax2.set_title('F1 Performance by Difficulty Level - Correct-Preferring Feature')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)
ax2.legend()

# Add value labels
for i, f1 in enumerate(f1s):
    ax2.text(i, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(output_dir / 'metrics_by_difficulty_correct.png')
plt.close()
```

### Phase 4: Evaluate Incorrect-Preferring Feature
```python
print("\n" + "="*60)
print("EVALUATING INCORRECT-PREFERRING FEATURE ACROSS DIFFICULTY LEVELS")
print("="*60)

# Load SAE for incorrect-preferring feature
incorrect_layer = best_features['incorrect']
sae_incorrect = load_gemma_scope_sae(incorrect_layer, device)
logger.info(f"Loaded SAE for layer {incorrect_layer} on {device}")

incorrect_results = calculate_difficulty_metrics(
    difficulty_groups, 
    best_features, 
    global_thresholds['incorrect'],
    'incorrect', 
    output_dir,
    sae_incorrect,
    device,
    temp_data
)

# Clean up SAE after use
del sae_incorrect
torch.cuda.empty_cache()

# Generate comparative visualization for incorrect-preferring
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
difficulties = list(incorrect_results.keys())
aurocs = [incorrect_results[d]['auroc'] for d in difficulties]
f1s = [incorrect_results[d]['f1'] for d in difficulties]

# AUROC plot
ax1.plot(difficulties, aurocs, marker='s', linewidth=2, markersize=8, 
         label='Incorrect-Preferring Feature', color='red')
ax1.set_xlabel('Difficulty Level')
ax1.set_ylabel('AUROC')
ax1.set_title('AUROC Performance by Difficulty Level - Incorrect-Preferring Feature')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)
ax1.legend()

# Add value labels
for i, auroc in enumerate(aurocs):
    ax1.text(i, auroc + 0.02, f'{auroc:.3f}', ha='center', va='bottom')

# F1 plot
ax2.plot(difficulties, f1s, marker='s', linewidth=2, markersize=8, 
         label='Incorrect-Preferring Feature', color='orange')
ax2.set_xlabel('Difficulty Level')
ax2.set_ylabel('F1 Score')
ax2.set_title('F1 Performance by Difficulty Level - Incorrect-Preferring Feature')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)
ax2.legend()

# Add value labels
for i, f1 in enumerate(f1s):
    ax2.text(i, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(output_dir / 'metrics_by_difficulty_incorrect.png')
plt.close()
```

### Phase 5: Comparative Analysis and Results
```python
# Generate side-by-side comparison
plt.figure(figsize=(12, 6))
difficulties = list(correct_results.keys())
correct_aurocs = [correct_results[d]['auroc'] for d in difficulties]
incorrect_aurocs = [incorrect_results[d]['auroc'] for d in difficulties]
correct_f1s = [correct_results[d]['f1'] for d in difficulties]
incorrect_f1s = [incorrect_results[d]['f1'] for d in difficulties]

# Create side-by-side comparison for both metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

x = np.arange(len(difficulties))
width = 0.35

# AUROC comparison
ax1.bar(x - width/2, correct_aurocs, width, label='Correct-Preferring', color='blue', alpha=0.7)
ax1.bar(x + width/2, incorrect_aurocs, width, label='Incorrect-Preferring', color='red', alpha=0.7)
ax1.set_xlabel('Difficulty Level')
ax1.set_ylabel('AUROC')
ax1.set_title('AUROC Performance Comparison Across Difficulty Levels')
ax1.set_xticks(x)
ax1.set_xticklabels([d.capitalize() for d in difficulties])
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Add value labels for AUROC
for i, (c_auroc, i_auroc) in enumerate(zip(correct_aurocs, incorrect_aurocs)):
    ax1.text(i - width/2, c_auroc + 0.01, f'{c_auroc:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.text(i + width/2, i_auroc + 0.01, f'{i_auroc:.3f}', ha='center', va='bottom', fontsize=9)

# F1 comparison
ax2.bar(x - width/2, correct_f1s, width, label='Correct-Preferring', color='green', alpha=0.7)
ax2.bar(x + width/2, incorrect_f1s, width, label='Incorrect-Preferring', color='orange', alpha=0.7)
ax2.set_xlabel('Difficulty Level')
ax2.set_ylabel('F1 Score')
ax2.set_title('F1 Performance Comparison Across Difficulty Levels')
ax2.set_xticks(x)
ax2.set_xticklabels([d.capitalize() for d in difficulties])
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

# Add value labels for F1
for i, (c_f1, i_f1) in enumerate(zip(correct_f1s, incorrect_f1s)):
    ax2.text(i - width/2, c_f1 + 0.01, f'{c_f1:.3f}', ha='center', va='bottom', fontsize=9)
    ax2.text(i + width/2, i_f1 + 0.01, f'{i_f1:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'metrics_comparison_by_difficulty.png')
plt.close()

# Generate additional comparative visualizations
# Note: We need to reload SAEs since they were deleted after individual analyses
sae_correct = load_gemma_scope_sae(best_features['correct'], device)
plot_roc_curves_by_difficulty(difficulty_groups, 'correct', correct_results, output_dir, 
                               best_features, sae_correct, device, temp_data)
del sae_correct
torch.cuda.empty_cache()

sae_incorrect = load_gemma_scope_sae(best_features['incorrect'], device)
plot_roc_curves_by_difficulty(difficulty_groups, 'incorrect', incorrect_results, output_dir,
                               best_features, sae_incorrect, device, temp_data)
del sae_incorrect
torch.cuda.empty_cache()

plot_auroc_trends(correct_results, incorrect_results, output_dir)

# Compile comprehensive results
results = {
    'phase': '3.12',
    'analysis_type': 'difficulty_based_auroc',
    'difficulty_groups': {
        group_name: {
            'complexity_range': group_data['cyclomatic_complexity'].min(),
            'complexity_max': group_data['cyclomatic_complexity'].max(),
            'n_tasks': len(group_data),
            'percentage': len(group_data) / len(validation_data) * 100
        }
        for group_name, group_data in difficulty_groups.items()
    },
    'best_features': best_features,
    'correct_preferring_results': correct_results,
    'incorrect_preferring_results': incorrect_results,
    'insights': {
        'correct_feature_trend': 'decreasing' if correct_aurocs[0] > correct_aurocs[-1] else 'increasing',
        'incorrect_feature_trend': 'decreasing' if incorrect_aurocs[0] > incorrect_aurocs[-1] else 'increasing',
        'most_effective_difficulty': {
            'correct': max(correct_results.keys(), key=lambda k: correct_results[k]['auroc']),
            'incorrect': max(incorrect_results.keys(), key=lambda k: incorrect_results[k]['auroc'])
        }
    },
    'creation_timestamp': datetime.now().isoformat()
}

# Save results
with open(output_dir / 'difficulty_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Generate human-readable summary
print("\n" + "="*60)
print("DIFFICULTY-BASED AUROC ANALYSIS SUMMARY")
print("="*60)
print(f"\nDataset: {len(validation_data)} validation tasks")
print(f"Difficulty Groups: Easy ({len(difficulty_groups['easy'])}), Medium ({len(difficulty_groups['medium'])}), Hard ({len(difficulty_groups['hard'])})")
print(f"\nCorrect-Preferring Feature (Layer {best_features['correct']}, Feature {best_features['correct_feature_idx']}):")
for difficulty, result in correct_results.items():
    print(f"  {difficulty.capitalize()}: AUROC = {result['auroc']:.4f}, F1 = {result['f1']:.4f} (n={result['n_samples']})")
print(f"\nIncorrect-Preferring Feature (Layer {best_features['incorrect']}, Feature {best_features['incorrect_feature_idx']}):")
for difficulty, result in incorrect_results.items():
    print(f"  {difficulty.capitalize()}: AUROC = {result['auroc']:.4f}, F1 = {result['f1']:.4f} (n={result['n_samples']})")
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `easy_complexity_threshold` | 1 | Cyclomatic complexity for easy problems |
| `medium_complexity_range` | [2, 3] | Cyclomatic complexity range for medium problems |
| `hard_complexity_threshold` | 4+ | Minimum cyclomatic complexity for hard problems |
| `output_dir` | data/phase3_12/ | Directory for saving results and visualizations |
| `random_seed` | 42 | For reproducible results |
| `plot_visualizations` | True | Generate and save ROC curves and comparisons |

## Visualization Components

### 1. Difficulty Distribution
```python
def plot_difficulty_distribution(difficulty_groups, output_dir):
    """Visualize the distribution of tasks across difficulty levels."""
    plt.figure(figsize=(10, 6))
    
    group_sizes = [len(group) for group in difficulty_groups.values()]
    group_names = [name.capitalize() for name in difficulty_groups.keys()]
    
    bars = plt.bar(group_names, group_sizes, color=['lightgreen', 'orange', 'lightcoral'])
    plt.xlabel('Difficulty Level')
    plt.ylabel('Number of Tasks')
    plt.title('Task Distribution by Difficulty Level (Cyclomatic Complexity)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels and percentages
    total_tasks = sum(group_sizes)
    for i, (bar, size) in enumerate(zip(bars, group_sizes)):
        percentage = size / total_tasks * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{size}\n({percentage:.1f}%)', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'difficulty_distribution.png')
    plt.close()
```

### 2. ROC Curves per Difficulty Group
```python
def plot_roc_curves_by_difficulty(difficulty_groups, feature_type, results, output_dir, 
                                   best_features, sae, device, temp_data):
    """Plot ROC curves for each difficulty group on the same plot."""
    plt.figure(figsize=(10, 8))
    
    colors = ['green', 'orange', 'red']
    layer = best_features[feature_type]
    feature_idx = best_features[f'{feature_type}_feature_idx']
    
    for i, (group_name, group_result) in enumerate(results.items()):
        # Re-calculate ROC curve points for plotting
        y_true, scores = load_group_activations(
            difficulty_groups[group_name], 
            layer,
            feature_idx, 
            feature_type,
            sae, device, temp_data
        )
        
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = group_result['auroc']
        
        plt.plot(fpr, tpr, linewidth=2, color=colors[i],
                label=f'{group_name.capitalize()} (AUC = {auroc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves by Difficulty - {feature_type.capitalize()}-Preferring Feature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_curves_by_difficulty_{feature_type}.png')
    plt.close()
```

### 3. AUROC Trend Analysis
```python
def plot_auroc_trends(correct_results, incorrect_results, output_dir):
    """Plot AUROC trends across difficulty levels for both feature types."""
    plt.figure(figsize=(12, 6))
    
    difficulties = list(correct_results.keys())
    correct_aurocs = [correct_results[d]['auroc'] for d in difficulties]
    incorrect_aurocs = [incorrect_results[d]['auroc'] for d in difficulties]
    
    plt.plot(difficulties, correct_aurocs, marker='o', linewidth=2, markersize=8,
             label='Correct-Preferring Feature', color='blue')
    plt.plot(difficulties, incorrect_aurocs, marker='s', linewidth=2, markersize=8,
             label='Incorrect-Preferring Feature', color='red')
    
    plt.xlabel('Difficulty Level')
    plt.ylabel('AUROC')
    plt.title('AUROC Performance Trends Across Difficulty Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add value labels
    for i, (c_auroc, i_auroc) in enumerate(zip(correct_aurocs, incorrect_aurocs)):
        plt.text(i, c_auroc + 0.02, f'{c_auroc:.3f}', ha='center', va='bottom', color='blue')
        plt.text(i, i_auroc - 0.05, f'{i_auroc:.3f}', ha='center', va='top', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'auroc_trends_by_difficulty.png')
    plt.close()
```

## Implementation Checklist

### Setup Phase
- [ ] Import scikit-learn metrics: `roc_auc_score`, `roc_curve`, `f1_score`, `precision_score`, `recall_score`
- [ ] Import visualization tools: matplotlib, seaborn
- [ ] Set random seed for reproducibility
- [ ] Create output directory: `data/phase3_12/`

### Data Loading
- [ ] Load Phase 3.8 results to get best features and F1-optimal thresholds for both directions
- [ ] Load Phase 0.1 validation split with cyclomatic complexity
- [ ] Verify cyclomatic complexity column exists and has valid values
- [ ] Group validation tasks by difficulty (Easy/Medium/Hard)

### Difficulty Analysis
- [ ] Calculate and visualize difficulty distribution
- [ ] Validate group sizes are reasonable (no empty groups)
- [ ] Log complexity ranges for each difficulty group
- [ ] Save difficulty grouping metadata

### Correct-Preferring Feature Analysis
- [ ] Load activations for each difficulty group
- [ ] Calculate AUROC for each difficulty group
- [ ] Calculate F1 for each difficulty group using Phase 3.8's global threshold
- [ ] Generate ROC curves for each difficulty group
- [ ] Create comparative AUROC and F1 trend visualization
- [ ] Save per-group results

### Incorrect-Preferring Feature Analysis
- [ ] Load activations for each difficulty group
- [ ] Calculate AUROC for each difficulty group
- [ ] Calculate F1 for each difficulty group using Phase 3.8's global threshold
- [ ] Generate ROC curves for each difficulty group
- [ ] Create comparative AUROC and F1 trend visualization
- [ ] Save per-group results

### Comparative Analysis
- [ ] Generate side-by-side AUROC and F1 comparison
- [ ] Create combined ROC curve plots
- [ ] Calculate trend analysis (increasing/decreasing performance) for both metrics
- [ ] Identify most effective difficulty level per feature type per metric

### Output Generation
- [ ] Create comprehensive difficulty_analysis_results.json with both AUROC and F1 metrics
- [ ] Save all visualizations (distribution, ROC curves, AUROC/F1 trends)
- [ ] Generate human-readable summary report with both metrics
- [ ] Create insights and recommendations for both AUROC and F1 performance

## Data Dependencies

### Input Requirements
Phase 3.12 requires the following completed phases:

1. **Phase 0.1**: Problem splits with cyclomatic complexity
   - `data/phase0_1/validation_mbpp.parquet` - Validation split with complexity annotations
   - Must contain `cyclomatic_complexity` column with numeric values

2. **Phase 3.8**: AUROC and F1 evaluation results
   - `data/phase3_8/evaluation_results.json` - Contains best features and F1-optimal thresholds for both directions
   - Provides layer indices, feature indices, and global thresholds for optimal PVA features

3. **Phase 3.5**: Temperature robustness testing
   - `data/phase3_5/activations/task_activations/{task_id}_layer_{n}.npz` - Raw activation files
   - `data/phase3_5/dataset_temp_0_0.parquet` - Temperature 0.0 results with test_passed labels
   - Must contain activations for the layers identified in Phase 3.8

4. **GemmaScope SAE Models** (downloaded automatically):
   - Loaded from `google/gemma-scope-2b-pt-res` HuggingFace repository
   - Required for encoding raw activations into SAE features

### Output Structure
```
data/phase3_12/
├── difficulty_analysis_results.json           # Comprehensive results for all groups
├── difficulty_distribution.png                # Task distribution by difficulty
├── roc_curve_correct_easy.png                 # ROC curve for correct feature, easy group
├── roc_curve_correct_medium.png               # ROC curve for correct feature, medium group
├── roc_curve_correct_hard.png                 # ROC curve for correct feature, hard group
├── roc_curve_incorrect_easy.png               # ROC curve for incorrect feature, easy group
├── roc_curve_incorrect_medium.png             # ROC curve for incorrect feature, medium group
├── roc_curve_incorrect_hard.png               # ROC curve for incorrect feature, hard group
├── roc_curves_by_difficulty_correct.png       # Combined ROC curves for correct feature
├── roc_curves_by_difficulty_incorrect.png     # Combined ROC curves for incorrect feature
├── metrics_by_difficulty_correct.png          # AUROC and F1 trends for correct feature
├── metrics_by_difficulty_incorrect.png        # AUROC and F1 trends for incorrect feature
├── metrics_comparison_by_difficulty.png       # Side-by-side AUROC and F1 comparison
└── difficulty_summary.txt                     # Human-readable summary
```

## Reusable Utilities from Common Modules

### From `phase3_8/auroc_f1_evaluator.py`:
- `load_split_activations()` - SAE loading and activation processing (adapt for groups)
- AUROC and F1 calculation patterns using scikit-learn
- Visualization framework for ROC curves
- Device detection and memory management
- Threshold loading and application logic

### From `common_simplified/helpers.py`:
- `load_mbpp_from_phase0_1(split_name, phase0_1_dir)` - Load MBPP split data
- `save_json(data, filepath)` - Save results to JSON

### From `common/utils.py`:
- `detect_device()` - Detect available device (CUDA > MPS > CPU)
- `ensure_directory_exists(directory)` - Create output directories
- `discover_latest_phase_output(phase)` - Auto-discover previous phase outputs

### From `phase2_5_simplified/sae_analyzer.py`:
- `load_gemma_scope_sae(layer_idx, device)` - Load GemmaScope SAE for encoding
- `JumpReLUSAE` - SAE model class with encode() method

### From `phase0_difficulty_analysis/difficulty_analyzer.py`:
- `get_cyclomatic_complexity()` - Calculate cyclomatic complexity (if needed)
- Understanding of complexity scoring methodology

### To Be Implemented in Phase 3.12:
- Difficulty-based grouping logic
- Multi-group AUROC and F1 evaluation framework
- Comparative visualization across difficulty levels for both metrics
- Trend analysis and insights generation for AUROC and F1 performance

## Notes

- **Data Dependencies**: Phase 3.12 requires Phase 3.8 (best features + thresholds), Phase 0.1 (validation split), and Phase 3.5 (activations)
- **Grouping Strategy**: Three-tier difficulty based on cyclomatic complexity distribution
- **Dual Metrics**: AUROC (threshold-independent) and F1 (using Phase 3.8's global threshold)
- **SAE Encoding Required**: Phase 3.5 saves raw activations; Phase 3.12 must encode through GemmaScope SAE
- **Comparative Analysis**: Evaluates how problem complexity affects PVA feature effectiveness for both metrics
- **Research Insights**: Answers whether PVA features work better on simple or complex problems
- **Memory Management**: Load SAE once per difficulty group, clean up after encoding
- **Statistical Validity**: Ensure adequate sample sizes per difficulty group (Easy: 106, Medium: 181, Hard: 101)

## Development Note

For proper testing of Phase 3.12, ensure:
- Phase 3.8 has been completed with valid best features and F1-optimal thresholds
- Phase 0.1 validation split contains cyclomatic complexity annotations
- Phase 3.5 activations exist for the required layers
- Each difficulty group has a reasonable mix of correct and incorrect samples
- Sufficient samples per group for statistically meaningful AUROC and F1 calculations

## Expected Research Outcomes

Phase 3.12 will provide insights into:
1. **Complexity Sensitivity**: Whether PVA features are more effective on simple vs complex problems (both AUROC and F1)
2. **Feature Differentiation**: How correct vs incorrect-preferring features respond to complexity across both metrics
3. **Threshold Effects**: Whether there's a complexity level where PVA effectiveness changes
4. **Metric Consistency**: Whether AUROC and F1 show similar or different patterns across difficulty levels
5. **Generalization**: How well PVA features work across different problem types

This analysis will inform the broader understanding of how language models internally represent program validity across varying levels of code complexity, providing both threshold-independent (AUROC) and threshold-dependent (F1) perspectives.