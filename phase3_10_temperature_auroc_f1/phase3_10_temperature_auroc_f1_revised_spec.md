# Phase 3.10: Temperature-Based AUROC Analysis for PVA-SAE (Revised)

## Executive Summary

**Purpose**: Analyze how PVA feature effectiveness varies across different temperature settings in Python code generation

**Key Questions**:
1. Do PVA features maintain their discriminative power across temperature variations?
2. At what temperature do features lose reliability for predicting code correctness?
3. How do correct vs incorrect-preferring features respond to temperature changes?

**Approach**: Evaluate AUROC and F1 metrics for best PVA features across temperature levels using per-sample analysis of Phase 3.5 validation data

## Critical Change: Per-Sample Analysis

**Problem**: The original majority voting approach caused severe class imbalance at higher temperatures, making AUROC/F1 calculation impossible when nearly all tasks were classified as "correct".

**Solution**: Treat each generated sample independently:
- 5 samples per task × 50 tasks = 250 data points per temperature
- Each sample has its own pass/fail label
- Samples from same task share activation values (identical prompt)
- No aggregation - cleaner, simpler, more statistically robust

## Pipeline Overview

```
1. Load Phase 3.8 best features & thresholds
   └─> Extract layer indices, feature indices, and F1-optimal thresholds

2. Load Phase 3.5 temperature datasets
   └─> Process temperature levels: [0.0, 0.2, 0.4, 0.6, 0.8]
   └─> Each task has 5 generated solutions per temperature

3. Process each sample independently
   └─> No aggregation - each sample is a data point
   └─> Same activation values for samples from same task

4. Evaluate features across temperatures
   └─> Calculate AUROC and F1 for each temperature
   └─> Generate ROC curves and trend visualizations

5. Output comprehensive analysis to data/phase3_10/
```

## Temperature Range Decision

Using [0.0, 0.2, 0.4, 0.6, 0.8] instead of [0.0, 0.3, 0.6, 0.9, 1.2]:
- Better granularity in the critical 0.0-0.8 range
- Avoids extreme temperatures where most code fails
- Maintains sufficient variation for analysis

## Implementation Decisions

1. **Missing Activation Files**: Fail fast if any activation is missing - ensures complete analysis
2. **SAE Memory Management**: Load SAEs once and reuse across temperatures
3. **Visualization**: Simple matplotlib plots (following KISS principle)
4. **Class Imbalance**: With per-sample analysis, we expect better balance
5. **Results Structure**: One final JSON with all results

## Core Implementation Structure

### Data Processing
```python
def process_temperature_data(temp_dataset, best_features, sae):
    """Process data for a single temperature using per-sample analysis."""
    sample_features = []
    sample_labels = []
    
    # Process each row as an individual sample
    for _, row in temp_dataset.iterrows():
        task_id = row['task_id']
        
        # Load pre-saved activation from Phase 3.5
        activation_path = f'data/phase3_5/activations/task_activations/{task_id}_layer_{best_features["layer"]}.npz'
        
        # Fail fast if activation file missing
        if not os.path.exists(activation_path):
            raise FileNotFoundError(
                f"Missing activation file for task {task_id} at {activation_path}. "
                f"Phase 3.10 requires all activations from Phase 3.5 to be present."
            )
        
        try:
            raw_activation = np.load(activation_path)['arr_0']
            
            # Encode through SAE to get feature value
            with torch.no_grad():
                raw_tensor = torch.from_numpy(raw_activation).to(device)
                sae_features = sae.encode(raw_tensor)
                feature_value = sae_features[0, best_features['feature_idx']].item()
            
            # Each sample has its own label
            label = int(row['test_passed'])
            
            sample_features.append(feature_value)
            sample_labels.append(label)
            
        except Exception as e:
            logger.error(f"Error processing {task_id}: {str(e)}")
            raise
    
    return np.array(sample_features), np.array(sample_labels)
```

### Main Evaluation Loop
```python
def evaluate_across_temperatures(best_features):
    """Evaluate feature performance at each temperature."""
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8]  # Fixed temperature range
    results = {}
    
    # Load SAEs once for reuse
    sae_correct = load_gemma_scope_sae(best_features['correct']['layer'], device)
    sae_incorrect = load_gemma_scope_sae(best_features['incorrect']['layer'], device)
    
    for temp in temperatures:
        # Load temperature dataset
        temp_data = load_temperature_dataset(temp)
        results[temp] = {}
        
        logger.info(f"Processing temperature {temp}: {len(temp_data)} samples")
        
        # Process for both feature types
        for feature_type in ['correct', 'incorrect']:
            sae = sae_correct if feature_type == 'correct' else sae_incorrect
            
            features, labels = process_temperature_data(
                temp_data, 
                best_features[feature_type],
                sae
            )
            
            # Flip labels for correct-preferring features
            if feature_type == 'correct':
                labels = 1 - labels
            
            # Calculate metrics
            n_positive = sum(labels)
            n_negative = len(labels) - n_positive
            
            logger.info(f"  {feature_type}: {n_positive} positive, {n_negative} negative samples")
            
            if n_positive < 2 or n_negative < 2:
                # Still check for extreme imbalance
                logger.warning(f"Class imbalance at temp {temp} for {feature_type}: "
                             f"pos={n_positive}, neg={n_negative}")
                auroc = float('nan')
                f1 = float('nan')
            else:
                auroc = roc_auc_score(labels, features)
                threshold = best_features[feature_type]['threshold']
                predictions = (features > threshold).astype(int)
                f1 = f1_score(labels, predictions)
            
            results[temp][feature_type] = {
                'auroc': auroc,
                'f1': f1,
                'n_samples': len(features),  # Total samples, not tasks
                'n_positive': n_positive,
                'n_negative': n_negative,
                'feature_values': features.tolist(),  # Store for detailed analysis
                'labels': labels.tolist()
            }
    
    # Clean up SAEs
    del sae_correct, sae_incorrect
    torch.cuda.empty_cache()
    
    return results
```

### Visualization Functions
```python
def plot_temperature_trends(results):
    """Create temperature vs metric plots."""
    import matplotlib.pyplot as plt
    
    temperatures = sorted(results.keys())
    
    # Extract metrics
    correct_aurocs = [results[t]['correct']['auroc'] for t in temperatures]
    incorrect_aurocs = [results[t]['incorrect']['auroc'] for t in temperatures]
    correct_f1s = [results[t]['correct']['f1'] for t in temperatures]
    incorrect_f1s = [results[t]['incorrect']['f1'] for t in temperatures]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # AUROC plot
    ax1.plot(temperatures, correct_aurocs, 'b-o', label='Correct-preferring', markersize=8)
    ax1.plot(temperatures, incorrect_aurocs, 'r-s', label='Incorrect-preferring', markersize=8)
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('AUROC')
    ax1.set_title('AUROC vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # F1 plot
    ax2.plot(temperatures, correct_f1s, 'b-o', label='Correct-preferring', markersize=8)
    ax2.plot(temperatures, incorrect_f1s, 'r-s', label='Incorrect-preferring', markersize=8)
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Sample distribution plot
    n_positive_correct = [results[t]['correct']['n_positive'] for t in temperatures]
    n_negative_correct = [results[t]['correct']['n_negative'] for t in temperatures]
    
    ax3.bar(temperatures, n_positive_correct, width=0.15, label='Positive', alpha=0.7)
    ax3.bar(temperatures, n_negative_correct, width=0.15, bottom=n_positive_correct, 
            label='Negative', alpha=0.7)
    ax3.set_xlabel('Temperature')
    ax3.set_ylabel('Number of Samples')
    ax3.set_title('Sample Distribution (Correct-preferring Feature)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Feature value distribution
    for i, temp in enumerate(temperatures):
        feature_vals = results[temp]['correct']['feature_values']
        ax4.boxplot(feature_vals, positions=[i], widths=0.6)
    ax4.set_xticklabels(temperatures)
    ax4.set_xlabel('Temperature')
    ax4.set_ylabel('Feature Value')
    ax4.set_title('Feature Value Distribution (Correct-preferring)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/phase3_10/temperature_trends.png', dpi=150)
    plt.close()
```

## Output Structure
```
data/phase3_10/
├── temperature_analysis_results.json    # All results including per-sample data
├── temperature_trends.png              # AUROC, F1, and distribution plots
└── temperature_summary.txt             # Human-readable summary
```

## Results JSON Structure
```json
{
  "creation_timestamp": "...",
  "phase": "3.10",
  "description": "Temperature-Based AUROC Analysis (Per-Sample)",
  "temperatures_analyzed": [0.0, 0.2, 0.4, 0.6, 0.8],
  "analysis_method": "per_sample",
  "samples_per_task": 5,
  "best_features": {
    "correct": {...},
    "incorrect": {...}
  },
  "results_by_temperature": {
    "0.0": {
      "correct": {
        "auroc": 0.75,
        "f1": 0.68,
        "n_samples": 250,
        "n_positive": 120,
        "n_negative": 130,
        "feature_values": [...],  // 250 values
        "labels": [...]           // 250 labels
      },
      "incorrect": {...}
    },
    ...
  }
}
```

## Dependencies

- **Phase 3.8**: Best features and F1-optimal thresholds
- **Phase 3.5**: 
  - Temperature datasets with 5 samples per task (`dataset_temp_*.parquet`)
  - Pre-saved activations (`activations/task_activations/{task_id}_layer_{n}.npz`)
- **GemmaScope SAE**: For encoding raw activations to feature values

## Implementation Checklist

- [ ] Load Phase 3.8 results (best features + thresholds)
- [ ] Implement per-sample data processing (no aggregation)
- [ ] Load pre-saved activations from Phase 3.5 activation files
- [ ] Encode raw activations through GemmaScope SAE
- [ ] Calculate AUROC and F1 metrics for each temperature
- [ ] Generate enhanced visualizations including sample distributions
- [ ] Identify critical temperature thresholds
- [ ] Create comprehensive output report

## Expected Insights

1. **Temperature Tolerance**: Identify temperature where AUROC drops below 0.7
2. **Feature Robustness**: Compare stability of correct vs incorrect features
3. **Sample-Level Patterns**: Understand variation within tasks at same temperature
4. **Reliability Threshold**: Temperature range for practical PVA usage

## Reusable Components

Following DRY principle, Phase 3.10 reuses these existing functions/classes:

### From `common/utils.py`:
- `detect_device()` - Device detection
- `get_phase_dir()` - Get directory paths
- `discover_latest_phase_output()` - Auto-discover phase outputs
- `format_duration()` - Time formatting

### From `common_simplified/helpers.py`:
- `save_json()` - Save results to JSON
- `load_json()` - Load Phase 3.8 results

### From `phase2_5_simplified/sae_analyzer.py`:
- `JumpReLUSAE` - SAE model class
- `load_gemma_scope_sae()` - Load GemmaScope SAE

### From `common/config.py`:
- `Config` - Unified configuration
- `GEMMA_2B_SPARSITY` - Layer sparsity mappings

### From `common/logging.py`:
- `get_logger()` - Phase-specific logging

## Integration with run.py

### 1. Add Phase 3.10 to run.py

```python
# In setup_argument_parser(), add to choices:
phase_parser.add_argument(
    'phase',
    type=float,
    choices=[0, 0.1, 1, 2.2, 2.5, 3, 3.5, 3.6, 3.8, 3.10, 3.12],  # Add 3.10
    help='..., 3.10=Temperature-Based AUROC Analysis, ...'
)

# Add phase name mapping:
phase_names = {
    # ... existing phases ...
    3.10: "Temperature-Based AUROC Analysis",
    # ...
}

# Add run function:
def run_phase3_10(config: Config, logger, device: str):
    """Run Phase 3.10: Temperature-Based AUROC Analysis"""
    from phase3_10_temperature_auroc_f1.temperature_evaluator import TemperatureAUROCEvaluator
    
    logger.info("Starting Phase 3.10: Temperature-Based AUROC Analysis")
    logger.info("Using per-sample analysis (no aggregation)")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="3.10"))
    
    # Create and run evaluator
    evaluator = TemperatureAUROCEvaluator(config)
    results = evaluator.run()
    
    logger.info("\n✅ Phase 3.10 completed successfully")
    logger.info(f"Results saved to: {config.phase3_10_output_dir}")

# In main(), add case:
elif args.phase == 3.10:
    run_phase3_10(config, logger, device)
```

### 2. Update common/utils.py

```python
# Add to PHASE_CONFIGS dictionary:
"3.10": {
    "dir": "data/phase3_10",
    "patterns": ["temperature_analysis_results.json", "*.png", "temperature_summary.txt"],
    "exclude_keywords": None
}
```

### 3. Update common/config.py

```python
# Add Phase 3.10 output directory
@property
def phase3_10_output_dir(self) -> str:
    """Output directory for Phase 3.10"""
    return "data/phase3_10"

# In validate() method, add:
elif phase == "3.10":
    # Phase 3.10 requires Phase 3.8 (best features) and 3.5 (temperature data)
    required_phases = ["3.8", "3.5"]
    for req_phase in required_phases:
        output_path = discover_latest_phase_output(req_phase)
        if not output_path:
            raise ValueError(f"Phase 3.10 requires Phase {req_phase} to be completed first")
```