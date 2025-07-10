# Phase 3.10: Temperature-Based AUROC Analysis for PVA-SAE

## Executive Summary

**Purpose**: Analyze how PVA feature effectiveness varies across different temperature settings in Python code generation

**Key Questions**:
1. Do PVA features maintain their discriminative power across temperature variations?
2. At what temperature do features lose reliability for predicting code correctness?
3. How do correct vs incorrect-preferring features respond to temperature changes?

**Approach**: Evaluate AUROC and F1 metrics for best PVA features across 5 temperature levels using Phase 3.5 validation data with majority vote aggregation

## Pipeline Overview

```
1. Load Phase 3.8 best features & thresholds
   └─> Extract layer indices, feature indices, and F1-optimal thresholds

2. Load Phase 3.5 temperature datasets
   └─> Process all 5 temperature levels (0.0, 0.3, 0.6, 0.9, 1.2)
   └─> Each task has 5 generated solutions per temperature

3. Aggregate labels using majority vote
   └─> Task passes if ≥3/5 samples pass tests
   └─> Activations are identical across samples (same prompt)

4. Evaluate features across temperatures
   └─> Calculate AUROC and F1 for each temperature
   └─> Generate ROC curves and trend visualizations

5. Output comprehensive analysis to data/phase3_10/
```

## Key Design Decision: Majority Vote

Since each task has 5 generated solutions per temperature:
- **Activations**: Same for all 5 samples (deterministic prompt processing)
- **Labels**: Use majority vote - task passes if ≥3 out of 5 samples pass
- **Rationale**: Captures "reliable capability" rather than "lucky success"

## Implementation Decisions

1. **Missing Activation Files**: Skip missing tasks with warning, output skipped records to separate file
2. **SAE Memory Management**: Load SAEs once and reuse across temperatures
3. **Visualization**: Simple matplotlib plots (following KISS principle)
4. **Class Imbalance**: Plot NaN/N/A when AUROC cannot be calculated (similar to Phase 3.12)
5. **Results Structure**: One final JSON with all results

## Core Implementation Structure

### Data Processing
```python
def process_temperature_data(temp_dataset, best_features, sae, skipped_tasks):
    """Process data for a single temperature."""
    # Group by task_id (5 samples per task)
    task_groups = temp_dataset.groupby('task_id')
    
    task_features = []
    task_labels = []
    
    for task_id, samples in task_groups:
        # Load pre-saved activation from Phase 3.5
        activation_path = f'data/phase3_5/activations/task_activations/{task_id}_layer_{best_features["layer"]}.npz'
        
        # Skip if activation file missing
        if not os.path.exists(activation_path):
            logger.warning(f"Missing activation file for {task_id}")
            skipped_tasks.append({
                'task_id': task_id,
                'reason': 'missing_activation_file',
                'layer': best_features["layer"]
            })
            continue
        
        try:
            raw_activation = np.load(activation_path)['arr_0']
            
            # Encode through SAE to get feature value
            with torch.no_grad():
                raw_tensor = torch.from_numpy(raw_activation).to(device)
                sae_features = sae.encode(raw_tensor)
                feature_value = sae_features[0, best_features['feature_idx']].item()
            
            # Majority vote for label
            passes = samples['test_passed'].sum()
            label = int(passes >= 3)  # 1 if majority pass, 0 otherwise
            
            task_features.append(feature_value)
            task_labels.append(label)
            
        except Exception as e:
            logger.warning(f"Error processing {task_id}: {str(e)}")
            skipped_tasks.append({
                'task_id': task_id,
                'reason': f'processing_error: {str(e)}',
                'layer': best_features["layer"]
            })
    
    return np.array(task_features), np.array(task_labels)
```

### Main Evaluation Loop
```python
def evaluate_across_temperatures(best_features, temperatures):
    """Evaluate feature performance at each temperature."""
    results = {}
    skipped_tasks = []
    
    # Load SAEs once for reuse (Decision #2)
    sae_correct = load_gemma_scope_sae(best_features['correct']['layer'], device)
    sae_incorrect = load_gemma_scope_sae(best_features['incorrect']['layer'], device)
    
    for temp in temperatures:
        # Load temperature dataset
        temp_data = load_temperature_dataset(temp)
        results[temp] = {}
        
        # Process for both feature types
        for feature_type in ['correct', 'incorrect']:
            sae = sae_correct if feature_type == 'correct' else sae_incorrect
            
            features, labels = process_temperature_data(
                temp_data, 
                best_features[feature_type],
                sae,
                skipped_tasks
            )
            
            # Flip labels for correct-preferring features
            if feature_type == 'correct':
                labels = 1 - labels
            
            # Calculate metrics with edge case handling (Decision #4)
            n_positive = sum(labels)
            n_negative = len(labels) - n_positive
            
            if n_positive < 2 or n_negative < 2:
                # Not enough samples for AUROC
                logger.warning(f"Class imbalance at temp {temp} for {feature_type}: "
                             f"pos={n_positive}, neg={n_negative}")
                auroc = float('nan')  # Plot as NaN
                f1 = float('nan')
            else:
                auroc = roc_auc_score(labels, features)
                threshold = best_features[feature_type]['threshold']
                predictions = (features > threshold).astype(int)
                f1 = f1_score(labels, predictions)
            
            results[temp][feature_type] = {
                'auroc': auroc,
                'f1': f1,
                'n_tasks': len(features),
                'n_positive': n_positive,
                'n_negative': n_negative
            }
    
    # Clean up SAEs
    del sae_correct, sae_incorrect
    torch.cuda.empty_cache()
    
    # Save skipped tasks to separate file (Decision #1)
    if skipped_tasks:
        save_json({'skipped_tasks': skipped_tasks}, 
                  Path('data/phase3_10/skipped_tasks.json'))
        logger.info(f"Skipped {len(skipped_tasks)} tasks total")
    
    return results
```

### Visualization Functions
```python
def plot_temperature_trends(results):
    """Create temperature vs metric plots using matplotlib (Decision #3)."""
    import matplotlib.pyplot as plt
    
    temperatures = sorted(results.keys())
    
    # Extract metrics, handling NaN values
    correct_aurocs = [results[t]['correct']['auroc'] for t in temperatures]
    incorrect_aurocs = [results[t]['incorrect']['auroc'] for t in temperatures]
    correct_f1s = [results[t]['correct']['f1'] for t in temperatures]
    incorrect_f1s = [results[t]['incorrect']['f1'] for t in temperatures]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # AUROC plot
    ax1.plot(temperatures, correct_aurocs, 'b-o', label='Correct-preferring')
    ax1.plot(temperatures, incorrect_aurocs, 'r-s', label='Incorrect-preferring')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('AUROC')
    ax1.set_title('AUROC vs Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # F1 plot
    ax2.plot(temperatures, correct_f1s, 'b-o', label='Correct-preferring')
    ax2.plot(temperatures, incorrect_f1s, 'r-s', label='Incorrect-preferring')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Temperature')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('data/phase3_10/temperature_trends.png', dpi=150)
    plt.close()
```

## Output Structure
```
data/phase3_10/
├── temperature_analysis_results.json    # All results in one file (Decision #5)
├── temperature_trends.png              # AUROC and F1 trends
├── skipped_tasks.json                  # Records that were skipped (Decision #1)
└── temperature_summary.txt             # Human-readable summary
```

## Dependencies

- **Phase 3.8**: Best features and F1-optimal thresholds
- **Phase 3.5**: 
  - Temperature datasets with 5 samples per task (`dataset_temp_*.parquet`)
  - Pre-saved activations (`activations/task_activations/{task_id}_layer_{n}.npz`)
- **GemmaScope SAE**: For encoding raw activations to feature values

## Implementation Checklist

- [ ] Load Phase 3.8 results (best features + thresholds)
- [ ] Implement majority vote aggregation for labels
- [ ] Load pre-saved activations from Phase 3.5 activation files
- [ ] Encode raw activations through GemmaScope SAE
- [ ] Calculate AUROC and F1 metrics for each temperature
- [ ] Generate temperature trend visualizations
- [ ] Identify critical temperature thresholds
- [ ] Create comprehensive output report

## Expected Insights

1. **Temperature Tolerance**: Identify temperature where AUROC drops below 0.7
2. **Feature Robustness**: Compare stability of correct vs incorrect features
3. **Reliability Threshold**: Temperature range for practical PVA usage
4. **Degradation Pattern**: Linear vs exponential performance decline

## Reusable Components

Following DRY principle, Phase 3.10 should reuse these existing functions/classes:

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
    logger.info("Using majority vote aggregation for multi-sample tasks")
    
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