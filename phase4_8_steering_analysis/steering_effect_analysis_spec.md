# Phase 4.8: Steering Effect Analysis Specification

## Executive Summary

1. **Purpose**: Comprehensively analyze the effects of model steering using coefficients from Phase 4.5
2. **Method**: Apply selected steering coefficients to evaluate corruption and correction rates
3. **Metrics**: Corruption rate, correction rate, and binomial statistical testing
4. **Dataset**: MBPP validation split (200-400 problems for robust statistics)
5. **Statistical Validation**: Binomial tests to verify steering effects are significant
6. **Input**: Selected coefficients from Phase 4.5
7. **Output**: Statistical analysis of steering effectiveness and causal validation

## Pipeline Sequence

```
1. Load selected coefficients and PVA features
   └─> Read Phase 4.5 outputs → Load SAE features → Initialize models

2. Prepare evaluation dataset
   └─> Load validation split → Select subset for analysis → Build prompts

3. Generate baseline solutions
   └─> Generate without steering → Evaluate correctness → Store results

4. Apply steering interventions
   └─> Correct-preferring steering → Incorrect-preferring steering → Evaluate outcomes

5. Calculate effect metrics
   └─> Corruption rates → Correction rates → Statistical significance → Visualizations
```

## Core Metrics

### 1. Corruption Rate
Percentage of originally correct solutions that become incorrect when steered with incorrect-preferring features:

```python
corruption_rate = (correct_to_incorrect_count / total_originally_correct) * 100
```

### 2. Correction Rate
Percentage of originally incorrect solutions that become correct when steered with correct-preferring features:

```python
correction_rate = (incorrect_to_correct_count / total_originally_incorrect) * 100
```

### 3. Baseline Pass Rate
Natural pass rate without any steering intervention:

```python
baseline_pass_rate = (total_passed_baseline / total_problems) * 100
```

## Statistical Validation

### Binomial Testing
We use binomial tests to determine if the observed rates are significantly different from chance:

```python
from scipy.stats import binomtest

def test_steering_significance(successes: int, trials: int, baseline_rate: float, 
                             alternative: str = 'two-sided') -> Dict:
    """
    Test if steering effect is statistically significant.
    
    Args:
        successes: Number of successful steering outcomes
        trials: Total number of attempts
        baseline_rate: Expected rate under null hypothesis
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Dict with p-value, confidence interval, and interpretation
    """
    result = binomtest(successes, trials, baseline_rate, alternative=alternative)
    
    return {
        'p_value': result.pvalue,
        'confidence_interval': result.proportion_ci(confidence_level=0.95),
        'observed_rate': successes / trials,
        'expected_rate': baseline_rate,
        'significant': result.pvalue < 0.05,
        'effect_size': (successes / trials) - baseline_rate
    }
```

## Implementation

### Required Imports
```python
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.stats import binomtest
import matplotlib.pyplot as plt
import seaborn as sns

from common.prompt_utils import PromptBuilder
from common.logging import get_logger
from common.utils import discover_latest_phase_output
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase4_8.steering_effect_analysis")
```

### 1. Load Coefficients and Setup
```python
def load_steering_coefficients(phase4_5_dir: Path) -> Dict:
    """Load selected coefficients from Phase 4.5."""
    coeff_file = phase4_5_dir / 'selected_coefficients.json'
    if not coeff_file.exists():
        raise FileNotFoundError(
            f"Selected coefficients not found at {coeff_file}. "
            "Please run Phase 4.5 first."
        )
    return load_json(coeff_file)

def setup_steering_environment(coefficients: Dict, phase2_5_dir: Path, device):
    """Initialize SAEs and features for steering."""
    # Load PVA features
    pva_features = load_json(phase2_5_dir / 'top_20_features.json')
    
    # Extract best features
    best_correct = pva_features['correct'][0]
    best_incorrect = pva_features['incorrect'][0]
    
    # Load SAEs
    sae_correct = load_gemma_scope_sae(best_correct['layer'], device)
    sae_incorrect = load_gemma_scope_sae(best_incorrect['layer'], device)
    
    return {
        'correct': {
            'feature': best_correct,
            'sae': sae_correct,
            'coefficient': coefficients['correct']
        },
        'incorrect': {
            'feature': best_incorrect,
            'sae': sae_incorrect,
            'coefficient': coefficients['incorrect']
        }
    }
```

### 2. Baseline Generation
```python
def generate_baseline_solutions(problems: List[Dict], model, tokenizer, 
                              prompt_builder, device) -> List[Dict]:
    """Generate baseline solutions without steering."""
    baseline_results = []
    
    for problem in tqdm(problems, desc="Generating baseline"):
        prompt = prompt_builder.build_prompt(
            problem_description=problem['text'],
            test_cases=problem.get('test_list', [])
        )
        
        inputs = tokenizer(prompt, return_tensors='pt', 
                          truncation=True, max_length=2048).to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.0,
                do_sample=False
            )
        
        generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_code = extract_code(generated_code, prompt)
        
        test_passed = evaluate_code(generated_code, problem['test_list'])
        
        baseline_results.append({
            'task_id': problem['task_id'],
            'baseline_code': generated_code,
            'baseline_passed': test_passed
        })
    
    return baseline_results
```

### 3. Steering Application
```python
def apply_steering(problem: Dict, model, tokenizer, sae, feature_info, 
                  coefficient, prompt_builder, device) -> Dict:
    """Apply steering to a single problem."""
    prompt = prompt_builder.build_prompt(
        problem_description=problem['text'],
        test_cases=problem.get('test_list', [])
    )
    
    inputs = tokenizer(prompt, return_tensors='pt', 
                      truncation=True, max_length=2048).to(device)
    
    # Get decoder direction
    decoder_direction = sae.W_dec[feature_info['feature_idx']]
    
    # Create and apply steering hook
    from phase4_5_model_steering.pva_steering_spec import create_steering_hook
    hook = create_steering_hook(decoder_direction, coefficient)
    hook_handle = model.model.layers[feature_info['layer']].register_forward_pre_hook(hook)
    
    # Generate with steering
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.0,
            do_sample=False
        )
    
    # Remove hook
    hook_handle.remove()
    
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_code = extract_code(generated_code, prompt)
    
    test_passed = evaluate_code(generated_code, problem['test_list'])
    
    return {
        'task_id': problem['task_id'],
        'steered_code': generated_code,
        'steered_passed': test_passed
    }
```

### 4. Effect Analysis
```python
def analyze_steering_effects(baseline_results: List[Dict], 
                           correct_steering_results: List[Dict],
                           incorrect_steering_results: List[Dict]) -> Dict:
    """Calculate corruption and correction rates with statistical tests."""
    
    # Merge results by task_id
    results_df = pd.DataFrame(baseline_results)
    correct_df = pd.DataFrame(correct_steering_results)
    incorrect_df = pd.DataFrame(incorrect_steering_results)
    
    # Merge on task_id
    results_df = results_df.merge(
        correct_df[['task_id', 'steered_passed']], 
        on='task_id', 
        suffixes=('', '_correct_steer')
    )
    results_df = results_df.merge(
        incorrect_df[['task_id', 'steered_passed']], 
        on='task_id', 
        suffixes=('', '_incorrect_steer')
    )
    
    # Rename columns for clarity
    results_df.rename(columns={
        'steered_passed': 'correct_steering_passed',
        'steered_passed_incorrect_steer': 'incorrect_steering_passed'
    }, inplace=True)
    
    # Calculate metrics
    originally_correct = results_df[results_df['baseline_passed'] == True]
    originally_incorrect = results_df[results_df['baseline_passed'] == False]
    
    # Corruption rate: correct → incorrect with incorrect-preferring steering
    corrupted = originally_correct[
        originally_correct['incorrect_steering_passed'] == False
    ]
    corruption_rate = len(corrupted) / len(originally_correct) if len(originally_correct) > 0 else 0
    
    # Correction rate: incorrect → correct with correct-preferring steering  
    corrected = originally_incorrect[
        originally_incorrect['correct_steering_passed'] == True
    ]
    correction_rate = len(corrected) / len(originally_incorrect) if len(originally_incorrect) > 0 else 0
    
    # Baseline pass rate
    baseline_pass_rate = len(originally_correct) / len(results_df)
    
    # Statistical tests
    corruption_test = test_steering_significance(
        len(corrupted), 
        len(originally_correct),
        0.0,  # Null hypothesis: no corruption
        alternative='greater'
    )
    
    correction_test = test_steering_significance(
        len(corrected),
        len(originally_incorrect), 
        0.0,  # Null hypothesis: no correction
        alternative='greater'
    )
    
    return {
        'corruption_rate': corruption_rate,
        'correction_rate': correction_rate,
        'baseline_pass_rate': baseline_pass_rate,
        'corruption_count': len(corrupted),
        'correction_count': len(corrected),
        'originally_correct_count': len(originally_correct),
        'originally_incorrect_count': len(originally_incorrect),
        'total_problems': len(results_df),
        'corruption_significance': corruption_test,
        'correction_significance': correction_test,
        'detailed_results': results_df
    }
```

### 5. Visualization
```python
def create_steering_effect_plots(analysis_results: Dict, output_dir: Path):
    """Create visualizations of steering effects."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Rate comparison bar plot
    ax = axes[0, 0]
    rates = {
        'Baseline\nPass Rate': analysis_results['baseline_pass_rate'] * 100,
        'Corruption\nRate': analysis_results['corruption_rate'] * 100,
        'Correction\nRate': analysis_results['correction_rate'] * 100
    }
    bars = ax.bar(rates.keys(), rates.values(), color=['blue', 'red', 'green'])
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Steering Effect Rates')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 2. Statistical significance plot
    ax = axes[0, 1]
    sig_data = {
        'Corruption': analysis_results['corruption_significance']['p_value'],
        'Correction': analysis_results['correction_significance']['p_value']
    }
    bars = ax.bar(sig_data.keys(), sig_data.values(), 
                   color=['red' if p < 0.05 else 'gray' for p in sig_data.values()])
    ax.axhline(y=0.05, color='black', linestyle='--', label='p=0.05')
    ax.set_ylabel('p-value')
    ax.set_title('Statistical Significance')
    ax.set_yscale('log')
    ax.legend()
    
    # 3. Transition matrix heatmap
    ax = axes[1, 0]
    df = analysis_results['detailed_results']
    
    # Create transition matrix
    transitions = pd.crosstab(
        df['baseline_passed'].map({True: 'Correct', False: 'Incorrect'}),
        df['incorrect_steering_passed'].map({True: 'Correct', False: 'Incorrect'}),
        normalize='index'
    ) * 100
    
    sns.heatmap(transitions, annot=True, fmt='.1f', cmap='RdYlGn', 
                ax=ax, cbar_kws={'label': 'Percentage'})
    ax.set_title('Incorrect-Preferring Steering Transitions')
    ax.set_xlabel('After Steering')
    ax.set_ylabel('Baseline')
    
    # 4. Correction transitions
    ax = axes[1, 1]
    transitions_correct = pd.crosstab(
        df['baseline_passed'].map({True: 'Correct', False: 'Incorrect'}),
        df['correct_steering_passed'].map({True: 'Correct', False: 'Incorrect'}),
        normalize='index'
    ) * 100
    
    sns.heatmap(transitions_correct, annot=True, fmt='.1f', cmap='RdYlGn',
                ax=ax, cbar_kws={'label': 'Percentage'})
    ax.set_title('Correct-Preferring Steering Transitions')
    ax.set_xlabel('After Steering')
    ax.set_ylabel('Baseline')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'steering_effect_analysis.png', dpi=150)
    plt.close()
```

## Experimental Configuration

### Dataset Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_problems` | 200-400 | Sufficient for statistical power |
| `split` | validation | Independent from training |
| `temperature` | 0.0 | Deterministic generation |

### Statistical Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `alpha` | 0.05 | Significance threshold |
| `confidence_level` | 0.95 | For confidence intervals |
| `alternative` | 'greater' | One-tailed test for positive effects |

## Implementation Workflow

### Phase 1: Setup
```python
# Load coefficients from Phase 4.5
phase4_5_output = discover_latest_phase_output('4.5')
coefficients = load_steering_coefficients(Path(phase4_5_output).parent)
logger.info(f"Loaded coefficients: correct={coefficients['correct']}, "
           f"incorrect={coefficients['incorrect']}")

# Setup steering environment
phase2_5_output = discover_latest_phase_output('2.5')
steering_env = setup_steering_environment(
    coefficients, 
    Path(phase2_5_output).parent,
    device
)
```

### Phase 2: Baseline Generation
```python
# Load validation problems
validation_data = pd.read_parquet('data/phase0_1/validation_mbpp.parquet')
problems = validation_data.to_dict('records')[:300]  # Use 300 for good statistics

# Generate baseline
baseline_results = generate_baseline_solutions(
    problems, model, tokenizer, prompt_builder, device
)
```

### Phase 3: Apply Steering
```python
# Apply correct-preferring steering
correct_steering_results = []
for problem in tqdm(problems, desc="Correct-preferring steering"):
    result = apply_steering(
        problem, model, tokenizer,
        steering_env['correct']['sae'],
        steering_env['correct']['feature'],
        steering_env['correct']['coefficient'],
        prompt_builder, device
    )
    correct_steering_results.append(result)

# Apply incorrect-preferring steering
incorrect_steering_results = []
for problem in tqdm(problems, desc="Incorrect-preferring steering"):
    result = apply_steering(
        problem, model, tokenizer,
        steering_env['incorrect']['sae'],
        steering_env['incorrect']['feature'],
        steering_env['incorrect']['coefficient'],
        prompt_builder, device
    )
    incorrect_steering_results.append(result)
```

### Phase 4: Analysis and Reporting
```python
# Analyze effects
analysis = analyze_steering_effects(
    baseline_results,
    correct_steering_results,
    incorrect_steering_results
)

# Create visualizations
create_steering_effect_plots(analysis, output_dir)

# Save detailed results
save_json(analysis, output_dir / 'steering_effect_analysis.json')

# Print summary
print_analysis_summary(analysis)
```

## Expected Outcomes

### Successful Steering
- **Corruption Rate**: >20% of correct solutions fail with incorrect-preferring steering
- **Correction Rate**: >15% of incorrect solutions pass with correct-preferring steering
- **Statistical Significance**: p-values < 0.05 for both effects

### Validation Criteria
1. **Causal Effect**: Steering changes model behavior in predictable ways
2. **Feature Validity**: SAE features genuinely represent program validity awareness
3. **Practical Impact**: Effect sizes are meaningful, not just statistically significant

## Implementation Checklist

### Setup
- [ ] Load selected coefficients from Phase 4.5
- [ ] Load PVA features from Phase 2.5
- [ ] Initialize SAEs and model

### Execution
- [ ] Generate baseline solutions for 200-400 problems
- [ ] Apply correct-preferring steering
- [ ] Apply incorrect-preferring steering
- [ ] Calculate corruption and correction rates

### Analysis
- [ ] Run binomial statistical tests
- [ ] Create visualization plots
- [ ] Generate detailed report
- [ ] Save all results and analysis

## Notes

- **Memory Management**: Process problems in batches if needed
- **Error Handling**: Track and report any generation failures
- **Reproducibility**: Use same random seed as other phases
- **Single Prompt Processing**: Maintain consistency with other phases