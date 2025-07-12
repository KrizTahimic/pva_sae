# Phase 4.5: Steering Coefficient Selection Specification

## Executive Summary

1. **Purpose**: Find optimal steering coefficients for PVA features through empirical grid search
2. **Method**: Add SAE decoder directions to residual stream activations during generation
3. **Steering Coefficients**: Grid search [1, 5, 10, 15, 20, 30, 50, 100] with manual evaluation
4. **Application**: Continuous steering throughout generation process
5. **Evaluation**: Flip rate (pass/fail changes) and generation divergence metrics
6. **Dataset**: MBPP validation split from Phase 0.1 (20 problems for search)
7. **Output**: Selected coefficients saved for Phase 4.8 comprehensive analysis

## Pipeline Sequence

```
1. Load SAE features and decoder directions
   └─> Load best features from Phase 2.5 → Extract decoder directions → Initialize SAEs

2. Prepare MBPP prompts
   └─> Load validation split → Format problems → Identify steering positions

3. Set up steering hooks
   └─> Target residual stream (pre-hook) → Add direction * coefficient → Apply during generation

4. Generate steered outputs
   └─> Baseline generation → Steered generation → Execute tests → Compare results

5. Evaluate steering effect
   └─> Calculate flip rates → Measure generation divergence → Plot coefficient effects → Save selected coefficients
```

## Understanding Model Steering in PVA Context

### Steering Mechanism
The system modifies model activations by adding SAE decoder directions to the residual stream:
- **Direction**: Decoder vector from SAE corresponding to PVA features (correct/incorrect-preferring)
- **Coefficient**: Scalar multiplier controlling steering strength
- **Position**: Applied continuously to all positions (no specific targeting needed)
- **Timing**: Applied throughout the entire generation process
- **Processing**: Single prompt at a time (no batching) for consistent results

### Steering Targets
- **Incorrect → Correct**: Make model generate more correct code (using correct-preferring features)
- **Correct → Incorrect**: Make model generate more buggy code (using incorrect-preferring features)

## Core Implementation

### Required Imports
```python
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable
from tqdm import tqdm

from common.prompt_utils import PromptBuilder
from common.logging import get_logger
from common.utils import discover_latest_phase_output
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code, extract_code, load_json, save_json
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase4_5.steering_evaluator")
```

### 1. Loading PVA Features

```python
def load_pva_features(phase2_5_dir: Path) -> Dict:
    """
    Load pre-identified PVA features from Phase 2.5.
    
    Returns:
        Dict with 'correct' and 'incorrect' feature information
    """
    # Load top features from Phase 2.5
    top_features = load_json(phase2_5_dir / 'top_20_features.json')
    
    # Extract best feature for each type
    best_correct = top_features['correct'][0]  # Top correct-preferring
    best_incorrect = top_features['incorrect'][0]  # Top incorrect-preferring
    
    return {
        'correct': {
            'layer': best_correct['layer'],
            'feature_idx': best_correct['feature_idx'],
            'separation_score': best_correct['separation_score']
        },
        'incorrect': {
            'layer': best_incorrect['layer'],
            'feature_idx': best_incorrect['feature_idx'],
            'separation_score': best_incorrect['separation_score']
        }
    }
```

### 2. Steering Hook Implementation

```python
def create_steering_hook(sae_decoder_direction: torch.Tensor, 
                        coeff: float,
                        position_mask: Optional[torch.Tensor] = None) -> Callable:
    """
    Create a hook function that adds steering during generation.
    
    Args:
        sae_decoder_direction: Decoder direction from SAE [d_model]
        coeff: Steering coefficient
        position_mask: Boolean mask for positions to steer
    
    Returns:
        Hook function for model
    """
    def hook_fn(module, input):
        # input[0] is residual stream: [1, seq_len, d_model] (single prompt)
        residual = input[0]
        
        # Calculate steering vector
        steering = sae_decoder_direction.unsqueeze(0) * coeff
        
        # Apply steering to specified positions
        if position_mask is not None:
            # For single prompt, mask is [seq_len]
            if position_mask.dim() == 1:
                mask = position_mask.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            else:
                mask = position_mask.unsqueeze(-1)  # Already [1, seq_len]
            residual = residual + (steering.unsqueeze(0) * mask)
        else:
            # Apply to all positions (continuous steering)
            residual = residual + steering.unsqueeze(0)
        
        return (residual,) + input[1:]
    
    return hook_fn
```

### 3. Position Detection for MBPP

```python
def get_steering_position(input_ids: torch.Tensor, 
                         tokenizer) -> int:
    """
    Find the position to start steering (end of problem description).
    
    For MBPP, this is typically right before the function signature starts.
    We look for patterns like "\\n\\ndef" or the last token before code.
    
    Args:
        input_ids: Tokenized input [seq_len] (single prompt, no batch)
        tokenizer: Tokenizer for decoding
        
    Returns:
        Position index for steering
    """
    # Convert tokens to text to find code boundary
    text = tokenizer.decode(input_ids, skip_special_tokens=False)
    
    # Find where the code generation should begin
    # MBPP format: problem description followed by function signature
    code_markers = ["\\ndef ", "\\nclass ", "\\nimport "]
    
    position = len(input_ids) - 1  # Default to last token
    
    for marker in code_markers:
        if marker in text:
            # Find token position corresponding to this text position
            marker_pos = text.index(marker)
            # Convert character position to token position
            # ... (implementation details)
            break
    
    return position
```

### 4. Steering During Generation

```python
# Set up steering parameters
steering_config = {
    'model_name': 'google/gemma-2-2b',
    'feature_type': 'correct',  # Steer towards correct code
    'coefficients': [1, 5, 10, 15, 20, 30, 50, 100],
    'max_new_tokens': 500,
    # Single prompt processing - no batching
    'temperature': 0.0  # Deterministic for consistency
}

# Load PVA features
pva_features = load_pva_features(phase2_5_dir)
feature_info = pva_features[steering_config['feature_type']]

# Load SAE for the relevant layer
sae = load_gemma_scope_sae(feature_info['layer'], device)

# Get decoder direction
decoder_direction = sae.W_dec[feature_info['feature_idx']]

# For each coefficient in grid search
for coeff in steering_config['coefficients']:
    results = []
    
    # Process each problem individually (no batching)
    for idx, problem in enumerate(validation_problems):
        # Build prompt using common format
        prompt = prompt_builder.build_prompt(
            problem_description=problem['text'],
            test_cases=problem.get('test_list', [])
        )
        
        # Tokenize problem (single prompt)
        inputs = tokenizer(prompt, return_tensors='pt', 
                          truncation=True,
                          max_length=2048).to(device)  # Using standard max length
        
        # Generate baseline (no steering)
        baseline_output = model.generate(**inputs, max_new_tokens=500)
        baseline_code = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        
        # Apply steering hook for continuous steering
        # Note: We use continuous steering throughout generation
        # No need for position detection since we steer at all positions
        hook = create_steering_hook(decoder_direction, coeff, position_mask=None)
        hook_handle = model.model.layers[feature_info['layer']].register_forward_pre_hook(hook)
        
        # Generate with steering
        steered_output = model.generate(**inputs, max_new_tokens=500)
        steered_code = tokenizer.decode(steered_output[0], skip_special_tokens=True)
        
        # Remove hook
        hook_handle.remove()
        
        # Evaluate both outputs
        baseline_passed = run_tests(baseline_code, problem['tests'])
        steered_passed = run_tests(steered_code, problem['tests'])
        
        results.append({
            'task_id': problem['task_id'],
            'baseline_passed': baseline_passed,
            'steered_passed': steered_passed,
            'flipped': baseline_passed != steered_passed,
            'baseline_code': baseline_code,
            'steered_code': steered_code
        })
        
        # Memory cleanup periodically
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
```

### 5. Evaluation Metrics

```python
def calculate_flip_rate(results: List[Dict]) -> float:
    """
    Calculate percentage of problems where steering changed pass/fail outcome.
    """
    flipped = sum(1 for r in results if r['flipped'])
    return (flipped / len(results)) * 100

def calculate_generation_divergence(results: List[Dict]) -> Dict:
    """
    Measure how different steered generations are from baseline.
    
    Returns:
        Dict with multiple divergence metrics
    """
    import difflib
    from typing import List
    
    divergences = []
    
    for result in results:
        baseline = result['baseline_code']
        steered = result['steered_code']
        
        # Token-level edit distance
        baseline_tokens = baseline.split()
        steered_tokens = steered.split()
        
        # Calculate Levenshtein distance
        seq_matcher = difflib.SequenceMatcher(None, baseline_tokens, steered_tokens)
        similarity_ratio = seq_matcher.ratio()
        
        # Character-level differences
        char_diff = difflib.SequenceMatcher(None, baseline, steered).ratio()
        
        divergences.append({
            'token_similarity': similarity_ratio,
            'char_similarity': char_diff,
            'length_ratio': len(steered) / len(baseline) if len(baseline) > 0 else 1.0
        })
    
    return {
        'mean_token_similarity': np.mean([d['token_similarity'] for d in divergences]),
        'mean_char_similarity': np.mean([d['char_similarity'] for d in divergences]),
        'mean_length_ratio': np.mean([d['length_ratio'] for d in divergences])
    }
```

## Experimental Configuration

### Dataset Setup
```python
# Load validation split from Phase 0.1
validation_data = pd.read_parquet('data/phase0_1/validation_mbpp.parquet')

# Format prompts for generation using PromptBuilder
from common.prompt_utils import PromptBuilder
prompt_builder = PromptBuilder()

# Example prompt format:
# """Problem description..."""
# def function_name(args):
```

### Steering Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `coefficients` | [1, 5, 10, 15, 20, 30, 50, 100] | Grid search values |
| `position_type` | 'problem_end' | Apply steering at end of problem description |
| `processing` | Sequential | Single prompt at a time (no batching) |
| `n_examples_per_coeff` | 20 | Examples for coefficient selection |
| `n_final_eval` | 100 | Problems for final evaluation |
| `max_new_tokens` | 500 | Maximum tokens to generate |

### Evaluation Conditions
1. **Baseline**: No steering applied
2. **Correct-steering**: Steer with correct-preferring feature
3. **Incorrect-steering**: Steer with incorrect-preferring feature

## Implementation Workflow

### Phase 1: Feature Preparation
```python
# Load best PVA features from Phase 2.5
phase2_5_output = discover_latest_phase_output('2.5')
pva_features = load_pva_features(Path(phase2_5_output).parent)

# Initialize SAEs for both feature types
sae_correct = load_gemma_scope_sae(pva_features['correct']['layer'], device)
sae_incorrect = load_gemma_scope_sae(pva_features['incorrect']['layer'], device)
```

### Phase 2: Coefficient Grid Search
```python
def evaluate_steering_single_prompt(problem, model, tokenizer, sae, feature_info, 
                                  coefficient, prompt_builder, device):
    """Evaluate steering on a single problem."""
    # Build prompt
    prompt = prompt_builder.build_prompt(
        problem_description=problem['text'],
        test_cases=problem.get('test_list', [])
    )
    
    # Tokenize (single prompt)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # Generate baseline
    with torch.no_grad():
        baseline_output = model.generate(
            **inputs, 
            max_new_tokens=500,
            temperature=0.0,
            do_sample=False
        )
    baseline_code = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    baseline_code = extract_code(baseline_code, prompt)
    
    # Get decoder direction
    decoder_direction = sae.W_dec[feature_info['feature_idx']]
    
    # Apply steering hook
    hook = create_steering_hook(decoder_direction, coefficient)
    hook_handle = model.model.layers[feature_info['layer']].register_forward_pre_hook(hook)
    
    # Generate with steering
    with torch.no_grad():
        steered_output = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.0,
            do_sample=False
        )
    
    # Remove hook
    hook_handle.remove()
    
    steered_code = tokenizer.decode(steered_output[0], skip_special_tokens=True)
    steered_code = extract_code(steered_code, prompt)
    
    # Evaluate
    baseline_passed = evaluate_code(baseline_code, problem['test_list'])
    steered_passed = evaluate_code(steered_code, problem['test_list'])
    
    return {
        'task_id': problem['task_id'],
        'baseline_passed': baseline_passed,
        'steered_passed': steered_passed,
        'flipped': baseline_passed != steered_passed,
        'baseline_code': baseline_code,
        'steered_code': steered_code
    }

# Test each coefficient on subset of problems
coefficient_results = {}

for coeff in [1, 5, 10, 15, 20, 30, 50, 100]:
    logger.info(f"Testing coefficient: {coeff}")
    results = []
    
    # Process each problem individually
    for problem in tqdm(validation_problems[:20], desc=f"Coeff {coeff}"):
        try:
            result = evaluate_steering_single_prompt(
                problem, model, tokenizer, sae, feature_info,
                coeff, prompt_builder, device
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed on {problem['task_id']}: {e}")
            continue
    
    coefficient_results[coeff] = {
        'flip_rate': calculate_flip_rate(results),
        'divergence': calculate_generation_divergence(results),
        'examples': results[:5]  # Save examples for manual inspection
    }
```

### Phase 3: Manual Coefficient Selection
```python
# Plot flip rates and divergence metrics
plot_coefficient_effects(coefficient_results)

# Save examples for manual review
for coeff, data in coefficient_results.items():
    save_json(
        data['examples'], 
        output_dir / f'coefficient_examples/coeff_{coeff}_examples.json'
    )

# Print summary for manual selection
print("\nCoefficient Analysis Summary:")
print("Coeff | Flip Rate | Token Sim | Char Sim | Length Ratio")
print("-" * 60)
for coeff, data in sorted(coefficient_results.items()):
    print(f"{coeff:5d} | {data['flip_rate']:8.1f}% | "
          f"{data['divergence']['mean_token_similarity']:9.3f} | "
          f"{data['divergence']['mean_char_similarity']:8.3f} | "
          f"{data['divergence']['mean_length_ratio']:11.3f}")

# After manual review, save selected coefficients
selected_coefficients = {
    'correct': 20,  # Example: manually selected based on analysis
    'incorrect': 30,  # Example: may need different strength
    'selection_rationale': {
        'correct': 'Coefficient 20 showed 35% flip rate with good code quality',
        'incorrect': 'Coefficient 30 needed for stronger effect on incorrect steering'
    },
    'selection_timestamp': datetime.now().isoformat(),
    'phase': '4.5'
}

# Save for Phase 4.8
output_file = output_dir / 'selected_coefficients.json'
save_json(selected_coefficients, output_file)
logger.info(f"Saved selected coefficients to {output_file}")
```

### Phase 4: Output Summary
```python
# Create final summary for this phase
summary = {
    'phase': '4.5',
    'purpose': 'Steering Coefficient Selection',
    'coefficients_tested': [1, 5, 10, 15, 20, 30, 50, 100],
    'problems_evaluated': 20,
    'selected_coefficients': selected_coefficients,
    'coefficient_analysis': coefficient_results,
    'next_phase': '4.8 - Comprehensive steering effect analysis'
}

save_json(summary, output_dir / 'phase_4_5_summary.json')
logger.info("Phase 4.5 complete. Selected coefficients saved for Phase 4.8.")
```

### Phase 5: Results Visualization
```python
# Generate comparison plots
create_steering_comparison_plots(
    baseline_results,
    correct_steering_results,
    incorrect_steering_results
)

# Generate flip rate analysis
analyze_flip_patterns(all_results)
```

## Manual Coefficient Selection Guidelines

### What to Look For
1. **Flip Rate Sweet Spot**
   - Too low (< 10%): Steering too weak
   - Too high (> 80%): Likely breaking code structure
   - Ideal: 20-50% shows meaningful influence

2. **Code Quality Indicators**
   - Check if flipped codes maintain syntactic validity
   - Look for semantic changes vs. syntax errors
   - Verify function signatures remain intact

3. **Generation Coherence**
   - Token similarity > 0.5: Structure preserved
   - Token similarity < 0.3: Major structural changes
   - Character similarity helps identify formatting changes

### Example Analysis Output
```
Coefficient: 20
Flip Rate: 35.0%
Examples of flips:
  - Task 001: Pass→Fail (logic error introduced)
  - Task 005: Fail→Pass (edge case handled)
  - Task 012: Pass→Fail (syntax error in loop)

Generation Quality:
  - Average token similarity: 0.68
  - Average character similarity: 0.75
  - Most changes: variable names, operators, control flow
```

## Key Insights for Coefficient Selection

1. **Feature Effectiveness**: Look for meaningful flip rates without breaking code structure
2. **Asymmetric Coefficients**: Different features may need different steering strengths
3. **Quality vs. Effect**: Balance between steering effect and maintaining code coherence
4. **Manual Review**: Human judgment needed to select optimal coefficients

## Output Files

- `selected_coefficients.json`: Chosen coefficients for Phase 4.8
- `coefficient_examples/`: Example generations for each coefficient
- `coefficient_analysis_plots.png`: Visualization of coefficient effects
- `phase_4_5_summary.json`: Complete summary of this phase

## Implementation Checklist

### Setup
- [ ] Load Phase 2.5 outputs for best PVA features
- [ ] Load validation split from Phase 0.1
- [ ] Initialize SAEs and extract decoder directions

### Coefficient Search
- [ ] Implement steering hooks for generation
- [ ] Run grid search on 20 problems
- [ ] Calculate flip rates and divergence metrics
- [ ] Save examples for manual inspection
- [ ] Select optimal coefficients

### Output
- [ ] Save selected coefficients to JSON file
- [ ] Create phase summary with rationale
- [ ] Document coefficient selection process
- [ ] Prepare outputs for Phase 4.8

### Analysis
- [ ] Compare steering effects (flip rates, divergence)
- [ ] Analyze failure modes and success patterns
- [ ] Document generation quality observations
- [ ] Create final summary report

## Notes

- **Generation Time**: Steering during generation is slower than classification
- **Deterministic Generation**: Use temperature=0.0 for reproducibility
- **Memory Management**: Clear GPU cache between runs
- **Error Handling**: Some steered code may fail to parse - track these cases
- **Continuous Steering**: Apply throughout generation, not just at start
- **Single Prompt Processing**: Process one problem at a time for memory efficiency and consistency with other phases