# Attention Analysis Technical Specification

## Executive Summary

1. **Purpose**: Analyze how attention patterns change when steering model behavior with SAE features
2. **Method**: Compare attention scores captured at the final prompt token before and after applying SAE steering
3. **Capture Strategy**: ONE-TIME capture at the last prompt token position (no capture during autoregressive generation)
4. **Target Layers**: Only the best PVA layers identified in Phase 2.5 (not all layers)
5. **Evaluation**: Statistical comparison of attention differences with significance testing
6. **Dataset**: Validation split from Phase 0.1 (MBPP problems)
7. **Collection Phases**: 
   - Phase 3.5: Captures attention ONCE per task at temperature 0
   - Phase 4.8: Captures attention ONCE per task during steering
   - Phase 6.3: Analyzes the collected attention patterns

## Pipeline Sequence

```
Phase 3.5: Temperature Robustness (Data Collection)
├─> Generate at temperature 0 with activation extraction
├─> Extract attention patterns ONCE at final prompt token (before generation starts)
├─> Capture only from best PVA layers (from Phase 2.5)
├─> Save attention scores per task: {task_id}_attention.npz
└─> Include attention data in checkpoints

Phase 4.8: Steering Effect Analysis (Data Collection)
├─> Apply steering to validation problems
├─> Extract attention patterns ONCE at final prompt token during steered generation
├─> Capture only from steered layer (best correct/incorrect from Phase 2.5)
├─> Save both baseline and steered attention patterns
└─> No capture during autoregressive token generation

Phase 6.3: Attention Pattern Analysis (Analysis Only)
├─> Load attention data from Phase 3.5 (baseline patterns)
├─> Load attention data from Phase 4.8 (steered patterns)
├─> Compute statistical comparisons (steered - baseline)
├─> Test significance against random baselines
└─> Generate visualizations and reports
```

## Understanding Attention Analysis

### Critical Design Decision: One-Time Capture Strategy

**IMPORTANT**: Attention and activations are captured ONLY ONCE per task:
- **When**: At the final prompt token position (position -1)
- **Where**: Only in the best PVA layers identified by Phase 2.5
- **Why**: To analyze the model's attention state at the decision point before generation
- **NOT**: During autoregressive generation (no capture during token-by-token generation)

This is enforced by the hook guard in `activation_hooks.py`:
```python
if layer_idx not in self.activations:  # Only capture on first forward pass
    activation = residual_stream[:, self.position, :].detach().clone().cpu()
```

### Attention Pattern Types
The system supports multiple attention pattern extraction methods:
- **Standard Attention Weights**: Direct attention scores from attention matrices
- **Value-Weighted Patterns**: Attention weighted by value vectors
- **Output-Value-Weighted**: Full OV circuit decomposition
- **Distance-Based**: Similarity metrics in representation space

### Attention Aggregation  
For code generation tasks, attention is computed by:
1. Extracting attention patterns ONCE at the final prompt token
2. Focusing on the best PVA layers only (not all layers)
3. Comparing attention patterns between baseline and steered conditions
4. Storing raw attention patterns for statistical analysis

## Core Implementation

### 1. Attention Collection During Generation (Phase 3.5)

```python
# phase3_5_temperature_robustness/temperature_runner.py (modified)
def generate_temp0_with_attention(self, prompt: str) -> Tuple[str, Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Generate at temperature 0, extracting both activations and attention patterns.
    
    Args:
        prompt: The input prompt for generation
    
    Returns:
        Tuple of (generated_text, activations_dict, attention_dict)
        - generated_text: The generated code
        - activations_dict: {layer_num: activation_tensor}
        - attention_dict: {'attn_weights': tensor, 'positions': list}
    """
    # Setup hooks for activation and attention extraction
    self.activation_extractor.setup_hooks()
    self.attention_extractor.setup_hooks()
    
    try:
        # Generate with attention tracking enabled
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=0.0,
                max_new_tokens=self.config.model_max_new_tokens,
                do_sample=False,
                output_attentions=True,  # Enable attention output
                return_dict_in_generate=True
            )
        
        # Extract attention patterns from generation
        attention_patterns = self.attention_extractor.get_attention_patterns()
        
        return generated_text, activations, attention_patterns
    finally:
        # Clean up hooks
        self.activation_extractor.remove_hooks()
        self.attention_extractor.remove_hooks()
```

### 2. Attention Collection During Steering (Phase 4.8)

```python
# phase4_8_steering_analysis/steering_effect_analyzer.py (modified)
def generate_with_steering_and_attention(self, prompt: str, steering_type: str, coefficient: float) -> Dict:
    """
    Generate code with steering applied, capturing attention patterns.
    
    Args:
        prompt: Input prompt for generation
        steering_type: 'correct' or 'incorrect' steering
        coefficient: Steering strength
        
    Returns:
        Dict with generated_code, test_passed, and attention_patterns
    """
    # Setup steering hook
    decoder_direction = self.correct_decoder_direction if steering_type == 'correct' else self.incorrect_decoder_direction
    hook_fn = create_steering_hook(decoder_direction, coefficient)
    target_layer = self.best_correct_feature['layer'] if steering_type == 'correct' else self.best_incorrect_feature['layer']
    target_module = self.model.model.layers[target_layer]
    hook_handle = target_module.register_forward_pre_hook(hook_fn)
    
    # Setup attention extraction
    self.attention_extractor.setup_hooks()
    
    try:
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.model_max_new_tokens,
                temperature=0.0,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        # Get attention patterns with steering applied
        steered_attention = self.attention_extractor.get_attention_patterns()
        
        return {
            'generated_code': generated_code,
            'test_passed': test_passed,
            'attention_patterns': steered_attention
        }
    finally:
        hook_handle.remove()
        self.attention_extractor.remove_hooks()
```

### 3. Attention Pattern Storage with Raw Tensors and Boundaries

```python
# common_simplified/attention_hooks.py (new)
class AttentionExtractor:
    """
    Extract and store attention patterns during generation.
    CRITICAL: Captures ONLY ONCE at the final prompt token, NOT during autoregressive generation.
    """
    
    def __init__(self, model, layers: List[int] = None, position: int = -1):
        """
        Initialize attention extractor for specified layers.
        
        Args:
            model: The model to extract from
            layers: ONLY the best PVA layers from Phase 2.5 (not all layers!)
            position: Token position to extract (-1 for last prompt token)
        """
        self.model = model
        self.layers = layers  # Should be just 1-2 layers from Phase 2.5!
        self.position = position
        self.attention_patterns = {}
        self.hooks = []
        self.captured = set()  # Track which layers have been captured
    
    def setup_hooks(self):
        """Register hooks to capture attention during forward pass."""
        for layer_idx in self.layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.self_attn.register_forward_hook(
                self._attention_hook(layer_idx)
            )
            self.hooks.append(hook)
    
    def _attention_hook(self, layer_idx: int):
        """Hook function to capture attention patterns ONCE."""
        def hook(module, input, output):
            # CRITICAL: Only capture on first forward pass (prompt processing)
            # Skip all subsequent forward passes during autoregressive generation
            if layer_idx not in self.captured:
                # Extract attention weights from output at final prompt position
                if hasattr(output, 'attentions') and output.attentions is not None:
                    # Get attention at last prompt token only
                    attn = output.attentions[:, :, self.position, :]
                    self.attention_patterns[layer_idx] = attn.detach().cpu()
                    self.captured.add(layer_idx)  # Mark as captured
        return hook
    
    def get_attention_patterns(self) -> Dict[int, torch.Tensor]:
        """Return captured attention patterns and clear cache."""
        patterns = self.attention_patterns.copy()
        self.attention_patterns.clear()
        self.captured.clear()  # Reset for next task
        return patterns
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def save_raw_attention_with_boundaries(task_id: str, attention_tensor: torch.Tensor, 
                                       tokenized_prompt: torch.Tensor, tokenizer, 
                                       output_dir: Path, layer_idx: int):
    """
    Save raw attention patterns with section boundaries for flexible Phase 6.3 analysis.
    
    CRITICAL: Saves raw attention tensor (8 heads × seq_len) plus boundaries.
    This allows different aggregation strategies in Phase 6.3.
    """
    # Identify section boundaries
    prompt_text = tokenizer.decode(tokenized_prompt)
    parts = prompt_text.split('\n\n')
    
    # Calculate token boundaries for each section
    boundaries = calculate_section_boundaries(parts, tokenizer, tokenized_prompt)
    
    # Save raw attention and boundaries - NO pre-aggregation
    save_path = output_dir / f"{task_id}_layer_{layer_idx}_attention.npz"
    np.savez_compressed(
        save_path,
        raw_attention=attention_tensor.cpu().numpy(),  # Shape: (8, seq_len)
        boundaries=boundaries,  # Dict with problem_end, test_end indices
        prompt_length=len(tokenized_prompt),
        layer=layer_idx,
        prompt_text=prompt_text  # Save for verification/debugging
    )
    
    return save_path

def calculate_section_boundaries(parts: List[str], tokenizer, tokenized_prompt) -> Dict[str, int]:
    """
    Calculate precise token boundaries for prompt sections.
    
    Returns:
        Dict with 'problem_end', 'test_end' token indices
    """
    boundaries = {}
    
    # Problem description boundary
    if len(parts) > 0:
        problem_tokens = tokenizer.encode(parts[0], add_special_tokens=False)
        boundaries['problem_end'] = len(problem_tokens)
    else:
        boundaries['problem_end'] = 0
    
    # Test cases boundary (problem + tests)
    if len(parts) > 1:
        test_section = parts[0] + '\n\n' + parts[1]
        test_tokens = tokenizer.encode(test_section, add_special_tokens=False)
        boundaries['test_end'] = len(test_tokens)
    else:
        boundaries['test_end'] = boundaries['problem_end']
    
    # Total length for validation
    boundaries['total_length'] = len(tokenized_prompt)
    
    return boundaries

### 4. Value-Weighted Attention Patterns

```python
# mech_interp/mech_interp_utils.py:1341-1358
def get_value_weighted_patterns(model, cache, layer):
    """
    Compute attention-weighted value vectors: a_{i,j} * x_j * W_V
    
    Handles grouped query attention for models with different 
    numbers of attention and key-value heads.
    """
    pattern = cache[f'blocks.{layer}.attn.hook_pattern']
    v = cache[f'blocks.{layer}.attn.hook_v']
    
    # Handle grouped query attention
    if model.cfg.n_heads != v.shape[2]:
        repeat_kv_heads = model.cfg.n_heads // model.cfg.n_key_value_heads
        v = torch.repeat_interleave(v, dim=2, repeats=repeat_kv_heads)
    
    weighted_values = einsum(
        "batch key_pos head_index d_head, \
         batch head_index query_pos key_pos -> \
         batch query_pos key_pos head_index d_head",
        v, pattern
    )
    return weighted_values
```

### 5. Position Selection Strategy

```python
# mech_interp/hooks_utils.py:139-202
def get_batch_pos(batch_entity_pos, pos_type, tokenized_prompts):
    """
    Determine which token positions to analyze/steer.
    
    Position types:
    - 'entity_last': Only the last token of entity name
    - 'entity': All entity tokens
    - 'entity_to_end': Entity tokens through end of sequence
    - 'entity_last_to_end': Last entity token through end
    - 'all': Every position
    """
    if pos_type == 'entity_last':
        # Most common choice for attention analysis
        batch_pos = [[entity_pos_[-1]] for entity_pos_ in batch_entity_pos]
    
    elif pos_type == 'entity':
        batch_pos = batch_entity_pos
    
    elif pos_type == 'entity_to_end':
        batch_pos = []
        for idx, pos_ in enumerate(batch_entity_pos):
            len_seq = len(tokenized_prompts[idx])
            last_pos = pos_[-1]
            for j in range(1, len_seq - last_pos):
                pos_.append(last_pos + j)
            batch_pos.append(pos_)
    
    return batch_pos
```

### 6. Statistical Significance Testing

```python
# mech_interp/attn_analysis.py:189-227
def compare_means(sample1, sample2, alpha=0.05):
    """
    Performs one-tailed Welch's t-test to determine if sample1's mean
    is significantly larger than sample2's mean.
    
    Uses unequal variance assumption (Welch's test).
    """
    stats_dict = {
        'mean1': np.mean(sample1),
        'mean2': np.mean(sample2),
        'std1': np.std(sample1, ddof=1),
        'std2': np.std(sample2, ddof=1),
        'n1': len(sample1),
        'n2': len(sample2)
    }
    
    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    
    # Convert to one-tailed p-value
    one_tailed_p = p_value / 2 if t_stat > 0 else 1 - (p_value / 2)
    
    stats_dict.update({
        't_statistic': t_stat,
        'p_value': one_tailed_p,
        'significant': one_tailed_p < alpha
    })
    
    return stats_dict
```

### 7. Attention Difference Computation (Phase 6.3)

```python
# phase6_3_attention_analysis/attention_analyzer.py (new)
def compute_attention_differences(self) -> Dict[str, np.ndarray]:
    """
    Compare attention patterns between baseline and steered generations.
    
    Process:
    1. Load baseline attention from Phase 3.5
    2. Load steered attention from Phase 4.8
    3. Calculate differences per task
    4. Aggregate statistics across tasks
    """
    differences = {}
    
    # Load baseline attention patterns from Phase 3.5
    baseline_attention_dir = self.phase3_5_dir / "attention_patterns"
    
    # Load steered attention patterns from Phase 4.8
    steered_attention_dir = self.phase4_8_dir / "steered_attention"
    
    for task_id in self.task_ids:
        # Load baseline attention for this task
        baseline_path = baseline_attention_dir / f"{task_id}_attention.npz"
        baseline_attn = np.load(baseline_path)['attention']
        
        # Load steered attention for this task
        steered_path = steered_attention_dir / f"{task_id}_steered_attention.npz"
        steered_attn = np.load(steered_path)['attention']
        
        # Compute difference: steered - baseline
        differences[task_id] = steered_attn - baseline_attn
    
    return differences
```

## Experimental Configuration

### Main Analysis Parameters
```python
# phase6_3_attention_analysis/attention_analyzer.py
# Configuration parameters
type_pattern = 'attn_weights'  # or 'value_weighted'
attention_aggregation = 'last_to_key'  # Attention from last position to key tokens
significance_alpha = 0.05      # Statistical significance threshold

# Model-specific settings (from Phase 2.5)
best_layers = {
    'correct': self.best_correct_feature['layer'],
    'incorrect': self.best_incorrect_feature['layer']
}

# Heads to analyze - focus on layers identified in Phase 2.5
if model_alias == 'gemma-2-2b':
    # Analyze all heads in the best layers
    heads_to_analyze = [
        (best_layers['correct'], head_idx) 
        for head_idx in range(model.config.num_attention_heads)
    ]
```

### Data Loading
```python
# phase6_3_attention_analysis/attention_analyzer.py
def load_attention_data(self):
    """
    Load attention patterns from Phase 3.5 and Phase 4.8.
    
    Returns:
        Dict with baseline and steered attention patterns per task
    """
    attention_data = {}
    
    # Discover Phase 3.5 output directory
    phase3_5_output = discover_latest_phase_output("3.5")
    phase3_5_dir = Path(phase3_5_output).parent if phase3_5_output else Path(self.config.phase3_5_output_dir)
    
    # Discover Phase 4.8 output directory  
    phase4_8_output = discover_latest_phase_output("4.8")
    phase4_8_dir = Path(phase4_8_output).parent if phase4_8_output else Path(self.config.phase4_8_output_dir)
    
    # Load validation task IDs from Phase 0.1
    validation_data = pd.read_parquet(Path(self.config.phase0_1_output_dir) / "validation_mbpp.parquet")
    
    for task_id in validation_data['task_id']:
        attention_data[task_id] = {
            'baseline': self._load_task_attention(phase3_5_dir, task_id, 'baseline'),
            'steered_correct': self._load_task_attention(phase4_8_dir, task_id, 'correct'),
            'steered_incorrect': self._load_task_attention(phase4_8_dir, task_id, 'incorrect')
        }
    
    return attention_data
```

### Statistical Analysis
```python
# phase6_3_attention_analysis/attention_analyzer.py
def compute_statistical_significance(self, attention_differences: Dict[str, np.ndarray]):
    """
    Test if steering produces statistically significant attention changes.
    
    Process:
    1. Aggregate attention differences across tasks
    2. Compare against null hypothesis (no change)
    3. Use Welch's t-test for significance
    4. Report per-head statistics
    """
    results = {}
    
    # Aggregate differences across all tasks
    all_differences = np.stack(list(attention_differences.values()))
    
    # Test each attention head separately
    n_layers, n_heads = all_differences.shape[1:3]
    
    for layer in range(n_layers):
        for head in range(n_heads):
            head_differences = all_differences[:, layer, head]
            
            # Test if mean difference is significantly different from 0
            t_stat, p_value = stats.ttest_1samp(head_differences, 0)
            
            results[f'L{layer}H{head}'] = {
                'mean_difference': np.mean(head_differences),
                'std_difference': np.std(head_differences),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    return results
```

## Visualization Strategy: Flexible Analysis with Raw Attention

### Core Concept
Store raw attention tensors with section boundaries, enabling flexible aggregation in Phase 6.3:
1. **Raw Storage**: Save full attention tensor (8 heads × seq_len) 
2. **Boundary Markers**: Store token indices for problem_end and test_end
3. **Phase 6.3 Flexibility**: Perform different aggregations during analysis

### Data Storage Format
```python
# Saved in Phase 3.5 and 4.8 for each task:
{
    'raw_attention': np.ndarray,      # Shape: (8, seq_len) - full attention from last token
    'boundaries': {
        'problem_end': int,            # Token index where problem description ends
        'test_end': int,               # Token index where test cases end
        'total_length': int            # Total prompt length for validation
    },
    'layer': int,                      # Which layer this attention is from
    'prompt_text': str                 # For debugging/verification
}
```

### Phase 6.3: 3-Bin Aggregation During Analysis
```python
# phase6_3_attention_analysis/attention_analyzer.py
def load_and_aggregate_attention(self, task_id: str, phase_dir: Path) -> Dict:
    """
    Load raw attention and perform 3-bin aggregation during analysis.
    This happens in Phase 6.3, not during collection.
    
    Returns:
        Dict with both raw and aggregated attention data
    """
    # Load raw attention data
    attention_files = list((phase_dir / "attention_patterns").glob(f"{task_id}_layer_*_attention.npz"))
    
    aggregated_by_layer = {}
    for file_path in attention_files:
        data = np.load(file_path)
        raw_attention = data['raw_attention']  # Shape: (8, seq_len)
        boundaries = data['boundaries'].item()
        layer = data['layer'].item()
        
        # Perform 3-bin aggregation here in Phase 6.3
        section_attention = self.aggregate_to_3_bins(raw_attention, boundaries)
        
        aggregated_by_layer[layer] = {
            'raw': raw_attention,
            'aggregated': section_attention,
            'boundaries': boundaries
        }
    
    return aggregated_by_layer

def aggregate_to_3_bins(self, attention_tensor: np.ndarray, boundaries: Dict) -> Dict:
    """
    Aggregate raw attention into 3 bins based on section boundaries.
    
    Args:
        attention_tensor: [n_heads, sequence_length] - raw attention
        boundaries: Dict with 'problem_end', 'test_end' indices
    
    Returns:
        Dict with aggregated attention per section
    """
    # Sum attention within each section
    section_attention = {
        'problem': attention_tensor[:, :boundaries['problem_end']].sum(axis=-1),
        'tests': attention_tensor[:, boundaries['problem_end']:boundaries['test_end']].sum(axis=-1),
        'solution_marker': attention_tensor[:, boundaries['test_end']:].sum(axis=-1)
    }
    
    # Also compute normalized percentages
    total_per_head = np.array([section_attention['problem'],
                               section_attention['tests'],
                               section_attention['solution_marker']]).sum(axis=0)
    
    section_percentages = {
        section: (attention / total_per_head * 100)
        for section, attention in section_attention.items()
    }
    
    return {
        'raw_sums': section_attention,
        'percentages': section_percentages,
        'normalized': {  # Length-normalized attention
            'problem': section_attention['problem'] / max(boundaries['problem_end'], 1),
            'tests': section_attention['tests'] / max(boundaries['test_end'] - boundaries['problem_end'], 1),
            'solution_marker': section_attention['solution_marker'] / max(boundaries['total_length'] - boundaries['test_end'], 1)
        }
    }
```

### Alternative Aggregation Strategies (Available in Phase 6.3)
```python
def compute_alternative_aggregations(self, raw_attention: np.ndarray, boundaries: Dict) -> Dict:
    """
    Demonstrate flexibility of raw storage - multiple aggregation strategies.
    All computed in Phase 6.3 from the same raw data.
    """
    aggregations = {}
    
    # Strategy 1: Standard 3-bin sum
    aggregations['3_bin_sum'] = self.aggregate_to_3_bins(raw_attention, boundaries)
    
    # Strategy 2: Max attention per section
    aggregations['3_bin_max'] = {
        'problem': raw_attention[:, :boundaries['problem_end']].max(axis=-1),
        'tests': raw_attention[:, boundaries['problem_end']:boundaries['test_end']].max(axis=-1),
        'solution_marker': raw_attention[:, boundaries['test_end']:].max(axis=-1)
    }
    
    # Strategy 3: Attention to specific keywords (if needed)
    # Can identify important tokens and analyze attention to them specifically
    
    # Strategy 4: Fine-grained analysis (e.g., per test case if identifiable)
    # Can split test section further if patterns are consistent
    
    return aggregations

## Visualization Components

### 1. Attention Distribution Stacked Bar Chart
```python
def create_attention_distribution_chart(self, attention_data):
    """
    Create stacked bar chart showing attention distribution across 3 sections.
    One bar per condition: Baseline, Correct Steering, Incorrect Steering.
    """
    # Aggregate across all tasks
    distributions = self._aggregate_distributions(attention_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = ['Baseline', 'Correct\nSteering', 'Incorrect\nSteering']
    x = np.arange(len(conditions))
    
    # Stack the bars
    problem_heights = [distributions[c]['problem'] for c in conditions]
    test_heights = [distributions[c]['tests'] for c in conditions]
    solution_heights = [distributions[c]['solution_marker'] for c in conditions]
    
    ax.bar(x, problem_heights, label='Problem Description', color='#8dd3c7')
    ax.bar(x, test_heights, bottom=problem_heights, label='Test Cases', color='#ffffb3')
    ax.bar(x, solution_heights, 
           bottom=np.array(problem_heights) + np.array(test_heights),
           label='Solution Marker', color='#bebada')
    
    ax.set_ylabel('Attention Distribution (%)')
    ax.set_title('Attention Focus Across Prompt Sections')
    ax.legend()
    
    return fig
```

### 2. Head-Level Attention Heatmap
```python
def create_head_attention_heatmap(self, attention_data):
    """
    Heatmap showing each head's attention to each section.
    Rows: 8 attention heads (Gemma-2-2b)
    Columns: 3 sections × 3 conditions = 9 columns
    """
    n_heads = 8  # For Gemma-2-2b
    
    # Create matrix [n_heads, 9]
    attention_matrix = self._build_head_attention_matrix(attention_data, n_heads)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(attention_matrix, cmap='YlOrRd', aspect='auto')
    
    # Labels and formatting
    ax.set_xticks(np.arange(9))
    ax.set_xticklabels(['Prob', 'Test', 'Sol'] * 3)
    ax.set_yticks(np.arange(n_heads))
    ax.set_yticklabels([f'Head {i}' for i in range(n_heads)])
    
    # Add condition separators
    ax.axvline(x=2.5, color='white', linewidth=2)
    ax.axvline(x=5.5, color='white', linewidth=2)
    
    plt.colorbar(im, ax=ax, label='Attention Score')
    plt.title('Per-Head Attention Distribution')
    
    return fig
```

### 3. Attention Change Delta Plots
```python
def create_attention_delta_plots(self, attention_data):
    """
    Show how steering changes attention to each section.
    Separate plots for correct and incorrect steering.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate changes from baseline
    correct_deltas = self._calculate_deltas(attention_data, 'correct')
    incorrect_deltas = self._calculate_deltas(attention_data, 'incorrect')
    
    sections = ['Problem\nDesc.', 'Test\nCases', 'Solution\nMarker']
    x = np.arange(len(sections))
    
    # Plot correct steering effects
    self._plot_delta_bars(ax1, x, sections, correct_deltas, 'Correct Steering Effect')
    
    # Plot incorrect steering effects  
    self._plot_delta_bars(ax2, x, sections, incorrect_deltas, 'Incorrect Steering Effect')
    
    plt.suptitle('Attention Redistribution Due to Steering')
    return fig
```

### 4. Statistical Significance Table
```python
def create_significance_table(self, attention_data):
    """
    Table showing statistical significance of attention changes.
    Uses paired t-tests comparing baseline vs steered conditions.
    """
    results = self._compute_statistical_tests(attention_data)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Build table data
    headers = ['Section', 'Correct Δ', 'p-value', 'Sig?', 'Incorrect Δ', 'p-value', 'Sig?']
    table_data = []
    
    for section in ['problem', 'tests', 'solution_marker']:
        row = [
            section.title(),
            f"{results[section]['correct_change']:.1f}%",
            f"{results[section]['correct_pval']:.4f}",
            '✓' if results[section]['correct_sig'] else '✗',
            f"{results[section]['incorrect_change']:.1f}%",
            f"{results[section]['incorrect_pval']:.4f}",
            '✓' if results[section]['incorrect_sig'] else '✗'
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    
    # Highlight significant results
    self._highlight_significant_cells(table, table_data)
    
    plt.title('Statistical Significance of Attention Changes')
    return fig
```

### 5. Attention Transformation Scatter Plots
```python
def create_attention_transformation_scatter(self, baseline_attention, steered_attention, steering_type: str):
    """
    Scatter plot comparing baseline (x-axis) vs steered (y-axis) attention scores.
    Shows how steering transforms attention patterns across all tasks and heads.
    
    Args:
        baseline_attention: Dict of baseline attention by task
        steered_attention: Dict of steered attention by task  
        steering_type: 'correct' or 'incorrect'
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sections = ['problem', 'tests', 'solution_marker']
    colors = ['#8dd3c7', '#ffffb3', '#bebada']
    
    for idx, (ax, section, color) in enumerate(zip(axes, sections, colors)):
        # Collect all attention scores for this section
        baseline_scores = []
        steered_scores = []
        
        for task_id in baseline_attention.keys():
            # Get aggregated attention for this section
            baseline_agg = self.aggregate_to_3_bins(
                baseline_attention[task_id]['raw'], 
                baseline_attention[task_id]['boundaries']
            )
            steered_agg = self.aggregate_to_3_bins(
                steered_attention[task_id]['raw'],
                steered_attention[task_id]['boundaries']
            )
            
            # Collect scores for all heads
            baseline_scores.extend(baseline_agg['percentages'][section])
            steered_scores.extend(steered_agg['percentages'][section])
        
        # Create scatter plot
        ax.scatter(baseline_scores, steered_scores, alpha=0.5, s=20, c=color)
        
        # Add diagonal reference line (y=x)
        max_val = max(max(baseline_scores), max(steered_scores))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='No change')
        
        # Add trend line
        z = np.polyfit(baseline_scores, steered_scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(baseline_scores), max(baseline_scores), 100)
        ax.plot(x_trend, p(x_trend), 'r-', alpha=0.5, linewidth=2, 
                label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Formatting
        ax.set_xlabel('Baseline Attention (%)')
        ax.set_ylabel(f'{steering_type.title()} Steered Attention (%)')
        ax.set_title(f'{section.title()} Section')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Color regions to show increase/decrease
        ax.axhspan(0, max_val, where=ax.get_ylim()[0] > ax.get_xlim()[0], 
                  facecolor='green', alpha=0.1)  # Increased attention region
        ax.axhspan(0, max_val, where=ax.get_ylim()[0] < ax.get_xlim()[0],
                  facecolor='red', alpha=0.1)  # Decreased attention region
    
    plt.suptitle(f'Attention Transformation: Baseline → {steering_type.title()} Steering')
    plt.tight_layout()
    return fig

def create_head_specific_transformation_plot(self, baseline_attention, steered_attention):
    """
    More detailed scatter plot showing each head's transformation separately.
    Useful for identifying which heads are most affected by steering.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for head_idx in range(8):  # 8 heads for Gemma-2-2b
        ax = axes[head_idx]
        
        # Collect scores for this specific head
        baseline_problem, baseline_tests, baseline_solution = [], [], []
        steered_problem, steered_tests, steered_solution = [], [], []
        
        for task_id in baseline_attention.keys():
            baseline_agg = self.aggregate_to_3_bins(
                baseline_attention[task_id]['raw'],
                baseline_attention[task_id]['boundaries']
            )
            steered_agg = self.aggregate_to_3_bins(
                steered_attention[task_id]['raw'],
                steered_attention[task_id]['boundaries']
            )
            
            # Get scores for this head only
            baseline_problem.append(baseline_agg['percentages']['problem'][head_idx])
            baseline_tests.append(baseline_agg['percentages']['tests'][head_idx])
            baseline_solution.append(baseline_agg['percentages']['solution_marker'][head_idx])
            
            steered_problem.append(steered_agg['percentages']['problem'][head_idx])
            steered_tests.append(steered_agg['percentages']['tests'][head_idx])
            steered_solution.append(steered_agg['percentages']['solution_marker'][head_idx])
        
        # Plot all three sections with different colors
        ax.scatter(baseline_problem, steered_problem, alpha=0.6, s=15, 
                  c='#8dd3c7', label='Problem')
        ax.scatter(baseline_tests, steered_tests, alpha=0.6, s=15,
                  c='#ffffb3', label='Tests')
        ax.scatter(baseline_solution, steered_solution, alpha=0.6, s=15,
                  c='#bebada', label='Solution')
        
        # Add diagonal
        max_val = 100
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        
        ax.set_xlabel('Baseline %')
        ax.set_ylabel('Steered %')
        ax.set_title(f'Head {head_idx}')
        ax.grid(True, alpha=0.3)
        if head_idx == 0:
            ax.legend(loc='upper left', fontsize=7)
    
    plt.suptitle('Per-Head Attention Transformation Analysis')
    plt.tight_layout()
    return fig
```

### 6. Head-Specific Attention Change Bar Charts
```python
def create_head_attention_change_bars(self, baseline_attention, steered_attention, steering_type: str):
    """
    Bar charts showing attention score differences (steered - baseline) for each head.
    Separate subplots for each section (problem, tests, solution), with error bars.
    
    Args:
        baseline_attention: Dict of baseline attention by task
        steered_attention: Dict of steered attention by task
        steering_type: 'correct' or 'incorrect'
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    sections = ['problem', 'tests', 'solution_marker']
    section_labels = ['Problem Description', 'Test Cases', 'Solution Marker']
    colors = ['#8dd3c7', '#ffffb3', '#bebada']
    
    # Assuming we're analyzing the best layer from Phase 2.5
    layer_idx = self.best_layer  # e.g., 15
    
    for ax, section, label, color in zip(axes, sections, section_labels, colors):
        # Calculate differences for each head across all tasks
        head_differences = {f'L{layer_idx}H{h}': [] for h in range(8)}
        
        for task_id in baseline_attention.keys():
            # Get aggregated attention
            baseline_agg = self.aggregate_to_3_bins(
                baseline_attention[task_id]['raw'],
                baseline_attention[task_id]['boundaries']
            )
            steered_agg = self.aggregate_to_3_bins(
                steered_attention[task_id]['raw'],
                steered_attention[task_id]['boundaries']
            )
            
            # Calculate difference for each head
            for head_idx in range(8):
                diff = (steered_agg['percentages'][section][head_idx] - 
                       baseline_agg['percentages'][section][head_idx])
                head_differences[f'L{layer_idx}H{head_idx}'].append(diff)
        
        # Calculate means and standard deviations
        x_labels = []
        means = []
        stds = []
        
        for head_idx in range(8):
            label_key = f'L{layer_idx}H{head_idx}'
            x_labels.append(label_key)
            diffs = head_differences[label_key]
            means.append(np.mean(diffs))
            stds.append(np.std(diffs))
        
        # Create bar chart with error bars
        x_pos = np.arange(len(x_labels))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                      color=color, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Color bars based on positive/negative change
        for bar, mean in zip(bars, means):
            if mean > 0:
                bar.set_facecolor('green')
                bar.set_alpha(0.6)
            else:
                bar.set_facecolor('red')
                bar.set_alpha(0.6)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylabel('Attention Score Difference (%)')
        ax.set_title(f'{label} - {steering_type.title()} Steering Effect')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add significance markers for heads with substantial changes
        for i, (mean, std) in enumerate(zip(means, stds)):
            # Mark as significant if |mean| > 2*std (roughly 95% confidence)
            if abs(mean) > 2 * std and std > 0:
                y_pos = mean + np.sign(mean) * (std + 1)
                ax.text(i, y_pos, '*', ha='center', va='bottom' if mean > 0 else 'top',
                       fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Per-Head Attention Changes: {steering_type.title()} Steering\n(* indicates statistically significant change)')
    plt.tight_layout()
    return fig

def create_comparative_head_changes(self, baseline_attention, correct_steered, incorrect_steered):
    """
    Side-by-side comparison of head-specific changes for correct vs incorrect steering.
    Shows which heads respond differently to different steering types.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    sections = ['problem', 'tests', 'solution_marker']
    section_labels = ['Problem Description', 'Test Cases', 'Solution Marker']
    
    layer_idx = self.best_layer
    
    for row_idx, (section, label) in enumerate(zip(sections, section_labels)):
        # Calculate differences for both steering types
        for col_idx, (steered_data, steering_type) in enumerate(
            [(correct_steered, 'Correct'), (incorrect_steered, 'Incorrect')]
        ):
            ax = axes[row_idx, col_idx]
            
            # Calculate differences
            head_differences = []
            head_stds = []
            
            for head_idx in range(8):
                diffs = []
                for task_id in baseline_attention.keys():
                    baseline_agg = self.aggregate_to_3_bins(
                        baseline_attention[task_id]['raw'],
                        baseline_attention[task_id]['boundaries']
                    )
                    steered_agg = self.aggregate_to_3_bins(
                        steered_data[task_id]['raw'],
                        steered_data[task_id]['boundaries']
                    )
                    
                    diff = (steered_agg['percentages'][section][head_idx] - 
                           baseline_agg['percentages'][section][head_idx])
                    diffs.append(diff)
                
                head_differences.append(np.mean(diffs))
                head_stds.append(np.std(diffs))
            
            # Create grouped bar chart
            x_pos = np.arange(8)
            bars = ax.bar(x_pos, head_differences, yerr=head_stds, capsize=5,
                          alpha=0.7, edgecolor='black', linewidth=1)
            
            # Color based on expected behavior
            if steering_type == 'Correct' and section == 'tests':
                # Expect positive changes for test attention with correct steering
                expected_color = 'green'
            elif steering_type == 'Incorrect' and section == 'tests':
                # Expect negative changes for test attention with incorrect steering
                expected_color = 'red'
            else:
                expected_color = 'blue'
            
            for bar, diff in zip(bars, head_differences):
                if section == 'tests':
                    # Color test section bars based on actual change direction
                    bar.set_facecolor('green' if diff > 0 else 'red')
                else:
                    bar.set_facecolor('gray')
                bar.set_alpha(0.6)
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'H{i}' for i in range(8)])
            ax.set_ylabel('Δ Attention (%)' if col_idx == 0 else '')
            ax.set_xlabel('Head Index')
            ax.set_title(f'{label}\n{steering_type} Steering')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(-15, 15)  # Fixed scale for comparison
    
    plt.suptitle('Head-Specific Attention Changes: Correct vs Incorrect Steering Comparison')
    plt.tight_layout()
    return fig
```

## Implementation Workflow

### Phase 3.5: Attention Collection During Temperature Testing
```python
# phase3_5_temperature_robustness/temperature_runner.py (modified)
# In __init__ method:
# IMPORTANT: Only extract from best PVA layers, not all layers!
self.best_layers = self._discover_best_layers()  # From Phase 2.5
self.extraction_layers = list(set([
    self.best_layers['correct'], 
    self.best_layers['incorrect']
]))  # Usually just 1-2 layers total!

# In generate_temp0_with_activations method:
# Initialize attention extractor for ONLY the best layers
self.attention_extractor = AttentionExtractor(
    self.model,
    layers=self.extraction_layers,  # Just 1-2 layers from Phase 2.5!
    position=-1  # Last prompt token ONLY
)

# Setup hooks before generation
self.activation_extractor.setup_hooks()
self.attention_extractor.setup_hooks()

try:
    # Generate - attention captured ONCE at prompt's last token
    outputs = self.model.generate(
        **inputs,
        output_attentions=True,
        return_dict_in_generate=True,
        **generation_params
    )
    
    # Attention was captured during first forward pass only
    # No capture happened during autoregressive generation
    attention_patterns = self.attention_extractor.get_attention_patterns()
    self._save_attention_patterns(task_id, attention_patterns)
    
finally:
    # Clean up hooks
    self.attention_extractor.remove_hooks()
```

### Phase 4.8: Attention Collection During Steering
```python
# phase4_8_steering_analysis/steering_effect_analyzer.py (modified)
# In _apply_steering method:

# Initialize attention extractor
self.attention_extractor = AttentionExtractor(
    self.model,
    layers=[self.best_correct_feature['layer'], self.best_incorrect_feature['layer']]
)

# Setup steering hook
hook_fn = create_steering_hook(decoder_direction, coefficient)
hook_handle = target_module.register_forward_pre_hook(hook_fn)

# Setup attention extraction
self.attention_extractor.setup_hooks()

try:
    # Generate with steering and attention tracking
    outputs = self.model.generate(
        **inputs,
        output_attentions=True,
        return_dict_in_generate=True,
        **generation_params
    )
    
    # Save steered attention patterns
    steered_attention = self.attention_extractor.get_attention_patterns()
    self._save_steered_attention(task_id, steering_type, steered_attention)
    
finally:
    hook_handle.remove()
    self.attention_extractor.remove_hooks()
```

### Phase 6.3: Attention Analysis
```python
# phase6_3_attention_analysis/attention_analyzer.py (new)
class AttentionAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.phase6_3_output_dir)
        ensure_directory_exists(self.output_dir)
        
        # Load features from Phase 2.5
        self._load_pva_features()
        
    def run(self):
        """Main analysis pipeline."""
        # 1. Load attention data from Phases 3.5 and 4.8
        attention_data = self.load_attention_data()
        
        # 2. Compute attention differences
        differences_correct = self.compute_differences(attention_data, 'correct')
        differences_incorrect = self.compute_differences(attention_data, 'incorrect')
        
        # 3. Statistical analysis
        stats_correct = self.compute_statistical_significance(differences_correct)
        stats_incorrect = self.compute_statistical_significance(differences_incorrect)
        
        # 4. Generate visualizations
        self.create_attention_heatmaps(differences_correct, 'correct')
        self.create_attention_heatmaps(differences_incorrect, 'incorrect')
        
        # 5. Save results
        self.save_analysis_results({
            'correct_steering': stats_correct,
            'incorrect_steering': stats_incorrect,
            'n_tasks_analyzed': len(attention_data),
            'layers_analyzed': [self.best_correct_feature['layer'], 
                               self.best_incorrect_feature['layer']]
        })
```

## Expected Results and Hypotheses

### Primary Hypotheses
1. **Correct Steering → Increased Test Attention**: When steering toward correctness, the model should pay MORE attention to test cases
2. **Incorrect Steering → Decreased Test Attention**: When steering toward incorrectness, the model should pay LESS attention to test cases  
3. **Problem Description Stability**: Attention to problem description should remain relatively stable across conditions

### Expected 3-Bin Distribution (Baseline)
- **Problem Description**: 40-50% of total attention
- **Test Cases**: 30-40% of total attention
- **Solution Marker**: 10-20% of total attention

### Expected Changes with Steering
- **Correct Steering**:
  - Test Cases: +5-10 percentage points (e.g., 35% → 42%)
  - Problem Description: -3-5 percentage points
  - Solution Marker: -2-3 percentage points

- **Incorrect Steering**:
  - Test Cases: -5-10 percentage points (e.g., 35% → 28%)
  - Problem Description: +2-4 percentage points
  - Solution Marker: +2-4 percentage points

### Head-Level Expectations
- **Some heads will specialize**: Certain heads may focus primarily on test cases
- **Steering sensitivity varies**: Not all heads will respond equally to steering
- **Layer matters**: The best PVA layer should show strongest effects

## Key Insights

1. **Raw Storage Flexibility**: Saving raw attention with boundaries allows multiple aggregation strategies in Phase 6.3
2. **Dynamic Boundary Handling**: Section boundaries are computed per-task to handle variable prompt lengths
3. **Statistical Power**: Multiple aggregation options provide robust statistics despite task variability
4. **Mechanistic Understanding**: Shows HOW steering changes model's information processing
5. **Actionable Results**: Directly reveals if model focuses more/less on tests when steered
6. **Future-Proof Design**: Raw storage enables new analyses without re-running expensive generation

## Implementation Checklist

### Phase 3.5 Modifications
- [ ] Add AttentionExtractor initialization in __init__
- [ ] Modify generate_temp0_with_activations to extract attention
- [ ] Implement `save_raw_attention_with_boundaries()` function
- [ ] Call save function with tokenized prompt and layer index
- [ ] Update checkpoint system to include raw attention data
- [ ] Add attention metadata to output files

### Phase 4.8 Modifications  
- [ ] Add AttentionExtractor for steered generation
- [ ] Capture attention during steered generation (baseline and steered)
- [ ] Implement `save_raw_attention_with_boundaries()` for steered attention
- [ ] Ensure consistent boundary detection with Phase 3.5
- [ ] Save raw attention tensor with boundaries
- [ ] Include attention in checkpointing

### Phase 6.3 Implementation
- [ ] Create AttentionAnalyzer class
- [ ] Load raw attention data from Phases 3.5 and 4.8
- [ ] Implement flexible aggregation methods:
  - [ ] 3-bin sum aggregation
  - [ ] 3-bin max aggregation
  - [ ] Length-normalized aggregation
- [ ] Implement 8 main visualizations:
  - [ ] Stacked bar chart for attention distribution
  - [ ] Head-level heatmap (8 heads × 9 conditions)
  - [ ] Delta plots showing steering effects
  - [ ] Statistical significance table
  - [ ] Attention transformation scatter plots (baseline vs steered)
  - [ ] Per-head transformation analysis (2×4 grid)
  - [ ] Head-specific attention change bar charts with error bars
  - [ ] Comparative head changes (3×2 grid, correct vs incorrect)
- [ ] Compute paired t-tests for each section
- [ ] Generate summary statistics and insights
- [ ] Export figures and results

### Common Utilities
- [ ] Create common_simplified/attention_hooks.py
- [ ] Implement AttentionExtractor class with one-time capture
- [ ] Add `save_raw_attention_with_boundaries()` function
- [ ] Add `calculate_section_boundaries()` function
- [ ] Phase 6.3 aggregation utilities (not in common)

## Critical Implementation Notes

### One-Time Capture Summary
- **Frequency**: ONE capture per task (not per token during generation!)
- **Position**: Final prompt token only (position -1)
- **Layers**: ONLY best PVA layers from Phase 2.5 (typically 1-2 layers, not all 26/42!)
- **Timing**: During first forward pass only (prompt processing)
- **NOT captured**: During autoregressive generation tokens

### Memory and Performance
- **No Batching**: Process each generation individually
- **Minimal Overhead**: Only capturing from 1-2 layers at one position
- **Storage Consideration**: Raw attention is ~8 × prompt_length floats per task (vs 24 floats with pre-aggregation)
- **Memory Efficient**: Clear attention cache immediately after saving
- **Checkpoint Integration**: Include attention data in existing checkpoint system
- **Storage Format**: Use compressed .npz files for efficient disk usage

### Why This Design?
1. **Efficiency**: Capturing during every token generation would be wasteful and slow
2. **Relevance**: The decision point is at the last prompt token before generation begins
3. **Focus**: We only care about the best PVA layers, not all layers
4. **Consistency**: Matches how Phase 3.5 already captures activations (once at position -1)