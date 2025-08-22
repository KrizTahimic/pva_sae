# Attention Analysis Technical Specification

## Executive Summary

1. **Purpose**: Analyze how attention patterns change when steering model behavior with SAE features
2. **Method**: Compare attention scores to entity tokens before and after applying SAE steering
3. **Attention Types**: Standard attention weights and value-weighted attention patterns
4. **Target Position**: Entity last token position for steering interventions
5. **Evaluation**: Statistical comparison of attention differences with significance testing
6. **Dataset**: WikiData test split across 4 entity types (player, movie, song, city)
7. **Sample Sizes**: N=100 for quantitative evaluation, batch_size=16 for processing

## Pipeline Sequence

```
1. Load attention analysis components
   └─> Initialize model → Set attention result caching → Load tokenizer

2. Prepare entity data
   └─> Load WikiData queries → Tokenize prompts → Identify entity positions

3. Compute baseline attention
   └─> Run model with cache → Extract attention patterns → Aggregate to entity tokens

4. Apply steering and compute attention
   └─> Add steering hooks → Run with cache → Extract steered patterns → Compare to baseline

5. Statistical analysis
   └─> Compute attention differences → Test against random baselines → Calculate significance

6. Visualization
   └─> Generate scatter plots → Create boxplots → Export attention heatmaps
```

## Understanding Attention Analysis

### Attention Pattern Types
The system supports multiple attention pattern extraction methods:
- **Standard Attention Weights**: Direct attention scores from attention matrices
- **Value-Weighted Patterns**: Attention weighted by value vectors
- **Output-Value-Weighted**: Full OV circuit decomposition
- **Distance-Based**: Similarity metrics in representation space

### Attention Aggregation
Attention to entities is computed by:
1. Extracting attention patterns for each layer/head
2. Selecting entity token positions
3. Aggregating attention from final position to entity tokens
4. Computing mean attention scores across batches

## Core Implementation

### 1. Baseline Attention Computation

```python
# mech_interp/attn_analysis.py:47-69
def compute_attn_original(model, N, tokenized_prompts, pos_entities, 
                         pos_type='all', type_pattern='attn_weights', batch_size=4):
    """
    Compute attention patterns without any steering intervention.
    
    Args:
        N: Number of samples to process
        pos_entities: Entity token positions for each prompt
        pos_type: Which positions to analyze ('entity_last', 'entity', 'all', etc.)
        type_pattern: 'attn_weights' or 'value_weighted'
    
    Returns:
        Tensor of shape [N, n_layers, n_heads] with attention scores
    """
    mean_orig_attn_list = []
    for i in range(0, len(tokenized_prompts), batch_size):
        batch_pos = get_batch_pos(batch_entity_pos, pos_type, batch_tokenized_prompts)
        
        original_logits, original_cache = model.run_with_cache(batch_tokenized_prompts)
        clean_mean_attn = get_attn_to_entity_tok_mean(
            model, original_cache, batch_pos, model.cfg.n_layers, type_pattern
        )
        mean_orig_attn_list.append(clean_mean_attn)
    
    return torch.cat(mean_orig_attn_list, 0)
```

### 2. Steered Attention Computation

```python
# mech_interp/attn_analysis.py:72-103
def compute_attn_steered(model, N, tokenized_prompts, pos_entities, pos_type='all',
                        steering_latents=None, ablate_latents=None, 
                        coeff_value=Union[Literal['norm', 'mean'], int],
                        type_pattern='attn_weights', batch_size=4):
    """
    Compute attention patterns with SAE steering applied.
    
    Args:
        steering_latents: List of (layer, latent_idx, mean_act, direction) tuples
        coeff_value: Steering strength (100 for Gemma, 20 for Llama)
        
    Implementation uses cache_steered_latents from hooks_utils.py
    """
    mean_steered_attn_list = []
    for i in range(0, len(tokenized_prompts), batch_size):
        batch_pos = get_batch_pos(batch_entity_pos, pos_type, batch_tokenized_prompts)
        
        model.reset_hooks()
        # Apply steering via hooks (hooks_utils.py:109-134)
        steered_logits, steered_cache = cache_steered_latents(
            model, batch_tokenized_prompts, pos=batch_pos,
            steering_latents=steering_latents,
            ablate_latents=ablate_latents,
            coeff_value=coeff_value
        )
        
        steered_mean_attn = get_attn_to_entity_tok_mean(
            model, steered_cache, batch_pos, model.cfg.n_layers, type_pattern
        )
        mean_steered_attn_list.append(steered_mean_attn)
    
    return torch.cat(mean_steered_attn_list, 0)
```

### 3. Attention Pattern Extraction

```python
# mech_interp/mech_interp_utils.py:335-367
def get_attn_to_entity_tok_mean(model, cache, entity_token_pos, n_layers, 
                                type_pattern='attn_weights'):
    """
    Extract attention scores from last token to entity tokens.
    
    Processing steps:
    1. For each layer, get attention patterns
    2. Select attention from last position (-1) to all positions
    3. Sum attention to entity token positions
    4. Stack and rearrange to [batch, layers, heads]
    """
    attn_to_entity_tok_list = []
    for layer in range(n_layers):
        if type_pattern == 'attn_weights':
            attn_patterns = cache["pattern", layer]
            attn_to_entity_toks = attn_patterns[:,:,-1]  # Last position attention
            
        elif type_pattern == 'value_weighted':
            # Get value-weighted patterns (mech_interp_utils.py:1341-1358)
            weighted_values = get_value_weighted_patterns(model, cache, layer)
            raw_inter_token_attribution = torch.norm(weighted_values, dim=-1, p=2)
            attn_patterns = einops.rearrange(
                raw_inter_token_attribution, 
                'batch query_pos key_pos head_index -> batch head_index query_pos key_pos'
            )
            attn_to_entity_toks = attn_patterns[:,:,-1]
        
        # Aggregate attention to entity positions
        attn_to_entity_toks_sum = torch.zeros(attn_to_entity_toks.shape[0], attn_patterns.shape[1])
        for batch_idx in range(attn_patterns.shape[0]):
            attn_to_entity_toks_sum[batch_idx] = attn_to_entity_toks[
                batch_idx, :, entity_token_pos[batch_idx]
            ].sum(dim=-1)
        
        attn_to_entity_tok_list.append(attn_to_entity_toks_sum)
    
    attn_to_entity_tok_stack = torch.stack(attn_to_entity_tok_list, 0)
    # Rearrange from [n_layers, batch, n_heads] to [batch, n_layers, n_heads]
    return einops.rearrange(attn_to_entity_tok_stack, 'layers batch heads -> batch layers heads').cpu()
```

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

### 7. Attention Difference Computation

```python
# mech_interp/attn_analysis.py:265-305
def compute_attn_original_vs_steered(known_label, steering_latent, pos_type, 
                                     type_pattern, batch_size,
                                     tokenized_prompts_dict_entity_type, 
                                     pos_entities_dict_entity_type):
    """
    Compare attention patterns between original and steered models.
    
    Process:
    1. Compute attention for each entity type
    2. Apply steering with appropriate coefficient
    3. Calculate differences: steered - original
    4. Return difference dictionary by entity type
    """
    mean_attn_dict = {'Original': {}, 'Steered': {}}
    
    for entity_type in ['player', 'movie', 'song', 'city']:
        tokenized_prompts = tokenized_prompts_dict[known_label]
        pos_entities = pos_entities_dict[known_label]
        
        # Original attention
        orig_attn = compute_attn_original(
            model, N, tokenized_prompts, pos_entities, 
            pos_type=pos_type, type_pattern=type_pattern, batch_size=batch_size
        )
        
        # Steered attention
        steered_attn = compute_attn_steered(
            model, N, tokenized_prompts, pos_entities, pos_type=pos_type,
            steering_latents=[steering_latent], ablate_latents=None,
            coeff_value=coeff_value, type_pattern=type_pattern,
            batch_size=batch_size
        )
        
        mean_attn_dict['Original'][entity_type] = orig_attn
        mean_attn_dict['Steered'][entity_type] = steered_attn
    
    # Compute differences
    difference_dict = {}
    for entity_type in ['player', 'movie', 'song', 'city']:
        difference_dict[entity_type] = (
            mean_attn_dict['Steered'][entity_type] - 
            mean_attn_dict['Original'][entity_type]
        )
    
    return difference_dict
```

## Experimental Configuration

### Main Analysis Loop
```python
# mech_interp/attn_analysis.py:328-446
# Configuration parameters
batch_size = 16
N = 100
type_pattern = 'attn_weights'  # or 'value_weighted'
pos_type = 'entity_last'       # Position to steer
all_heads = True               # Analyze all heads or specific subset
split = 'test'
random_n_latents = 10          # Random baseline comparisons
top_latents = [0, 1, 2]        # Top 3 SAE latents

# Model-specific coefficients
coeff_values = {
    'known': 100 if 'gemma' in model_alias else 20,
    'unknown': 100 if 'gemma' in model_alias else 20
}

# Head selection for different models
if model_alias == 'gemma-2-2b':
    heads_dict[known_label] = [[15,5], [18,5], [20,3], [25,4]]
else:  # gemma-2-9b or llama
    heads_dict[known_label] = [[25,2], [26,2], [29,14], [33,7], [37,12], [39,7]]
```

### Dataset Loading
```python
# mech_interp/attn_analysis.py:105-129
def load_data(model_alias, entity_type, tokenizer, split='test'):
    """
    Load WikiData queries and prepare tokenized prompts.
    
    Template: 'The {entity_type} {entity_name} {relation}'
    
    Returns:
        tokenized_prompts_dict: Dict with 'known'/'unknown' keys
        pos_entities_dict: Entity positions for each prompt
        entity_token_pos: Fixed position for entity type
    """
    prompt_template = 'The {entity_type} {entity_name} {relation}'
    entity_token_pos = entity_type_to_token_pos[entity_type]
    queries = load_wikidata_queries(model_alias)
    
    valid_entities = get_valid_entities(
        queries, tokenizer, entity_type, 
        entity_token_pos, split=split, fixed_length=False
    )
    
    # Process known and unknown entities
    for known_label in ['known', 'unknown']:
        prompts, _, entities = create_prompts_and_answers(
            tokenizer, queries, entity_type, known_label, 
            valid_entities, prompt_template, relations_model_type='base'
        )
        tokenized_prompts_dict[known_label] = model.to_tokens(prompts)
        # Find entity positions in tokenized prompts
        pos_entities_dict[known_label] = find_entity_positions(tokenized_prompts, entities)
```

### Statistical Analysis Against Random Baseline
```python
# mech_interp/attn_analysis.py:461-483
def compute_significance_test_against_random(random_results, top_latent_results, known_label):
    """
    Test if top latent steering produces significantly different
    attention changes compared to random latent steering.
    
    Process:
    1. For each random latent result
    2. Flatten all head attention differences
    3. Compare distributions with Welch's t-test
    4. Count significant results
    """
    significant_count = 0
    for random_idx in range(len(random_results)):
        all_random_latents_list = []
        all_top_latent_list = []
        
        for head in top_latent_results[()].keys():
            all_random_latents_list.append(random_results[random_idx][head])
            all_top_latent_list.append(top_latent_results[()][head])
        
        all_random_latents = np.array(all_random_latents_list).flatten()
        all_top_latent = np.array(all_top_latent_list).flatten()
        
        # Direction of comparison depends on known_label
        if known_label == 'known':
            mean_1 = all_random_latents
            mean_2 = all_top_latent
        else:
            mean_1 = all_top_latent
            mean_2 = all_random_latents
        
        # Test if mean_1 > mean_2
        stats_dict = compare_means(mean_1, mean_2, alpha=0.05)
        significant_count += stats_dict['significant'] == True
    
    return significant_count
```

## Visualization Components

### Attention Pattern Visualization
```python
# mech_interp/mech_interp_utils.py:1360-1456
def visualize_attention_patterns(model, type_pattern, heads, local_cache, 
                                 title="", max_width=700, html=True):
    """
    Create visualization of attention patterns for specified heads.
    
    Supports multiple pattern types:
    - 'attn_weights': Standard attention scores
    - 'value_weighted': ||a_ij * v_j|| norms
    - 'output_value_weighted': ||a_ij * v_j * W_O|| norms
    - 'distance_based': Distance metrics in output space
    
    Returns tensor of shape [batch, n_heads, dest_pos, src_pos]
    """
```

### Statistical Plots
```python
# mech_interp/attn_analysis.py:131-187
def plot_head_differences(values, heads, width=20):
    """
    Create bar plot showing attention score differences with error bars.
    
    Args:
        values: Tensor of shape [n_samples, n_layers, n_heads]
        heads: List of (layer, head) tuples to plot
    
    Shows mean difference and standard deviation for each head.
    """
```

### Scatter Plot Comparison
```python
# mech_interp/mech_interp_utils.py:1045-1154
def plot_heads_scatter_entity_types(attn_heads_A, attn_heads_B, 
                                    A_label, B_label, heads, 
                                    head_colors, title):
    """
    Create scatter plot comparing attention scores between conditions.
    
    Features:
    - Different colors for different heads
    - Different shapes for entity types
    - Diagonal reference line
    - Separate legend for heads and entity types
    """
```

## Implementation Workflow

### Phase 1: Setup and Configuration
```python
# mech_interp/attn_analysis.py:308-324
model_alias = 'gemma-2-2b'
model = HookedTransformer.from_pretrained_no_processing(model_alias)
model.set_use_attn_result(True)  # Enable attention result caching
tokenizer = model.tokenizer
tokenizer.padding_side = 'left'

# Load data for all entity types
tokenized_prompts_dict_entity_type = {}
pos_entities_dict_entity_type = {}
for entity_type in ['player', 'movie', 'song', 'city']:
    tokenized_prompts, pos_entities, _ = load_data(
        model_alias, entity_type, tokenizer
    )
    tokenized_prompts_dict_entity_type[entity_type] = tokenized_prompts
    pos_entities_dict_entity_type[entity_type] = pos_entities
```

### Phase 2: Load Steering Features
```python
# mech_interp/attn_analysis.py:344-356
# Load SAE latents for steering
for filter_with_pile in [True]:
    for idx in top_latents:
        top_latents = {'known': idx, 'unknown': idx}
        
        known_latent, unknown_latent, random_latents_known, random_latents_unknown = load_latents(
            model_alias, top_latents,
            random_n_latents=random_n_latents,
            filter_with_pile=filter_with_pile
        )
        
        # Determine starting layers for analysis
        layer_start = {
            'known': unknown_latent[0][0],
            'unknown': known_latent[0][0]
        }
```

### Phase 3: Compute Attention Differences
```python
# mech_interp/attn_analysis.py:376-416
for known_label in ['known', 'unknown']:
    # Select steering direction (opposite of label)
    coeff_value = coeff_values['unknown'] if known_label == 'known' else coeff_values['known']
    steering_latent = unknown_latent[0] if known_label == 'known' else known_latent[0]
    
    # Compute attention differences for all entity types
    difference_dict = compute_attn_original_vs_steered(
        known_label, steering_latent, pos_type, type_pattern,
        batch_size, tokenized_prompts_dict_entity_type, 
        pos_entities_dict_entity_type
    )
    
    # Concatenate results across entity types
    all_values = []
    for entity_type in difference_dict:
        values = difference_dict[entity_type]  # Shape: [n, n_layers, n_heads]
        all_values.append(values)
    concatenated = np.concatenate(all_values, axis=0)
    
    # Save results
    save_path = f'./attn_steering_values/{model_alias}_coeff_{coeff_value}' \
                f'from_{known_label}_{top_latent_idx}_{pos_type}_' \
                f'pile_filtering_{filter_with_pile}.npy'
    np.save(save_path, data_save)
```

### Phase 4: Random Baseline Comparison
```python
# mech_interp/attn_analysis.py:419-446
# Compute attention differences with random steering
for known_label in ['known', 'unknown']:
    list_difference_dict = []
    for random_latent_idx in range(random_n_latents):
        coeff_value = coeff_values['unknown'] if known_label == 'known' else coeff_values['known']
        steering_latent = random_latents_unknown[random_latent_idx] if known_label == 'known' \
                         else random_latents_known[random_latent_idx]
        
        difference_dict = compute_attn_original_vs_steered(
            known_label, steering_latent, pos_type, type_pattern, 
            batch_size, tokenized_prompts_dict_entity_type, 
            pos_entities_dict_entity_type
        )
        
        # Process and store results
        all_values = []
        for entity_type in difference_dict:
            values = difference_dict[entity_type]
            all_values.append(values)
        concatenated = np.concatenate(all_values, axis=0)
        
        data_save = {}
        for layer, head in heads_dict[known_label]:
            values_head = concatenated[:, layer, head]
            data_save[f'L{layer}H{head}'] = values_head
        list_difference_dict.append(data_save)
    
    # Save random baseline results
    save_path = f'./attn_steering_values/{model_alias}_random_coeff_{coeff_value}' \
                f'from_{known_label}_{top_latent_idx}_{pos_type}_' \
                f'pile_filtering_{filter_with_pile}.npy'
    np.save(save_path, list_difference_dict)
```

### Phase 5: Statistical Testing
```python
# mech_interp/attn_analysis.py:486-504
# Run significance tests
significant_count_dict = {}
for model_alias in ['gemma-2-2b']:
    significant_count_dict[model_alias] = {}
    for known_label in ['known', 'unknown']:
        significant_count_dict[model_alias][known_label] = {}
        for top_latent_idx in range(3):  # Top 3 latents
            coeff_value = 100 if 'gemma' in model_alias else 20
            
            # Load results
            random_results, top_latent_results = load_attn_diff_results(
                model_alias, coeff_value, known_label, 
                top_latent_idx, pos_type, filter_with_pile
            )
            
            # Test significance
            significant_count = compute_significance_test_against_random(
                random_results, top_latent_results, known_label
            )
            
            significant_count_dict[model_alias][known_label][filter_with_pile][top_latent_idx] = significant_count
            print(f'Significant count {model_alias} {known_label} {filter_with_pile}: {significant_count}')
```

## Key Insights

1. **Attention Focus Changes**: Steering with SAE features significantly alters attention patterns to entity tokens
2. **Head Specificity**: Certain attention heads (e.g., L20H3, L18H5 for Gemma-2-2b) show stronger responses to steering
3. **Asymmetric Effects**: Known→unknown steering may produce different attention changes than unknown→known
4. **Statistical Validation**: Random latent steering provides baseline for significance testing
5. **Entity Type Consistency**: Attention changes are consistent across different entity types
6. **Position Importance**: Steering at entity_last position is sufficient to affect attention patterns

## Implementation Checklist

### Setup
- [ ] Initialize model with attention result caching enabled
- [ ] Load WikiData queries for all entity types
- [ ] Configure steering coefficients (100 for Gemma, 20 for Llama)

### Data Preparation
- [ ] Tokenize prompts for known/unknown entities
- [ ] Identify entity token positions
- [ ] Prepare batch processing structures

### Attention Analysis
- [ ] Compute baseline attention patterns
- [ ] Apply SAE steering with appropriate coefficients
- [ ] Extract steered attention patterns
- [ ] Calculate attention differences

### Statistical Validation
- [ ] Generate random latent baselines
- [ ] Perform significance testing
- [ ] Document p-values and effect sizes

### Visualization
- [ ] Create attention difference bar plots
- [ ] Generate scatter plots for known vs unknown
- [ ] Export attention heatmaps
- [ ] Save numerical results for further analysis

## Notes

- **Batch Processing**: Use batch_size=16 for memory efficiency
- **Cache Management**: Clear caches after each batch to prevent OOM
- **Position Selection**: entity_last is most commonly used and effective
- **Coefficient Values**: Model-specific, requires manual tuning
- **Statistical Power**: N=100 provides sufficient power for significance testing
- **Attention Types**: Standard attention weights are primary metric, value-weighted provides additional insights