# SAE Analysis Implementation Specification

## Executive Summary

1. **Goal**: Identify SAE features that distinguish between known and unknown entities
2. **Feature Pool**: All layers × 16,384 features per layer (e.g., 25 × 16,384 = 409,600 total features)
3. **Scoring**: Each individual feature gets a separation score (not per layer)
4. **Selection**: Top 20 features selected globally across ALL layers
5. **Key Insight**: A "top feature" could be from any layer - selection is global, not per-layer
6. **Output**: Ranked list of features with their layer, index, and discrimination scores

## Pipeline Sequence

```
1. Cache entity activations
   └─> Load prompts → Run model → Extract residual stream (d_model) at entity positions → Save

2. Compute SAE features for each layer
   └─> For layers 1-25: Load SAE → Encode activations (d_model → 16k features) → Save

3. Calculate separation scores PER FEATURE
   └─> For each of 16k features per layer:
       - Compute activation frequency on known entities
       - Compute activation frequency on unknown entities  
       - Calculate separation score (e.g., freq_known - freq_unknown)

4. Global feature ranking
   └─> Collect all 409,600 scores → Sort globally → Take top 20

5. Optional: Apply pile filtering
   └─> Check top features against generic text → Keep if < 2% activation
```

**Critical insight**: Features compete globally, not within layers.

## Core Implementation

### 1. Multi-Layer Feature Collection

```python
def get_features_layers(model_alias, acts_labels_dict, layers, sae_width, repo_id, save=True, **kwargs):
    """
    Process multiple layers and compute features for each.
    
    Args:
        layers: List of layer indices to process (e.g., [1,2,3,...,25])
        acts_labels_dict: Cached activations for each layer
    
    Returns:
        Dict with features for all layers: {layer: {known/unknown: feature_data}}
    """
    feats_per_layer = {}
    
    for layer in layers:  # Process EACH layer
        # Load layer-specific SAE
        sae = load_sae(repo_id, get_sae_id(layer))
        
        # Get cached activations for this layer
        acts_labels = acts_labels_dict[layer]
        
        # Encode: residual stream → 16k sparse features
        sae_acts = sae.encode(acts_labels['acts'])  # Shape: [n_samples, 16384]
        
        # Split by known/unknown
        known_acts = sae_acts[acts_labels['labels'] == 0]
        unknown_acts = sae_acts[acts_labels['labels'] == 1]
        
        # Compute per-feature scores (16,384 scores for this layer)
        scores_dict, freq_acts_dict, mean_acts = get_features(
            {'known': known_acts, 'unknown': unknown_acts},
            metric='absolute_difference'
        )
        
        feats_per_layer[layer] = (scores_dict, freq_acts_dict, mean_acts)
    
    return format_layer_features(feats_per_layer, save=save)
```

### 2. Per-Feature Scoring Computation

```python
def get_features(sae_acts, metric='absolute_difference', eps=1e-6):
    """
    Compute separation scores for EACH of the 16,384 features.
    
    Args:
        sae_acts: Dict with 'known' and 'unknown' activations
                  Each is shape [n_samples, 16384]
    
    Returns:
        scores_dict: Separation score for each feature
        freq_acts_dict: Activation frequencies
        mean_features_acts: Mean activation values
    """
    # Compute activation frequency for EACH feature
    # freq_acts shape: [16384] - one value per feature
    freq_acts_known = (sae_acts['known'] > eps).float().mean(dim=0)
    freq_acts_unknown = (sae_acts['unknown'] > eps).float().mean(dim=0)
    
    # Compute separation score for EACH feature
    if metric == 'absolute_difference':
        # Simple difference in activation rates
        scores_known = freq_acts_known - freq_acts_unknown  # Shape: [16384]
        scores_unknown = freq_acts_unknown - freq_acts_known  # Shape: [16384]
    
    elif metric == 'relative_difference':
        # Normalized by baseline frequency
        scores_known = (freq_acts_known - freq_acts_unknown) / (freq_acts_unknown + eps)
        scores_unknown = (freq_acts_unknown - freq_acts_known) / (freq_acts_known + eps)
    
    # Return scores for all 16,384 features
    return {
        'known': scores_known.tolist(),
        'unknown': scores_unknown.tolist()
    }, {
        'known': (freq_acts_known.tolist(), freq_acts_unknown.tolist()),
        'unknown': (freq_acts_unknown.tolist(), freq_acts_known.tolist())
    }, {
        'known': sae_acts['known'].mean(0).tolist(),
        'unknown': sae_acts['unknown'].mean(0).tolist()
    }
```

### 3. Global Feature Selection Across Layers

```python
def get_top_k_features(feats_layers, k=20):
    """
    Select top k features GLOBALLY across ALL layers.
    
    Args:
        feats_layers: Dict mapping layer → feature data
        k: Number of top features to select
    
    Returns:
        Top k features with their layer info
    """
    # Aggregate features from ALL layers
    all_features = {'known': [], 'unknown': []}
    
    for layer in feats_layers.keys():
        layer_data = feats_layers[layer]
        
        # Add ALL 16,384 features from this layer to global pool
        for feature_idx in range(16384):
            for category in ['known', 'unknown']:
                feature_info = {
                    'layer': layer,
                    'feature_idx': feature_idx,
                    'score': layer_data[category][feature_idx]['score'],
                    'freq_known': layer_data[category][feature_idx]['freq_acts_known'],
                    'freq_unknown': layer_data[category][feature_idx]['freq_acts_unknown'],
                }
                all_features[category].append(feature_info)
    
    # Sort ALL features globally by score
    top_features = {}
    for category in ['known', 'unknown']:
        # Sort all ~409,600 features by score
        sorted_features = sorted(all_features[category], 
                               key=lambda x: x['score'], 
                               reverse=True)
        
        # Take top k
        top_features[category] = sorted_features[:k]
    
    return top_features
```

### 4. Feature ID Convention

```python
def format_feature_id(layer, feature_idx):
    """
    Create standard feature identifier.
    
    Examples:
        Layer 12, Feature 5432 → "L12F5432"
        Layer 3, Feature 891 → "L3F891"
    """
    return f"L{layer}F{feature_idx}"

def parse_feature_id(feature_id):
    """
    Parse feature identifier back to components.
    
    Example:
        "L12F5432" → (12, 5432)
    """
    layer = int(feature_id[1:feature_id.find('F')])
    feature_idx = int(feature_id[feature_id.find('F')+1:])
    return layer, feature_idx
```

## Data Flow Example

### Input: Model with 25 layers, 16k SAE width

```
Total feature pool:
- Layer 1: Features 0-16383 (16,384 features)
- Layer 2: Features 0-16383 (16,384 features)
- ...
- Layer 25: Features 0-16383 (16,384 features)
- TOTAL: 409,600 features
```

### Processing:

```
1. Each feature gets a separation score:
   - L1F0: score = 0.02
   - L1F1: score = 0.15
   - ...
   - L12F5432: score = 0.89 (high score!)
   - ...
   - L25F16383: score = 0.03

2. Global sorting of all 409,600 scores

3. Top 20 might be:
   1. L12F5432 (score: 0.89)
   2. L12F891 (score: 0.87)
   3. L8F3421 (score: 0.85)
   4. L19F7823 (score: 0.84)
   5. L12F1092 (score: 0.83)
   ... (notice multiple from L12 if it has good features)
```

## Implementation Checklist

### Phase 1: Setup
- [ ] Identify layers with available SAEs (e.g., 1-25 for Gemma-2-2b)
- [ ] Load model configuration (d_model, n_layers)
- [ ] Prepare entity data (known vs unknown labels)

### Phase 2: Activation Collection
- [ ] Cache activations at entity token positions
- [ ] Ensure activations saved for ALL specified layers
- [ ] Verify shape: [n_samples, n_layers, d_model]

### Phase 3: Per-Layer SAE Processing
- [ ] For each layer:
  - [ ] Load layer-specific SAE
  - [ ] Encode activations (d_model → 16,384 features)
  - [ ] Split by known/unknown entities
  - [ ] Compute 16,384 separation scores
  - [ ] Save all scores (not just top ones)

### Phase 4: Global Feature Selection
- [ ] Aggregate features from all layers (~409,600 total)
- [ ] Sort globally by separation score
- [ ] Select top 20 features
- [ ] Record which layer each top feature comes from

### Phase 5: Analysis
- [ ] Analyze layer distribution of top features
- [ ] Check if certain layers dominate
- [ ] Validate scores make sense

## Key Implementation Details

### Memory Efficiency
```python
# Don't load all features into memory at once
def get_top_k_features_efficient(layers, k=20):
    import heapq
    
    # Use heap for efficient top-k selection
    top_known = []
    top_unknown = []
    
    for layer in layers:
        # Load one layer at a time
        layer_features = load_layer_features(layer)
        
        for feat_idx, feat_data in enumerate(layer_features['known']):
            score = feat_data['score']
            item = (-score, layer, feat_idx)  # Negative for max-heap
            
            if len(top_known) < k:
                heapq.heappush(top_known, item)
            elif -score > top_known[0][0]:
                heapq.heapreplace(top_known, item)
    
    # Extract final top k
    return sorted(top_known, key=lambda x: x[0])
```

### Parallel Processing
```python
# Process multiple layers in parallel
from concurrent.futures import ProcessPoolExecutor

def process_layers_parallel(layers, acts_labels_dict):
    with ProcessPoolExecutor() as executor:
        futures = []
        for layer in layers:
            future = executor.submit(process_single_layer, 
                                   layer, 
                                   acts_labels_dict[layer])
            futures.append((layer, future))
        
        results = {}
        for layer, future in futures:
            results[layer] = future.result()
    
    return results
```

## Common Pitfalls

1. **Assuming top features come from one layer**
   - Wrong: "Get top 20 features from best layer"
   - Right: "Get top 20 features across all layers"

2. **Computing layer-level scores**
   - Wrong: "Average all features in layer for layer score"
   - Right: "Each feature has individual score"

3. **Per-layer selection**
   - Wrong: "Take top feature from each layer"
   - Right: "Take top features globally (might be 5 from one layer)"

## Quick Integration Example

```python
# 1. Setup
layers = list(range(1, 26))  # Layers 1-25
model_alias = 'gemma-2-2b'

# 2. Load cached activations
acts_labels_dict = load_cached_activations(model_alias, layers)

# 3. Compute features for all layers
all_features = get_features_layers(
    model_alias=model_alias,
    acts_labels_dict=acts_labels_dict,
    layers=layers,
    sae_width='16k',
    repo_id='google/gemma-scope-2b-pt-res'
)

# 4. Global feature selection
top_features = get_top_k_features(all_features, k=20)

# 5. Analyze results
print(f"Top feature: {top_features['known'][0]}")
# Output: {'layer': 12, 'feature_idx': 5432, 'score': 0.89, ...}

# 6. For steering: load specific feature
layer, feat_idx = 12, 5432
sae = load_sae(repo_id, get_sae_id(layer))
steering_direction = sae.W_dec[feat_idx]  # Shape: [d_model]
```

## Data Structures

### Layer Features Format
```json
{
  "layer_12": {
    "known": {
      "0": {"score": 0.15, "freq_acts_known": 0.18, "freq_acts_unknown": 0.03},
      "1": {"score": 0.02, "freq_acts_known": 0.05, "freq_acts_unknown": 0.03},
      ...
      "16383": {"score": 0.41, "freq_acts_known": 0.52, "freq_acts_unknown": 0.11}
    },
    "unknown": {...}
  },
  "layer_13": {...}
}
```

### Global Top Features Format
```json
{
  "known": [
    {"layer": 12, "feature_idx": 5432, "score": 0.89, "id": "L12F5432"},
    {"layer": 12, "feature_idx": 891, "score": 0.87, "id": "L12F891"},
    {"layer": 8, "feature_idx": 3421, "score": 0.85, "id": "L8F3421"},
    ...
  ],
  "unknown": [...]
}
```

## Notes

- Global selection ensures we get the absolute best features regardless of layer
- Layer distribution of top features can reveal which layers encode entity knowledge
- Separation scores measure discriminative power, not just activation frequency
- All 16,384 features per layer are scored before selection
- Implementation should handle ~400k features efficiently