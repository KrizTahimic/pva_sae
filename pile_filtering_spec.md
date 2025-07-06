# Pile Filtering Technical Specification for PVA-SAE

## Executive Summary

1. **Dataset**: NeelNanda/pile-10k (10,000 diverse texts from The Pile - web pages, academic papers, code, books, etc.)
2. **Process**: Feed 10,000 texts through model → Extract internal activations (NOT generation) → SAE encoding → Feature analysis
3. **Model Runs**: Each of the 10,000 texts is run through the model ONCE, but only ONE position's activation is saved (the random word position)
4. **SAE**: Transforms d_model activations (e.g., 2048) into 16k sparse features using GemmaScope
5. **Threshold**: Features activating on ≥2% of these 10,000 random positions are filtered out as "general language features"
6. **Timing**: Filtering happens AFTER feature selection (post-argmax) in Phase 2.5 (current Phase 2)
7. **Output**: Up to 20 features per category (correct/incorrect code) that pass the filter
8. **Purpose**: Remove general language features, keep only Python validity-specific features
9. **Integration**: Phase 2.2 collects pile activations, Phase 2.5 (current Phase 2) applies filtering

## Pipeline Sequence (Integrated with PVA-SAE Phases)

```
Phase 1.0: Build dataset & cache Python code activations ✓ ALREADY IMPLEMENTED
   └─> Generate Python solutions → Extract activations → Save to data/phase1_0/activations/

Phase 2.2: Cache pile data activations ← NEW PHASE TO IMPLEMENT
   └─> Load 10,000 texts → Feed through model → Extract activations at random positions → Save as .npz

Phase 2.5: SAE analysis with pile filtering (current Phase 2) ← MODIFY EXISTING
   └─> Load Python activations → Encode through SAE → Compute separation scores
   └─> Load pile activations → Encode through SAE → Compute generic frequencies  
   └─> Filter features: Keep if < 2% pile activation → Max 20 per category

Phase 3: Validation ✓ ALREADY IMPLEMENTED
   └─> Use filtered features for steering experiments
```

**Key changes from original spec:**
- Uses correct/incorrect (Python validity) instead of known/unknown entities
- Integrates with existing phase structure
- Leverages already-cached activations from Phase 1
- Uses .npz format consistent with current codebase

## Understanding Pile Dataset Usage

### What is pile-10k?
The NeelNanda/pile-10k dataset contains the first 10,000 text samples from The Pile - a massive collection of diverse text sources:
- Web pages (Common Crawl)
- Academic papers (ArXiv, PubMed)
- Books (Project Gutenberg, Books3)
- Code (GitHub)
- Legal documents (FreeLaw)
- Q&A (StackExchange)
- Wikipedia articles
- Movie subtitles
- Chat logs
- And more...

### Key Point: Activation Extraction, NOT Text Generation
The variable name `prompts` in the code is misleading. These texts are NOT used to prompt the model for generation. Instead:
1. A random word is selected from each text
2. The text is truncated to 128 tokens (pre-filtering step)
3. If the random word survives truncation, the text is processed
4. The text is fed through the model ONCE (no generation)
5. Model activations are extracted at the position of the random word

### Example Process
```
Text #1: "The marine ecosystem consists of complex interactions between..."
Random word selected: "marine"
Pre-check: Truncate to 128 tokens → "marine" still present ✓
Process: Feed through model → Find "marine" at token position 2
Extract d_model activation vector at position 2
Save activation

Text #2: "def calculate_total(items):\n    return sum(item.price for item in items)"
Random word selected: "return"
Pre-check: Truncate to 128 tokens → "return" still present ✓
Process: Feed through model → Find "return" at token position 8
Extract d_model activation vector at position 8
Save activation

Text #3: "Long text with many tokens... [200+ tokens] ... important conclusion"
Random word selected: "conclusion"
Pre-check: Truncate to 128 tokens → "conclusion" NOT present ✗
Skip this text (random word was truncated)

... repeat for all texts, only processing those where random word survives truncation
```

### Why Random Words? The Key Insight
**The codebase processes all 10,000 texts BUT only saves 1 activation per text:**
- Each text is tokenized to 128 tokens
- A random word is selected from each text  
- Model processes the FULL text (all 128 tokens)
- But only the activation at the random word's position is saved
- Result: 10,000 activation vectors (not 1,280,000!)

### Why This Matters
By extracting activations from 10,000 random positions across diverse text types, they establish a baseline for "general language features". If an SAE feature activates frequently (>2%) across these random positions, it's likely capturing general language patterns rather than entity-specific information.

## Core Implementation

### 1. Load Pile Data
```python
def load_pile_data(tokens_to_cache: str) -> Tuple[List[str], List[str]]:
    # tokens_to_cache must be 'random' for pile
    prompts = load_dataset("NeelNanda/pile-10k")['train']['text']  # 10,000 diverse texts
    substrings = [random.choice(prompt.split()) for prompt in prompts]  # Random word from each text
    return prompts, substrings  # Returns: (10,000 texts, 10,000 random words)
```

### 2. Cache Model Activations (Critical Step!)
```python
# Feed texts through model to extract activations (NOT for generation!)
def cache_activations(model_base, prompts, substrings, compute_activation_fn, ...):
    # Process all 10,000 texts in batches
    for i in range(0, len(prompts), batch_size):  # len(prompts) = 10,000
        batch_prompts = prompts[i:i+batch_size]
        batch_substrings = substrings[i:i+batch_size]  # Random words
        
        # STEP 1: Pre-check which texts still contain the random word after truncation
        # This avoids processing texts where the random word gets cut off
        inputs_ = tokenizer(batch_prompts, truncation=True, max_length=128)
        filtered_prompts = []
        filtered_substrings = []
        for p in range(len(batch_prompts)):
            truncated_text = tokenizer.decode(inputs_.input_ids[p])
            if batch_substrings[p] in truncated_text:  # Random word survived truncation
                filtered_prompts.append(batch_prompts[p])
                filtered_substrings.append(batch_substrings[p])
        
        # STEP 2: Get activations for texts that passed the filter
        # compute_activation_fn internally calls _get_activations_fixed_seq_len which:
        # - Re-tokenizes with: tokenizer(prompts, truncation=True, max_length=128)
        # - Runs model forward pass
        # - Extracts activations using hooks
        input_ids, activations = compute_activation_fn(prompts=filtered_prompts)
        # activations shape: [batch, n_layers, seq_len, d_model]
        
        # STEP 3: Extract activation at the random word position
        for j, substring in enumerate(filtered_substrings):
            position = find_string_in_tokens(substring, input_ids[j], tokenizer)
            if position is not None:
                # Extract single position activation (n_positions=1 for pile)
                activation_at_position = activations[j, :, position, :]  # [n_layers, d_model]
                # Save to .npz file
            
    # Result: Up to 10,000 activation vectors (less if some random words were truncated)
    # Save format: data/phase2_2/pile_activations/{idx}_layer_{layer_idx}.npz
```

### 3. Compute Feature Statistics
```python
# For Python code data
def get_code_features(sae_acts, metric='absolute_difference'):
    # Compute activation frequencies on correct vs incorrect code
    freq_acts_correct = (sae_acts['correct'] > eps).float().mean(dim=0)
    freq_acts_incorrect = (sae_acts['incorrect'] > eps).float().mean(dim=0)
    
    # Compute separation scores
    scores_correct = freq_acts_correct - freq_acts_incorrect
    scores_incorrect = freq_acts_incorrect - freq_acts_correct
    
    return scores_dict, freq_acts_dict, mean_features_acts

# For pile data
def get_pile_features(sae_acts):
    # Compute activation frequencies on generic text
    freq_acts_generic = (sae_acts > eps).float().mean(dim=0)
    return freq_acts_generic
```

### 4. Load Cached Activations and Encode through SAE
```python
def get_features_layers(model_alias, acts_labels_dict, layers, sae_width, ...):
    for layer in layers:
        # Load SAE for this layer
        sae = load_sae(repo_id, sae_id)
        
        # Get cached activations (d_model dimensions)
        acts_labels = acts_labels_dict[layer]
        
        # Encode through SAE: d_model → 16k sparse features
        sae_acts = sae.encode(acts_labels['acts'])
        
        # Compute statistics on 16k features
        scores_dict, freq_acts_dict, mean_features_acts = get_features(sae_acts, ...)
```

### 5. The Filtering Algorithm (Critical Logic)
```python
def apply_pile_filter(sorted_features, pile_frequencies, threshold=0.02, max_features=20):
    # Process both 'correct' and 'incorrect' code features
    filtered_features = {'correct': {}, 'incorrect': {}}
    
    for code_category in ['correct', 'incorrect']:
        counter = 0
        
        # For each top code feature (already sorted by score)
        for feature_info in sorted_features[code_category]:
            layer = feature_info['layer']
            feat_idx = feature_info['feature_idx']
            
            # Get this feature's frequency on generic text
            generic_freq = pile_frequencies[layer][feat_idx]
            
            # THE KEY FILTER: Keep only if rarely activates on generic text
            if generic_freq < threshold:
                filtered_features[code_category].append(feature_info)
                counter += 1
                
            # Maximum features per category
            if counter >= max_features:
                break
    
    return filtered_features
```

## Implementation Checklist

### Phase 1.0: Cache Python Code Activations ✓ ALREADY IMPLEMENTED
- [x] Generate Python solutions for MBPP problems
- [x] Extract residual stream activations at final token position
- [x] Save to: `data/phase1_0/activations/{correct|incorrect}/{task_id}_layer_{n}.npz`

### Phase 2.2: Cache Pile Activations (NEW - TO IMPLEMENT)
- [ ] Load NeelNanda/pile-10k dataset (10,000 diverse texts)
- [ ] Set `max_length = 128` for pile processing (truncate texts)
- [ ] Support `--run_count` argument for development (default: 10000)
  - [ ] Development: `--run_count 3` for quick testing
  - [ ] Production: `--run_count 10000` or omit for default
- [ ] For each text (up to run_count):
  - [ ] Select a random word from the text
  - [ ] Feed text through model (forward pass only, NO generation)
  - [ ] Find position of random word in tokenized text
  - [ ] Extract residual stream activation at that specific position
- [ ] Total model runs: run_count (3 for dev, 10,000 for production)
- [ ] Save to: `data/phase2_2/pile_activations/{idx}_layer_{n}.npz`

### Phase 2.5: SAE Analysis with Pile Filtering (MODIFY EXISTING Phase 2)
#### Already Implemented:
- [x] Load cached Python code activations from Phase 1
- [x] Load GemmaScope SAE for each layer
- [x] Encode activations through SAE (d_model → 16k features)
- [x] Calculate separation scores (correct vs incorrect) for all 16k features
- [x] Sort features globally by separation score
- [x] Save top 20 features

#### To Add:
- [ ] Load cached pile activations from Phase 2.2
- [ ] Encode pile activations through SAE
- [ ] Compute activation frequencies on generic text for each of 16k features
- [ ] Apply pile filter to top features:
  - [ ] Check each top feature's frequency on generic text
  - [ ] Keep if frequency < 0.02 (2% of generic text samples)
  - [ ] Stop at 20 features per category (correct/incorrect)
- [ ] Save filtered results

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `pile_dataset` | "NeelNanda/pile-10k" | 10,000 diverse texts for baseline |
| `pile_texts_processed` | 10,000 | Number of model forward passes |
| `pile_threshold` | 0.02 (2%) | Features activating on >200/10,000 positions are filtered |
| `max_features` | 20 | Max features kept per category (correct/incorrect) |
| `seq_len` | 128 | Token sequence length for pile |
| `d_model` | Model-dependent (e.g., 2048) | Model's hidden dimension |
| `d_sae` | 16384 (16k) | SAE feature dimension |
| `scoring_method` | 'absolute_difference' | How to compute separation scores |

## Quick Integration Guide

```bash
# 1. Run Phase 1.0 if not already done (builds dataset and caches Python code activations)
python3 run.py phase 1

# 2. Run Phase 2.2 to cache pile activations (NEW)
# Development mode - run with just 3 samples for testing:
python3 run.py phase 2.2 --run_count 3

# Production mode - run with full 10k samples:
python3 run.py phase 2.2
# Or specify custom count:
python3 run.py phase 2.2 --run_count 1000

# Multi-GPU for production:
python3 multi_gpu_launcher.py --phase 2.2

# 3. Run Phase 2 with pile filtering (enabled by default)
python3 run.py phase 2
# Or explicitly disable pile filtering:
python3 run.py phase 2 --no-pile-filter

# 4. Continue with Phase 3 validation using filtered features
python3 run.py phase 3
```

### Implementation Code Structure

```python
# phase2_2_pile_setup/runner.py (NEW)
def run_phase2_2(config: Config, logger, device: str):
    """Cache pile dataset activations for filtering."""
    from datasets import load_dataset
    import random
    
    # Get run count from config (defaults to 10000)
    run_count = getattr(config, '_run_count', config.pile_samples)
    
    # Load pile-10k dataset
    dataset = load_dataset("NeelNanda/pile-10k")['train']
    texts = dataset['text'][:run_count]  # Use specified count
    
    # Pre-select random words from each text
    substrings = []
    for text in texts:
        words = text.split()
        if words:
            substrings.append(random.choice(words))
        else:
            substrings.append(None)  # Handle empty texts
    
    logger.info(f"Processing {run_count} pile samples (use --run_count to change)")
    
    # Process in batches
    batch_size = config.activation_batch_size
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_substrings = substrings[i:i+batch_size]
        
        # STEP 1: Pre-filter texts where random word survives truncation
        inputs_ = tokenizer(batch_texts, truncation=True, max_length=128)
        filtered_texts = []
        filtered_substrings = []
        filtered_indices = []
        
        for j, (text, substring) in enumerate(zip(batch_texts, batch_substrings)):
            if substring is None:
                continue
            truncated_text = tokenizer.decode(inputs_.input_ids[j])
            if substring in truncated_text:
                filtered_texts.append(text)
                filtered_substrings.append(substring)
                filtered_indices.append(i + j)
        
        if not filtered_texts:
            continue
            
        # STEP 2: Extract activations for filtered texts
        # This will internally tokenize with max_length=128
        input_ids, activations = extract_activations(
            model, filtered_texts, tokenizer, config.activation_layers
        )
        
        # STEP 3: Save activations at random word positions
        for j, (substring, idx) in enumerate(zip(filtered_substrings, filtered_indices)):
            position = find_string_in_tokens(substring, input_ids[j], tokenizer)
            if position is not None:
                # Save activation for each layer
                for layer_idx in config.activation_layers:
                    activation = activations[j, layer_idx, position, :]  # [d_model]
                    save_path = f"data/phase2_2/pile_activations/{idx}_layer_{layer_idx}.npz"
                    np.savez_compressed(save_path, activation=activation.cpu().numpy())

# phase2_simplified/sae_analyzer.py (modify existing)
def apply_pile_filter(self, top_features: Dict, pile_frequencies: Dict) -> Dict:
    """Apply pile filtering to remove general language features."""
    if not self.config.pile_filter_enabled:
        logger.info("Pile filtering disabled, returning original features")
        return top_features
        
    logger.info(f"Applying pile filter with threshold {self.config.pile_threshold}")
    filtered = {'correct': [], 'incorrect': []}
    
    for category in ['correct', 'incorrect']:
        for feature in top_features[category]:
            layer = feature['layer']
            feat_idx = feature['feature_idx']
            
            # Check pile frequency
            if pile_frequencies[layer][feat_idx] < self.config.pile_threshold:
                filtered[category].append(feature)
                
            if len(filtered[category]) >= 20:
                break
                
    logger.info(f"Filtered to {len(filtered['correct'])} correct, "
                f"{len(filtered['incorrect'])} incorrect features")
    return filtered
```

## Data Structure

### Pile Frequencies Storage
```python
# Structure: pile_frequencies[layer][feature_idx] = frequency
pile_frequencies = {
    0: [0.001, 0.045, 0.002, ...],  # 16k frequencies for layer 0
    1: [0.003, 0.001, 0.089, ...],  # 16k frequencies for layer 1
    ...
}
```

### Code Features Storage (Current Phase 2 Format)
```python
# Structure: top_features[category] = list of feature dicts
top_features = {
    'correct': [
        {'layer': 5, 'feature_idx': 1234, 'separation_score': 0.85, 'f_correct': 0.76, ...},
        {'layer': 3, 'feature_idx': 5678, 'separation_score': 0.82, 'f_correct': 0.71, ...},
        ...
    ],
    'incorrect': [
        {'layer': 7, 'feature_idx': 9012, 'separation_score': 0.79, 'f_incorrect': 0.68, ...},
        ...
    ]
}
```

### Abstraction for Clean Implementation
```python
# Clean abstraction that hides implementation details
class PileFrequencyLoader:
    def __init__(self, model_alias, layers):
        self.model_alias = model_alias
        self.layers = layers
        
    def load_frequencies(self):
        """Load generic text activation frequencies for all features."""
        frequencies = {}
        for layer in self.layers:
            layer_data = self._load_layer_data(layer)
            frequencies[layer] = self._extract_frequencies(layer_data)
        return frequencies
    
    def _extract_frequencies(self, layer_data):
        """Extract feature frequencies from saved data."""
        # Implementation handles the data format details
        return [feat['frequency'] for feat in layer_data['features']]
```

## Notes

- **Critical Understanding**: The pile texts are fed through the model to extract activations, NOT to generate text
- **Two-stage tokenization**: First check if random word survives truncation, then only process texts where it does
- Each text contributes one activation vector from a random position - may be <10,000 if some random words get truncated
- **Development Mode**: Use `--run_count 3` for quick testing during development
- **Production Mode**: Use full 10,000 samples (default) for accurate filtering
- Pile filtering ensures Python validity-specific features aren't just general language patterns
- The 2% threshold means: if a feature activates on >200 of the 10,000 random positions (or >2% of run_count), it's too generic
- Processing order matters: always filter AFTER scoring and sorting in Phase 2
- All 16k SAE features are computed first, then filtered down to max 20 per category
- Activation caching is crucial for efficiency - avoids re-running model inference multiple times
- GemmaScope SAEs transform dense d_model activations into sparse 16k features for interpretability
- The diverse nature of pile texts (web, code, academic, books) ensures comprehensive coverage of general language
- Pile filtering is enabled by default in Phase 2 but can be disabled with `--no-pile-filter`