# SAE Analysis Implementation Guide

This guide provides a high-level overview of implementing SAE (Sparse Autoencoder) analysis for language models using HuggingFace transformers, with a focus on clean architecture and organized code structure.

## Overview

The general workflow for SAE analysis involves:
1. Loading a pre-trained language model
2. Collecting activations from specific layers during inference
3. Loading pre-trained SAEs that correspond to model layers
4. Analyzing SAE features and their activations
5. Steering model behavior by modifying SAE latent activations

## 1. Model Loading

### Setup
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    torch_dtype=torch.float16,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
```

### Creating a Clean Model Interface
To avoid scattered code, create a single class that handles all model operations:

```python
class ModelWithHooks:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = self.model.device
        
    def get_layer(self, layer_idx):
        """Get a specific transformer layer - handles architecture differences"""
        if hasattr(self.model, 'model'):  # Llama, Gemma style
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer'):  # GPT style
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError("Unknown model architecture")
    
    def num_layers(self):
        """Get total number of layers"""
        if hasattr(self.model, 'model'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer'):
            return len(self.model.transformer.h)
```

## 2. Activation Collection

### Centralized Hook System
Keep all activation collection logic in one place:

```python
class ActivationCollector:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.activations = {}
        self.hooks = []
    
    def collect(self, input_ids, layer_indices, positions="all"):
        """Collect activations at specified layers and positions"""
        self.activations.clear()
        
        # Register hooks
        for layer_idx in layer_indices:
            layer = self.model_wrapper.get_layer(layer_idx)
            hook = layer.register_forward_hook(
                self._create_hook(layer_idx, positions)
            )
            self.hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            self.model_wrapper.model(input_ids)
        
        # Clean up
        self._remove_hooks()
        
        return self.activations
    
    def _create_hook(self, layer_idx, positions):
        def hook_fn(module, input, output):
            # output shape: (batch, seq_len, hidden_dim)
            if positions == "all":
                self.activations[layer_idx] = output.detach()
            elif positions == "last":
                self.activations[layer_idx] = output[:, -1, :].detach()
            elif isinstance(positions, list):
                # Specific token positions
                self.activations[layer_idx] = output[:, positions, :].detach()
            return output
        return hook_fn
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
```

### Batch Processing and Caching
For large-scale analysis:

```python
class ActivationCache:
    def __init__(self, cache_dir="./activation_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def process_dataset(self, model_wrapper, dataset, layers, batch_size=32):
        """Process dataset in batches and cache activations"""
        collector = ActivationCollector(model_wrapper)
        
        for batch_idx in range(0, len(dataset), batch_size):
            batch = dataset[batch_idx:batch_idx + batch_size]
            
            # Tokenize batch
            inputs = model_wrapper.tokenizer(
                batch["text"], 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Collect activations
            activations = collector.collect(
                inputs["input_ids"], 
                layers,
                positions="all"
            )
            
            # Save to disk
            self._save_batch(activations, batch_idx)
    
    def _save_batch(self, activations, batch_idx):
        """Save activations using memory-mapped arrays"""
        for layer_idx, acts in activations.items():
            filename = self.cache_dir / f"layer_{layer_idx}_batch_{batch_idx}.npy"
            np.save(filename, acts.cpu().numpy())
```

## 3. SAE Integration

### SAE Loading
```python
from sae_lens import SAE

class SAEManager:
    def __init__(self):
        self.saes = {}
    
    def load_saes_for_model(self, model_name, layers):
        """Load SAEs for specific model layers"""
        for layer in layers:
            if "gemma-2" in model_name:
                # Gemma-specific SAE loading
                self.saes[layer] = SAE.from_pretrained(
                    release="gemma-scope-2b-pt-res",
                    sae_id=f"layer_{layer}/width_16k/average_l0_82"
                )
            else:
                # Generic SAE loading
                self.saes[layer] = SAE.from_pretrained(
                    f"sae-{model_name}-layer-{layer}"
                )
    
    def encode(self, layer_idx, activations):
        """Encode activations to SAE latent space"""
        return self.saes[layer_idx].encode(activations)
    
    def decode(self, layer_idx, latents):
        """Decode from SAE latent space back to activations"""
        return self.saes[layer_idx].decode(latents)
```

## 4. Analysis Tools

### Feature Analysis
```python
class SAEAnalyzer:
    def __init__(self, model_wrapper, sae_manager):
        self.model_wrapper = model_wrapper
        self.sae_manager = sae_manager
        self.collector = ActivationCollector(model_wrapper)
    
    def analyze_features(self, text, layers, top_k=10):
        """Analyze which SAE features activate for given text"""
        # Tokenize
        inputs = self.model_wrapper.tokenizer(
            text, return_tensors="pt"
        ).to(self.model_wrapper.device)
        
        # Collect activations
        activations = self.collector.collect(
            inputs["input_ids"], layers
        )
        
        results = {}
        for layer_idx, acts in activations.items():
            # Encode to SAE space
            latents = self.sae_manager.encode(layer_idx, acts)
            
            # Find top activating features
            # latents shape: (batch, seq_len, n_features)
            max_activations = latents.max(dim=1).values  # Max over sequence
            top_features = torch.topk(max_activations, k=top_k, dim=-1)
            
            results[layer_idx] = {
                "feature_indices": top_features.indices,
                "feature_values": top_features.values,
                "latents": latents
            }
        
        return results
```

## 5. Model Steering

### Clean Steering Implementation
```python
class ModelSteering:
    def __init__(self, model_wrapper, sae_manager):
        self.model_wrapper = model_wrapper
        self.sae_manager = sae_manager
        self.steering_hooks = []
    
    def steer(self, layer_idx, feature_idx, coefficient, positions=None):
        """Add steering to specific SAE features"""
        def steering_hook(module, input, output):
            # Encode to SAE space
            latents = self.sae_manager.encode(layer_idx, output)
            
            # Apply steering
            if positions is None:
                # Steer all positions
                latents[:, :, feature_idx] += coefficient
            else:
                # Steer specific positions
                latents[:, positions, feature_idx] += coefficient
            
            # Decode back
            steered_output = self.sae_manager.decode(layer_idx, latents)
            return steered_output
        
        # Register hook
        layer = self.model_wrapper.get_layer(layer_idx)
        hook = layer.register_forward_hook(steering_hook)
        self.steering_hooks.append(hook)
        
        return hook
    
    def generate_with_steering(self, prompt, **generation_kwargs):
        """Generate text with current steering hooks active"""
        inputs = self.model_wrapper.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.model_wrapper.device)
        
        with torch.no_grad():
            outputs = self.model_wrapper.model.generate(
                inputs["input_ids"],
                **generation_kwargs
            )
        
        return self.model_wrapper.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
    
    def clear_steering(self):
        """Remove all steering hooks"""
        for hook in self.steering_hooks:
            hook.remove()
        self.steering_hooks.clear()
```

## 6. Complete Example Workflow

```python
# Initialize components
model_wrapper = ModelWithHooks("google/gemma-2-2b")
sae_manager = SAEManager()
sae_manager.load_saes_for_model("gemma-2-2b", layers=[10, 15, 20])

# Analysis
analyzer = SAEAnalyzer(model_wrapper, sae_manager)
results = analyzer.analyze_features(
    "The capital of France is",
    layers=[10, 15, 20]
)

# Steering
steering = ModelSteering(model_wrapper, sae_manager)

# Add steering to increase "factual knowledge" feature
steering.steer(
    layer_idx=15,
    feature_idx=results[15]["feature_indices"][0],  # Top feature
    coefficient=2.0
)

# Generate with steering
output = steering.generate_with_steering(
    "The capital of France is",
    max_length=50,
    temperature=0.7
)

# Clean up
steering.clear_steering()
```

## Best Practices

### Code Organization
```
project/
├── models/
│   ├── __init__.py
│   ├── model_wrapper.py      # ModelWithHooks class
│   └── architectures.py      # Model-specific details
├── activation_collection/
│   ├── __init__.py
│   ├── collector.py          # ActivationCollector class
│   └── cache.py              # ActivationCache class
├── sae/
│   ├── __init__.py
│   ├── manager.py            # SAEManager class
│   └── analysis.py           # SAEAnalyzer class
├── steering/
│   ├── __init__.py
│   └── steering.py           # ModelSteering class
└── experiments/
    └── run_analysis.py       # Your experiments
```

### Key Principles
1. **Centralize functionality**: Each component has a single responsibility
2. **Clean interfaces**: Hide implementation details behind simple methods
3. **Explicit cleanup**: Always remove hooks when done
4. **Type hints and documentation**: Make code self-documenting
5. **Error handling**: Handle architecture differences gracefully

### Common Pitfalls to Avoid
- Don't scatter hook logic across multiple files
- Don't mix model-specific code with analysis logic
- Don't forget to remove hooks (memory leaks)
- Don't hardcode layer access patterns
- Don't assume all models have the same architecture

## Key Libraries

- **transformers**: Model loading and inference
- **torch**: Deep learning framework
- **sae-lens**: Pre-trained SAEs
- **einops**: Tensor manipulation
- **numpy**: Efficient array storage
- **tqdm**: Progress bars for batch processing

This architecture provides a clean, maintainable framework for SAE analysis without the complexity of multiple libraries or scattered implementations.