# HuggingFace Hooking Tutorial

This tutorial provides a comprehensive guide to hooking into HuggingFace transformer models, with special emphasis on accessing the residual stream.

## Why HuggingFace Instead of TransformerLens?

This codebase uses HuggingFace transformers for hooking in production scenarios because:

1. **Direct Model Support**: Works with any model from HuggingFace Hub without conversion
2. **Production Ready**: Optimized for deployment, batch processing, and memory efficiency
3. **Latest Models**: Immediate access to new models as they're released
4. **Minimal Dependencies**: Uses standard PyTorch hooks, no extra framework needed
5. **Scalability**: Better for large-scale data collection and generation tasks

While TransformerLens is excellent for research and provides convenient named hooks, HuggingFace is the choice for production systems that need efficiency and broad model compatibility.

## Table of Contents
1. [Understanding PyTorch Hooks](#understanding-pytorch-hooks)
2. [HuggingFace Model Architecture](#huggingface-model-architecture)
3. [Step-by-Step Hooking Process](#step-by-step-hooking-process)
4. [Hooking the Residual Stream](#hooking-the-residual-stream)
5. [Complete Examples](#complete-examples)
6. [Best Practices](#best-practices)

## Understanding PyTorch Hooks

PyTorch provides two types of hooks for modules:

### Pre-hooks
- Called BEFORE the module's forward pass
- Signature: `hook(module, input) -> modified_input or None`
- Can modify inputs before processing

### Post-hooks
- Called AFTER the module's forward pass
- Signature: `hook(module, input, output) -> modified_output or None`
- Can modify outputs after processing

## HuggingFace Model Architecture

### Model Structure
HuggingFace transformers follow a consistent structure:

```
model
├── model.embed_tokens          # Token embeddings
├── model.layers[0...n-1]       # Transformer blocks
│   ├── self_attn              # Multi-head attention
│   │   ├── q_proj, k_proj, v_proj
│   │   └── o_proj
│   ├── mlp                    # Feed-forward network
│   │   ├── up_proj, gate_proj
│   │   └── down_proj
│   └── [layer_norms]          # Normalization layers
└── model.norm                 # Final layer norm
```

### Key Insight: The Residual Stream

**The input to each transformer block IS the residual stream at that layer.**

When you hook into `model.model.layers[i]` with a pre-hook, you're accessing the residual stream!

## Step-by-Step Hooking Process

### Step 1: Define Your Hook Function

```python
def my_hook_function(module, input, **kwargs):
    """
    Args:
        module: The PyTorch module being hooked
        input: Tuple or Tensor of inputs to the module
    Returns:
        Modified input (or None to leave unchanged)
    """
    # Handle both tuple and tensor inputs
    if isinstance(input, tuple):
        activation = input[0]  # Shape: [batch_size, seq_len, d_model]
    else:
        activation = input
    
    # Your modification here
    modified_activation = activation  # Modify as needed
    
    # Return in the same format
    if isinstance(input, tuple):
        return (modified_activation, *input[1:])
    else:
        return modified_activation
```

### Step 2: Identify Target Modules

```python
# Access different parts of the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Residual stream (transformer block inputs)
residual_modules = model.model.layers  # List of transformer blocks

# Attention outputs
attention_modules = [layer.self_attn for layer in model.model.layers]

# MLP outputs
mlp_modules = [layer.mlp for layer in model.model.layers]
```

### Step 3: Register Hooks Using Context Manager

```python
from utils.hf_patching_utils import add_hooks

# Create list of (module, hook_function) pairs
hook_pairs = [(module, my_hook_function) for module in target_modules]

# Use context manager for automatic cleanup
with add_hooks(module_forward_pre_hooks=hook_pairs, module_forward_hooks=[]):
    # Run your model - hooks will be active
    output = model(input_ids=input_ids)
# Hooks are automatically removed here
```

## Hooking the Residual Stream

### Understanding Residual Stream Access

The residual stream flows through the transformer as follows:

```
[Embeddings] → [Residual₀] → [Block₀] → [Residual₁] → [Block₁] → ... → [Output]
                    ↑                          ↑
                Pre-hook here            Pre-hook here
```

### Method 1: Hook All Residual Streams

```python
def residual_stream_hook(module, input):
    """Hook that captures the residual stream at a specific layer."""
    if isinstance(input, tuple):
        residual_stream = input[0]  # Shape: [batch, seq, d_model]
    else:
        residual_stream = input
    
    # The residual_stream variable now contains the residual stream
    # at this layer, before any processing by the transformer block
    
    # Example: Print statistics
    print(f"Residual stream shape: {residual_stream.shape}")
    print(f"Residual stream norm: {residual_stream.norm(dim=-1).mean()}")
    
    # Return unmodified (or modify if needed)
    return input

# Hook into all layers' residual streams
residual_hooks = [
    (model.model.layers[i], residual_stream_hook) 
    for i in range(len(model.model.layers))
]

with add_hooks(module_forward_pre_hooks=residual_hooks, module_forward_hooks=[]):
    output = model(input_ids=input_ids)
```

### Method 2: Hook Specific Layer's Residual Stream

```python
def get_residual_stream_at_layer(model, layer_idx, input_ids):
    """Extract residual stream at a specific layer."""
    residual_stream = None
    
    def capture_hook(module, input):
        nonlocal residual_stream
        if isinstance(input, tuple):
            residual_stream = input[0].clone()  # Clone to preserve
        else:
            residual_stream = input.clone()
        return input  # Don't modify
    
    # Hook only the target layer
    hooks = [(model.model.layers[layer_idx], capture_hook)]
    
    with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
        model(input_ids=input_ids)
    
    return residual_stream
```

### Method 3: Modify Residual Stream

```python
def add_vector_to_residual_stream(model, layer_idx, vector, positions=None):
    """Add a steering vector to the residual stream at specific positions."""
    
    def steering_hook(module, input):
        if isinstance(input, tuple):
            residual = input[0]
        else:
            residual = input
            
        # Add vector to specific positions or all positions
        if positions is not None:
            residual[:, positions, :] += vector
        else:
            residual += vector
            
        if isinstance(input, tuple):
            return (residual, *input[1:])
        else:
            return residual
    
    return [(model.model.layers[layer_idx], steering_hook)]
```

## Complete Examples

### Example 1: Caching Residual Streams

```python
def cache_residual_streams(model, input_ids, layers=None):
    """Cache residual streams from multiple layers."""
    if layers is None:
        layers = range(len(model.model.layers))
    
    cache = {}
    
    def make_cache_hook(layer_idx):
        def hook(module, input):
            if isinstance(input, tuple):
                cache[f'layer_{layer_idx}'] = input[0].clone()
            else:
                cache[f'layer_{layer_idx}'] = input.clone()
            return input
        return hook
    
    # Create hooks for each layer
    hooks = [
        (model.model.layers[layer_idx], make_cache_hook(layer_idx))
        for layer_idx in layers
    ]
    
    # Run model with hooks
    with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
        output = model(input_ids=input_ids)
    
    return cache, output
```

### Example 2: Direction Ablation in Residual Stream

```python
def ablate_direction_from_residual(model, direction, layers=None):
    """Remove a specific direction from residual streams."""
    if layers is None:
        layers = range(len(model.model.layers))
    
    # Normalize direction
    direction = direction / direction.norm()
    
    def ablation_hook(module, input):
        if isinstance(input, tuple):
            residual = input[0]
        else:
            residual = input
        
        # Project out the direction
        projection = (residual @ direction).unsqueeze(-1) * direction
        residual = residual - projection
        
        if isinstance(input, tuple):
            return (residual, *input[1:])
        else:
            return residual
    
    # Create hooks for specified layers
    hooks = [
        (model.model.layers[layer_idx], ablation_hook)
        for layer_idx in layers
    ]
    
    return hooks
```

### Example 3: Position-Specific Interventions

```python
def position_aware_residual_hook(positions_by_batch):
    """Apply modifications only at specific token positions."""
    
    def hook(module, input):
        if isinstance(input, tuple):
            residual = input[0]  # [batch, seq, d_model]
        else:
            residual = input
        
        # Apply modifications only at specified positions
        for batch_idx, positions in enumerate(positions_by_batch):
            # Example: amplify activation at entity positions
            residual[batch_idx, positions, :] *= 1.5
        
        if isinstance(input, tuple):
            return (residual, *input[1:])
        else:
            return residual
    
    return hook
```

## Best Practices

### 1. Always Use Context Managers
```python
# Good - hooks are automatically cleaned up
with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
    output = model(input_ids)

# Bad - hooks persist and accumulate
for module, hook in hooks:
    module.register_forward_pre_hook(hook)
output = model(input_ids)  # Hooks still active!
```

### 2. Handle Input Formats Correctly
```python
def robust_hook(module, input):
    # Always check if input is tuple
    if isinstance(input, tuple):
        activation = input[0]
        # Process activation
        return (processed_activation, *input[1:])
    else:
        activation = input
        # Process activation
        return processed_activation
```

### 3. Clone Tensors When Caching
```python
# Good - preserves gradient computation
cached_activation = activation.clone()

# Bad - may interfere with autograd
cached_activation = activation
```

### 4. Use Descriptive Hook Names
```python
def get_residual_stream_layer_10_hook():
    # Clear purpose from function name
    pass
```

### 5. Be Careful with In-Place Operations
```python
# Good - creates new tensor
residual = residual + vector

# Risky - modifies in place
residual += vector  # May cause gradient issues
```

## Common Pitfalls

1. **Forgetting that pre-hooks on blocks give residual stream**
   - The input to transformer blocks IS the residual stream

2. **Not handling tuple inputs**
   - Some models pass additional arguments as tuples

3. **Accumulating hooks**
   - Always remove hooks after use or use context managers

4. **Modifying shared tensors**
   - Clone before modifying to avoid side effects

## Summary

To hook the residual stream in HuggingFace models:

1. Target `model.model.layers[i]` with pre-hooks
2. The input you receive IS the residual stream
3. Use the context manager for clean hook management
4. Handle both tuple and tensor inputs
5. Clone tensors when caching

The residual stream is the backbone of transformer computation - by hooking into it, you can analyze, modify, or steer model behavior at any layer.