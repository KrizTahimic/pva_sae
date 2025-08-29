#!/usr/bin/env python3
"""
Debug script to understand the flow of data through a transformer model.
Specifically: what comes first - residual stream or attention heads?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List

class ModelFlowTracer:
    """Trace the flow of data through model layers."""
    
    def __init__(self, model_name: str = "google/gemma-2-2b"):
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # Use CPU for debugging
            torch_dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hooks = []
        self.trace = []
        
    def create_trace_hook(self, name: str, hook_type: str):
        """Create a hook that logs when it's called."""
        def hook_fn(module, input, output=None):
            self.trace.append({
                'name': name,
                'type': hook_type,
                'input_shape': input[0].shape if isinstance(input, tuple) else input.shape,
                'output_shape': output[0].shape if output and isinstance(output, tuple) else (output.shape if output else None)
            })
            print(f"  {len(self.trace):3d}. {hook_type:10s} | {name:30s} | Input: {input[0].shape if isinstance(input, tuple) else 'N/A'}")
            
            # For pre-hooks, we must return the input unchanged
            if hook_type == "PRE":
                return input
                
        return hook_fn
    
    def setup_hooks_for_layer(self, layer_idx: int = 0):
        """Setup hooks to trace a single layer's execution flow."""
        print(f"\nSetting up hooks for layer {layer_idx}...")
        
        layer = self.model.model.layers[layer_idx]
        
        # 1. Pre-hook on the entire layer (captures residual stream BEFORE layer)
        hook = layer.register_forward_pre_hook(
            self.create_trace_hook(f"Layer_{layer_idx}_ENTIRE", "PRE")
        )
        self.hooks.append(hook)
        
        # 2. Post-hook on the entire layer (captures residual stream AFTER layer)
        hook = layer.register_forward_hook(
            self.create_trace_hook(f"Layer_{layer_idx}_ENTIRE", "POST")
        )
        self.hooks.append(hook)
        
        # 3. Hooks on attention module
        if hasattr(layer, 'self_attn'):
            # Pre-hook on attention
            hook = layer.self_attn.register_forward_pre_hook(
                self.create_trace_hook(f"Layer_{layer_idx}_Attention", "PRE")
            )
            self.hooks.append(hook)
            
            # Post-hook on attention
            hook = layer.self_attn.register_forward_hook(
                self.create_trace_hook(f"Layer_{layer_idx}_Attention", "POST")
            )
            self.hooks.append(hook)
        
        # 4. Hooks on MLP/FFN module
        if hasattr(layer, 'mlp'):
            # Pre-hook on MLP
            hook = layer.mlp.register_forward_pre_hook(
                self.create_trace_hook(f"Layer_{layer_idx}_MLP", "PRE")
            )
            self.hooks.append(hook)
            
            # Post-hook on MLP
            hook = layer.mlp.register_forward_hook(
                self.create_trace_hook(f"Layer_{layer_idx}_MLP", "POST")
            )
            self.hooks.append(hook)
            
        # 5. Layer norm hooks if present
        if hasattr(layer, 'input_layernorm'):
            hook = layer.input_layernorm.register_forward_hook(
                self.create_trace_hook(f"Layer_{layer_idx}_InputLayerNorm", "POST")
            )
            self.hooks.append(hook)
            
        if hasattr(layer, 'post_attention_layernorm'):
            hook = layer.post_attention_layernorm.register_forward_hook(
                self.create_trace_hook(f"Layer_{layer_idx}_PostAttnLayerNorm", "POST")
            )
            self.hooks.append(hook)
    
    def trace_forward_pass(self, text: str = "Hello world"):
        """Run a forward pass and trace the execution."""
        print(f"\nTracing forward pass with input: '{text}'")
        print("="*80)
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Clear trace
        self.trace = []
        
        # Run forward pass
        print("\nExecution Order:")
        print("-"*80)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        print("-"*80)
        
        return self.trace
    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("\nCleaned up all hooks")
    
    def analyze_flow(self):
        """Analyze and explain the flow."""
        print("\n" + "="*80)
        print("ANALYSIS: How Residual Stream and Attention Interact")
        print("="*80)
        
        print("""
In a typical transformer layer, the flow is:

1. RESIDUAL STREAM enters the layer (this is what we capture with pre-hooks)
   ↓
2. Input LayerNorm applied (normalizes the residual stream)
   ↓
3. ATTENTION MODULE processes the normalized input
   - Multi-head attention computes attention patterns
   - Applies attention to create new representations
   ↓
4. Attention output is ADDED back to residual stream (residual connection #1)
   ↓
5. Post-attention LayerNorm applied
   ↓
6. MLP/FFN MODULE processes the normalized input
   ↓
7. MLP output is ADDED back to residual stream (residual connection #2)
   ↓
8. RESIDUAL STREAM exits the layer (goes to next layer)

KEY INSIGHTS:
- The RESIDUAL STREAM is the "highway" that flows through the model
- ATTENTION HEADS operate ON the residual stream (after normalization)
- Attention output gets ADDED BACK to the residual stream
- When we use pre-hooks on a layer, we capture the residual stream BEFORE attention
- When we steer with pre-hooks, we modify the residual stream that FEEDS INTO attention

HOOK INTERACTION ISSUE:
- If we have 2 pre-hooks on the same layer:
  1. Steering hook modifies residual stream
  2. Another pre-hook might not properly pass through the modification
- The order matters: First hook modifies, second hook must preserve the modification
""")


def main():
    """Run the debugging script."""
    print("Model Flow Debugging Script")
    print("="*80)
    
    # Create tracer
    tracer = ModelFlowTracer("google/gemma-2-2b")
    
    # Setup hooks for layer 0 and 1 to see the flow
    tracer.setup_hooks_for_layer(0)
    tracer.setup_hooks_for_layer(1)
    
    # Trace execution
    trace = tracer.trace_forward_pass("def hello():")
    
    # Analyze
    tracer.analyze_flow()
    
    # Show simplified trace
    print("\nSimplified Execution Trace:")
    print("-"*80)
    for i, step in enumerate(trace[:20], 1):  # Show first 20 steps
        print(f"{i:3d}. {step['name']:35s} | Type: {step['type']:5s}")
    
    # Cleanup
    tracer.cleanup()
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("""
The residual stream comes FIRST (it's the input to each layer).
Attention heads operate ON the residual stream.
When we modify the residual stream with a pre-hook, we're modifying
what the attention heads will see and process.

If hooks don't properly preserve modifications, steering can fail!
""")


if __name__ == "__main__":
    main()