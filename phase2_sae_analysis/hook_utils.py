"""
Hook utilities for SAE analysis with both TransformerLens and Hugging Face models.

This module provides context managers and utilities for managing forward hooks
in different model frameworks, enabling activation extraction, steering, and ablation.
"""

import contextlib
import functools
import logging
from typing import List, Tuple, Callable, Optional, Union
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def add_hf_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]] = None,
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]] = None,
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to Hugging Face models.
    
    Args:
        module_forward_pre_hooks: List of (module, hook_function) tuples for pre-hooks
        module_forward_hooks: List of (module, hook_function) tuples for post-hooks
        **kwargs: Additional arguments passed to hook functions via functools.partial
    """
    module_forward_pre_hooks = module_forward_pre_hooks or []
    module_forward_hooks = module_forward_hooks or []
    
    try:
        handles = []
        
        # Register pre-hooks
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
            
        # Register post-hooks
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
            
        logger.debug(f"Registered {len(handles)} hooks")
        yield
        
    finally:
        # Always clean up hooks
        for handle in handles:
            handle.remove()
        logger.debug(f"Removed {len(handles)} hooks")


@contextlib.contextmanager
def add_tl_hooks(
    model: HookedTransformer,
    fwd_hooks: List[Tuple[str, Callable]] = None,
    **kwargs
):
    """
    Context manager for temporarily adding hooks to TransformerLens models.
    
    Args:
        model: HookedTransformer model
        fwd_hooks: List of (hook_name, hook_function) tuples
        **kwargs: Additional arguments passed to hook functions
    """
    fwd_hooks = fwd_hooks or []
    
    try:
        # Clear any existing hooks
        model.reset_hooks()
        
        # Add new hooks
        for hook_name, hook_fn in fwd_hooks:
            partial_hook = functools.partial(hook_fn, **kwargs)
            model.add_hook(hook_name, partial_hook, "fwd")
            
        logger.debug(f"Added {len(fwd_hooks)} TransformerLens hooks")
        yield model
        
    finally:
        # Clean up hooks
        model.reset_hooks()
        logger.debug("Removed all TransformerLens hooks")


class ActivationCache:
    """Simple activation cache for storing and retrieving activations."""
    
    def __init__(self):
        self.cache = {}
        
    def store(self, name: str, activation: torch.Tensor):
        """Store activation with given name."""
        self.cache[name] = activation.detach().clone()
        
    def get(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve activation by name."""
        return self.cache.get(name)
    
    def clear(self):
        """Clear all cached activations."""
        self.cache.clear()
        
    def keys(self):
        """Get all cache keys."""
        return self.cache.keys()
    
    def __contains__(self, name: str) -> bool:
        """Check if activation is cached."""
        return name in self.cache
    
    def __getitem__(self, name: str) -> torch.Tensor:
        """Get activation by name (raises KeyError if not found)."""
        return self.cache[name]


def get_activation_extraction_hook(
    cache: ActivationCache,
    hook_name: str,
    position: Union[int, str] = -1,
    extract_fn: Optional[Callable] = None
):
    """
    Create a hook function for extracting activations.
    
    Args:
        cache: ActivationCache to store activations
        hook_name: Name to store activation under
        position: Token position to extract (-1 for last token, 'all' for all tokens)
        extract_fn: Optional function to transform activation before caching
    
    Returns:
        Hook function compatible with both TL and HF models
    """
    def hook_fn(activation, hook=None):
        """
        Hook function that extracts and caches activations.
        Compatible with both TransformerLens and direct PyTorch hooks.
        """
        # Handle different input formats
        if isinstance(activation, tuple):
            # For some hooks, activation might be a tuple
            act = activation[0]
        else:
            act = activation
            
        # Extract specified position
        if position == 'all':
            extracted = act.detach().clone()
        elif isinstance(position, int):
            # Extract specific token position
            extracted = act[:, position, :].detach().clone()
        else:
            raise ValueError(f"Invalid position: {position}")
            
        # Apply extraction function if provided
        if extract_fn is not None:
            extracted = extract_fn(extracted)
            
        # Store in cache
        cache.store(hook_name, extracted)
        
        # Return original activation (no modification)
        return activation
    
    return hook_fn


def get_sae_reconstruction_hook(
    sae: nn.Module,
    reconstruct_bos_token: bool = False
):
    """
    Create a hook for SAE reconstruction during forward pass.
    
    Args:
        sae: SAE model with encode/decode methods
        reconstruct_bos_token: Whether to reconstruct BOS token
    
    Returns:
        Hook function for SAE reconstruction
    """
    def hook_fn(module, input, **kwargs):
        """
        SAE reconstruction hook for pre-forward hooks.
        Replaces activations with SAE reconstructions.
        """
        # Handle input format
        if isinstance(input, tuple):
            activation = input[0]
            other_inputs = input[1:]
        else:
            activation = input
            other_inputs = ()
            
        # Get batch and sequence dimensions
        batch_size, seq_len, d_model = activation.shape
        
        # Skip BOS token if specified
        if not reconstruct_bos_token and seq_len > 1:
            # Only reconstruct non-BOS tokens
            to_reconstruct = activation[:, 1:, :]
            reconstructed = sae(to_reconstruct.reshape(-1, d_model))
            
            # Handle different SAE output formats
            if hasattr(reconstructed, 'sae_out'):
                reconstructed = reconstructed.sae_out
            
            # Reshape back and combine with original BOS
            reconstructed = reconstructed.reshape(batch_size, seq_len - 1, d_model)
            final_activation = torch.cat([
                activation[:, :1, :],  # Keep original BOS
                reconstructed
            ], dim=1)
        else:
            # Reconstruct all tokens
            reshaped = activation.reshape(-1, d_model)
            reconstructed = sae(reshaped)
            
            # Handle different SAE output formats
            if hasattr(reconstructed, 'sae_out'):
                reconstructed = reconstructed.sae_out
                
            final_activation = reconstructed.reshape(batch_size, seq_len, d_model)
        
        # Return modified input
        if other_inputs:
            return (final_activation,) + other_inputs
        else:
            return final_activation
    
    return hook_fn


# Utility functions for common hook patterns
def create_layer_hook_name(layer_idx: int, hook_type: str = "resid_pre") -> str:
    """
    Create TransformerLens hook name for a specific layer.
    
    Args:
        layer_idx: Layer index
        hook_type: Type of hook (resid_pre, resid_mid, resid_post, etc.)
    
    Returns:
        Full hook name string
    """
    return f"blocks.{layer_idx}.hook_{hook_type}"


def get_layer_module(model: nn.Module, layer_idx: int) -> nn.Module:
    """
    Get layer module from Hugging Face model.
    
    Args:
        model: Hugging Face model
        layer_idx: Layer index
    
    Returns:
        Layer module
    """
    # Common patterns for different model architectures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Gemma, Llama style
        return model.model.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT style
        return model.transformer.h[layer_idx]
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        # GPT-NeoX style
        return model.gpt_neox.layers[layer_idx]
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")