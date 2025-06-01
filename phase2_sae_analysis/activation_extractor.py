"""
Improved activation extraction using TransformerLens and modern hook patterns.

This module provides a more robust and flexible activation extraction system
that supports both TransformerLens and Hugging Face models.
"""

import logging
from typing import List, Dict, Union, Optional, Tuple
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from .hook_utils import (
    ActivationCache, 
    add_tl_hooks, 
    add_hf_hooks,
    get_activation_extraction_hook,
    create_layer_hook_name,
    get_layer_module
)

logger = logging.getLogger(__name__)


class ImprovedActivationExtractor:
    """
    Improved activation extractor supporting both TransformerLens and Hugging Face models.
    
    Features:
    - Better memory management with context managers
    - Support for multiple positions and layers
    - Flexible hook management
    - Proper cleanup guarantees
    """
    
    def __init__(
        self, 
        model: Union[HookedTransformer, torch.nn.Module], 
        tokenizer: AutoTokenizer,
        device: str = "mps"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.is_hooked_transformer = isinstance(model, HookedTransformer)
        self.cache = ActivationCache()
        
        logger.info(f"Initialized extractor for {'TransformerLens' if self.is_hooked_transformer else 'Hugging Face'} model")
    
    def extract_activations(
        self,
        prompts: List[str],
        layer_idx: int,
        position: Union[int, str] = -1,
        hook_type: str = "resid_pre"
    ) -> torch.Tensor:
        """
        Extract activations for given prompts at specified layer and position.
        
        Args:
            prompts: List of text prompts
            layer_idx: Layer index to extract from
            position: Token position (-1 for last, 'all' for all tokens)
            hook_type: Type of hook for TransformerLens (resid_pre, resid_mid, resid_post)
            
        Returns:
            Tensor of extracted activations
        """
        self.cache.clear()
        
        if self.is_hooked_transformer:
            return self._extract_with_transformerlens(prompts, layer_idx, position, hook_type)
        else:
            return self._extract_with_huggingface(prompts, layer_idx, position)
    
    def _extract_with_transformerlens(
        self,
        prompts: List[str],
        layer_idx: int,
        position: Union[int, str],
        hook_type: str
    ) -> torch.Tensor:
        """Extract activations using TransformerLens hooks."""
        hook_name = create_layer_hook_name(layer_idx, hook_type)
        
        def extraction_hook(activation, hook):
            """Hook function for TransformerLens."""
            if position == 'all':
                extracted = activation.detach().clone()
            elif isinstance(position, int):
                extracted = activation[:, position, :].detach().clone()
            else:
                raise ValueError(f"Invalid position: {position}")
                
            # Store each sample separately for easier concatenation
            for i in range(extracted.shape[0]):
                cache_key = f"sample_{len(self.cache.cache)}"
                self.cache.store(cache_key, extracted[i:i+1])
            
            return activation
        
        # Use context manager for safe hook management
        with add_tl_hooks(self.model, [(hook_name, extraction_hook)]):
            self._process_prompts_batch(prompts)
        
        # Concatenate all cached activations
        activations = []
        for key in sorted(self.cache.keys()):
            activations.append(self.cache[key])
        
        if not activations:
            raise RuntimeError("No activations were captured")
            
        result = torch.cat(activations, dim=0)
        logger.info(f"Extracted activations shape: {result.shape}")
        return result
    
    def _extract_with_huggingface(
        self,
        prompts: List[str],
        layer_idx: int,
        position: Union[int, str]
    ) -> torch.Tensor:
        """Extract activations using Hugging Face hooks."""
        layer_module = get_layer_module(self.model, layer_idx)
        
        def extraction_hook(module, input, output=None):
            """Hook function for Hugging Face models."""
            # For pre-hooks, we get input; for post-hooks, we get output
            activation = input[0] if isinstance(input, tuple) else input
            
            if position == 'all':
                extracted = activation.detach().clone()
            elif isinstance(position, int):
                extracted = activation[:, position, :].detach().clone()
            else:
                raise ValueError(f"Invalid position: {position}")
                
            # Store each sample separately
            for i in range(extracted.shape[0]):
                cache_key = f"sample_{len(self.cache.cache)}"
                self.cache.store(cache_key, extracted[i:i+1])
        
        # Use pre-hook to get residual stream input
        hooks = [(layer_module, extraction_hook)]
        
        with add_hf_hooks(module_forward_pre_hooks=hooks):
            self._process_prompts_batch(prompts)
        
        # Concatenate all cached activations
        activations = []
        for key in sorted(self.cache.keys()):
            activations.append(self.cache[key])
        
        if not activations:
            raise RuntimeError("No activations were captured")
            
        result = torch.cat(activations, dim=0)
        logger.info(f"Extracted activations shape: {result.shape}")
        return result
    
    def _process_prompts_batch(self, prompts: List[str], batch_size: int = 1):
        """Process prompts in batches to trigger hooks."""
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Forward pass to trigger hooks
                if self.is_hooked_transformer:
                    _ = self.model(inputs.input_ids)
                else:
                    _ = self.model(**inputs)
    
    def extract_multi_layer(
        self,
        prompts: List[str],
        layer_indices: List[int],
        position: Union[int, str] = -1,
        hook_type: str = "resid_pre"
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from multiple layers simultaneously.
        
        Args:
            prompts: List of text prompts
            layer_indices: List of layer indices to extract from
            position: Token position
            hook_type: Type of hook for TransformerLens
            
        Returns:
            Dictionary mapping layer_idx to activation tensors
        """
        results = {}
        
        if self.is_hooked_transformer:
            # Can extract from multiple layers in one pass with TransformerLens
            self.cache.clear()
            hooks = []
            
            for layer_idx in layer_indices:
                hook_name = create_layer_hook_name(layer_idx, hook_type)
                
                def make_extraction_hook(layer_id):
                    sample_counter = 0
                    def extraction_hook(activation, hook):
                        nonlocal sample_counter
                        if position == 'all':
                            extracted = activation.detach().clone()
                        elif isinstance(position, int):
                            extracted = activation[:, position, :].detach().clone()
                        else:
                            raise ValueError(f"Invalid position: {position}")
                        
                        # Store with layer-specific cache key
                        for i in range(extracted.shape[0]):
                            cache_key = f"layer_{layer_id}_sample_{sample_counter}"
                            self.cache.store(cache_key, extracted[i:i+1])
                            sample_counter += 1
                        
                        return activation
                    return extraction_hook
                
                hooks.append((hook_name, make_extraction_hook(layer_idx)))
            
            with add_tl_hooks(self.model, hooks):
                self._process_prompts_batch(prompts)
            
            # Organize results by layer
            for layer_idx in layer_indices:
                layer_activations = []
                # Get all samples for this layer
                sample_count = 0
                while f"layer_{layer_idx}_sample_{sample_count}" in self.cache:
                    cache_key = f"layer_{layer_idx}_sample_{sample_count}"
                    layer_activations.append(self.cache[cache_key])
                    sample_count += 1
                
                if layer_activations:
                    results[layer_idx] = torch.cat(layer_activations, dim=0)
        else:
            # Extract one layer at a time for Hugging Face models
            for layer_idx in layer_indices:
                results[layer_idx] = self.extract_activations(
                    prompts, layer_idx, position
                )
        
        logger.info(f"Extracted activations from {len(results)} layers")
        return results
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        if self.is_hooked_transformer:
            return {
                "model_type": "TransformerLens",
                "model_name": self.model.cfg.model_name,
                "n_layers": self.model.cfg.n_layers,
                "d_model": self.model.cfg.d_model,
                "device": str(self.model.cfg.device)
            }
        else:
            return {
                "model_type": "Hugging Face",
                "model_name": getattr(self.model.config, 'model_type', 'unknown'),
                "n_layers": getattr(self.model.config, 'num_hidden_layers', 'unknown'),
                "d_model": getattr(self.model.config, 'hidden_size', 'unknown'),
                "device": str(next(self.model.parameters()).device)
            }