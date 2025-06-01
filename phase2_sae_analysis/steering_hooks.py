"""
Steering and ablation hooks for SAE-based model interventions.

This module implements hooks for:
1. Activation steering using PVA latent directions
2. SAE latent ablation for causal analysis
3. SAE reconstruction during forward passes
4. Generation with steering/ablation applied
"""

import logging
from typing import List, Tuple, Union, Optional, Callable
import torch
import torch.nn as nn
from functools import partial
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from .hook_utils import add_tl_hooks, add_hf_hooks

logger = logging.getLogger(__name__)


def create_steering_hook(
    direction: torch.Tensor,
    position: Union[int, str] = -1,
    coeff_value: Union[float, str] = 1.0
):
    """
    Create a steering hook that adds a direction to activations.
    
    Args:
        direction: Direction vector to add to activations [d_model]
        position: Token position to steer (-1 for last, 'all' for all tokens)
        coeff_value: Steering coefficient (float or 'norm' for norm-based scaling)
    
    Returns:
        Hook function for steering
    """
    def steering_hook(activation, hook=None):
        """Steering hook function."""
        # Skip during generation if sequence length is 1
        if activation.shape[1] == 1:
            return activation
        
        if position != 'all':
            pos = position if position != -1 else activation.shape[1] - 1
            
            if coeff_value == 'norm':
                # Scale by norm of residual stream
                norm_res_streams = torch.norm(activation[:, pos, :], dim=-1)
                steering_vector = direction.unsqueeze(0) * norm_res_streams.unsqueeze(-1)
            else:
                # Fixed coefficient
                steering_vector = direction.unsqueeze(0) * coeff_value
            
            activation[:, pos, :] += steering_vector
        else:
            # Steer all positions
            if coeff_value == 'norm':
                norms = torch.norm(activation, dim=-1, keepdim=True)
                steering_vector = direction.unsqueeze(0).unsqueeze(0) * norms
            else:
                steering_vector = direction.unsqueeze(0).unsqueeze(0) * coeff_value
            
            activation += steering_vector
        
        return activation
    
    return steering_hook


def create_ablation_hook(
    direction: torch.Tensor,
    position: Union[int, str] = -1
):
    """
    Create an ablation hook that removes a direction from activations.
    
    Args:
        direction: Direction vector to remove from activations [d_model]
        position: Token position to ablate (-1 for last, 'all' for all tokens)
    
    Returns:
        Hook function for ablation
    """
    def ablation_hook(activation, hook=None):
        """Ablation hook function."""
        # Skip during generation if sequence length is 1
        if activation.shape[1] == 1:
            return activation
        
        # Normalize direction vector
        normalized_direction = direction / torch.norm(direction)
        
        if position != 'all':
            pos = position if position != -1 else activation.shape[1] - 1
            
            # Project out the direction
            projection = torch.sum(
                activation[:, pos, :] * normalized_direction.unsqueeze(0), 
                dim=-1, 
                keepdim=True
            )
            activation[:, pos, :] -= projection * normalized_direction.unsqueeze(0)
        else:
            # Ablate all positions
            projection = torch.sum(
                activation * normalized_direction.unsqueeze(0).unsqueeze(0), 
                dim=-1, 
                keepdim=True
            )
            activation -= projection * normalized_direction.unsqueeze(0).unsqueeze(0)
        
        return activation
    
    return ablation_hook


def create_sae_reconstruction_hook(
    sae: nn.Module,
    reconstruct_bos_token: bool = False
):
    """
    Create a hook for SAE reconstruction during forward pass.
    
    Args:
        sae: SAE model with encode/decode or forward methods
        reconstruct_bos_token: Whether to reconstruct BOS token
    
    Returns:
        Hook function for SAE reconstruction
    """
    def sae_hook(activation, hook=None):
        """SAE reconstruction hook function."""
        batch_size, seq_len, d_model = activation.shape
        
        if not reconstruct_bos_token and seq_len > 1:
            # Skip BOS token
            to_reconstruct = activation[:, 1:, :]
            bos_token = activation[:, :1, :]
            
            # Reshape and apply SAE
            reshaped = to_reconstruct.reshape(-1, d_model)
            with torch.no_grad():
                if hasattr(sae, 'forward'):
                    reconstructed = sae(reshaped)
                elif hasattr(sae, 'decode'):
                    sae_acts = sae.encode(reshaped)
                    reconstructed = sae.decode(sae_acts)
                else:
                    raise ValueError("SAE must have either 'forward' or 'encode'/'decode' methods")
            
            # Handle different output formats
            if hasattr(reconstructed, 'sae_out'):
                reconstructed = reconstructed.sae_out
            
            # Reshape back and combine with BOS
            reconstructed = reconstructed.reshape(batch_size, seq_len - 1, d_model)
            return torch.cat([bos_token, reconstructed], dim=1)
        else:
            # Reconstruct all tokens
            reshaped = activation.reshape(-1, d_model)
            with torch.no_grad():
                if hasattr(sae, 'forward'):
                    reconstructed = sae(reshaped)
                elif hasattr(sae, 'decode'):
                    sae_acts = sae.encode(reshaped)
                    reconstructed = sae.decode(sae_acts)
                else:
                    raise ValueError("SAE must have either 'forward' or 'encode'/'decode' methods")
            
            # Handle different output formats
            if hasattr(reconstructed, 'sae_out'):
                reconstructed = reconstructed.sae_out
                
            return reconstructed.reshape(batch_size, seq_len, d_model)
    
    return sae_hook


class ModelSteerer:
    """
    Class for applying steering, ablation, and SAE reconstruction to models.
    
    Supports both TransformerLens and Hugging Face models with context management
    for safe hook application and removal.
    """
    
    def __init__(
        self, 
        model: Union[HookedTransformer, torch.nn.Module], 
        tokenizer: AutoTokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.is_hooked_transformer = isinstance(model, HookedTransformer)
        
        logger.info(f"Initialized steerer for {'TransformerLens' if self.is_hooked_transformer else 'Hugging Face'} model")
    
    def generate_with_steering(
        self,
        prompts: List[str],
        steering_vectors: List[Tuple[int, torch.Tensor, float]] = None,
        ablation_vectors: List[Tuple[int, torch.Tensor]] = None,
        sae_reconstructions: List[Tuple[int, nn.Module]] = None,
        max_new_tokens: int = 30,
        position: Union[int, str] = -1,
        do_sample: bool = False,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate text with steering, ablation, and/or SAE reconstruction applied.
        
        Args:
            prompts: Input prompts for generation
            steering_vectors: List of (layer_idx, direction, coefficient) tuples
            ablation_vectors: List of (layer_idx, direction) tuples  
            sae_reconstructions: List of (layer_idx, sae_model) tuples
            max_new_tokens: Maximum tokens to generate
            position: Token position for interventions
            do_sample: Whether to use sampling
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        steering_vectors = steering_vectors or []
        ablation_vectors = ablation_vectors or []
        sae_reconstructions = sae_reconstructions or []
        
        if self.is_hooked_transformer:
            return self._generate_with_tl(
                prompts, steering_vectors, ablation_vectors, sae_reconstructions,
                max_new_tokens, position, do_sample, **generation_kwargs
            )
        else:
            return self._generate_with_hf(
                prompts, steering_vectors, ablation_vectors, sae_reconstructions,
                max_new_tokens, position, do_sample, **generation_kwargs
            )
    
    def _generate_with_tl(
        self,
        prompts: List[str],
        steering_vectors: List[Tuple[int, torch.Tensor, float]],
        ablation_vectors: List[Tuple[int, torch.Tensor]],
        sae_reconstructions: List[Tuple[int, nn.Module]],
        max_new_tokens: int,
        position: Union[int, str],
        do_sample: bool,
        **generation_kwargs
    ) -> List[str]:
        """Generate with TransformerLens hooks."""
        hooks = []
        
        # Add steering hooks
        for layer_idx, direction, coeff in steering_vectors:
            hook_name = f"blocks.{layer_idx}.hook_resid_pre"
            hook_fn = create_steering_hook(direction, position, coeff)
            hooks.append((hook_name, hook_fn))
        
        # Add ablation hooks
        for layer_idx, direction in ablation_vectors:
            hook_name = f"blocks.{layer_idx}.hook_resid_pre"
            hook_fn = create_ablation_hook(direction, position)
            hooks.append((hook_name, hook_fn))
        
        # Add SAE reconstruction hooks
        for layer_idx, sae in sae_reconstructions:
            hook_name = f"blocks.{layer_idx}.hook_resid_pre"
            hook_fn = create_sae_reconstruction_hook(sae)
            hooks.append((hook_name, hook_fn))
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.cfg.device)
        
        # Generate with hooks
        with add_tl_hooks(self.model, hooks):
            # TransformerLens generate method has different interface
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **generation_kwargs
            )
        
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove prompt tokens
            new_tokens = output[len(inputs.input_ids[i]):]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def _generate_with_hf(
        self,
        prompts: List[str],
        steering_vectors: List[Tuple[int, torch.Tensor, float]],
        ablation_vectors: List[Tuple[int, torch.Tensor]],
        sae_reconstructions: List[Tuple[int, nn.Module]],
        max_new_tokens: int,
        position: Union[int, str],
        do_sample: bool,
        **generation_kwargs
    ) -> List[str]:
        """Generate with Hugging Face hooks."""
        from .hook_utils import get_layer_module
        
        pre_hooks = []
        
        # Add hooks for each layer that needs intervention
        all_layers = set()
        all_layers.update(layer_idx for layer_idx, _, _ in steering_vectors)
        all_layers.update(layer_idx for layer_idx, _ in ablation_vectors)
        all_layers.update(layer_idx for layer_idx, _ in sae_reconstructions)
        
        for layer_idx in all_layers:
            layer_module = get_layer_module(self.model, layer_idx)
            
            def create_combined_hook(l_idx):
                def combined_hook(module, input):
                    activation = input[0] if isinstance(input, tuple) else input
                    
                    # Apply SAE reconstruction first
                    for sae_layer, sae in sae_reconstructions:
                        if sae_layer == l_idx:
                            activation = create_sae_reconstruction_hook(sae)(activation)
                    
                    # Apply steering
                    for steer_layer, direction, coeff in steering_vectors:
                        if steer_layer == l_idx:
                            activation = create_steering_hook(direction, position, coeff)(activation)
                    
                    # Apply ablation
                    for abl_layer, direction in ablation_vectors:
                        if abl_layer == l_idx:
                            activation = create_ablation_hook(direction, position)(activation)
                    
                    return activation if not isinstance(input, tuple) else (activation,) + input[1:]
                
                return combined_hook
            
            pre_hooks.append((layer_module, create_combined_hook(layer_idx)))
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(next(self.model.parameters()).device)
        
        # Generate with hooks
        with add_hf_hooks(module_forward_pre_hooks=pre_hooks):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs
            )
        
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove prompt tokens
            new_tokens = output[len(inputs.input_ids[i]):]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts