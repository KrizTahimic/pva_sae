"""
Unified activation extraction utilities for the PVA-SAE project.

This module provides centralized activation collection functionality with support for
multiple model types (TransformerLens, HuggingFace) and memory-efficient extraction.
"""

import torch
import torch.nn as nn
import contextlib
import logging
from typing import List, Tuple, Dict, Union, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import gc

from transformer_lens import HookedTransformer
import numpy as np

from common.utils import torch_memory_cleanup, torch_no_grad_and_cleanup
from common.config import Config


logger = logging.getLogger(__name__)


@dataclass
class ActivationData:
    """Container for extracted activation data with metadata."""
    layer: int
    position: Union[int, str]  # -1 for last, 'all' for all positions
    hook_type: str  # 'resid_pre', 'resid_mid', 'resid_post', etc.
    activations: torch.Tensor
    prompt_count: int
    
    @property
    def shape(self) -> torch.Size:
        """Get activation tensor shape."""
        return self.activations.shape
    
    def to_device(self, device: str) -> 'ActivationData':
        """Move activations to specified device."""
        return ActivationData(
            layer=self.layer,
            position=self.position,
            hook_type=self.hook_type,
            activations=self.activations.to(device),
            prompt_count=self.prompt_count
        )


def save_activation_data(data: ActivationData, filepath: Path) -> None:
    """
    Save activation data to disk using numpy compressed format.
    
    Args:
        data: ActivationData object to save
        filepath: Path to save file (.npz format)
        
    Raises:
        IOError: If file cannot be written
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert tensor to numpy for efficient storage
        activations_np = data.activations.detach().cpu().numpy()
        
        # Save both data and metadata
        np.savez_compressed(
            filepath,
            activations=activations_np,
            layer=data.layer,
            position=str(data.position),  # Convert to string for JSON compatibility
            hook_type=data.hook_type,
            prompt_count=data.prompt_count
        )
        logger.debug(f"Saved activation data to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save activation data to {filepath}: {e}")
        raise IOError(f"Cannot write to {filepath}: {e}")


def load_activation_data(filepath: Path) -> ActivationData:
    """
    Load activation data from disk.
    
    Args:
        filepath: Path to saved activation file
        
    Returns:
        ActivationData object
        
    Raises:
        FileNotFoundError: If activation file doesn't exist
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Activation file not found: {filepath}")
    
    try:
        # Load numpy archive
        data = np.load(filepath)
        
        # Reconstruct tensor from numpy
        activations = torch.from_numpy(data['activations'])
        
        # Parse position (convert back from string if needed)
        position = data['position'].item()
        if position.isdigit() or position == '-1':
            position = int(position)
        
        # Reconstruct ActivationData
        activation_data = ActivationData(
            layer=int(data['layer'].item()),
            position=position,
            hook_type=str(data['hook_type'].item()),
            activations=activations,
            prompt_count=int(data['prompt_count'].item())
        )
        
        logger.debug(f"Loaded activation data from {filepath}")
        return activation_data
        
    except Exception as e:
        logger.error(f"Failed to load activation data from {filepath}: {e}")
        raise ValueError(f"Invalid activation file format: {e}")


class ActivationCache:
    """
    Thread-safe activation cache with memory management.
    
    Provides efficient caching of activations with automatic cleanup
    and memory tracking to prevent OOM errors.
    """
    
    def __init__(self, max_cache_size_gb: float = 10.0):
        """
        Initialize activation cache.
        
        Args:
            max_cache_size_gb: Maximum cache size in GB before automatic cleanup
        """
        self.cache: Dict[str, torch.Tensor] = {}
        self.max_cache_size_gb = max_cache_size_gb
        self.logger = logging.getLogger(__name__)
    
    def store(self, key: str, activation: torch.Tensor) -> None:
        """Store activation with given key."""
        # Check cache size before storing
        if self._get_cache_size_gb() > self.max_cache_size_gb * 0.9:
            self.logger.warning(f"Cache approaching size limit ({self.max_cache_size_gb}GB), clearing...")
            self.clear()
        
        self.cache[key] = activation.detach().clone()
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get activation by key."""
        return self.cache.get(key)
    
    def batch_store(self, activations: Dict[str, torch.Tensor]) -> None:
        """Store multiple activations at once."""
        for key, activation in activations.items():
            self.store(key, activation)
    
    def clear(self) -> None:
        """Clear all cached activations and free memory."""
        self.cache.clear()
        torch_memory_cleanup()
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self.cache.keys())
    
    def _get_cache_size_gb(self) -> float:
        """Calculate current cache size in GB."""
        total_bytes = sum(
            tensor.element_size() * tensor.nelement() 
            for tensor in self.cache.values()
        )
        return total_bytes / (1024 ** 3)
    
    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self.cache:
            raise KeyError(f"Key '{key}' not found in cache")
        return self.cache[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.cache
    
    def __len__(self) -> int:
        return len(self.cache)


class BaseActivationExtractor:
    """Base class for activation extraction with common functionality."""
    
    def __init__(
        self,
        model: Any,
        device: str,
        config: Optional[Config] = None
    ):
        """
        Initialize base extractor.
        
        Args:
            model: Model instance (HookedTransformer or HuggingFace model)
            device: Device to use for extraction
            config: Activation extraction configuration
        """
        self.model = model
        self.device = device
        self.config = config or Config()
        self.cache = ActivationCache(max_cache_size_gb=self.config.activation_max_cache_gb)
        self.logger = logging.getLogger(__name__)
    
    def clear_cache(self) -> None:
        """Clear activation cache."""
        self.cache.clear()
    
    @torch_no_grad_and_cleanup
    def extract_activations(
        self,
        prompts: List[str],
        layer_idx: int,
        position: Union[int, str] = -1,
        hook_type: str = "resid_pre"
    ) -> ActivationData:
        """
        Extract activations for given prompts at specified layer and position.
        
        Args:
            prompts: List of text prompts
            layer_idx: Layer index to extract from
            position: Token position (-1 for last, 'all' for all tokens)
            hook_type: Type of hook (implementation-specific)
            
        Returns:
            ActivationData containing extracted activations
        """
        raise NotImplementedError("Subclasses must implement extract_activations")
    
    def extract_multi_layer(
        self,
        prompts: List[str],
        layers: List[int],
        position: Union[int, str] = -1,
        hook_type: str = "resid_pre"
    ) -> Dict[int, ActivationData]:
        """
        Extract activations from multiple layers.
        
        Args:
            prompts: List of text prompts
            layers: List of layer indices
            position: Token position
            hook_type: Type of hook
            
        Returns:
            Dictionary mapping layer index to ActivationData
        """
        results = {}
        
        for layer_idx in layers:
            self.logger.info(f"Extracting from layer {layer_idx}")
            
            # Clear cache between layers to manage memory
            if self.config.activation_clear_cache_between_layers:
                self.clear_cache()
            
            activation_data = self.extract_activations(
                prompts=prompts,
                layer_idx=layer_idx,
                position=position,
                hook_type=hook_type
            )
            results[layer_idx] = activation_data
        
        return results


class TransformerLensExtractor(BaseActivationExtractor):
    """Activation extractor for TransformerLens models."""
    
    def __init__(
        self,
        model: HookedTransformer,
        device: str,
        config: Optional[Config] = None
    ):
        """Initialize TransformerLens-specific extractor."""
        super().__init__(model, device, config)
        
        if not isinstance(model, HookedTransformer):
            raise TypeError("Model must be a HookedTransformer instance")
        
        self.logger.info("Initialized TransformerLens activation extractor")
    
    @contextlib.contextmanager
    def _hook_context(self, hooks: List[Tuple[str, Callable]]):
        """Context manager for safe hook management with TransformerLens."""
        try:
            # Clear any existing hooks first
            self.model.reset_hooks()
            
            # Add new hooks
            for hook_name, hook_fn in hooks:
                self.model.add_hook(hook_name, hook_fn)
            
            yield
            
        finally:
            # Always clean up hooks
            self.model.reset_hooks()
    
    @torch_no_grad_and_cleanup
    def extract_activations(
        self,
        prompts: List[str],
        layer_idx: int,
        position: Union[int, str] = -1,
        hook_type: str = "resid_pre"
    ) -> ActivationData:
        """Extract activations using TransformerLens hooks."""
        if not prompts:
            raise ValueError("No prompts provided")
        
        # Clear cache for this extraction
        self.cache.clear()
        
        # Construct hook name based on TransformerLens conventions
        hook_name = f"blocks.{layer_idx}.hook_{hook_type}"
        
        def extraction_hook(activation, hook):
            """Hook function for TransformerLens."""
            # Handle different position specifications
            if position == 'all':
                extracted = activation.detach().clone()
            elif isinstance(position, int):
                pos = position if position != -1 else activation.shape[1] - 1
                extracted = activation[:, pos, :].detach().clone()
            else:
                raise ValueError(f"Invalid position: {position}")
            
            # Store each sample separately for easier concatenation
            for i in range(extracted.shape[0]):
                cache_key = f"sample_{len(self.cache.cache)}"
                self.cache.store(cache_key, extracted[i:i+1])
            
            return activation
        
        # Process prompts with hook
        with self._hook_context([(hook_name, extraction_hook)]):
            self._process_prompts_batch(prompts)
        
        # Collect and concatenate cached activations
        activations = self._collect_cached_activations()
        
        return ActivationData(
            layer=layer_idx,
            position=position,
            hook_type=hook_type,
            activations=activations,
            prompt_count=len(prompts)
        )
    
    def _process_prompts_batch(self, prompts: List[str]) -> None:
        """Process prompts in batches for memory efficiency."""
        batch_size = self.config.activation_batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Run model forward pass
            self.model(batch_prompts)
            
            # Memory cleanup after each batch
            if self.config.activation_cleanup_after_batch:
                torch_memory_cleanup()
    
    def _collect_cached_activations(self) -> torch.Tensor:
        """Collect and concatenate all cached activations."""
        # Get all cache keys in order
        keys = sorted(self.cache.keys(), key=lambda x: int(x.split('_')[1]))
        
        if not keys:
            raise RuntimeError("No activations were cached")
        
        # Concatenate all activations
        activations = []
        for key in keys:
            activations.append(self.cache.get(key))
        
        return torch.cat(activations, dim=0)


class HuggingFaceExtractor(BaseActivationExtractor):
    """Activation extractor for HuggingFace models."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        config: Optional[Config] = None
    ):
        """
        Initialize HuggingFace-specific extractor.
        
        Args:
            model: HuggingFace model instance
            tokenizer: HuggingFace tokenizer
            device: Device for extraction
            config: Extraction configuration
        """
        super().__init__(model, device, config)
        self.tokenizer = tokenizer
        self.logger.info("Initialized HuggingFace activation extractor")
    
    @contextlib.contextmanager
    def _hook_context(self, layer_idx: int, hook_fn: Callable):
        """Context manager for HuggingFace model hooks."""
        handles = []
        
        try:
            # Get the target layer
            if hasattr(self.model, 'transformer'):
                # GPT-style models
                layer = self.model.transformer.h[layer_idx]
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # LLaMA/Gemma-style models
                layer = self.model.model.layers[layer_idx]
            else:
                raise ValueError(f"Unknown model architecture for hook registration")
            
            # Register hook on the layer output
            handle = layer.register_forward_hook(hook_fn)
            handles.append(handle)
            
            yield
            
        finally:
            # Remove all hooks
            for handle in handles:
                handle.remove()
    
    @torch_no_grad_and_cleanup
    def extract_activations(
        self,
        prompts: List[str],
        layer_idx: int,
        position: Union[int, str] = -1,
        hook_type: str = "output"
    ) -> ActivationData:
        """Extract activations using PyTorch hooks on HuggingFace models."""
        if not prompts:
            raise ValueError("No prompts provided")
        
        # Clear cache for this extraction
        self.cache.clear()
        
        def extraction_hook(module, input, output):
            """Hook function for HuggingFace models."""
            # HuggingFace models typically return tuples
            if isinstance(output, tuple):
                activation = output[0]  # Hidden states are usually first
            else:
                activation = output
            
            # Handle position extraction
            if position == 'all':
                extracted = activation.detach().clone()
            elif isinstance(position, int):
                pos = position if position != -1 else activation.shape[1] - 1
                extracted = activation[:, pos, :].detach().clone()
            else:
                raise ValueError(f"Invalid position: {position}")
            
            # Store activations
            for i in range(extracted.shape[0]):
                cache_key = f"sample_{len(self.cache.cache)}"
                self.cache.store(cache_key, extracted[i:i+1])
        
        # Process prompts with hook
        with self._hook_context(layer_idx, extraction_hook):
            self._process_prompts_batch(prompts)
        
        # Collect cached activations
        activations = self._collect_cached_activations()
        
        return ActivationData(
            layer=layer_idx,
            position=position,
            hook_type=hook_type,
            activations=activations,
            prompt_count=len(prompts)
        )
    
    def _process_prompts_batch(self, prompts: List[str]) -> None:
        """Process prompts in batches."""
        batch_size = self.config.activation_batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.activation_max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            _ = self.model(**inputs)
            
            # Memory cleanup
            if self.config.activation_cleanup_after_batch:
                torch_memory_cleanup()
    
    def _collect_cached_activations(self) -> torch.Tensor:
        """Collect and concatenate cached activations."""
        keys = sorted(self.cache.keys(), key=lambda x: int(x.split('_')[1]))
        
        if not keys:
            raise RuntimeError("No activations were cached")
        
        activations = []
        for key in keys:
            activations.append(self.cache.get(key))
        
        return torch.cat(activations, dim=0)


def create_activation_extractor(
    model: Any,
    model_type: str = "auto",
    device: str = "cuda",
    config: Optional[Config] = None,
    **kwargs
) -> BaseActivationExtractor:
    """
    Factory function to create appropriate activation extractor.
    
    Args:
        model: Model instance
        model_type: Type of model ("transformerlens", "huggingface", "auto")
        device: Device for extraction
        config: Extraction configuration
        **kwargs: Additional arguments (e.g., tokenizer for HuggingFace)
        
    Returns:
        Appropriate activation extractor instance
    """
    if model_type == "auto":
        # Auto-detect model type
        if isinstance(model, HookedTransformer):
            model_type = "transformerlens"
        else:
            model_type = "huggingface"
    
    if model_type == "transformerlens":
        return TransformerLensExtractor(model, device, config)
    elif model_type == "huggingface":
        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None:
            raise ValueError("HuggingFace extractor requires tokenizer")
        return HuggingFaceExtractor(model, tokenizer, device, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")