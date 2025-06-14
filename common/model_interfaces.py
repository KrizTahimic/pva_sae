"""
Unified model interfaces for the PVA-SAE project.

This module provides high-level interfaces that combine generation and activation
extraction capabilities for seamless use across all project phases.
"""

import torch
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Removed transformer_lens import - only HuggingFace models supported

from common.models import ModelManager
from common.generation import RobustGenerator, GenerationResult
from common.activation_extraction import (
    create_activation_extractor,
    ActivationData
)
from common.config import Config
from common.utils import torch_memory_cleanup


logger = logging.getLogger(__name__)


@dataclass
class GenerationWithActivations:
    """Container for generation results with corresponding activations."""
    generation_result: GenerationResult
    activations: Dict[str, ActivationData]  # key: "layer_X_hooktype"
    
    def get_activation(self, layer: int, hook_type: str = "resid_pre") -> Optional[ActivationData]:
        """Get activation for specific layer and hook type."""
        key = f"layer_{layer}_{hook_type}"
        return self.activations.get(key)


class UnifiedModelInterface:
    """
    Unified interface for model operations combining generation and activation extraction.
    
    This class provides a high-level API for:
    - Code generation with various robustness features
    - Activation extraction at multiple layers
    - Combined generation + activation collection
    - Model steering capabilities
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        config: Optional[Config] = None
    ):
        """
        Initialize unified model interface.
        
        Args:
            model_name: Name/path of the model to load
            device: Device to use
            config: Unified configuration
        """
        self.model_name = model_name
        self.device = device
        
        # Initialize configuration
        self.config = config or Config(model_name=model_name)
        
        # Initialize components
        self.model_manager = None
        self.generator = None
        self.activation_extractor = None
        self.model = None
        self.tokenizer = None
        
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, num_gpus: int = 1, as_transformer_lens: bool = False) -> None:
        """
        Load model with specified configuration.
        
        Args:
            num_gpus: Number of GPUs to use
            as_transformer_lens: Whether to load as TransformerLens model
        """
        if as_transformer_lens:
            self._load_transformer_lens_model()
        else:
            self._load_huggingface_model(num_gpus)
        
        # Initialize generator
        self.generator = RobustGenerator(
            self.model_manager,
            self.config
        )
        
        # Initialize activation extractor
        model_type = "transformerlens" if as_transformer_lens else "huggingface"
        self.activation_extractor = create_activation_extractor(
            model=self.model,
            model_type=model_type,
            device=self.device,
            config=self.config,
            tokenizer=self.tokenizer
        )
        
        self.logger.info(f"Unified interface loaded for {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        extract_activations: bool = False,
        activation_layers: Optional[List[int]] = None,
        activation_position: Union[int, str] = -1,
        hook_type: str = "resid_pre"
    ) -> Union[GenerationResult, GenerationWithActivations]:
        """
        Generate text with optional activation extraction.
        
        Args:
            prompt: Input prompt
            temperature: Generation temperature
            max_new_tokens: Maximum new tokens
            extract_activations: Whether to extract activations
            activation_layers: Layers to extract from (None = all)
            activation_position: Token position for extraction
            hook_type: Hook type for extraction
            
        Returns:
            GenerationResult or GenerationWithActivations if extracting
        """
        if not extract_activations:
            # Simple generation without activation extraction
            return self.generator.generate(
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
        
        # Generation with activation extraction
        return self._generate_with_activations(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            layers=activation_layers,
            position=activation_position,
            hook_type=hook_type
        )
    
    def extract_activations_only(
        self,
        prompts: List[str],
        layers: List[int],
        position: Union[int, str] = -1,
        hook_type: str = "resid_pre"
    ) -> Dict[int, ActivationData]:
        """
        Extract activations without generation.
        
        Args:
            prompts: List of prompts
            layers: Layers to extract from
            position: Token position
            hook_type: Hook type
            
        Returns:
            Dictionary mapping layer to ActivationData
        """
        if not self.activation_extractor:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.activation_extractor.extract_multi_layer(
            prompts=prompts,
            layers=layers,
            position=position,
            hook_type=hook_type
        )
    
    def steer_generation(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer: int,
        coefficient: float = 1.0,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> GenerationResult:
        """
        Generate with model steering applied.
        
        Args:
            prompt: Input prompt
            steering_vector: Steering vector to apply
            layer: Layer to apply steering at
            coefficient: Steering strength
            temperature: Generation temperature
            max_new_tokens: Maximum new tokens
            
        Returns:
            GenerationResult with steered output
        """
        # This is a placeholder for steering functionality
        # Will be implemented based on specific steering approach
        raise NotImplementedError("Model steering will be implemented in Phase 3")
    
    
    def _load_huggingface_model(self, num_gpus: int) -> None:
        """Load HuggingFace model."""
        self.model_manager = ModelManager(self.config)
        self.model_manager.load_model(num_gpus=num_gpus)
        self.model = self.model_manager.model
        self.tokenizer = self.model_manager.tokenizer
    
    # Removed _load_transformer_lens_model (YAGNI) - only HuggingFace models supported
    
    def _generate_with_activations(
        self,
        prompt: str,
        temperature: Optional[float],
        max_new_tokens: Optional[int],
        layers: Optional[List[int]],
        position: Union[int, str],
        hook_type: str
    ) -> GenerationWithActivations:
        """Generate text and extract activations in one pass."""
        # For now, we'll do this in two steps
        # In future, could optimize to extract during generation
        
        # Step 1: Generate
        generation_result = self.generator.generate(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        if not generation_result.success:
            # Return empty activations if generation failed
            return GenerationWithActivations(
                generation_result=generation_result,
                activations={}
            )
        
        # Step 2: Extract activations from the full prompt + generated text
        full_text = prompt + generation_result.generated_text
        
        # Determine layers to extract from
        if layers is None:
            # Extract from all layers
            if hasattr(self.model, 'cfg'):
                layers = list(range(self.model.cfg.n_layers))
            else:
                # Estimate based on model size
                layers = list(range(32))  # Default assumption
        
        # Extract activations
        activation_data = self.activation_extractor.extract_multi_layer(
            prompts=[full_text],
            layers=layers,
            position=position,
            hook_type=hook_type
        )
        
        # Format activations with descriptive keys
        formatted_activations = {}
        for layer, data in activation_data.items():
            key = f"layer_{layer}_{hook_type}"
            formatted_activations[key] = data
        
        return GenerationWithActivations(
            generation_result=generation_result,
            activations=formatted_activations
        )


class ModelSteeringInterface:
    """
    Interface for model steering experiments.
    
    Provides high-level API for steering language models using
    latent directions identified from SAE analysis.
    """
    
    def __init__(self, unified_interface: UnifiedModelInterface):
        """
        Initialize steering interface.
        
        Args:
            unified_interface: Initialized UnifiedModelInterface
        """
        self.interface = unified_interface
        self.logger = logging.getLogger(__name__)
    
    def apply_pva_steering(
        self,
        prompt: str,
        pva_direction: torch.Tensor,
        layer: int,
        coefficients: List[float],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> Dict[float, GenerationResult]:
        """
        Apply PVA steering with multiple coefficient values.
        
        Args:
            prompt: Input prompt
            pva_direction: PVA latent direction vector
            layer: Layer to apply steering
            coefficients: List of steering coefficients to try
            temperature: Generation temperature
            max_new_tokens: Maximum new tokens
            
        Returns:
            Dictionary mapping coefficient to generation result
        """
        results = {}
        
        for coeff in coefficients:
            self.logger.info(f"Applying PVA steering with coefficient={coeff}")
            
            # Apply steering and generate
            result = self._steer_and_generate(
                prompt=prompt,
                steering_vector=pva_direction,
                layer=layer,
                coefficient=coeff,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
            
            results[coeff] = result
        
        return results
    
    def _steer_and_generate(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer: int,
        coefficient: float,
        temperature: Optional[float],
        max_new_tokens: Optional[int]
    ) -> GenerationResult:
        """Apply steering and generate (placeholder for implementation)."""
        # This will be implemented based on the specific steering approach
        # For now, return a placeholder
        return self.interface.generator.generate(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )


def create_unified_interface(
    model_name: str,
    device: str = "cuda",
    num_gpus: int = 1,
    **config_kwargs
) -> UnifiedModelInterface:
    """
    Factory function to create and initialize unified model interface.
    
    Args:
        model_name: Model name/path
        device: Device to use
        num_gpus: Number of GPUs
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Initialized UnifiedModelInterface
    """
    interface = UnifiedModelInterface(
        model_name=model_name,
        device=device,
        model_config=config_kwargs.get('model_config'),
        robustness_config=config_kwargs.get('robustness_config'),
        activation_config=config_kwargs.get('activation_config')
    )
    
    interface.load_model(num_gpus=num_gpus)
    
    return interface