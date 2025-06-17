"""
Unified generation utilities for the PVA-SAE project.

This module provides centralized code generation functionality with robustness features,
temperature variation support, and advanced error handling for all project phases.
"""

import time
from typing import Optional, List, Dict
from dataclasses import dataclass

from common.models import ModelManager
from common.config import Config
from common.utils import torch_memory_cleanup
from common.logging import get_logger
from common.activation_extraction import ActivationData

# No module-level logger - initialized per instance to respect phase context


@dataclass
class GenerationResult:
    """Container for generation results with metadata."""
    prompt: str
    generated_text: str
    generation_time: float
    temperature: float
    success: bool
    error_message: Optional[str] = None
    attempt_count: int = 1
    activations: Optional[Dict[int, ActivationData]] = None


class RobustGenerator:
    """
    Robust code generation with temperature variation and retry logic.
    
    This class provides production-ready generation capabilities with:
    - Automatic retry on failure with exponential backoff
    - Temperature variation for robustness testing
    - Memory management and cleanup
    - Detailed error tracking and logging
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        config: Optional[Config] = None,
        default_max_new_tokens: int = 2000
    ):
        """
        Initialize robust generator.
        
        Args:
            model_manager: Initialized ModelManager instance
            config: Configuration for robustness features
            default_max_new_tokens: Default max tokens for generation
        """
        self.model_manager = model_manager
        self.config = config or Config()
        self.default_max_new_tokens = default_max_new_tokens
        self.logger = get_logger("generation")  # Create logger per instance
        
        if not self.model_manager.model:
            raise RuntimeError("Model not loaded in ModelManager")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        retry_on_failure: bool = True,
        extract_activations: bool = False,
        activation_layers: Optional[List[int]] = None
    ) -> GenerationResult:
        """
        Generate text with robust error handling and retry logic.
        
        Args:
            prompt: Input prompt for generation
            temperature: Generation temperature (None uses model default)
            max_new_tokens: Maximum new tokens to generate
            retry_on_failure: Whether to retry on generation failure
            extract_activations: Whether to extract activations during generation
            activation_layers: Layers to extract from (None uses config default)
            
        Returns:
            GenerationResult with generated text, metadata, and optional activations
        """
        temperature = temperature if temperature is not None else self.model_manager.config.model_temperature
        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        
        # Handle activation layer defaults
        if extract_activations and activation_layers is None:
            activation_layers = self.config.activation_layers
        
        if retry_on_failure:
            return self._generate_with_retry(
                prompt, temperature, max_new_tokens, extract_activations, activation_layers
            )
        else:
            return self._single_generation_attempt(
                prompt, temperature, max_new_tokens, extract_activations, activation_layers
            )
    
    def generate_with_temperature_variation(
        self,
        prompt: str,
        temperatures: List[float],
        max_new_tokens: Optional[int] = None
    ) -> List[GenerationResult]:
        """
        Generate multiple outputs with different temperatures for robustness testing.
        
        Args:
            prompt: Input prompt for generation
            temperatures: List of temperatures to try
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            List of GenerationResult objects, one per temperature
        """
        results = []
        
        for temp in temperatures:
            self.logger.info(f"Generating with temperature={temp}")
            
            # Cleanup memory before each generation
            if self.config.memory_cleanup_frequency > 0:
                torch_memory_cleanup()
            
            result = self.generate(
                prompt=prompt,
                temperature=temp,
                max_new_tokens=max_new_tokens,
                retry_on_failure=True
            )
            results.append(result)
            
            # Small delay between generations to avoid overwhelming the model
            time.sleep(0.1)
        
        return results
    
    def _generate_with_retry(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        extract_activations: bool = False,
        activation_layers: Optional[List[int]] = None
    ) -> GenerationResult:
        """Generate with retry logic on failure."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return self._single_generation_attempt(
                    prompt, temperature, max_new_tokens, extract_activations, activation_layers, attempt + 1
                )
            except Exception as e:
                last_error = e
                retry_delay = self.config.retry_backoff * (2 ** attempt)
                
                self.logger.warning(
                    f"Generation attempt {attempt + 1}/{self.config.max_retries} failed: {str(e)}"
                )
                
                if attempt < self.config.max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                    time.sleep(retry_delay)
                    
                    # Clear GPU cache before retry
                    torch_memory_cleanup()
        
        # All retries failed
        error_msg = f"All {self.config.max_retries} generation attempts failed: {str(last_error)}"
        self.logger.error(error_msg)
        
        return GenerationResult(
            prompt=prompt,
            generated_text="",
            generation_time=0.0,
            temperature=temperature,
            success=False,
            error_message=error_msg,
            attempt_count=self.config.max_retries,
            activations=None
        )
    
    def _single_generation_attempt(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        extract_activations: bool = False,
        activation_layers: Optional[List[int]] = None,
        attempt: int = 1
    ) -> GenerationResult:
        """Single generation attempt with timing and error handling."""
        start_time = time.time()
        
        try:
            if extract_activations and activation_layers:
                # Use hook-enabled generation
                generated_text, activations = self.model_manager.generate_with_activation_hooks(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    activation_layers=activation_layers,
                    hook_type=self.config.activation_hook_type
                )
            else:
                # Regular generation
                generated_text = self.model_manager.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                activations = None
            
            if not generated_text.strip():
                raise ValueError("Generated empty text")
            
            generation_time = time.time() - start_time
            
            return GenerationResult(
                prompt=prompt,
                generated_text=generated_text,
                generation_time=generation_time,
                temperature=temperature,
                success=True,
                attempt_count=attempt,
                activations=activations
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Generation failed: {str(e)}"
            
            return GenerationResult(
                prompt=prompt,
                generated_text="",
                generation_time=generation_time,
                temperature=temperature,
                success=False,
                error_message=error_msg,
                attempt_count=attempt,
                activations=None
            )


def create_generator(
    model_manager: ModelManager,
    config: Optional[Config] = None
) -> RobustGenerator:
    """
    Factory function to create RobustGenerator instance.
    
    Args:
        model_manager: Initialized ModelManager
        config: Optional configuration
        
    Returns:
        RobustGenerator instance
    """
    return RobustGenerator(model_manager, config)