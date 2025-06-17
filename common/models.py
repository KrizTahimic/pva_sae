"""
Model management utilities for the PVA-SAE project.

This module handles model loading, configuration, and generation
for language models used in the project.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, Union, List, Tuple
import gc
from contextlib import contextmanager

from common.utils import detect_device, get_optimal_dtype, get_memory_usage
from common.config import Config
from common.logging import get_logger
from common.activation_extraction import ActivationData

# No module-level logger - initialized per instance to respect phase context


class ModelManager:
    """Manages language model loading and generation"""
    
    def __init__(self, config: Config):
        """
        Initialize model manager
        
        Args:
            config: Model configuration object (required)
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.dtype = None
        self.logger = get_logger("models")  # Create logger per instance
        
        # Setup device and dtype
        self._setup_device_and_dtype()
    
    def _setup_device_and_dtype(self):
        """Setup device and dtype based on configuration or auto-detection"""
        if self.config.model_device:
            self.device = torch.device(self.config.model_device)
        else:
            self.device = detect_device()
        
        if self.config.model_dtype:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            self.dtype = dtype_map.get(self.config.model_dtype, torch.float32)
        else:
            self.dtype = get_optimal_dtype(self.device)
        
        self.logger.info(f"Using device: {self.device}, dtype: {self.dtype}")
    
    def load_model(self, num_gpus: int = 1):
        """
        Load model and tokenizer with multi-GPU support
        
        Args:
            num_gpus: Number of GPUs to use (1 for single GPU, >1 for DataParallel)
        """
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.model_trust_remote_code
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto" if self.device.type == "cuda" and num_gpus == 1 else None,
            "trust_remote_code": self.config.model_trust_remote_code
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Handle multi-GPU setup
        if self.device.type == "cuda" and num_gpus > 1:
            if torch.cuda.device_count() < num_gpus:
                self.logger.warning(
                    f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available"
                )
                num_gpus = torch.cuda.device_count()
            
            if num_gpus > 1:
                # Move model to primary GPU first
                self.model = self.model.to(self.device)
                # Wrap with DataParallel
                self.model = torch.nn.DataParallel(self.model, device_ids=list(range(num_gpus)))
                self.logger.info(f"Model loaded with DataParallel on {num_gpus} GPUs")
            else:
                self.logger.info(f"Model loaded on single GPU")
        elif self.device.type != "cuda":
            self.model = self.model.to(self.device)
            self.logger.info(f"Model loaded on {self.device}")
        else:
            self.logger.info(f"Model loaded successfully on {self.device}")
    
    def generate(self, 
                 prompt: str, 
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            
        Returns:
            str: Generated text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use provided values or fall back to config
        max_new_tokens = max_new_tokens or self.config.model_max_new_tokens
        temperature = temperature if temperature is not None else self.config.model_temperature
        
        # Automatically determine do_sample based on temperature
        do_sample = temperature > 0.0
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Setup generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def generate_with_activation_hooks(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        activation_layers: List[int] = None,
        hook_type: str = "output"
    ) -> Tuple[str, Dict[int, ActivationData]]:
        """
        Generate text while extracting activations from specified layers in single pass.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            activation_layers: Layers to extract activations from
            hook_type: Type of hook (currently 'output')
            
        Returns:
            Tuple of (generated_text, activations_dict)
            
        Raises:
            RuntimeError: If model not loaded or activation extraction fails
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not activation_layers:
            raise ValueError("activation_layers cannot be empty")
        
        # Use provided values or fall back to config
        max_new_tokens = max_new_tokens or self.config.model_max_new_tokens
        temperature = temperature if temperature is not None else self.config.model_temperature
        
        # Automatically determine do_sample based on temperature
        do_sample = temperature > 0.0
        
        # Tokenize prompt to determine last prompt position
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_length = prompt_inputs['input_ids'].shape[-1]
        last_prompt_position = prompt_length - 1  # 0-indexed
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
        
        # Setup generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Storage for captured activations
        captured_activations = {}
        forward_pass_count = 0
        
        def activation_hook(layer_idx):
            """Create hook function for specific layer."""
            def hook_fn(module, input, output):
                nonlocal forward_pass_count, captured_activations
                
                # Only capture on first forward pass (prompt processing)
                # Note: counter increments during first pass, so check <= 1
                if forward_pass_count <= 1:
                    # HuggingFace models typically return tuples
                    if isinstance(output, tuple):
                        activation = output[0]  # Hidden states are usually first
                    else:
                        activation = output
                    
                    # Extract last prompt token position
                    last_prompt_activation = activation[:, last_prompt_position, :].detach().clone()
                    
                    # Store activation data
                    captured_activations[layer_idx] = ActivationData(
                        layer=layer_idx,
                        position=last_prompt_position,
                        hook_type=hook_type,
                        activations=last_prompt_activation,
                        prompt_count=1
                    )
                    
                    self.logger.info(f"Captured activations for layer {layer_idx} (pass_count={forward_pass_count}, shape={last_prompt_activation.shape})")
            
            return hook_fn
        
        # Hook counter function to track forward passes
        def forward_pass_counter_hook(module, input, output):
            nonlocal forward_pass_count
            forward_pass_count += 1
        
        # Register hooks on specified layers
        handles = []
        
        try:
            # Register forward pass counter on first layer
            first_layer = self._get_layer(activation_layers[0])
            counter_handle = first_layer.register_forward_hook(forward_pass_counter_hook)
            handles.append(counter_handle)
            
            # Register activation hooks on all specified layers
            for layer_idx in activation_layers:
                layer = self._get_layer(layer_idx)
                hook_fn = activation_hook(layer_idx)
                handle = layer.register_forward_hook(hook_fn)
                handles.append(handle)
            
            # Generate with hooks active
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode generated text (excluding prompt)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            )
            
            # Validate that we captured activations
            if len(captured_activations) != len(activation_layers):
                missing_layers = set(activation_layers) - set(captured_activations.keys())
                self.logger.error(f"Activation capture failed - expected {len(activation_layers)} layers, got {len(captured_activations)}")
                self.logger.error(f"Expected layers: {activation_layers}")
                self.logger.error(f"Captured layers: {list(captured_activations.keys())}")
                self.logger.error(f"Final forward_pass_count: {forward_pass_count}")
                raise RuntimeError(f"Failed to capture activations for layers: {missing_layers}")
            
            self.logger.debug(f"Successfully captured activations from {len(captured_activations)} layers after {forward_pass_count} forward passes")
            return generated_text, captured_activations
            
        except Exception as e:
            self.logger.error(f"Hook-enabled generation failed: {e}")
            raise RuntimeError(f"Hook-enabled generation failed: {e}") from e
        
        finally:
            # Clean up all hooks
            for handle in handles:
                handle.remove()
            self.logger.debug(f"Cleaned up {len(handles)} hooks")
    
    def _get_layer(self, layer_idx: int):
        """Get layer module by index with architecture detection."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-style models (GPT-2, GPT-J, etc.)
            return self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA/Gemma-style models
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'layers'):
            # Direct layers attribute (some models)
            return self.model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown model architecture for layer access. Model type: {type(self.model)}")
    
    @contextmanager
    def generation_context(self):
        """Context manager for generation with automatic cleanup"""
        try:
            yield self
        finally:
            # Clear GPU cache if using CUDA
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"status": "not_loaded"}
        
        info = {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "model_size": sum(p.numel() for p in self.model.parameters()) / 1e9,  # in billions
            "memory_usage": get_memory_usage()
        }
        
        return info
    
    def cleanup(self):
        """Cleanup model and free memory"""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        self.logger.info("Model cleanup completed")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()


