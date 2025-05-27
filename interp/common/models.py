"""
Model management utilities for the PVA-SAE project.

This module handles model loading, configuration, and generation
for language models used in the project.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import Optional, Dict, Any, Union
import logging
import gc
from contextlib import contextmanager

from .utils import detect_device, get_optimal_dtype, get_memory_usage
from .config import ModelConfiguration


class ModelManager:
    """Manages language model loading and generation"""
    
    def __init__(self, config: Optional[ModelConfiguration] = None):
        """
        Initialize model manager
        
        Args:
            config: Model configuration object
        """
        self.config = config or ModelConfiguration()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.dtype = None
        self.logger = logging.getLogger(__name__)
        
        # Setup device and dtype
        self._setup_device_and_dtype()
    
    def _setup_device_and_dtype(self):
        """Setup device and dtype based on configuration or auto-detection"""
        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = detect_device()
        
        if self.config.dtype:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            self.dtype = dtype_map.get(self.config.dtype, torch.float32)
        else:
            self.dtype = get_optimal_dtype(self.device)
        
        self.logger.info(f"Using device: {self.device}, dtype: {self.dtype}")
    
    def load_model(self):
        """Load model and tokenizer"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Load model
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto" if self.device.type == "cuda" else None,
            "trust_remote_code": self.config.trust_remote_code
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        
        self.logger.info(f"Model loaded successfully on {self.device}")
    
    def generate(self, 
                 prompt: str, 
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 do_sample: Optional[bool] = None,
                 stream: bool = False) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            stream: Whether to stream output
            
        Returns:
            str: Generated text
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use provided values or fall back to config
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Setup generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Add streamer if requested
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def batch_generate(self, 
                       prompts: list[str],
                       **generation_kwargs) -> list[str]:
        """
        Generate text for multiple prompts
        
        Args:
            prompts: List of input prompts
            **generation_kwargs: Additional generation parameters
            
        Returns:
            list[str]: List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **generation_kwargs)
            results.append(result)
        return results
    
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


class ModelPool:
    """Manages a pool of models for experiments"""
    
    def __init__(self):
        """Initialize model pool"""
        self.models = {}
        self.logger = logging.getLogger(__name__)
    
    def add_model(self, name: str, config: ModelConfiguration) -> ModelManager:
        """
        Add a model to the pool
        
        Args:
            name: Name for the model in the pool
            config: Model configuration
            
        Returns:
            ModelManager: The model manager instance
        """
        if name in self.models:
            self.logger.warning(f"Model {name} already exists, replacing...")
            self.models[name].cleanup()
        
        manager = ModelManager(config)
        manager.load_model()
        self.models[name] = manager
        
        return manager
    
    def get_model(self, name: str) -> Optional[ModelManager]:
        """Get a model from the pool"""
        return self.models.get(name)
    
    def remove_model(self, name: str):
        """Remove a model from the pool"""
        if name in self.models:
            self.models[name].cleanup()
            del self.models[name]
    
    def cleanup_all(self):
        """Cleanup all models in the pool"""
        for name, model in self.models.items():
            model.cleanup()
        self.models.clear()