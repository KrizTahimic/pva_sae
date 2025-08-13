"""Simple activation extraction using PyTorch hooks."""

import torch
from typing import Dict, List, Callable
from common.logging import get_logger

logger = get_logger("common_simplified.activation_hooks")


class ActivationExtractor:
    """Simple activation extractor using PyTorch hooks."""
    
    def __init__(self, model: torch.nn.Module, layers: List[int], position: int = -1):
        """
        Initialize activation extractor.
        
        Args:
            model: The model to extract activations from
            layers: List of layer indices to extract
            position: Token position to extract (-1 for last token)
        """
        self.model = model
        self.layers = layers
        self.position = position
        self.hooks = []
        self.activations = {}
        
    def setup_hooks(self) -> None:
        """Attach hooks to specified layers to capture residual stream."""
        self.remove_hooks()  # Clean up any existing hooks
        
        for layer_idx in self.layers:
            try:
                # Get the layer module
                layer = self.model.model.layers[layer_idx]
                
                # Register pre-hook to capture the residual stream BEFORE layer processing
                # This gives us the residual stream activations that feed into the layer
                hook = layer.register_forward_pre_hook(self._create_hook(layer_idx))
                self.hooks.append(hook)
            except (AttributeError, IndexError) as e:
                logger.error(f"Failed to access layer {layer_idx}: {type(e).__name__}: {e}")
                logger.error(f"Model type: {type(self.model).__name__}")
                logger.error(f"Model has 'model' attribute: {hasattr(self.model, 'model')}")
                if hasattr(self.model, 'model'):
                    logger.error(f"Model.model has 'layers' attribute: {hasattr(self.model.model, 'layers')}")
                    if hasattr(self.model.model, 'layers'):
                        logger.error(f"Number of layers: {len(self.model.model.layers)}")
                raise
            
        logger.info(f"Set up hooks for {len(self.layers)} layers to capture residual stream")
    
    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for a specific layer."""
        def hook_fn(module, input):
            # Only capture if we haven't captured for this layer yet
            # This ensures we only capture activations from the first forward pass (prompt processing)
            # and ignore subsequent forward passes during autoregressive generation
            if layer_idx not in self.activations:
                # For pre-hooks, input is a tuple with the residual stream
                # input[0] is the residual stream tensor: (batch_size, seq_len, hidden_size)
                residual_stream = input[0]
                
                # Get activation at specified position (usually -1 for last token)
                # This extracts the residual stream activation for the last token in the prompt
                activation = residual_stream[:, self.position, :].detach()
                self.activations[layer_idx] = activation
            
        return hook_fn
    
    def extract(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract activations for given input.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Dictionary mapping layer index to activation tensor
        """
        self.activations.clear()
        
        # Run forward pass (no gradients needed)
        with torch.no_grad():
            _ = self.model(input_ids)
        
        # Return copy of activations
        return self.activations.copy()
    
    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()
    
    def __enter__(self):
        """Enter context manager - setup hooks."""
        self.setup_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - remove hooks."""
        self.remove_hooks()
        return False
    
    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Get captured activations."""
        return self.activations.copy()


def extract_activations_simple(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    layers: List[int],
    position: int = -1,
    device: str = "cuda"
) -> Dict[int, torch.Tensor]:
    """
    Simple function to extract activations from text.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        text: Input text
        layers: Layers to extract from
        position: Token position (-1 for last)
        device: Device to run on
        
    Returns:
        Dictionary of layer -> activation tensor
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    
    # Create extractor and setup hooks
    extractor = ActivationExtractor(model, layers, position)
    extractor.setup_hooks()
    
    # Extract activations
    activations = extractor.extract(input_ids)
    
    # Clean up
    extractor.remove_hooks()
    
    return activations