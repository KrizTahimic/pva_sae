"""Simple activation extraction using PyTorch hooks."""

import torch
from pathlib import Path
from typing import Callable, Optional
from common.logging import get_logger
from common.tensor_utils import save_attention

logger = get_logger("common.activation_hooks")


class ActivationExtractor:
    """Simple activation extractor using PyTorch hooks."""
    
    def __init__(self, model: torch.nn.Module, layers: list[int], position: int = -1):
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
                activation = residual_stream[:, self.position, :].detach().clone().cpu()
                self.activations[layer_idx] = activation
                logger.debug(f"Layer {layer_idx}: Captured activation with mean={activation.mean():.6f}, std={activation.std():.6f}, first_val={activation[0,0]:.6f}")
            
            # CRITICAL: Return the input unchanged so steering hooks can still work
            # If we don't return anything (None), PyTorch will use the original unmodified input,
            # which would cancel out any steering modifications from previous hooks
            return input
            
        return hook_fn
    
    def extract(self, input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
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
    
    def get_activations(self) -> dict[int, torch.Tensor]:
        """Get captured activations."""
        return self.activations.copy()


def extract_activations_simple(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    layers: list[int],
    position: int = -1,
    device: str = "cuda"
) -> dict[int, torch.Tensor]:
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


class AttentionExtractor:
    """
    Extract and store attention patterns during generation.
    CRITICAL: Captures ONLY ONCE at the final prompt token, NOT during autoregressive generation.
    """
    
    def __init__(self, model: torch.nn.Module, layers: list[int], position: int = -1):
        """
        Initialize attention extractor for specified layers.

        Args:
            model: The model to extract from
            layers: ONLY the best PVA layers from Phase 2.5 (not all layers!)
            position: Token position to extract (-1 for last prompt token)
        """
        self.model = model
        self.layers = layers  # Should be just 1-2 layers from Phase 2.5!
        self.position = position
        self.attention_patterns = {}
        self.hooks = []
        self.captured = set()  # Track which layers have been captured
        
        logger.info(f"Initialized AttentionExtractor for layers {layers} at position {position}")
    
    def setup_hooks(self) -> None:
        """Register hooks to capture attention during forward pass."""
        self.remove_hooks()  # Clean up any existing hooks
        
        for layer_idx in self.layers:
            try:
                layer = self.model.model.layers[layer_idx]
                # Hook into the attention module's forward method
                hook = layer.self_attn.register_forward_hook(
                    self._attention_hook(layer_idx)
                )
                self.hooks.append(hook)
                logger.debug(f"Set up attention hook for layer {layer_idx}")
            except (AttributeError, IndexError) as e:
                logger.error(f"Failed to set up attention hook for layer {layer_idx}: {e}")
                raise
        
        logger.info(f"Set up attention hooks for {len(self.layers)} layers")
    
    def _attention_hook(self, layer_idx: int):
        """Hook function to capture attention patterns ONCE."""
        def hook(module, input, output):
            # Guard: Already captured (skip subsequent forward passes during autoregressive generation)
            if layer_idx in self.captured:
                return

            try:
                # Guard: Wrong output format
                if not (isinstance(output, tuple) and len(output) >= 2):
                    logger.warning(f"Layer {layer_idx}: Unexpected output format from attention layer")
                    return

                attn_weights = output[1]  # Attention weights are second element

                # Guard: No weights
                if attn_weights is None:
                    logger.warning(f"Layer {layer_idx}: Attention weights were None")
                    return

                # Happy path - capture attention pattern
                # attn_weights shape: (batch, num_heads, seq_len, seq_len)
                # Extract attention FROM the last token position TO all positions
                attn_from_last = attn_weights[:, :, self.position, :].detach().cpu()

                # Store with shape (num_heads, seq_len)
                self.attention_patterns[layer_idx] = attn_from_last.squeeze(0)  # Remove batch dim
                self.captured.add(layer_idx)

                logger.debug(f"Layer {layer_idx}: Captured attention pattern with shape {attn_from_last.shape}")

            except Exception as e:
                logger.error(f"Failed to capture attention for layer {layer_idx}: {e}")

        return hook
    
    def get_attention_patterns(self) -> dict[int, torch.Tensor]:
        """Return captured attention patterns and clear cache."""
        patterns = self.attention_patterns.copy()
        self.attention_patterns.clear()
        self.captured.clear()  # Reset for next task
        return patterns
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.captured.clear()
        self.attention_patterns.clear()
        logger.debug(f"Removed {len(self.hooks)} attention hooks")
    
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


def save_raw_attention_with_boundaries(
    task_id: str,
    attention_tensor: torch.Tensor,
    tokenized_prompt: torch.Tensor,
    tokenizer,
    output_dir: Path,
    layer_idx: int
) -> Path:
    """
    Save raw attention patterns with section boundaries for flexible Phase 6.3 analysis.

    CRITICAL: Saves raw attention tensor (n_heads × seq_len) plus boundaries.
    This allows different aggregation strategies in Phase 6.3.

    Args:
        task_id: Task identifier
        attention_tensor: Raw attention weights (n_heads, seq_len)
        tokenized_prompt: Tokenized prompt tensor
        tokenizer: Tokenizer for decoding
        output_dir: Directory to save attention patterns
        layer_idx: Layer index

    Returns:
        Path to saved tensor file (.safetensors)
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decode prompt to identify sections
    prompt_text = tokenizer.decode(tokenized_prompt.squeeze(0), skip_special_tokens=True)

    # Calculate token boundaries
    boundaries = calculate_section_boundaries(prompt_text, tokenizer, tokenized_prompt)

    # Save using safetensors (tensor) + JSON (metadata)
    save_path = output_dir / f"{task_id}_layer_{layer_idx}_attention"

    metadata = {
        "boundaries": boundaries,
        "prompt_length": len(tokenized_prompt.squeeze(0)),
        "layer": layer_idx,
        "task_id": task_id,
        "prompt_text": prompt_text,
    }

    save_attention(attention_tensor, save_path, metadata=metadata)

    logger.debug(f"Saved raw attention for task {task_id}, layer {layer_idx} to {save_path}")
    return save_path.with_suffix(".safetensors")


def calculate_section_boundaries(prompt_text: str, tokenizer, tokenized_prompt: torch.Tensor) -> dict[str, int]:
    """
    Calculate precise token boundaries for prompt sections.
    
    The prompt structure is:
    {problem_description}\\n\\n{test_cases}\\n\\n# Solution:
    
    Args:
        prompt_text: Decoded prompt text
        tokenizer: Tokenizer for encoding sections
        tokenized_prompt: Original tokenized prompt
    
    Returns:
        dict with 'problem_end', 'test_end' token indices
    """
    boundaries = {}
    
    # Split by double newline to identify sections
    parts = prompt_text.split('\n\n')
    
    if len(parts) >= 1:
        # Problem description is the first part
        problem_tokens = tokenizer.encode(parts[0], add_special_tokens=False)
        boundaries['problem_end'] = len(problem_tokens)
    else:
        boundaries['problem_end'] = 0
    
    if len(parts) >= 2:
        # Test cases are the second part
        # Reconstruct text up to end of test cases
        test_section_text = parts[0] + '\n\n' + parts[1]
        test_tokens = tokenizer.encode(test_section_text, add_special_tokens=False)
        boundaries['test_end'] = len(test_tokens)
    else:
        boundaries['test_end'] = boundaries['problem_end']
    
    # Total length for validation
    boundaries['total_length'] = len(tokenized_prompt.squeeze(0))
    
    # Solution marker starts after test cases
    # The rest is "# Solution:" or similar
    boundaries['solution_start'] = boundaries['test_end']
    
    logger.debug(f"Section boundaries: problem_end={boundaries['problem_end']}, "
                f"test_end={boundaries['test_end']}, total={boundaries['total_length']}")
    
    return boundaries