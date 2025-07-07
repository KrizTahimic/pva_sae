"""
Specialized activation hook for extracting activations at specific token positions.
Used for pile dataset processing where we only need activation at the random word position.
"""

import torch


class PileActivationHook:
    """Extract activation at a specific token position during forward pass."""
    
    def __init__(self, position: int):
        """
        Initialize hook for specific position extraction.
        
        Args:
            position: Token position to extract activation from (0-indexed)
        """
        self.position = position
        self.activation = None
        
    def hook_fn(self, module, input, output):
        """
        Hook function to extract activation at the specified position.
        
        Args:
            module: The layer being hooked
            input: Input to the layer (tuple)
            output: Output from the layer - shape [batch_size, seq_len, d_model]
        """
        # Handle different output types (some layers return tuples)
        if isinstance(output, tuple):
            output = output[0]
            
        # Extract activation at the specific position only
        # output shape: [batch_size=1, seq_len, d_model]
        if output.shape[1] > self.position:
            # Extract and detach to prevent memory issues
            self.activation = output[0, self.position, :].detach().cpu()
        else:
            # Position is out of bounds - this shouldn't happen if preprocessing is correct
            self.activation = None