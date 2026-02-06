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
        
    def hook_fn(self, module, input):
        """
        Pre-hook function to extract activation at the specified position.

        Uses register_forward_pre_hook pattern (input[0] = resid_pre) to match
        Phase 1's activation extraction, ensuring comparable activations.

        Args:
            module: The layer being hooked
            input: Input to the layer (tuple) - input[0] shape: [batch_size, seq_len, d_model]
        """
        # input is a tuple; first element is the hidden state (resid_pre)
        hidden_state = input[0]

        # Extract activation at the specific position only
        # hidden_state shape: [batch_size=1, seq_len, d_model]
        if hidden_state.shape[1] > self.position:
            # Extract and detach to prevent memory issues
            self.activation = hidden_state[0, self.position, :].detach().clone().cpu()
        else:
            # Position is out of bounds - this shouldn't happen if preprocessing is correct
            self.activation = None