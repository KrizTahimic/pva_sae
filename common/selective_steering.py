"""
Shared state for selective steering (Phases 8.2 and 8.3).

The SteeringState class enables real-time threshold checking during generation:
- A predicting hook captures activation on the first new token and sets should_steer
- A steering hook applies steering only if should_steer is True
"""


class SteeringState:
    """
    Shared state between predicting (activation capture) and steering hooks.

    Used to enable real-time threshold checking during generation:
    - Predicting hook captures activation on first new token and sets should_steer flag
    - Steering hook applies steering only if should_steer is True
    """

    def __init__(self, prompt_length: int):
        """
        Initialize steering state.

        Args:
            prompt_length: Length of the prompt (to detect first new token)
        """
        self.prompt_length = prompt_length
        self.first_token_checked = False  # Has predicting activation been captured?
        self.incorrect_pred_activation = None  # Captured incorrect-predicting latent activation
        self.should_steer = False  # Should we apply steering?
