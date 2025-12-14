"""
Phase 2.3 Runner: Compute pile SAE frequencies.

Entry point for computing per-feature activation frequencies on the pile dataset.
"""

from common.config import Config
from common.gpu_utils import get_device
from common.logging import get_logger

from .pile_frequency_computer import PileFrequencyComputer

logger = get_logger("phase2_3.runner", phase="2.3")


def run_phase_2_3(config: Config) -> dict:
    """
    Run Phase 2.3: Pile SAE Frequency Computation.

    Args:
        config: Configuration object

    Returns:
        dict with computation results
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    computer = PileFrequencyComputer(config, device)
    return computer.run()
