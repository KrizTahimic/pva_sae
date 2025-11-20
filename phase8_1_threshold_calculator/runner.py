"""
Phase 8.1 Runner: Percentile Threshold Calculator

Simple runner interface for Phase 8.1 threshold calculation.
"""

from common.config import Config
from common.logging import get_logger
from phase8_1_threshold_calculator.threshold_calculator import ThresholdCalculator

logger = get_logger(__name__)


def run_phase_8_1(config: Config):
    """
    Run Phase 8.1: Calculate percentile thresholds from Phase 3.6 data.

    Args:
        config: Global configuration object

    Returns:
        Dict containing threshold calculation results
    """
    logger.info("Starting Phase 8.1: Percentile Threshold Calculator")

    calculator = ThresholdCalculator(config)
    results = calculator.run()

    logger.info("Phase 8.1 completed successfully")

    return results
