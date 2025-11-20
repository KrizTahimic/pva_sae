"""
Phase 8.2 Runner: Percentile Threshold Optimizer

Simple runner interface for Phase 8.2 threshold optimization.
"""

from common.config import Config
from common.logging import get_logger
from phase8_2_threshold_optimizer.threshold_optimizer import ThresholdOptimizer

logger = get_logger(__name__)


def run_phase_8_2(config: Config):
    """
    Run Phase 8.2: Optimize threshold selection across percentiles.

    Performs grid search across percentiles [50, 75, 80, 85, 90, 95] from Phase 8.1,
    testing each on the hyperparameter dataset to find the threshold that maximizes
    net benefit (correction_rate - corruption_rate).

    Args:
        config: Global configuration object

    Returns:
        Dict containing optimization results with optimal threshold and comparison data
    """
    logger.info("Starting Phase 8.2: Percentile Threshold Optimizer")

    optimizer = ThresholdOptimizer(config)
    results = optimizer.run()

    logger.info("Phase 8.2 completed successfully")

    return results
