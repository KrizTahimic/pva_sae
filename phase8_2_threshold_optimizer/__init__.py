"""
Phase 8.2: Percentile Threshold Optimizer

Finds the optimal percentile threshold that maximizes net benefit
(correction_rate - corruption_rate) by testing all percentiles from Phase 8.1
on the hyperparameter dataset.
"""

from phase8_2_threshold_optimizer.threshold_optimizer import ThresholdOptimizer
from phase8_2_threshold_optimizer.runner import run_phase_8_2

__all__ = ['ThresholdOptimizer', 'run_phase_8_2']
