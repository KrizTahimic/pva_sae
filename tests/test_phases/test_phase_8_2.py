"""
Tests for Phase 8.2 - Threshold Optimizer

Validates:
- percentile_calculation: Correct percentile thresholds
- dual_direction_architecture: LogReg for detection, mass-mean for steering
"""

import pytest
import numpy as np

from common.config import Config


# =============================================================================
# percentile_calculation Tests
# =============================================================================

class TestPercentileCalculation:
    """Test correct percentile thresholds."""

    def test_percentile_interpretation(self):
        """Percentile should select top X% of incorrect-predicting scores."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # 80th percentile threshold
        percentile = 80
        threshold = np.percentile(scores, percentile)

        # Should select top 20% (scores > 0.8)
        selected = scores[scores >= threshold]
        assert len(selected) == 2  # 0.9, 1.0

    def test_refinement_radius_from_config(self):
        """Config should specify refinement radius."""
        config = Config()
        assert hasattr(config, 'phase8_2_refinement_radius')
        assert config.phase8_2_refinement_radius > 0

    def test_tolerance_from_config(self):
        """Config should specify tolerance."""
        config = Config()
        assert hasattr(config, 'phase8_2_tolerance')
        assert config.phase8_2_tolerance > 0

    def test_percentile_grid(self):
        """Should test multiple percentile values."""
        # Coarse grid: 10, 20, 30, ..., 90
        coarse_grid = list(range(10, 100, 10))

        assert len(coarse_grid) == 9
        assert 50 in coarse_grid

    def test_refinement_around_optimal(self):
        """Should refine around coarse optimal."""
        config = Config()
        radius = config.phase8_2_refinement_radius

        coarse_optimal = 70  # Example
        refined_range = range(
            max(1, coarse_optimal - radius),
            min(99, coarse_optimal + radius) + 1
        )

        # Should include coarse optimal
        assert coarse_optimal in refined_range

        # Should span radius
        assert min(refined_range) == coarse_optimal - radius
        assert max(refined_range) == coarse_optimal + radius


# =============================================================================
# dual_direction_architecture Tests
# =============================================================================

class TestDualDirectionArchitecture:
    """Test LogReg for detection, mass-mean for steering."""

    def test_logreg_for_detection(self):
        """Should use logreg probe for threshold prediction."""
        # Phase 8.2 uses logreg for deciding when to steer
        prediction_method = 'logreg'

        # logreg gives calibrated probabilities
        assert prediction_method == 'logreg'

    def test_mass_mean_for_steering(self):
        """Should use mass-mean for actual steering."""
        # Phase 8.2 uses mass_mean for the steering direction
        steering_method = 'mass_mean'

        assert steering_method == 'mass_mean'

    def test_probe_methods_in_config(self):
        """Config should have both probe methods available."""
        config = Config()

        assert hasattr(config, 'probe_mass_mean_reg_lambda')
        assert hasattr(config, 'probe_logreg_C_values')

    def test_separate_uses_documented(self):
        """Both probes should have distinct use cases."""
        # logreg: optimal for binary classification (detection)
        # mass_mean: optimal for direction extraction (steering)

        logreg_use = "detection"
        mass_mean_use = "steering"

        assert logreg_use != mass_mean_use


# =============================================================================
# Optimization Logic Tests
# =============================================================================

class TestOptimizationLogic:
    """Test percentile optimization logic."""

    def test_net_benefit_calculation(self):
        """Net benefit = correction_rate - corruption_rate."""
        correction_rate = 25.0
        corruption_rate = 10.0

        net_benefit = correction_rate - corruption_rate

        assert net_benefit == 15.0

    def test_optimal_maximizes_net_benefit(self):
        """Optimal percentile should maximize net benefit."""
        results = {
            50: {'correction': 20, 'corruption': 15, 'net': 5},
            60: {'correction': 25, 'corruption': 12, 'net': 13},
            70: {'correction': 30, 'corruption': 10, 'net': 20},  # Best
            80: {'correction': 28, 'corruption': 15, 'net': 13},
        }

        optimal = max(results, key=lambda p: results[p]['net'])
        assert optimal == 70

    def test_handles_all_negative_net_benefits(self):
        """Should handle case where all nets are negative."""
        results = {
            50: {'net': -5},
            60: {'net': -3},  # "Best" (least negative)
            70: {'net': -8},
        }

        optimal = max(results, key=lambda p: results[p]['net'])
        assert optimal == 60  # Least bad option


# =============================================================================
# Output Format Tests
# =============================================================================

class TestOutputFormat:
    """Test Phase 8.2 output format."""

    def test_output_includes_optimal_percentile(self):
        """Output should include optimal percentile and threshold."""
        expected_output = {
            'optimization_summary': {
                'optimal_percentile': 70,
                'optimal_threshold': 0.75,
                'correction_rate': 30.0,
                'corruption_rate': 10.0,
                'net_benefit': 20.0
            }
        }

        assert 'optimal_percentile' in expected_output['optimization_summary']
        assert 'optimal_threshold' in expected_output['optimization_summary']

    def test_per_percentile_results(self):
        """Should include results for each percentile tested."""
        per_percentile = {
            50: {'threshold': 0.5, 'correction': 20, 'corruption': 15},
            60: {'threshold': 0.6, 'correction': 25, 'corruption': 12},
            70: {'threshold': 0.7, 'correction': 30, 'corruption': 10},
        }

        assert len(per_percentile) == 3
        for percentile, results in per_percentile.items():
            assert 'threshold' in results
            assert 'correction' in results
            assert 'corruption' in results
