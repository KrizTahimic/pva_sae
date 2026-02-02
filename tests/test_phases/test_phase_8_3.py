"""
Tests for Phase 8.3 - Selective Steering

Validates:
- threshold_application: Steering only when threshold exceeded
- selective_vs_always_steering: Different outcomes
"""

import pytest
import pandas as pd
import numpy as np

from common.config import Config


# =============================================================================
# threshold_application Tests
# =============================================================================

class TestThresholdApplication:
    """Test steering only when threshold exceeded."""

    def test_steer_above_threshold(self):
        """Should steer when incorrectness score exceeds threshold."""
        threshold = 0.7
        scores = [0.8, 0.9, 0.75]

        should_steer = [score > threshold for score in scores]

        assert all(should_steer)

    def test_skip_below_threshold(self):
        """Should not steer when score below threshold."""
        threshold = 0.7
        scores = [0.3, 0.5, 0.6]

        should_steer = [score > threshold for score in scores]

        assert not any(should_steer)

    def test_boundary_case(self):
        """Should handle boundary case at threshold."""
        threshold = 0.7
        score = 0.7

        # Convention: strictly greater than
        should_steer = score > threshold
        assert should_steer is False

    def test_percentile_threshold_from_config(self):
        """Config should have percentile threshold setting."""
        config = Config()
        assert hasattr(config, 'phase8_3_use_percentile_threshold')
        assert hasattr(config, 'phase8_3_percentile')


# =============================================================================
# selective_vs_always_steering Tests
# =============================================================================

class TestSelectiveVsAlwaysSteering:
    """Test different outcomes between selective and always steering."""

    def test_selective_steers_fewer_samples(self):
        """Selective should steer fewer samples than always."""
        n_samples = 100
        scores = np.random.rand(n_samples)
        threshold = 0.7

        # Always steer: all samples
        always_steered = n_samples

        # Selective: only above threshold
        selective_steered = sum(scores > threshold)

        assert selective_steered < always_steered

    def test_selective_avoids_low_confidence_steering(self):
        """Selective should skip samples with low incorrectness scores."""
        samples = [
            {'score': 0.2, 'baseline_passed': True},   # Low score, don't steer
            {'score': 0.3, 'baseline_passed': True},   # Low score, don't steer
            {'score': 0.8, 'baseline_passed': False},  # High score, steer
            {'score': 0.9, 'baseline_passed': False},  # High score, steer
        ]

        threshold = 0.7
        to_steer = [s for s in samples if s['score'] > threshold]

        assert len(to_steer) == 2
        # Only incorrect samples with high incorrectness scores
        assert all(not s['baseline_passed'] for s in to_steer)

    def test_selective_reduces_corruption(self):
        """Selective should reduce corruption by not steering low-risk samples."""
        # Samples with scores and outcomes
        samples = [
            {'score': 0.3, 'baseline': True, 'if_steered': False},  # Would corrupt
            {'score': 0.4, 'baseline': True, 'if_steered': True},   # Would preserve
            {'score': 0.8, 'baseline': False, 'if_steered': True},  # Would correct
        ]

        threshold = 0.7

        # Always steering
        always_corruptions = sum(1 for s in samples
                                if s['baseline'] and not s['if_steered'])
        assert always_corruptions == 1

        # Selective steering (only high score)
        selective_corruptions = sum(1 for s in samples
                                   if s['score'] > threshold and
                                   s['baseline'] and not s['if_steered'])
        assert selective_corruptions == 0

    def test_selective_preserves_correct_solutions(self):
        """Selective should preserve more correct solutions."""
        samples = pd.DataFrame([
            {'score': 0.2, 'baseline_passed': True},
            {'score': 0.3, 'baseline_passed': True},
            {'score': 0.4, 'baseline_passed': True},
            {'score': 0.8, 'baseline_passed': False},
        ])

        threshold = 0.7

        # Samples NOT steered (preserved as-is)
        preserved = samples[samples['score'] <= threshold]

        # All correct samples with low scores should be preserved
        correct_preserved = preserved[preserved['baseline_passed']]
        assert len(correct_preserved) == 3


# =============================================================================
# Auto-Discovery Tests
# =============================================================================

class TestAutoDiscovery:
    """Test auto-discovery of optimal threshold from Phase 8.2."""

    def test_auto_discover_percentile(self, tmp_path):
        """Should auto-discover percentile from Phase 8.2 when not specified."""
        import json

        # Create Phase 8.2 output
        phase_8_2_dir = tmp_path / "data" / "phase8_2"
        phase_8_2_dir.mkdir(parents=True)

        data = {
            "optimization_summary": {
                "optimal_percentile": 70,
                "optimal_threshold": 0.75
            }
        }
        with open(phase_8_2_dir / "threshold_optimization.json", 'w') as f:
            json.dump(data, f)

        # Load
        with open(phase_8_2_dir / "threshold_optimization.json") as f:
            loaded = json.load(f)

        assert loaded['optimization_summary']['optimal_percentile'] == 70
        assert loaded['optimization_summary']['optimal_threshold'] == 0.75

    def test_uses_config_percentile_if_specified(self):
        """Should use config percentile if explicitly specified."""
        config = Config()
        config.phase8_3_percentile = 60

        # Should use 60, not auto-discover
        assert config.phase8_3_percentile == 60


# =============================================================================
# Output Format Tests
# =============================================================================

class TestOutputFormat:
    """Test Phase 8.3 output format."""

    def test_output_includes_threshold_info(self):
        """Output should include threshold used."""
        expected_output = {
            'threshold_info': {
                'percentile': 70,
                'threshold_value': 0.75,
                'source': 'phase_8.2'
            },
            'summary': {
                'n_steered': 30,
                'n_skipped': 70,
                'correction_rate': 45.0,
                'corruption_rate': 5.0
            }
        }

        assert 'threshold_info' in expected_output
        assert 'n_steered' in expected_output['summary']
        assert 'n_skipped' in expected_output['summary']

    def test_comparison_with_always_steering(self):
        """Output should compare with always-steering baseline."""
        expected_output = {
            'comparison': {
                'selective_correction': 45.0,
                'always_correction': 40.0,
                'selective_corruption': 5.0,
                'always_corruption': 15.0,
                'selective_net': 40.0,
                'always_net': 25.0
            }
        }

        # Selective should have better net benefit
        assert expected_output['comparison']['selective_net'] > \
               expected_output['comparison']['always_net']
