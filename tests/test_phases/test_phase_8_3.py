"""
Tests for Phase 8.3 - Selective Steering

Validates:
- SteeringState: Shared state initialization and mutability
- _calculate_correction_metrics: Production metric calculation for correction experiment
- _calculate_preservation_metrics: Production metric calculation for preservation experiment
- _calculate_combined_metrics: Combined metrics across both experiments
- config: Phase 8.3 configuration values
"""

import pytest
import numpy as np

from common.config import Config
from phase8_3_selective_steering.selective_steering_analyzer import (
    SelectiveSteeringAnalyzer,
    SteeringState,
)


# =============================================================================
# SteeringState Tests
# =============================================================================

class TestSteeringState:
    """Test SteeringState initialization and defaults."""

    def test_initial_state(self):
        """SteeringState should start with correct defaults."""
        state = SteeringState(prompt_length=42)

        assert state.prompt_length == 42
        assert state.first_token_checked is False
        assert state.incorrect_pred_activation is None
        assert state.should_steer is False

    def test_prompt_length_stored(self):
        """prompt_length should be stored for first-token detection."""
        state = SteeringState(prompt_length=128)
        assert state.prompt_length == 128

    def test_state_mutable(self):
        """State should be mutable for hook communication."""
        state = SteeringState(prompt_length=10)

        state.first_token_checked = True
        state.incorrect_pred_activation = 0.85
        state.should_steer = True

        assert state.first_token_checked is True
        assert state.incorrect_pred_activation == 0.85
        assert state.should_steer is True


# =============================================================================
# _calculate_correction_metrics Tests
# =============================================================================

class TestCalculateCorrectionMetrics:
    """Test _calculate_correction_metrics with mock result dicts."""

    @pytest.fixture
    def analyzer(self):
        """Create a SelectiveSteeringAnalyzer with mocked dependencies."""
        opt = object.__new__(SelectiveSteeringAnalyzer)
        opt.config = Config()
        opt.threshold = 0.5
        opt.phase4_8_rates = None
        return opt

    def test_returns_all_expected_keys(self, analyzer):
        """Should return all required metric keys."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': False, 'incorrect_pred_activation': 0.8},
        ]

        metrics = analyzer._calculate_correction_metrics(results)

        expected_keys = ['total_problems', 'valid_problems', 'n_steered',
                         'n_not_steered', 'n_corrected', 'correction_rate',
                         'steering_trigger_rate', 'correction_efficiency',
                         'activation_stats']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_correction_rate_calculation(self, analyzer):
        """correction_rate = n_corrected / total_problems."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': False, 'incorrect_pred_activation': 0.8},
            {'task_id': 't2', 'was_steered': True, 'steered_correct': False,
             'baseline_passed': False, 'incorrect_pred_activation': 0.9},
            {'task_id': 't3', 'was_steered': False, 'steered_correct': False,
             'baseline_passed': False, 'incorrect_pred_activation': 0.3},
            {'task_id': 't4', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': False, 'incorrect_pred_activation': 0.7},
        ]

        metrics = analyzer._calculate_correction_metrics(results)

        assert metrics['n_corrected'] == 2
        assert metrics['correction_rate'] == pytest.approx(0.5, abs=1e-4)

    def test_steering_trigger_rate(self, analyzer):
        """steering_trigger_rate = n_steered / total_problems."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': False, 'incorrect_pred_activation': 0.8},
            {'task_id': 't2', 'was_steered': False, 'steered_correct': False,
             'baseline_passed': False, 'incorrect_pred_activation': 0.3},
            {'task_id': 't3', 'was_steered': True, 'steered_correct': False,
             'baseline_passed': False, 'incorrect_pred_activation': 0.7},
        ]

        metrics = analyzer._calculate_correction_metrics(results)

        assert metrics['n_steered'] == 2
        assert metrics['steering_trigger_rate'] == pytest.approx(2 / 3, abs=1e-4)

    def test_correction_efficiency(self, analyzer):
        """correction_efficiency = n_corrected / n_steered."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': False, 'incorrect_pred_activation': 0.8},
            {'task_id': 't2', 'was_steered': True, 'steered_correct': False,
             'baseline_passed': False, 'incorrect_pred_activation': 0.9},
            {'task_id': 't3', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': False, 'incorrect_pred_activation': 0.7},
            {'task_id': 't4', 'was_steered': True, 'steered_correct': False,
             'baseline_passed': False, 'incorrect_pred_activation': 0.6},
        ]

        metrics = analyzer._calculate_correction_metrics(results)

        # 2 corrected out of 4 steered
        assert metrics['correction_efficiency'] == pytest.approx(0.5, abs=1e-4)

    def test_errors_excluded_from_valid_count(self, analyzer):
        """Results with 'error' key should be excluded from valid count."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': False, 'incorrect_pred_activation': 0.8},
            {'task_id': 't2', 'was_steered': False, 'steered_correct': False,
             'baseline_passed': False, 'incorrect_pred_activation': None,
             'error': 'timeout'},
        ]

        metrics = analyzer._calculate_correction_metrics(results)

        assert metrics['total_problems'] == 2
        assert metrics['valid_problems'] == 1

    def test_zero_problems_no_division_error(self, analyzer):
        """Should handle empty results without ZeroDivisionError."""
        metrics = analyzer._calculate_correction_metrics([])

        assert metrics['correction_rate'] == 0
        assert metrics['steering_trigger_rate'] == 0
        assert metrics['n_corrected'] == 0

    def test_zero_steered_efficiency(self, analyzer):
        """correction_efficiency should be 0 when no problems were steered."""
        results = [
            {'task_id': 't1', 'was_steered': False, 'steered_correct': False,
             'baseline_passed': False, 'incorrect_pred_activation': 0.3},
        ]

        metrics = analyzer._calculate_correction_metrics(results)

        assert metrics['correction_efficiency'] == 0

    def test_activation_stats_present(self, analyzer):
        """Should include activation statistics."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': False, 'incorrect_pred_activation': 0.5},
            {'task_id': 't2', 'was_steered': True, 'steered_correct': False,
             'baseline_passed': False, 'incorrect_pred_activation': 0.9},
        ]

        metrics = analyzer._calculate_correction_metrics(results)

        stats = metrics['activation_stats']
        assert stats['mean'] == pytest.approx(0.7, abs=1e-5)
        assert stats['min'] == pytest.approx(0.5, abs=1e-5)
        assert stats['max'] == pytest.approx(0.9, abs=1e-5)
        assert stats['threshold'] == 0.5


# =============================================================================
# _calculate_preservation_metrics Tests
# =============================================================================

class TestCalculatePreservationMetrics:
    """Test _calculate_preservation_metrics with mock result dicts."""

    @pytest.fixture
    def analyzer(self):
        """Create a SelectiveSteeringAnalyzer with mocked dependencies."""
        opt = object.__new__(SelectiveSteeringAnalyzer)
        opt.config = Config()
        opt.threshold = 0.5
        opt.phase4_8_rates = None
        return opt

    def test_returns_all_expected_keys(self, analyzer):
        """Should return all required metric keys."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': True, 'incorrect_pred_activation': 0.8},
        ]

        metrics = analyzer._calculate_preservation_metrics(results)

        expected_keys = ['total_problems', 'valid_problems', 'n_steered',
                         'n_not_steered', 'n_preserved', 'n_corrupted',
                         'preservation_rate', 'corruption_rate',
                         'steering_avoidance_rate', 'activation_stats']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_preservation_rate_calculation(self, analyzer):
        """preservation_rate = n_preserved / total_problems."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': True, 'incorrect_pred_activation': 0.8},
            {'task_id': 't2', 'was_steered': True, 'steered_correct': False,
             'baseline_passed': True, 'incorrect_pred_activation': 0.9},
            {'task_id': 't3', 'was_steered': False, 'steered_correct': True,
             'baseline_passed': True, 'incorrect_pred_activation': 0.3},
        ]

        metrics = analyzer._calculate_preservation_metrics(results)

        assert metrics['n_preserved'] == 2
        assert metrics['preservation_rate'] == pytest.approx(2 / 3, abs=1e-4)

    def test_corruption_rate_calculation(self, analyzer):
        """corruption_rate = n_corrupted / total_problems."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': True, 'incorrect_pred_activation': 0.8},
            {'task_id': 't2', 'was_steered': True, 'steered_correct': False,
             'baseline_passed': True, 'incorrect_pred_activation': 0.9},
            {'task_id': 't3', 'was_steered': True, 'steered_correct': False,
             'baseline_passed': True, 'incorrect_pred_activation': 0.7},
        ]

        metrics = analyzer._calculate_preservation_metrics(results)

        assert metrics['n_corrupted'] == 2
        assert metrics['corruption_rate'] == pytest.approx(2 / 3, abs=1e-4)

    def test_steering_avoidance_rate(self, analyzer):
        """steering_avoidance_rate = n_not_steered / total_problems."""
        results = [
            {'task_id': 't1', 'was_steered': False, 'steered_correct': True,
             'baseline_passed': True, 'incorrect_pred_activation': 0.3},
            {'task_id': 't2', 'was_steered': False, 'steered_correct': True,
             'baseline_passed': True, 'incorrect_pred_activation': 0.2},
            {'task_id': 't3', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': True, 'incorrect_pred_activation': 0.8},
            {'task_id': 't4', 'was_steered': True, 'steered_correct': False,
             'baseline_passed': True, 'incorrect_pred_activation': 0.9},
        ]

        metrics = analyzer._calculate_preservation_metrics(results)

        assert metrics['steering_avoidance_rate'] == pytest.approx(0.5, abs=1e-4)

    def test_errors_excluded_from_valid_count(self, analyzer):
        """Results with 'error' key should be excluded from valid count."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': True, 'incorrect_pred_activation': 0.8},
            {'task_id': 't2', 'was_steered': False, 'steered_correct': True,
             'baseline_passed': True, 'incorrect_pred_activation': None,
             'error': 'timeout'},
        ]

        metrics = analyzer._calculate_preservation_metrics(results)

        assert metrics['total_problems'] == 2
        assert metrics['valid_problems'] == 1

    def test_zero_problems_no_division_error(self, analyzer):
        """Should handle empty results without ZeroDivisionError."""
        metrics = analyzer._calculate_preservation_metrics([])

        assert metrics['preservation_rate'] == 0
        assert metrics['corruption_rate'] == 0


# =============================================================================
# _calculate_combined_metrics Tests
# =============================================================================

class TestCalculateCombinedMetrics:
    """Test _calculate_combined_metrics across both experiments."""

    @pytest.fixture
    def analyzer(self):
        """Create a SelectiveSteeringAnalyzer with mocked dependencies."""
        opt = object.__new__(SelectiveSteeringAnalyzer)
        opt.config = Config()
        opt.threshold = 0.5
        opt.phase4_8_rates = None
        return opt

    def test_returns_all_expected_keys(self, analyzer):
        """Should return all required combined metric keys."""
        correction_results = [
            {'task_id': 't1', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': False},
        ]
        preservation_results = [
            {'task_id': 't2', 'was_steered': False, 'steered_correct': True,
             'baseline_passed': True},
        ]

        metrics = analyzer._calculate_combined_metrics(
            correction_results, preservation_results
        )

        expected_keys = ['total_problems', 'total_steered',
                         'overall_steering_rate', 'comparison_to_phase4_8']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_total_problems_sum(self, analyzer):
        """total_problems should be sum of both experiments."""
        correction_results = [
            {'task_id': f't{i}', 'was_steered': True, 'steered_correct': True,
             'baseline_passed': False}
            for i in range(3)
        ]
        preservation_results = [
            {'task_id': f'p{i}', 'was_steered': False, 'steered_correct': True,
             'baseline_passed': True}
            for i in range(5)
        ]

        metrics = analyzer._calculate_combined_metrics(
            correction_results, preservation_results
        )

        assert metrics['total_problems'] == 8

    def test_overall_steering_rate(self, analyzer):
        """overall_steering_rate = total_steered / total_problems."""
        correction_results = [
            {'task_id': 't1', 'was_steered': True, 'baseline_passed': False},
            {'task_id': 't2', 'was_steered': True, 'baseline_passed': False},
        ]
        preservation_results = [
            {'task_id': 'p1', 'was_steered': False, 'baseline_passed': True},
            {'task_id': 'p2', 'was_steered': True, 'baseline_passed': True},
        ]

        metrics = analyzer._calculate_combined_metrics(
            correction_results, preservation_results
        )

        # 3 steered out of 4 total
        assert metrics['total_steered'] == 3
        assert metrics['overall_steering_rate'] == pytest.approx(0.75, abs=1e-4)

    def test_zero_problems_no_division_error(self, analyzer):
        """Should handle empty results without ZeroDivisionError."""
        metrics = analyzer._calculate_combined_metrics([], [])

        assert metrics['total_problems'] == 0
        assert metrics['overall_steering_rate'] == 0

    def test_includes_phase4_8_comparison_when_available(self, analyzer):
        """Should include Phase 4.8 rates when loaded dynamically."""
        analyzer.phase4_8_rates = {
            'correction_rate': 4.04,
            'corruption_rate': 14.66,
            'preservation_rate': 85.34,
        }
        metrics = analyzer._calculate_combined_metrics(
            [{'task_id': 't1', 'was_steered': True, 'baseline_passed': False}],
            [{'task_id': 'p1', 'was_steered': False, 'baseline_passed': True}],
        )

        comparison = metrics['comparison_to_phase4_8']
        assert 'phase4_8_correction_rate' in comparison
        assert 'phase4_8_corruption_rate' in comparison

    def test_phase4_8_comparison_unavailable(self, analyzer):
        """Should show unavailable note when Phase 4.8 not loaded."""
        metrics = analyzer._calculate_combined_metrics(
            [{'task_id': 't1', 'was_steered': True, 'baseline_passed': False}],
            [{'task_id': 'p1', 'was_steered': False, 'baseline_passed': True}],
        )

        comparison = metrics['comparison_to_phase4_8']
        assert 'note' in comparison
        assert 'unavailable' in comparison['note']


# =============================================================================
# Config Tests
# =============================================================================

class TestPhase83Config:
    """Test Phase 8.3 configuration values."""

    def test_percentile_threshold_setting(self):
        """Config should have percentile threshold settings."""
        config = Config()
        assert hasattr(config, 'phase8_3_use_percentile_threshold')

    def test_percentile_setting(self):
        """Config should have percentile setting."""
        config = Config()
        assert hasattr(config, 'phase8_3_percentile')

    def test_percentile_is_overridable(self):
        """phase8_3_percentile should be overridable."""
        config = Config()
        config.phase8_3_percentile = 60
        assert config.phase8_3_percentile == 60


# =============================================================================
# Threshold Decision Tests
# =============================================================================

from unittest.mock import patch, MagicMock
import torch

from common.selective_steering import SteeringState


class TestThresholdDecision:
    """Test that the threshold correctly determines should_steer."""

    def test_activation_above_threshold_should_steer(self):
        """When activation > threshold, should_steer should be True."""
        threshold = 0.5
        activation_value = 0.8

        state = SteeringState(prompt_length=10)

        # Simulate the threshold check logic from _generate_with_selective_steering
        state.incorrect_pred_activation = activation_value
        state.should_steer = activation_value > threshold
        state.first_token_checked = True

        assert state.should_steer is True

    def test_activation_below_threshold_should_not_steer(self):
        """When activation < threshold, should_steer should be False."""
        threshold = 0.5
        activation_value = 0.3

        state = SteeringState(prompt_length=10)

        state.incorrect_pred_activation = activation_value
        state.should_steer = activation_value > threshold
        state.first_token_checked = True

        assert state.should_steer is False

    def test_activation_equal_threshold_should_not_steer(self):
        """When activation == threshold exactly, should_steer should be False.

        The Phase 8.3 comparison is strictly greater-than (activation > threshold),
        so activation == threshold means no steering.
        """
        threshold = 0.5
        activation_value = 0.5

        state = SteeringState(prompt_length=10)

        state.incorrect_pred_activation = activation_value
        state.should_steer = activation_value > threshold
        state.first_token_checked = True

        assert state.should_steer is False

    def test_threshold_zero_steers_on_any_positive(self):
        """With threshold = 0.0, any positive activation should trigger steering."""
        threshold = 0.0

        state = SteeringState(prompt_length=10)
        state.incorrect_pred_activation = 0.001
        state.should_steer = state.incorrect_pred_activation > threshold
        state.first_token_checked = True

        assert state.should_steer is True

    def test_threshold_zero_no_steer_at_zero(self):
        """With threshold = 0.0, activation of exactly 0.0 should NOT steer."""
        threshold = 0.0

        state = SteeringState(prompt_length=10)
        state.incorrect_pred_activation = 0.0
        state.should_steer = state.incorrect_pred_activation > threshold
        state.first_token_checked = True

        assert state.should_steer is False

    def test_negative_activation_never_steers(self):
        """Negative activation should never trigger steering (threshold >= 0)."""
        threshold = 0.0

        state = SteeringState(prompt_length=10)
        state.incorrect_pred_activation = -1.5
        state.should_steer = state.incorrect_pred_activation > threshold
        state.first_token_checked = True

        assert state.should_steer is False

    def test_high_threshold_requires_high_activation(self):
        """A high threshold should only trigger on very high activations."""
        threshold = 10.0

        # Below threshold
        state = SteeringState(prompt_length=10)
        state.incorrect_pred_activation = 5.0
        state.should_steer = state.incorrect_pred_activation > threshold
        assert state.should_steer is False

        # Above threshold
        state2 = SteeringState(prompt_length=10)
        state2.incorrect_pred_activation = 15.0
        state2.should_steer = state2.incorrect_pred_activation > threshold
        assert state2.should_steer is True

    def test_was_steered_false_returns_baseline(self):
        """When should_steer is False, result should use baseline (passthrough)."""
        # This tests the result dict pattern from _generate_with_selective_steering
        baseline_passed = True
        state = SteeringState(prompt_length=10)
        state.incorrect_pred_activation = 0.3
        state.should_steer = 0.3 > 0.5  # False

        # Simulate the result dict when not steered
        result = {
            'task_id': 'test_1',
            'baseline_passed': baseline_passed,
            'was_steered': state.should_steer,
            'steered_correct': baseline_passed,  # Passthrough baseline
            'steered_code': None,
            'source': 'phase3_5_baseline'
        }

        assert result['was_steered'] is False
        assert result['steered_correct'] == baseline_passed
        assert result['steered_code'] is None
        assert result['source'] == 'phase3_5_baseline'


# =============================================================================
# Dual Direction Config Tests
# =============================================================================


class TestDualDirectionConfig:
    """Test that Phase 8.3 uses both probe directions (logreg + mass_mean)."""

    def test_config_direction_source_default_is_sae(self):
        """Default direction_source should be 'sae'."""
        config = Config()
        assert config.direction_source == 'sae'

    def test_probe_mode_detection(self):
        """When direction_source is probe_logreg or probe_mass_mean, use_probe should be True."""
        config = Config()

        # Simulate the logic from SelectiveSteeringAnalyzer.__init__
        for source in ('probe_logreg', 'probe_mass_mean'):
            config.direction_source = source
            use_probe = config.direction_source in ('probe_logreg', 'probe_mass_mean')
            assert use_probe is True, f"Failed for direction_source={source}"

    def test_sae_mode_detection(self):
        """When direction_source is 'sae', use_probe should be False."""
        config = Config()
        config.direction_source = 'sae'
        use_probe = config.direction_source in ('probe_logreg', 'probe_mass_mean')
        assert use_probe is False

    @patch('common.steering_setup.load_probe_directions_for_steering')
    @patch('common.steering_setup.load_probe_directions_for_predicting')
    @patch('common.steering_setup.load_file')
    def test_dual_probe_loads_both_methods(self, mock_load_file, mock_pred, mock_steer):
        """load_dual_probe_directions should load logreg for prediction and mass_mean for steering."""
        from common.steering_setup import (
            load_dual_probe_directions,
            ProbeDirections,
        )

        # Mock predicting probe (logreg)
        pred_direction = torch.randn(256)
        mock_pred.return_value = ProbeDirections(
            correct_direction=pred_direction,
            incorrect_direction=-pred_direction,
            layer=19,
            method="logreg",
            bias=0.42,
            phase_dir="/fake/phase2_6",
        )

        # Mock steering probe (mass_mean)
        steer_direction = torch.randn(256)
        mock_steer.return_value = ProbeDirections(
            correct_direction=steer_direction,
            incorrect_direction=-steer_direction,
            layer=19,
            method="mass_mean",
            bias=0.0,
            phase_dir="/fake/phase2_6",
        )

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.randn(2, 2)])

        config = Config()
        device = torch.device("cpu")

        dual = load_dual_probe_directions(config, device, mock_model)

        # Verify logreg was used for prediction
        mock_pred.assert_called_once()
        pred_call_kwargs = mock_pred.call_args
        assert pred_call_kwargs[1].get('method', pred_call_kwargs[0][2] if len(pred_call_kwargs[0]) > 2 else None) == "logreg" or \
               (len(pred_call_kwargs[1]) > 0 and pred_call_kwargs[1].get('method') == "logreg")

        # Verify mass_mean was used for steering
        mock_steer.assert_called_once()
        steer_call_kwargs = mock_steer.call_args
        assert steer_call_kwargs[1].get('method', steer_call_kwargs[0][3] if len(steer_call_kwargs[0]) > 3 else None) == "mass_mean" or \
               (len(steer_call_kwargs[1]) > 0 and steer_call_kwargs[1].get('method') == "mass_mean")

    @patch('common.steering_setup.load_probe_directions_for_steering')
    @patch('common.steering_setup.load_probe_directions_for_predicting')
    def test_dual_probe_returns_correct_structure(self, mock_pred, mock_steer):
        """DualProbeDirections should have both predicting and steering attributes."""
        from common.steering_setup import (
            load_dual_probe_directions,
            ProbeDirections,
            DualProbeDirections,
        )

        pred_direction = torch.randn(256)
        mock_pred.return_value = ProbeDirections(
            correct_direction=pred_direction,
            incorrect_direction=-pred_direction,
            layer=19,
            method="logreg",
            bias=-0.3,
            phase_dir="/fake/phase2_6",
        )

        steer_direction = torch.randn(256)
        mock_steer.return_value = ProbeDirections(
            correct_direction=steer_direction,
            incorrect_direction=-steer_direction,
            layer=19,
            method="mass_mean",
            bias=0.0,
            phase_dir="/fake/phase2_6",
        )

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.randn(2, 2)])

        config = Config()
        device = torch.device("cpu")

        dual = load_dual_probe_directions(config, device, mock_model)

        # Check structure
        assert isinstance(dual, DualProbeDirections)
        assert dual.predicting_probe.method == "logreg"
        assert dual.steering_probe.method == "mass_mean"
        assert dual.predicting_bias == -0.3
        assert dual.predicting_layer == 19
        assert dual.steering_layer == 19

    @patch('common.steering_setup.load_probe_directions_for_steering')
    @patch('common.steering_setup.load_probe_directions_for_predicting')
    def test_dual_probe_predicting_direction_is_incorrect(self, mock_pred, mock_steer):
        """Predicting direction should be the incorrect direction (negated correct).

        Phase 8.3 checks if the incorrect-predicting activation exceeds a threshold,
        so it needs the incorrect direction for the dot product.
        """
        from common.steering_setup import (
            load_dual_probe_directions,
            ProbeDirections,
        )

        pred_direction = torch.tensor([1.0, 0.0, 0.0])
        mock_pred.return_value = ProbeDirections(
            correct_direction=pred_direction,
            incorrect_direction=-pred_direction,
            layer=19,
            method="logreg",
            bias=0.0,
            phase_dir="/fake/phase2_6",
        )

        steer_direction = torch.tensor([0.0, 1.0, 0.0])
        mock_steer.return_value = ProbeDirections(
            correct_direction=steer_direction,
            incorrect_direction=-steer_direction,
            layer=19,
            method="mass_mean",
            bias=0.0,
            phase_dir="/fake/phase2_6",
        )

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.randn(2, 2)])

        config = Config()
        device = torch.device("cpu")

        dual = load_dual_probe_directions(config, device, mock_model)

        # predicting_direction should be the incorrect direction from logreg
        assert torch.allclose(dual.predicting_direction, -pred_direction)

    @patch('common.steering_setup.load_probe_directions_for_steering')
    @patch('common.steering_setup.load_probe_directions_for_predicting')
    def test_dual_probe_steering_direction_is_correct(self, mock_pred, mock_steer):
        """Steering direction should be the correct direction from mass_mean.

        Phase 8.3 steers toward correctness using the mass_mean correct direction.
        """
        from common.steering_setup import (
            load_dual_probe_directions,
            ProbeDirections,
        )

        pred_direction = torch.tensor([1.0, 0.0, 0.0])
        mock_pred.return_value = ProbeDirections(
            correct_direction=pred_direction,
            incorrect_direction=-pred_direction,
            layer=19,
            method="logreg",
            bias=0.0,
            phase_dir="/fake/phase2_6",
        )

        steer_direction = torch.tensor([0.0, 1.0, 0.0])
        mock_steer.return_value = ProbeDirections(
            correct_direction=steer_direction,
            incorrect_direction=-steer_direction,
            layer=19,
            method="mass_mean",
            bias=0.0,
            phase_dir="/fake/phase2_6",
        )

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.randn(2, 2)])

        config = Config()
        device = torch.device("cpu")

        dual = load_dual_probe_directions(config, device, mock_model)

        # correct_latent_direction should be the correct direction from mass_mean
        assert torch.allclose(dual.correct_latent_direction, steer_direction)
