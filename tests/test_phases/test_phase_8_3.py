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
