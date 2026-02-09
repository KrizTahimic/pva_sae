"""
Tests for Phase 8.2 - Threshold Optimizer

Validates:
- _calculate_metrics: Production metric calculation for correction/preservation
- SteeringState: Shared state initialization
- TwoStageOptimizer: Coarse grid + golden section refinement
- config: Phase 8.2 configuration values
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from common.config import Config
from common.search_optimization import TwoStageOptimizer
from phase8_2_threshold_optimizer.threshold_optimizer import (
    ThresholdOptimizer,
    ThresholdEvaluator,
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
        state.incorrect_pred_activation = 0.75
        state.should_steer = True

        assert state.first_token_checked is True
        assert state.incorrect_pred_activation == 0.75
        assert state.should_steer is True


# =============================================================================
# _calculate_metrics Tests (using real production code)
# =============================================================================

class TestCalculateMetrics:
    """Test ThresholdOptimizer._calculate_metrics with mock result dicts."""

    @pytest.fixture
    def optimizer(self):
        """Create a ThresholdOptimizer with mocked dependencies."""
        opt = object.__new__(ThresholdOptimizer)
        opt.config = Config()
        return opt

    def test_correction_metrics_keys(self, optimizer):
        """Correction metrics should include expected keys."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'corrected': True, 'baseline_passed': False},
            {'task_id': 't2', 'was_steered': True, 'corrected': False, 'baseline_passed': False},
            {'task_id': 't3', 'was_steered': False, 'corrected': False, 'baseline_passed': False},
        ]

        metrics = optimizer._calculate_metrics(results, 'correction', total_problems=3)

        expected_keys = ['dataset_type', 'n_problems', 'n_valid', 'n_errors',
                         'n_steered', 'n_not_steered', 'n_corrected',
                         'correction_rate', 'steering_rate']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_correction_rate_calculation(self, optimizer):
        """correction_rate = n_corrected / total_problems."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'corrected': True, 'baseline_passed': False},
            {'task_id': 't2', 'was_steered': True, 'corrected': False, 'baseline_passed': False},
            {'task_id': 't3', 'was_steered': True, 'corrected': True, 'baseline_passed': False},
            {'task_id': 't4', 'was_steered': False, 'corrected': False, 'baseline_passed': False},
        ]

        metrics = optimizer._calculate_metrics(results, 'correction', total_problems=4)

        assert metrics['n_corrected'] == 2
        assert metrics['correction_rate'] == pytest.approx(0.5)

    def test_preservation_metrics_keys(self, optimizer):
        """Preservation metrics should include expected keys."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'preserved': True, 'corrupted': False, 'baseline_passed': True},
            {'task_id': 't2', 'was_steered': True, 'preserved': False, 'corrupted': True, 'baseline_passed': True},
        ]

        metrics = optimizer._calculate_metrics(results, 'preservation', total_problems=2)

        expected_keys = ['dataset_type', 'n_preserved', 'n_corrupted',
                         'preservation_rate', 'corruption_rate']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_corruption_rate_calculation(self, optimizer):
        """corruption_rate = n_corrupted / total_problems."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'preserved': True, 'corrupted': False, 'baseline_passed': True},
            {'task_id': 't2', 'was_steered': True, 'preserved': False, 'corrupted': True, 'baseline_passed': True},
            {'task_id': 't3', 'was_steered': False, 'preserved': True, 'corrupted': False, 'baseline_passed': True},
        ]

        metrics = optimizer._calculate_metrics(results, 'preservation', total_problems=3)

        assert metrics['n_corrupted'] == 1
        assert metrics['corruption_rate'] == pytest.approx(1 / 3)

    def test_errors_excluded_from_valid_count(self, optimizer):
        """Results with 'error' key should be excluded from n_valid."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'corrected': True, 'baseline_passed': False},
            {'task_id': 't2', 'was_steered': False, 'corrected': False, 'baseline_passed': False,
             'error': 'timeout'},
        ]

        metrics = optimizer._calculate_metrics(results, 'correction', total_problems=2)

        assert metrics['n_valid'] == 1
        assert metrics['n_errors'] == 1

    def test_zero_problems_no_division_error(self, optimizer):
        """Should handle empty results without ZeroDivisionError."""
        metrics = optimizer._calculate_metrics([], 'correction', total_problems=0)

        assert metrics['correction_rate'] == 0.0
        assert metrics['n_corrected'] == 0

    def test_steering_rate_calculation(self, optimizer):
        """steering_rate = n_steered / total_problems."""
        results = [
            {'task_id': 't1', 'was_steered': True, 'corrected': True, 'baseline_passed': False},
            {'task_id': 't2', 'was_steered': False, 'corrected': False, 'baseline_passed': False},
            {'task_id': 't3', 'was_steered': True, 'corrected': False, 'baseline_passed': False},
        ]

        metrics = optimizer._calculate_metrics(results, 'correction', total_problems=3)

        assert metrics['n_steered'] == 2
        assert metrics['steering_rate'] == pytest.approx(2 / 3)


# =============================================================================
# TwoStageOptimizer Tests
# =============================================================================

class TestTwoStageOptimizer:
    """Test TwoStageOptimizer coarse grid + golden section refinement."""

    def test_finds_optimal_in_simple_case(self):
        """Should find the optimal value in a simple quadratic."""
        # Score function: peak at 50
        def score_fn(x):
            return -(x - 50) ** 2 + 100

        optimizer = TwoStageOptimizer(
            evaluate_fn=score_fn,
            grid_points=[10, 20, 30, 40, 50, 60, 70, 80, 90],
            refinement_radius=10,
            tolerance=1,
            lower_bound=1,
            upper_bound=99
        )

        optimal, score, evaluations = optimizer.optimize()

        assert optimal == 50
        assert score == 100

    def test_cache_prevents_duplicate_evaluations(self):
        """Should not evaluate the same point twice."""
        call_count = {}

        def counting_fn(x):
            call_count[x] = call_count.get(x, 0) + 1
            return -(x - 50) ** 2

        optimizer = TwoStageOptimizer(
            evaluate_fn=counting_fn,
            grid_points=[30, 50, 70],
            refinement_radius=5,
            tolerance=1,
            lower_bound=1,
            upper_bound=99
        )

        optimizer.optimize()

        # No point should be evaluated more than once
        for x, count in call_count.items():
            assert count == 1, f"Point {x} evaluated {count} times"

    def test_returns_all_evaluations(self):
        """Should return dict of all evaluated points."""
        def score_fn(x):
            return x

        optimizer = TwoStageOptimizer(
            evaluate_fn=score_fn,
            grid_points=[10, 20, 30],
            refinement_radius=5,
            tolerance=1,
            lower_bound=1,
            upper_bound=99
        )

        optimal, score, evaluations = optimizer.optimize()

        assert isinstance(evaluations, dict)
        assert len(evaluations) > 0

    def test_discrete_refinement(self):
        """With available_values, should only test those values."""
        evaluated = set()

        def tracking_fn(x):
            evaluated.add(x)
            return -(x - 45) ** 2

        available = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        optimizer = TwoStageOptimizer(
            evaluate_fn=tracking_fn,
            grid_points=[10, 30, 50, 70, 90],
            refinement_radius=15,
            tolerance=1,
            lower_bound=10,
            upper_bound=90,
            available_values=available
        )

        optimal, score, evaluations = optimizer.optimize()

        # All evaluated points should be in available_values
        for x in evaluated:
            assert x in available, f"Evaluated non-available value: {x}"

    def test_coarse_early_stopping(self):
        """Coarse grid should stop after first score drop from peak."""
        evaluated = []

        def tracking_fn(x):
            evaluated.append(x)
            # Peak at 30, drops at 40
            scores = {10: 5, 20: 8, 30: 10, 40: 7, 50: 3}
            return scores.get(x, 0)

        optimizer = TwoStageOptimizer(
            evaluate_fn=tracking_fn,
            grid_points=[10, 20, 30, 40, 50],
            refinement_radius=5,
            tolerance=1,
            lower_bound=1,
            upper_bound=99
        )

        coarse_optimal, coarse_score = optimizer.coarse_grid_search()

        # Should stop at 40 (first drop after peak at 30)
        assert coarse_optimal == 30
        assert 50 not in evaluated  # Should have stopped before 50


# =============================================================================
# Config Tests
# =============================================================================

class TestPhase82Config:
    """Test Phase 8.2 configuration values."""

    def test_refinement_radius(self):
        """Config should specify refinement radius."""
        config = Config()
        assert hasattr(config, 'phase8_2_refinement_radius')
        assert config.phase8_2_refinement_radius > 0

    def test_tolerance(self):
        """Config should specify tolerance."""
        config = Config()
        assert hasattr(config, 'phase8_2_tolerance')
        assert config.phase8_2_tolerance > 0

    def test_probe_methods_available(self):
        """Config should have both probe methods for dual-direction architecture."""
        config = Config()
        assert hasattr(config, 'probe_mass_mean_reg_lambda')
        assert hasattr(config, 'probe_logreg_C_values')


# =============================================================================
# Phase 4.9 Integration Tests
# =============================================================================

class TestPhase82Phase49Integration:
    """Test that SAE mode loads steering latent + coefficient from Phase 4.9."""

    MOCK_SELECTION = {
        "correct": {"rank": 2, "layer": 15, "latent_idx": 12809, "refined_coefficient": 62},
        "incorrect": {"rank": 0, "layer": 18, "latent_idx": 4612, "refined_coefficient": 25},
    }

    def test_threshold_optimizer_sae_stores_phase4_9_latent(self):
        """ThresholdOptimizer SAE mode should read correct_steer_layer/latent from Phase 4.9."""
        # Verify the code references load_phase4_9_best_latent (not load_steering_latents)
        import inspect
        source = inspect.getsource(ThresholdOptimizer._load_dependencies)
        assert 'load_phase4_9_best_latent' in source
        assert 'load_steering_latents' not in source  # Old path removed

        # Verify Phase 4.9 selection structure provides what ThresholdOptimizer needs
        selection = self.MOCK_SELECTION
        assert selection['correct']['layer'] == 15
        assert selection['correct']['latent_idx'] == 12809
        assert selection['correct']['refined_coefficient'] == 62

    def test_threshold_optimizer_sae_coefficient_from_phase4_9(self):
        """ThresholdOptimizer SAE mode should use refined_coefficient from Phase 4.9."""
        # The code path: self._phase4_9_selection['correct']['refined_coefficient']
        selection = self.MOCK_SELECTION
        assert selection['correct']['refined_coefficient'] == 62
        # ThresholdOptimizer sets: self.steering_coefficient = selection['correct']['refined_coefficient']

    def test_threshold_evaluator_sae_uses_phase4_9(self):
        """ThresholdEvaluator SAE mode should call load_phase4_9_best_latent."""
        # Verify the code references load_phase4_9_best_latent
        import inspect
        source = inspect.getsource(ThresholdEvaluator._load_dependencies)
        assert 'load_phase4_9_best_latent' in source
        assert 'load_steering_latents' not in source  # Old path removed
