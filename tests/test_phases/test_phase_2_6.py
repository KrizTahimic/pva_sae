"""
Tests for Phase 2.6 - Probe Directions

Validates:
- mass_mean_calculation: compute_mass_mean_direction returns unit-normalized direction
- logistic_regression_training: compute_logreg_direction returns direction, bias, best_C
- direction_metrics: compute_direction_metrics returns all required metrics
- config: Config provides probe hyperparameters
"""

import pytest
import numpy as np

from common.config import Config
from phase2_6_probe_directions.probe_direction_computer import (
    ProbeDirectionComputer,
)


# =============================================================================
# Helper: Create a ProbeDirectionComputer without filesystem dependencies
# =============================================================================

@pytest.fixture
def computer():
    """Create a ProbeDirectionComputer with minimal init (no Phase 1 loading)."""
    comp = object.__new__(ProbeDirectionComputer)
    comp.config = Config()
    return comp


@pytest.fixture
def training_data():
    """Create synthetic training data with known separation."""
    np.random.seed(42)
    d_model = 64  # Small for speed

    # Correct samples: shifted in first dimension
    correct = np.random.randn(50, d_model) * 0.5
    correct[:, 0] += 2.0

    # Incorrect samples: shifted opposite
    incorrect = np.random.randn(50, d_model) * 0.5
    incorrect[:, 0] -= 2.0

    X = np.vstack([correct, incorrect])
    y = np.array([1] * 50 + [0] * 50)

    return X, y


# =============================================================================
# compute_mass_mean_direction Tests
# =============================================================================

class TestComputeMassMeanDirection:
    """Test compute_mass_mean_direction returns unit-normalized direction."""

    def test_returns_unit_norm(self, computer, training_data):
        """Mass-mean direction should have unit L2 norm."""
        X, y = training_data
        direction = computer.compute_mass_mean_direction(X, y, reg_lambda=1e-4)

        norm = np.linalg.norm(direction)
        assert norm == pytest.approx(1.0, rel=1e-5)

    def test_correct_shape(self, computer, training_data):
        """Direction should be [d_model] shaped."""
        X, y = training_data
        direction = computer.compute_mass_mean_direction(X, y, reg_lambda=1e-4)

        assert direction.shape == (X.shape[1],)

    def test_points_toward_correct_class(self, computer, training_data):
        """Direction should point toward correct class (positive projection)."""
        X, y = training_data
        direction = computer.compute_mass_mean_direction(X, y, reg_lambda=1e-4)

        # Mean projection for correct samples should be higher
        correct_projection = X[y == 1] @ direction
        incorrect_projection = X[y == 0] @ direction

        assert np.mean(correct_projection) > np.mean(incorrect_projection)

    def test_regularization_affects_direction(self, computer, training_data):
        """Different regularization should produce different (but valid) directions."""
        X, y = training_data

        dir_small_reg = computer.compute_mass_mean_direction(X, y, reg_lambda=1e-6)
        dir_large_reg = computer.compute_mass_mean_direction(X, y, reg_lambda=1.0)

        # Both should be unit norm
        assert np.linalg.norm(dir_small_reg) == pytest.approx(1.0, rel=1e-5)
        assert np.linalg.norm(dir_large_reg) == pytest.approx(1.0, rel=1e-5)

        # But they should differ (unless data is perfectly aligned)
        cosine_sim = abs(np.dot(dir_small_reg, dir_large_reg))
        assert cosine_sim < 1.0  # Not exactly the same


# =============================================================================
# compute_logreg_direction Tests
# =============================================================================

class TestComputeLogregDirection:
    """Test compute_logreg_direction returns direction, bias, best_C."""

    def test_returns_three_values(self, computer, training_data):
        """Should return (direction, bias, best_C) tuple."""
        X, y = training_data
        result = computer.compute_logreg_direction(X, y)

        assert len(result) == 3
        direction, bias, best_C = result

    def test_direction_shape(self, computer, training_data):
        """Direction should match d_model."""
        X, y = training_data
        direction, bias, best_C = computer.compute_logreg_direction(X, y)

        assert direction.shape == (X.shape[1],)

    def test_bias_is_scalar(self, computer, training_data):
        """Bias should be a scalar float."""
        X, y = training_data
        direction, bias, best_C = computer.compute_logreg_direction(X, y)

        assert isinstance(bias, (float, np.floating))

    def test_best_c_from_config_values(self, computer, training_data):
        """Best C should be from the configured C values."""
        X, y = training_data
        direction, bias, best_C = computer.compute_logreg_direction(X, y)

        assert best_C in computer.config.probe_logreg_C_values

    def test_direction_is_discriminative(self, computer, training_data):
        """LogReg direction should separate correct from incorrect."""
        X, y = training_data
        direction, bias, best_C = computer.compute_logreg_direction(X, y)

        logits = X @ direction + bias
        # Correct samples should generally have higher logits
        correct_mean = np.mean(logits[y == 1])
        incorrect_mean = np.mean(logits[y == 0])

        assert correct_mean > incorrect_mean


# =============================================================================
# compute_direction_metrics Tests
# =============================================================================

class TestComputeDirectionMetrics:
    """Test compute_direction_metrics returns all required metrics."""

    def test_returns_all_keys(self, computer, training_data):
        """Metrics dict should have auroc, f1, t_statistic, p_value, separation, direction_norm."""
        X, y = training_data
        direction = computer.compute_mass_mean_direction(X, y, reg_lambda=1e-4)

        metrics = computer.compute_direction_metrics(X, y, direction, bias=0.0)

        expected_keys = ['auroc', 'f1', 't_statistic', 'p_value', 'separation', 'direction_norm']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_auroc_high_for_separable_data(self, computer, training_data):
        """AUROC should be high for well-separated data."""
        X, y = training_data
        direction = computer.compute_mass_mean_direction(X, y, reg_lambda=1e-4)

        metrics = computer.compute_direction_metrics(X, y, direction, bias=0.0)

        assert metrics['auroc'] > 0.9

    def test_separation_positive(self, computer, training_data):
        """Separation should be positive for correct-pointing direction."""
        X, y = training_data
        direction = computer.compute_mass_mean_direction(X, y, reg_lambda=1e-4)

        metrics = computer.compute_direction_metrics(X, y, direction, bias=0.0)

        assert metrics['separation'] > 0

    def test_direction_norm_matches(self, computer, training_data):
        """direction_norm should match np.linalg.norm of input direction."""
        X, y = training_data
        direction = computer.compute_mass_mean_direction(X, y, reg_lambda=1e-4)

        metrics = computer.compute_direction_metrics(X, y, direction, bias=0.0)

        expected_norm = float(np.linalg.norm(direction))
        assert metrics['direction_norm'] == pytest.approx(expected_norm, rel=1e-5)

    def test_f1_in_valid_range(self, computer, training_data):
        """F1 should be in [0, 1]."""
        X, y = training_data
        direction = computer.compute_mass_mean_direction(X, y, reg_lambda=1e-4)

        metrics = computer.compute_direction_metrics(X, y, direction, bias=0.0)

        assert 0.0 <= metrics['f1'] <= 1.0


# =============================================================================
# Config Tests
# =============================================================================

class TestProbeConfig:
    """Test Config provides probe hyperparameters."""

    def test_logreg_c_values(self):
        """Config should provide C values for cross-validation."""
        config = Config()
        assert hasattr(config, 'probe_logreg_C_values')
        assert len(config.probe_logreg_C_values) > 0

    def test_cv_folds(self):
        """Config should provide number of CV folds."""
        config = Config()
        assert hasattr(config, 'probe_cv_folds')
        assert config.probe_cv_folds == 5

    def test_mass_mean_reg_lambda(self):
        """Config should provide regularization lambda for mass mean."""
        config = Config()
        assert hasattr(config, 'probe_mass_mean_reg_lambda')
        assert config.probe_mass_mean_reg_lambda == 1e-4

    def test_c_values_span_range(self):
        """C values should span several orders of magnitude."""
        config = Config()
        c_values = config.probe_logreg_C_values

        assert min(c_values) <= 1e-4
        assert max(c_values) >= 1e4
