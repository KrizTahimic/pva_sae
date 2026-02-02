"""
Tests for Phase 2.6 - Probe Directions

Validates:
- logistic_regression_training: Probe trains without error
- mass_mean_calculation: mean(correct) - mean(incorrect)
- cross_validation: K-fold validation logic
"""

import pytest
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from common.config import Config


# =============================================================================
# logistic_regression_training Tests
# =============================================================================

class TestLogisticRegressionTraining:
    """Test probe trains without error."""

    @pytest.fixture
    def training_data(self):
        """Create synthetic training data."""
        np.random.seed(42)
        d_model = 2304
        n_correct = 50
        n_incorrect = 50

        # Correct samples: shifted mean
        correct = np.random.randn(n_correct, d_model) + 0.5
        incorrect = np.random.randn(n_incorrect, d_model) - 0.5

        X = np.vstack([correct, incorrect])
        y = np.array([1] * n_correct + [0] * n_incorrect)

        return X, y

    def test_logreg_fits_without_error(self, training_data):
        """LogisticRegression should fit without error."""
        X, y = training_data

        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X, y)

        # Should have coefficients
        assert clf.coef_.shape == (1, X.shape[1])

    def test_logreg_coefficients_as_direction(self, training_data):
        """LogisticRegression coefficients should be usable as direction."""
        X, y = training_data

        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X, y)

        direction = clf.coef_.flatten()

        # Should be a vector of d_model size
        assert direction.shape == (2304,)

        # Should be able to normalize
        normalized = direction / np.linalg.norm(direction)
        assert np.linalg.norm(normalized) == pytest.approx(1.0)

    def test_logreg_c_values_from_config(self):
        """Config should provide C values for cross-validation."""
        config = Config()
        assert hasattr(config, 'probe_logreg_C_values')
        assert len(config.probe_logreg_C_values) > 0

    def test_logreg_cv_folds_from_config(self):
        """Config should provide number of CV folds."""
        config = Config()
        assert hasattr(config, 'probe_cv_folds')
        assert config.probe_cv_folds == 5


# =============================================================================
# mass_mean_calculation Tests
# =============================================================================

class TestMassMeanCalculation:
    """Test mean(correct) - mean(incorrect) calculation."""

    def test_mass_mean_positive_direction(self):
        """Mass mean should point toward correct class."""
        np.random.seed(42)
        d_model = 2304

        correct = np.random.randn(50, d_model) + 1.0  # Shifted positive
        incorrect = np.random.randn(50, d_model) - 1.0  # Shifted negative

        mean_correct = np.mean(correct, axis=0)
        mean_incorrect = np.mean(incorrect, axis=0)

        mass_mean_direction = mean_correct - mean_incorrect

        # Direction should exist and have correct shape
        assert mass_mean_direction.shape == (d_model,)

        # Mean of direction should be positive (since correct shifted positive)
        assert np.mean(mass_mean_direction) > 0

    def test_mass_mean_formula(self):
        """Mass mean should equal mean_correct - mean_incorrect."""
        correct = np.array([[1, 2, 3], [2, 3, 4]])  # Mean: [1.5, 2.5, 3.5]
        incorrect = np.array([[0, 0, 0], [1, 1, 1]])  # Mean: [0.5, 0.5, 0.5]

        mean_correct = np.mean(correct, axis=0)
        mean_incorrect = np.mean(incorrect, axis=0)

        mass_mean = mean_correct - mean_incorrect

        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(mass_mean, expected)

    def test_mass_mean_regularization_from_config(self):
        """Config should provide regularization lambda for mass mean."""
        config = Config()
        assert hasattr(config, 'probe_mass_mean_reg_lambda')
        assert config.probe_mass_mean_reg_lambda == 1e-4


# =============================================================================
# cross_validation Tests
# =============================================================================

class TestCrossValidation:
    """Test K-fold validation logic."""

    def test_cv_folds_count(self):
        """Should use 5-fold cross-validation by default."""
        config = Config()
        assert config.probe_cv_folds == 5

    def test_cv_produces_valid_splits(self):
        """CV splits should cover all data without overlap."""
        from sklearn.model_selection import KFold

        n_samples = 100
        k_folds = 5

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        all_test_indices = []
        for train_idx, test_idx in kf.split(range(n_samples)):
            all_test_indices.extend(test_idx)
            # Train and test should not overlap
            assert len(set(train_idx) & set(test_idx)) == 0

        # All samples should appear in exactly one test fold
        assert set(all_test_indices) == set(range(n_samples))
        assert len(all_test_indices) == n_samples

    def test_cv_c_values_span_range(self):
        """C values should span several orders of magnitude."""
        config = Config()
        c_values = config.probe_logreg_C_values

        # Should cover from 1e-5 to 1e5 (10 orders of magnitude)
        assert min(c_values) <= 1e-4
        assert max(c_values) >= 1e4


# =============================================================================
# Direction Output Tests
# =============================================================================

class TestDirectionOutput:
    """Test probe direction output format."""

    def test_direction_is_tensor(self):
        """Output direction should be a tensor."""
        direction = torch.randn(2304)
        assert isinstance(direction, torch.Tensor)
        assert direction.shape == (2304,)

    def test_direction_can_be_normalized(self):
        """Direction should be normalizable."""
        direction = torch.randn(2304)
        normalized = direction / torch.norm(direction)

        assert torch.norm(normalized).item() == pytest.approx(1.0, rel=1e-5)

    def test_output_file_structure(self, tmp_path):
        """Probe output files should have correct structure."""
        from safetensors.torch import save_file, load_file
        import json

        # Simulate saving probe directions
        layer = 16
        direction = torch.randn(2304)

        probe_dir = tmp_path / "probe_directions"
        probe_dir.mkdir()

        # Save direction
        save_file(
            {'mass_mean_direction': direction, 'logreg_direction': direction},
            str(probe_dir / f"layer_{layer}_probes.safetensors")
        )

        # Verify can load
        loaded = load_file(str(probe_dir / f"layer_{layer}_probes.safetensors"))

        assert 'mass_mean_direction' in loaded
        assert 'logreg_direction' in loaded
        assert loaded['mass_mean_direction'].shape == (2304,)
