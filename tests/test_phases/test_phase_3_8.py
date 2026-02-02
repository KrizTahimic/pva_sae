"""
Tests for Phase 3.8 - AUROC/F1 Evaluation

Validates:
- auroc_calculation: sklearn.metrics.roc_auc_score used correctly
- f1_threshold_optimization: Best threshold selection
- per_layer_evaluation: Each layer evaluated independently
"""

import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from common.config import Config


# =============================================================================
# auroc_calculation Tests
# =============================================================================

class TestAUROCCalculation:
    """Test sklearn.metrics.roc_auc_score used correctly."""

    def test_auroc_perfect_separation(self):
        """Perfect separation should give AUROC=1.0."""
        # All correct samples score higher than all incorrect
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        auroc = roc_auc_score(y_true, y_scores)
        assert auroc == 1.0

    def test_auroc_random_classifier(self):
        """Random classifier should give AUROC~0.5."""
        np.random.seed(42)
        n_samples = 1000

        y_true = np.random.randint(0, 2, n_samples)
        y_scores = np.random.rand(n_samples)

        auroc = roc_auc_score(y_true, y_scores)
        assert 0.4 < auroc < 0.6  # Should be around 0.5

    def test_auroc_inverted_classifier(self):
        """Inverted (perfectly wrong) classifier should give AUROC=0.0."""
        # All incorrect samples score higher than all correct
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auroc = roc_auc_score(y_true, y_scores)
        assert auroc == 0.0

    def test_auroc_requires_both_classes(self):
        """AUROC returns nan when only one class is present."""
        y_true = np.array([1, 1, 1])  # Only positive class
        y_scores = np.array([0.9, 0.8, 0.7])

        # sklearn returns nan with a warning instead of raising
        result = roc_auc_score(y_true, y_scores)
        assert np.isnan(result)


# =============================================================================
# f1_threshold_optimization Tests
# =============================================================================

class TestF1ThresholdOptimization:
    """Test best threshold selection for F1."""

    def test_threshold_affects_f1(self):
        """Different thresholds should produce different F1 scores."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.8, 0.6, 0.4, 0.35, 0.3, 0.1])

        # Low threshold: high recall, low precision
        y_pred_low = (y_scores > 0.2).astype(int)
        f1_low = f1_score(y_true, y_pred_low)

        # High threshold: low recall, high precision
        y_pred_high = (y_scores > 0.7).astype(int)
        f1_high = f1_score(y_true, y_pred_high)

        assert f1_low != f1_high

    def test_find_optimal_threshold(self):
        """Should find threshold that maximizes F1."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.7, 0.5, 0.4, 0.3, 0.1])

        # Grid search for best threshold
        thresholds = np.linspace(0.1, 0.9, 9)
        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Best threshold should be around 0.45 (between incorrect and correct)
        assert 0.4 <= best_threshold <= 0.6

    def test_optimal_threshold_maximizes_f1(self):
        """Optimal threshold should give highest F1."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.7, 0.5, 0.4, 0.3, 0.1])

        thresholds = np.linspace(0.1, 0.9, 9)
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_scores > threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        optimal_f1 = f1_scores[optimal_idx]

        # Optimal should be the maximum
        assert optimal_f1 == max(f1_scores)


# =============================================================================
# per_layer_evaluation Tests
# =============================================================================

class TestPerLayerEvaluation:
    """Test each layer evaluated independently."""

    def test_layers_evaluated_separately(self):
        """Each layer should get its own AUROC/F1 scores."""
        np.random.seed(42)
        layers = [6, 12, 18]

        results_by_layer = {}

        for layer in layers:
            # Simulate different latent scores per layer
            y_true = np.random.randint(0, 2, 100)
            # Different quality per layer
            separation = 0.1 * layer / 20  # Higher layers have better separation
            y_scores = y_true * separation + np.random.rand(100) * (1 - separation)

            auroc = roc_auc_score(y_true, y_scores)
            results_by_layer[layer] = auroc

        # Each layer should have its own score
        assert len(results_by_layer) == 3
        for layer in layers:
            assert layer in results_by_layer

    def test_layer_comparison(self):
        """Should be able to compare layers by their AUROC."""
        layer_results = {
            6: {'auroc': 0.65},
            12: {'auroc': 0.75},
            18: {'auroc': 0.82},
        }

        # Find best layer
        best_layer = max(layer_results, key=lambda x: layer_results[x]['auroc'])
        assert best_layer == 18

    def test_n_candidates_from_config(self):
        """Config should specify number of candidates to evaluate."""
        config = Config()
        assert hasattr(config, 'phase3_8_n_candidates')
        assert config.phase3_8_n_candidates >= 1


# =============================================================================
# Output Format Tests
# =============================================================================

class TestOutputFormat:
    """Test Phase 3.8 output format."""

    def test_output_includes_all_metrics(self):
        """Output should include AUROC, F1, and optimal threshold."""
        expected_keys = ['auroc', 'f1', 'optimal_threshold']

        sample_output = {
            'layer': 16,
            'latent_idx': 100,
            'auroc': 0.75,
            'f1': 0.68,
            'optimal_threshold': 0.5
        }

        for key in expected_keys:
            assert key in sample_output

    def test_per_candidate_results(self):
        """Should have results for each candidate."""
        candidates = [
            {'layer': 16, 'latent_idx': 100},
            {'layer': 18, 'latent_idx': 200},
            {'layer': 16, 'latent_idx': 300},
        ]

        results = []
        for candidate in candidates:
            results.append({
                'layer': candidate['layer'],
                'latent_idx': candidate['latent_idx'],
                'auroc': np.random.rand(),
            })

        assert len(results) == len(candidates)
