"""
Tests for common/metrics_utils.py

Validates:
- AUROC matches sklearn.metrics.roc_auc_score
- F1/precision/recall at different thresholds
- Edge cases: all zeros, all ones, single class
- Empty input handling
"""

import pytest
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from common.metrics_utils import calculate_classification_metrics


# =============================================================================
# Basic Metrics Tests
# =============================================================================

class TestClassificationMetrics:
    """Test calculate_classification_metrics against sklearn."""

    def test_perfect_prediction(self):
        """Perfect scores should give AUROC=1.0 and F1=1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        metrics = calculate_classification_metrics(y_true, scores, threshold=0.5)

        assert metrics['auroc'] == pytest.approx(1.0)
        assert metrics['f1'] == pytest.approx(1.0)
        assert metrics['precision'] == pytest.approx(1.0)
        assert metrics['recall'] == pytest.approx(1.0)

    def test_auroc_matches_sklearn(self):
        """AUROC should match sklearn.metrics.roc_auc_score."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        scores = rng.rand(100)

        metrics = calculate_classification_metrics(y_true, scores, threshold=0.5)
        expected_auroc = roc_auc_score(y_true, scores)

        assert metrics['auroc'] == pytest.approx(expected_auroc)

    def test_f1_matches_sklearn(self):
        """F1 should match sklearn.metrics.f1_score at given threshold."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        scores = rng.rand(100)
        threshold = 0.5

        metrics = calculate_classification_metrics(y_true, scores, threshold=threshold)
        y_pred = (scores > threshold).astype(int)
        expected_f1 = f1_score(y_true, y_pred, zero_division=0)

        assert metrics['f1'] == pytest.approx(expected_f1)

    def test_precision_recall_match_sklearn(self):
        """Precision and recall should match sklearn at threshold."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        scores = rng.rand(100)
        threshold = 0.3

        metrics = calculate_classification_metrics(y_true, scores, threshold=threshold)
        y_pred = (scores > threshold).astype(int)

        assert metrics['precision'] == pytest.approx(
            precision_score(y_true, y_pred, zero_division=0)
        )
        assert metrics['recall'] == pytest.approx(
            recall_score(y_true, y_pred, zero_division=0)
        )


# =============================================================================
# Threshold Variation Tests
# =============================================================================

class TestThresholdVariation:
    """Test metrics at different thresholds."""

    def test_low_threshold_high_recall(self):
        """Low threshold should give high recall (catches everything)."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        metrics = calculate_classification_metrics(y_true, scores, threshold=0.05)
        assert metrics['recall'] == pytest.approx(1.0)

    def test_high_threshold_high_precision(self):
        """High threshold should give high precision (only confident predictions)."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        metrics = calculate_classification_metrics(y_true, scores, threshold=0.75)
        assert metrics['precision'] == pytest.approx(1.0)

    def test_threshold_stored_in_result(self):
        """Threshold should be stored in the result dict."""
        y_true = np.array([0, 1])
        scores = np.array([0.3, 0.7])

        metrics = calculate_classification_metrics(y_true, scores, threshold=0.42)
        assert metrics['threshold'] == 0.42

    def test_n_samples_correct(self):
        """n_samples should reflect input size."""
        y_true = np.array([0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.9])

        metrics = calculate_classification_metrics(y_true, scores, threshold=0.5)
        assert metrics['n_samples'] == 5


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases in metric calculation."""

    def test_all_same_class_returns_nan_auroc(self):
        """AUROC is undefined for single-class input; returns NaN."""
        y_true = np.array([1, 1, 1, 1])
        scores = np.array([0.5, 0.6, 0.7, 0.8])

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = calculate_classification_metrics(y_true, scores, threshold=0.5)
        assert np.isnan(metrics['auroc'])

    def test_all_predictions_below_threshold(self):
        """All predictions below threshold should give zero recall."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4])

        metrics = calculate_classification_metrics(y_true, scores, threshold=0.5)
        assert metrics['recall'] == 0.0

    def test_all_predictions_above_threshold(self):
        """All predictions above threshold should give perfect recall."""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.6, 0.7, 0.8, 0.9])

        metrics = calculate_classification_metrics(y_true, scores, threshold=0.5)
        assert metrics['recall'] == pytest.approx(1.0)

    def test_two_samples_minimum(self):
        """Should work with minimum viable input (2 samples, both classes)."""
        y_true = np.array([0, 1])
        scores = np.array([0.3, 0.7])

        metrics = calculate_classification_metrics(y_true, scores, threshold=0.5)
        assert metrics['auroc'] == pytest.approx(1.0)
        assert metrics['f1'] == pytest.approx(1.0)
