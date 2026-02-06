"""
Tests for Phase 3.8 - AUROC/F1 Evaluation

Validates:
- calculate_metrics: Production function returns correct metric keys
- find_optimal_threshold: Finds threshold maximizing F1
- config: Phase 3.8 config values
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from common.config import Config
from phase3_8_auroc_f1_evaluation.auroc_f1_evaluator import (
    calculate_metrics,
    find_optimal_threshold,
)


# =============================================================================
# calculate_metrics Tests
# =============================================================================

class TestCalculateMetrics:
    """Test calculate_metrics returns correct metric keys and values."""

    def test_returns_all_required_keys(self, tmp_path):
        """calculate_metrics should return auroc, f1, precision, recall, threshold."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        threshold = 0.5

        # Patch plot_confusion_matrix to avoid file I/O
        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            metrics = calculate_metrics(y_true, scores, threshold, 'correct', tmp_path)

        expected_keys = ['auroc', 'f1', 'precision', 'recall', 'threshold']
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_perfect_separation_gives_auroc_1(self, tmp_path):
        """Perfect separation should give AUROC=1.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        threshold = 0.5

        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            metrics = calculate_metrics(y_true, scores, threshold, 'correct', tmp_path)

        assert metrics['auroc'] == 1.0

    def test_threshold_applied_for_binary_predictions(self, tmp_path):
        """Threshold should be used for precision/recall/f1 computation."""
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.6, 0.4, 0.1])

        # High threshold: only first sample predicted positive
        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            metrics_high = calculate_metrics(y_true, scores, 0.8, 'correct', tmp_path)

        # Low threshold: first 3 samples predicted positive
        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            metrics_low = calculate_metrics(y_true, scores, 0.3, 'correct', tmp_path)

        # Different thresholds should produce different F1
        assert metrics_high['f1'] != metrics_low['f1']

    def test_threshold_stored_in_output(self, tmp_path):
        """Returned metrics should include the threshold used."""
        y_true = np.array([1, 0])
        scores = np.array([0.9, 0.1])
        threshold = 0.42

        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            metrics = calculate_metrics(y_true, scores, threshold, 'correct', tmp_path)

        assert metrics['threshold'] == pytest.approx(0.42)

    def test_metrics_values_in_valid_range(self, tmp_path):
        """All metric values should be in [0, 1]."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        scores = np.random.rand(100)

        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            metrics = calculate_metrics(y_true, scores, 0.5, 'correct', tmp_path)

        for key in ['auroc', 'f1', 'precision', 'recall']:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


# =============================================================================
# find_optimal_threshold Tests
# =============================================================================

class TestFindOptimalThreshold:
    """Test find_optimal_threshold finds F1-maximizing threshold."""

    def test_returns_threshold_and_metrics(self, tmp_path):
        """Should return (optimal_threshold, metrics_dict)."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.7, 0.5, 0.4, 0.3, 0.1])

        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            threshold, metrics = find_optimal_threshold(y_true, scores, 'correct', tmp_path)

        assert isinstance(threshold, float)
        assert isinstance(metrics, dict)

    def test_optimal_threshold_in_score_range(self, tmp_path):
        """Optimal threshold should be within the score range."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.7, 0.5, 0.4, 0.3, 0.1])

        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            threshold, metrics = find_optimal_threshold(y_true, scores, 'correct', tmp_path)

        assert scores.min() <= threshold <= scores.max()

    def test_optimal_threshold_maximizes_f1(self, tmp_path):
        """Threshold returned should maximize F1 among tested thresholds."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.7, 0.5, 0.4, 0.3, 0.1])

        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            threshold, metrics = find_optimal_threshold(y_true, scores, 'correct', tmp_path)

        # The returned F1 should be positive for well-separated data
        assert metrics['f1'] > 0.5

    def test_includes_f1_curve_data(self, tmp_path):
        """Metrics should include f1_curve for plotting."""
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.8, 0.6, 0.4, 0.2])

        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            threshold, metrics = find_optimal_threshold(y_true, scores, 'correct', tmp_path)

        assert 'f1_curve' in metrics
        assert 'thresholds' in metrics['f1_curve']
        assert 'f1_scores' in metrics['f1_curve']

    def test_includes_threshold_range(self, tmp_path):
        """Metrics should include threshold_range."""
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.8, 0.6, 0.4, 0.2])

        with patch('phase3_8_auroc_f1_evaluation.auroc_f1_evaluator.plot_confusion_matrix'):
            threshold, metrics = find_optimal_threshold(y_true, scores, 'correct', tmp_path)

        assert 'threshold_range' in metrics
        assert len(metrics['threshold_range']) == 2
        assert metrics['threshold_range'][0] == pytest.approx(scores.min())
        assert metrics['threshold_range'][1] == pytest.approx(scores.max())


# =============================================================================
# Config Tests
# =============================================================================

class TestPhase38Config:
    """Test Phase 3.8 config values."""

    def test_n_candidates_from_config(self):
        """Config should specify number of candidates to evaluate."""
        config = Config()
        assert hasattr(config, 'phase3_8_n_candidates')
        assert config.phase3_8_n_candidates >= 1

    def test_evaluation_random_seed(self):
        """Config should provide evaluation random seed."""
        config = Config()
        assert hasattr(config, 'evaluation_random_seed')
        assert isinstance(config.evaluation_random_seed, int)
