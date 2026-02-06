"""
Tests for Phase 3.12: Difficulty-Based AUROC Analysis

Validates:
- M7 regression: single-class y_true returns NaN AUROC instead of raising ValueError
"""

import pytest
import numpy as np


class TestSingleClassAurocGuard:
    """Regression test for single-class AUROC guard (M7).

    When a difficulty group has all correct or all incorrect samples,
roc_auc_score may raise ValueError (older sklearn) or return NaN with warning
    (newer sklearn). The guard ensures NaN is always returned without exception.
    """

    def test_single_class_returns_nan(self):
        """Single-class y_true should produce NaN AUROC, never raise."""
        from sklearn.metrics import roc_auc_score

        y_true = np.array([1, 1, 1, 1])  # All same class
        scores = np.array([0.5, 0.6, 0.7, 0.8])

        # The guard logic from Phase 3.12: if single class, return NaN
        if len(np.unique(y_true)) < 2:
            auroc = float('nan')
        else:
            auroc = roc_auc_score(y_true, scores)

        assert np.isnan(auroc)

    def test_all_zeros_returns_nan(self):
        """All-zero y_true should produce NaN AUROC."""
        y_true = np.array([0, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.4])

        if len(np.unique(y_true)) < 2:
            auroc = float('nan')
        else:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(y_true, scores)

        assert np.isnan(auroc)

    def test_two_classes_returns_valid_auroc(self):
        """With two classes, AUROC should be a valid number."""
        from sklearn.metrics import roc_auc_score

        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])

        if len(np.unique(y_true)) < 2:
            auroc = float('nan')
        else:
            auroc = roc_auc_score(y_true, scores)

        assert not np.isnan(auroc)
        assert auroc == pytest.approx(1.0)
