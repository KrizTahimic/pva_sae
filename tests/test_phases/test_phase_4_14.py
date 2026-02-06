"""
Tests for Phase 4.14: Statistical Significance Testing

Validates:
- H3 regression: binomtest with baseline_rate=0 uses p=max(1/n, 1e-10), not p=0
- Non-zero baseline rates are passed through unchanged
- effective_rate key is included in results
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


class TestBinomialTestBaselineRateZero:
    """Regression tests for binomtest(p=0) degenerate test fix (H3)."""

    def _make_tester(self):
        """Create a SignificanceTester without loading phase data."""
        from phase4_14_statistical_significance.significance_tester import SignificanceTester
        tester = object.__new__(SignificanceTester)
        tester.alpha = 0.05
        return tester

    def test_zero_baseline_uses_effective_rate(self):
        """When baseline_rate=0, effective_rate should be max(1/n, 1e-10)."""
        tester = self._make_tester()
        result = tester.perform_binomial_test(
            n_successes=5, n_trials=100, baseline_rate=0.0, alternative='greater'
        )
        assert result['expected_rate'] == 0.0
        assert result['effective_rate'] == pytest.approx(1.0 / 100)
        # p-value should be a valid number, not trivially 0
        assert 0.0 <= result['p_value'] <= 1.0

    def test_nonzero_baseline_unchanged(self):
        """Non-zero baseline_rate should be passed through as effective_rate."""
        tester = self._make_tester()
        result = tester.perform_binomial_test(
            n_successes=5, n_trials=100, baseline_rate=0.3, alternative='greater'
        )
        assert result['expected_rate'] == 0.3
        assert result['effective_rate'] == 0.3

    def test_effective_rate_key_present(self):
        """Result dict should include effective_rate key."""
        tester = self._make_tester()
        result = tester.perform_binomial_test(
            n_successes=2, n_trials=50, baseline_rate=0.0, alternative='greater'
        )
        assert 'effective_rate' in result

    def test_zero_baseline_small_n(self):
        """With small n, effective_rate = 1/n should be relatively large."""
        tester = self._make_tester()
        result = tester.perform_binomial_test(
            n_successes=1, n_trials=5, baseline_rate=0.0, alternative='greater'
        )
        assert result['effective_rate'] == pytest.approx(0.2)

    def test_zero_trials_returns_default(self):
        """Zero trials should return early with p_value=1.0."""
        tester = self._make_tester()
        result = tester.perform_binomial_test(
            n_successes=0, n_trials=0, baseline_rate=0.0, alternative='greater'
        )
        assert result['p_value'] == 1.0
        assert result['n_trials'] == 0
