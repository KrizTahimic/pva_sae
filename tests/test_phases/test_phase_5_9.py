"""
Tests for Phase 5.9: Orthogonalization Significance Testing

Validates:
- H3 regression: binomtest with baseline_rate=0 uses p=max(1/n, 1e-10), not p=0
- Same fix pattern as Phase 4.14
"""

import pytest


class TestBinomialTestBaselineRateZero:
    """Regression tests for binomtest(p=0) degenerate test fix (H3) in Phase 5.9."""

    def _make_tester(self):
        """Create an OrthogonalizationSignificanceTester without loading phase data."""
        from phase5_9_orthogonalization_significance.orthogonalization_significance_tester import (
            OrthogonalizationSignificanceTester
        )
        tester = object.__new__(OrthogonalizationSignificanceTester)
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
