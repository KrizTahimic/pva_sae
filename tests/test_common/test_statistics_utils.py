"""Tests for common.statistics_utils."""

import pytest

from common.statistics_utils import binomial_significance_test, calculate_effect_size


class TestBinomialSignificanceTest:
    """Tests for binomial_significance_test()."""

    def test_normal_inputs(self):
        """15 successes out of 100 trials with expected_rate=0.10 should return valid result."""
        result = binomial_significance_test(15, 100, expected_rate=0.10)

        assert result['n_successes'] == 15
        assert result['n_trials'] == 100
        assert result['observed_rate'] == pytest.approx(0.15)
        assert result['expected_rate'] == 0.10
        assert result['effect_size'] == pytest.approx(0.05)
        assert result['alternative'] == 'greater'
        assert isinstance(result['p_value'], float)
        assert 0.0 <= result['p_value'] <= 1.0
        assert result['significant'] in (True, False)
        ci_low, ci_high = result['confidence_interval']
        assert ci_low <= ci_high

    def test_expected_rate_zero_raises(self):
        """expected_rate=0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="expected_rate must be in"):
            binomial_significance_test(5, 100, expected_rate=0.0)

    def test_expected_rate_one_raises(self):
        """expected_rate=1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="expected_rate must be in"):
            binomial_significance_test(5, 100, expected_rate=1.0)

    def test_zero_trials_returns_safe_result(self):
        """n_trials=0 should return a safe non-significant result."""
        result = binomial_significance_test(0, 0, expected_rate=0.10)

        assert result['n_successes'] == 0
        assert result['n_trials'] == 0
        assert result['observed_rate'] == 0.0
        assert result['p_value'] == 1.0
        assert result['significant'] is False
        assert result['effect_size'] == 0.0
        assert result['confidence_interval'] == (0.0, 0.0)


class TestCalculateEffectSize:
    """Tests for calculate_effect_size()."""

    def test_basic_correctness(self):
        """0.15 - 0.10 should equal 0.05."""
        effect = calculate_effect_size(observed_rate=0.15, expected_rate=0.10)
        assert effect == pytest.approx(0.05)
