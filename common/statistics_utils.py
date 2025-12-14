"""
Statistical testing utilities.

This module provides utilities for:
- Binomial significance testing for steering experiments
- Confidence interval calculation
"""


from scipy.stats import binomtest

from .logging import get_logger

logger = get_logger("common.statistics_utils")


def binomial_significance_test(
    n_successes: int,
    n_trials: int,
    expected_rate: float,
    alpha: float = 0.05,
    alternative: str = 'greater'
) -> dict:
    """
    Perform binomial test comparing observed vs expected rate.

    This is commonly used to test whether steering interventions
    produce statistically significant improvements over baseline.

    Args:
        n_successes: Number of successful outcomes (e.g., corrections)
        n_trials: Total number of trials
        expected_rate: Expected success rate under null hypothesis
        alpha: Significance level (default: 0.05)
        alternative: Type of alternative hypothesis
            - 'greater': observed > expected (for correction experiments)
            - 'less': observed < expected (for corruption experiments)
            - 'two-sided': observed != expected

    Returns:
        dict containing:
            - n_successes: Input count
            - n_trials: Input count
            - observed_rate: Actual success rate
            - expected_rate: Expected rate under null
            - p_value: Test p-value
            - significant: Whether p < alpha
            - alternative: Type of test performed
            - confidence_interval: (lower, upper) bounds
            - effect_size: observed_rate - expected_rate

    Example:
        >>> # Test if 15/100 corrections is better than 10% baseline
        >>> result = binomial_significance_test(15, 100, 0.10, alternative='greater')
        >>> if result['significant']:
        ...     print(f"Significant improvement: p={result['p_value']:.4f}")
    """
    if n_trials == 0:
        return {
            'n_successes': 0,
            'n_trials': 0,
            'observed_rate': 0.0,
            'expected_rate': expected_rate,
            'p_value': 1.0,
            'significant': False,
            'alternative': alternative,
            'effect_size': 0.0,
            'confidence_interval': (0.0, 0.0)
        }

    result = binomtest(n_successes, n_trials, p=expected_rate, alternative=alternative)
    observed_rate = n_successes / n_trials

    # Get confidence interval
    ci = result.proportion_ci(confidence_level=1 - alpha)

    return {
        'n_successes': n_successes,
        'n_trials': n_trials,
        'observed_rate': observed_rate,
        'expected_rate': expected_rate,
        'p_value': result.pvalue,
        'significant': result.pvalue < alpha,
        'alternative': alternative,
        'confidence_interval': (ci.low, ci.high),
        'effect_size': observed_rate - expected_rate
    }


def calculate_effect_size(
    observed_rate: float,
    expected_rate: float
) -> float:
    """
    Calculate simple effect size as difference in rates.

    Args:
        observed_rate: Actual observed success rate
        expected_rate: Expected rate under null hypothesis

    Returns:
        Effect size (positive = improvement, negative = degradation)
    """
    return observed_rate - expected_rate


def format_significance_result(result: dict) -> str:
    """
    Format binomial test result as human-readable string.

    Args:
        result: Output from binomial_significance_test()

    Returns:
        Formatted string describing the result
    """
    sig_marker = "*" if result['significant'] else ""
    ci_low, ci_high = result['confidence_interval']

    return (
        f"Observed: {result['observed_rate']:.1%} "
        f"(n={result['n_successes']}/{result['n_trials']}), "
        f"Expected: {result['expected_rate']:.1%}, "
        f"p={result['p_value']:.4f}{sig_marker}, "
        f"95% CI: [{ci_low:.1%}, {ci_high:.1%}]"
    )
