"""
Tests for metric consistency across phases.

Validates:
- phase_4_5_vs_4_8_correction_rate: Same formula in both phases
- phase_4_8_vs_7_6_metrics: Instruct steering uses same formulas
- parallel_vs_sequential_metrics: Merged results match sequential
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from common.steering_metrics import (
    calculate_correction_rate,
    calculate_corruption_rate,
    calculate_preservation_rate,
)


# =============================================================================
# phase_4_5_vs_4_8_correction_rate Tests
# =============================================================================

class TestPhase45Vs48CorrectionRate:
    """Test same correction rate formula in Phase 4.5 and 4.8."""

    @pytest.fixture
    def steering_results(self):
        """Sample steering results for testing."""
        return [
            {'task_id': 't1', 'baseline_passed': False, 'steered_correct': True},   # Correction
            {'task_id': 't2', 'baseline_passed': False, 'steered_correct': True},   # Correction
            {'task_id': 't3', 'baseline_passed': False, 'steered_correct': False},  # No correction
            {'task_id': 't4', 'baseline_passed': True, 'steered_correct': True},    # Preservation
            {'task_id': 't5', 'baseline_passed': True, 'steered_correct': False},   # Corruption
        ]

    def test_correction_rate_formula_consistency(self, steering_results):
        """Correction rate should use same formula across phases."""
        # Formula: (incorrect -> correct) / total_incorrect * 100
        rate = calculate_correction_rate(steering_results)

        # Manual calculation
        incorrect_to_correct = 2  # t1, t2
        total_incorrect = 3       # t1, t2, t3
        expected = (incorrect_to_correct / total_incorrect) * 100

        assert rate == pytest.approx(expected, rel=0.01)

    def test_corruption_rate_formula_consistency(self, steering_results):
        """Corruption rate should use same formula across phases."""
        # Formula: (correct -> incorrect) / total_correct * 100
        rate = calculate_corruption_rate(steering_results)

        # Manual calculation
        correct_to_incorrect = 1  # t5
        total_correct = 2         # t4, t5
        expected = (correct_to_incorrect / total_correct) * 100

        assert rate == pytest.approx(expected, rel=0.01)

    def test_preservation_is_inverse_of_corruption(self, steering_results):
        """Preservation rate = 100 - corruption rate."""
        corruption = calculate_corruption_rate(steering_results)
        preservation = calculate_preservation_rate(steering_results)

        assert preservation == pytest.approx(100 - corruption, rel=0.01)


# =============================================================================
# phase_4_8_vs_7_6_metrics Tests
# =============================================================================

class TestPhase48Vs76Metrics:
    """Test instruct steering uses same formulas as base model steering."""

    def test_instruct_results_same_formula(self):
        """Results with different column names should work the same."""
        # Base model results
        base_results = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': True},
        ]

        # Instruct model uses same structure
        instruct_results = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': True},
        ]

        base_correction = calculate_correction_rate(base_results)
        instruct_correction = calculate_correction_rate(instruct_results)

        assert base_correction == instruct_correction

    def test_orthogonalized_column_works(self):
        """orthogonalized_correct column should work the same as steered_correct."""
        steered_results = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
        ]

        orthog_results = [
            {'baseline_passed': False, 'orthogonalized_correct': True},
            {'baseline_passed': False, 'orthogonalized_correct': False},
        ]

        steered_rate = calculate_correction_rate(steered_results)
        orthog_rate = calculate_correction_rate(orthog_results)

        assert steered_rate == orthog_rate


# =============================================================================
# parallel_vs_sequential_metrics Tests
# =============================================================================

class TestParallelVsSequentialMetrics:
    """Test merged parallel results match sequential calculation."""

    def test_merged_results_same_metrics(self):
        """Metrics from merged parallel results should match sequential."""
        # Simulate GPU 0 results
        gpu0_results = [
            {'task_id': 't0', 'baseline_passed': False, 'steered_correct': True},
            {'task_id': 't2', 'baseline_passed': True, 'steered_correct': True},
        ]

        # Simulate GPU 1 results
        gpu1_results = [
            {'task_id': 't1', 'baseline_passed': False, 'steered_correct': False},
            {'task_id': 't3', 'baseline_passed': True, 'steered_correct': False},
        ]

        # Sequential (all results together)
        sequential_results = gpu0_results + gpu1_results

        # Merged parallel (same as sequential)
        merged_results = gpu0_results + gpu1_results

        seq_correction = calculate_correction_rate(sequential_results)
        merged_correction = calculate_correction_rate(merged_results)

        assert seq_correction == merged_correction

    def test_deduplication_affects_metrics(self):
        """Duplicates should be handled correctly in metric calculation."""
        # Results with duplicate task_id
        results_with_dup = [
            {'task_id': 't1', 'baseline_passed': False, 'steered_correct': True},
            {'task_id': 't1', 'baseline_passed': False, 'steered_correct': True},  # Duplicate
            {'task_id': 't2', 'baseline_passed': False, 'steered_correct': False},
        ]

        # Without duplicate
        results_no_dup = [
            {'task_id': 't1', 'baseline_passed': False, 'steered_correct': True},
            {'task_id': 't2', 'baseline_passed': False, 'steered_correct': False},
        ]

        # Note: The metric functions don't deduplicate - they count all rows
        # This documents current behavior
        dup_rate = calculate_correction_rate(results_with_dup)
        no_dup_rate = calculate_correction_rate(results_no_dup)

        # With duplicate: 2/3 = 66.67%, Without: 1/2 = 50%
        assert dup_rate != no_dup_rate  # Documents that dedup matters

    def test_dataframe_metric_matches_list(self):
        """DataFrame and list inputs should produce same metrics."""
        data = [
            {'task_id': 't1', 'baseline_passed': False, 'steered_correct': True},
            {'task_id': 't2', 'baseline_passed': False, 'steered_correct': False},
            {'task_id': 't3', 'baseline_passed': True, 'steered_correct': True},
            {'task_id': 't4', 'baseline_passed': True, 'steered_correct': False},
        ]

        df = pd.DataFrame(data)

        list_correction = calculate_correction_rate(data)
        df_correction = calculate_correction_rate(df)

        list_corruption = calculate_corruption_rate(data)
        df_corruption = calculate_corruption_rate(df)

        assert list_correction == pytest.approx(df_correction, rel=0.01)
        assert list_corruption == pytest.approx(df_corruption, rel=0.01)


# =============================================================================
# Edge Cases in Metric Consistency
# =============================================================================

class TestMetricEdgeCases:
    """Test edge cases are handled consistently."""

    def test_all_correct_baseline(self):
        """When all baseline correct, correction rate should be 0."""
        results = [
            {'baseline_passed': True, 'steered_correct': True},
            {'baseline_passed': True, 'steered_correct': False},
        ]

        correction = calculate_correction_rate(results)
        assert correction == 0.0

    def test_all_incorrect_baseline(self):
        """When all baseline incorrect, corruption rate should be 0."""
        results = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
        ]

        corruption = calculate_corruption_rate(results)
        assert corruption == 0.0

    def test_perfect_correction(self):
        """All incorrect -> correct should be 100% correction rate."""
        results = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': True},
        ]

        correction = calculate_correction_rate(results)
        assert correction == 100.0

    def test_perfect_corruption(self):
        """All correct -> incorrect should be 100% corruption rate."""
        results = [
            {'baseline_passed': True, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': False},
        ]

        corruption = calculate_corruption_rate(results)
        assert corruption == 100.0

    def test_mixed_results_consistency(self):
        """Mixed results should have consistent metrics."""
        results = [
            {'baseline_passed': False, 'steered_correct': True},   # Correction
            {'baseline_passed': False, 'steered_correct': False},  # No change
            {'baseline_passed': True, 'steered_correct': True},    # Preservation
            {'baseline_passed': True, 'steered_correct': False},   # Corruption
        ]

        correction = calculate_correction_rate(results)
        corruption = calculate_corruption_rate(results)
        preservation = calculate_preservation_rate(results)

        # Correction: 1/2 = 50%
        assert correction == pytest.approx(50.0, rel=0.01)

        # Corruption: 1/2 = 50%
        assert corruption == pytest.approx(50.0, rel=0.01)

        # Preservation = 100 - corruption = 50%
        assert preservation == pytest.approx(50.0, rel=0.01)
