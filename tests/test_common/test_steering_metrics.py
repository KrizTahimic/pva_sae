"""
Tests for common/steering_metrics.py

Validates:
- Correction rate formula: (incorrect->correct) / total_incorrect * 100
- Corruption rate formula: (correct->incorrect) / total_correct * 100
- Preservation rate consistency: preservation_rate == 100 - corruption_rate
- Empty data handling: Zero division prevention
- DataFrame vs dict input: Both formats produce same result
- Code similarity tokenization: Handles valid and broken Python code
"""

import pytest
import pandas as pd

from common.steering_metrics import (
    calculate_correction_rate,
    calculate_corruption_rate,
    calculate_preservation_rate,
    calculate_code_similarity,
    create_last_position_steering_hook,
)


# =============================================================================
# Correction Rate Tests
# =============================================================================

class TestCorrectionRateFormula:
    """Test correction rate calculation: (incorrect->correct) / total_incorrect * 100"""

    def test_basic_correction_rate(self):
        """Test with simple known values."""
        results = [
            {'baseline_passed': False, 'steered_correct': True},   # Corrected
            {'baseline_passed': False, 'steered_correct': True},   # Corrected
            {'baseline_passed': False, 'steered_correct': False},  # Not corrected
            {'baseline_passed': True, 'steered_correct': True},    # Not counted (was correct)
        ]
        # 2 corrections / 3 incorrect = 66.67%
        rate = calculate_correction_rate(results)
        assert rate == pytest.approx(66.67, rel=0.01)

    def test_all_corrected(self):
        """Test when all incorrect problems are corrected."""
        results = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': True},
        ]
        rate = calculate_correction_rate(results)
        assert rate == 100.0

    def test_none_corrected(self):
        """Test when no incorrect problems are corrected."""
        results = [
            {'baseline_passed': False, 'steered_correct': False},
            {'baseline_passed': False, 'steered_correct': False},
        ]
        rate = calculate_correction_rate(results)
        assert rate == 0.0

    def test_correction_rate_dataframe(self):
        """Test correction rate with DataFrame input."""
        df = pd.DataFrame([
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': True},
        ])
        rate = calculate_correction_rate(df)
        assert rate == pytest.approx(50.0, rel=0.01)

    def test_orthogonalized_column_detection(self):
        """Test that orthogonalized_correct column is detected."""
        results = [
            {'baseline_passed': False, 'orthogonalized_correct': True},
            {'baseline_passed': False, 'orthogonalized_correct': False},
        ]
        rate = calculate_correction_rate(results)
        assert rate == pytest.approx(50.0, rel=0.01)


# =============================================================================
# Corruption Rate Tests
# =============================================================================

class TestCorruptionRateFormula:
    """Test corruption rate calculation: (correct->incorrect) / total_correct * 100"""

    def test_basic_corruption_rate(self):
        """Test with simple known values."""
        results = [
            {'baseline_passed': True, 'steered_correct': False},   # Corrupted
            {'baseline_passed': True, 'steered_correct': True},    # Preserved
            {'baseline_passed': True, 'steered_correct': True},    # Preserved
            {'baseline_passed': False, 'steered_correct': False},  # Not counted (was incorrect)
        ]
        # 1 corruption / 3 correct = 33.33%
        rate = calculate_corruption_rate(results)
        assert rate == pytest.approx(33.33, rel=0.01)

    def test_all_corrupted(self):
        """Test when all correct problems are corrupted."""
        results = [
            {'baseline_passed': True, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': False},
        ]
        rate = calculate_corruption_rate(results)
        assert rate == 100.0

    def test_none_corrupted(self):
        """Test when no correct problems are corrupted."""
        results = [
            {'baseline_passed': True, 'steered_correct': True},
            {'baseline_passed': True, 'steered_correct': True},
        ]
        rate = calculate_corruption_rate(results)
        assert rate == 0.0

    def test_corruption_rate_dataframe(self):
        """Test corruption rate with DataFrame input."""
        df = pd.DataFrame([
            {'baseline_passed': True, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
        ])
        rate = calculate_corruption_rate(df)
        assert rate == pytest.approx(50.0, rel=0.01)


# =============================================================================
# Preservation Rate Tests
# =============================================================================

class TestPreservationRateConsistency:
    """Test preservation_rate == 100 - corruption_rate"""

    def test_preservation_equals_inverse_corruption(self):
        """Preservation rate should be 100 - corruption rate."""
        results = [
            {'baseline_passed': True, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': True},
            {'baseline_passed': True, 'steered_correct': True},
        ]
        corruption = calculate_corruption_rate(results)
        preservation = calculate_preservation_rate(results)
        assert preservation == pytest.approx(100 - corruption, rel=0.01)

    def test_full_preservation(self):
        """All preserved means 100% preservation, 0% corruption."""
        results = [
            {'baseline_passed': True, 'steered_correct': True},
            {'baseline_passed': True, 'steered_correct': True},
        ]
        assert calculate_preservation_rate(results) == 100.0
        assert calculate_corruption_rate(results) == 0.0

    def test_zero_preservation(self):
        """All corrupted means 0% preservation, 100% corruption."""
        results = [
            {'baseline_passed': True, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': False},
        ]
        assert calculate_preservation_rate(results) == 0.0
        assert calculate_corruption_rate(results) == 100.0


# =============================================================================
# Empty Data Handling Tests
# =============================================================================

class TestEmptyDataHandling:
    """Test zero division prevention with empty/edge-case data."""

    def test_empty_list(self):
        """Empty list should return 0.0 without error."""
        assert calculate_correction_rate([]) == 0.0
        assert calculate_corruption_rate([]) == 0.0
        assert calculate_preservation_rate([]) == 100.0  # 100 - 0

    def test_empty_dataframe(self):
        """Empty DataFrame should return 0.0 without error."""
        df = pd.DataFrame(columns=['baseline_passed', 'steered_correct'])
        assert calculate_correction_rate(df) == 0.0
        assert calculate_corruption_rate(df) == 0.0

    def test_no_incorrect_baseline(self):
        """No incorrect problems should return 0.0 for correction rate."""
        results = [
            {'baseline_passed': True, 'steered_correct': True},
            {'baseline_passed': True, 'steered_correct': True},
        ]
        assert calculate_correction_rate(results) == 0.0

    def test_no_correct_baseline(self):
        """No correct problems should return 0.0 for corruption rate."""
        results = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
        ]
        assert calculate_corruption_rate(results) == 0.0


# =============================================================================
# DataFrame vs Dict Input Tests
# =============================================================================

class TestDataFrameVsDictInput:
    """Test that both input formats produce same result."""

    def test_same_correction_rate(self):
        """List and DataFrame inputs should give same correction rate."""
        data = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': True},
        ]
        df = pd.DataFrame(data)

        list_rate = calculate_correction_rate(data)
        df_rate = calculate_correction_rate(df)
        assert list_rate == pytest.approx(df_rate, rel=0.01)

    def test_same_corruption_rate(self):
        """List and DataFrame inputs should give same corruption rate."""
        data = [
            {'baseline_passed': True, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
        ]
        df = pd.DataFrame(data)

        list_rate = calculate_corruption_rate(data)
        df_rate = calculate_corruption_rate(df)
        assert list_rate == pytest.approx(df_rate, rel=0.01)


# =============================================================================
# Code Similarity Tests
# =============================================================================

class TestCodeSimilarityTokenization:
    """Test code similarity handles valid and broken Python code."""

    def test_identical_code(self):
        """Identical code should have similarity 1.0."""
        code = "def foo(x):\n    return x + 1"
        assert calculate_code_similarity(code, code) == 1.0

    def test_different_whitespace_same_tokens(self):
        """Different whitespace should still be similar (tokenizer ignores formatting)."""
        code1 = "def foo(x):\n    return x + 1"
        code2 = "def foo(x):\n        return x + 1"  # Extra indent
        similarity = calculate_code_similarity(code1, code2)
        assert similarity > 0.9  # Should be very similar

    def test_completely_different_code(self):
        """Completely different code should have low similarity."""
        code1 = "def add(a, b): return a + b"
        code2 = "class MyClass: pass"
        similarity = calculate_code_similarity(code1, code2)
        assert similarity < 0.5

    def test_empty_code_handling(self):
        """Empty code strings should be handled correctly."""
        assert calculate_code_similarity("", "") == 1.0  # Both empty = identical
        assert calculate_code_similarity("", "def foo(): pass") == 0.0  # One empty

    def test_broken_python_code(self):
        """Broken Python code should fall back to simple splitting."""
        code1 = "def foo(x:\n    incomplete"  # Invalid syntax
        code2 = "def foo(x):\n    return x"
        # Should not raise, just use fallback
        similarity = calculate_code_similarity(code1, code2)
        assert 0.0 <= similarity <= 1.0

    def test_comments_ignored(self):
        """Comments should be ignored in similarity calculation."""
        code1 = "def foo(x):\n    # comment\n    return x"
        code2 = "def foo(x):\n    return x"
        similarity = calculate_code_similarity(code1, code2)
        assert similarity > 0.9  # Should be very similar


# =============================================================================
# Steering Hook Tests
# =============================================================================

import torch
from common.direction_utils import normalize_direction


class TestSteeringHookNormalization:
    """Test that steering hook requires pre-normalized direction.

    Hook validates that directions are unit-normalized, ensuring coefficient
    directly controls perturbation magnitude. Callers must normalize directions
    using normalize_direction() before passing to hook.
    """

    def test_direction_must_be_pre_normalized(self):
        """Hook should use pre-normalized direction for steering."""
        # Create pre-normalized direction
        direction = normalize_direction(torch.tensor([3.0, 4.0]))  # normalized = [0.6, 0.8]
        coefficient = 10.0

        hook = create_last_position_steering_hook(direction, coefficient)

        # Create mock input: [batch=1, seq_len=2, d_model=2]
        residual = torch.zeros(1, 2, 2)
        mock_input = (residual,)

        # Call hook
        output = hook(None, mock_input)
        modified_residual = output[0]

        # Expected: direction * coefficient at last position
        # Since direction is pre-normalized: [0.6, 0.8] * 10 = [6.0, 8.0]
        expected_steering = torch.tensor([6.0, 8.0])

        actual_steering = modified_residual[0, -1, :]
        assert torch.allclose(actual_steering, expected_steering, atol=1e-5)

    def test_rejects_non_unit_norm_direction(self):
        """Hook should reject directions that are not unit-normalized."""
        # Non-normalized direction
        direction = torch.tensor([3.0, 4.0])  # norm = 5.0
        coefficient = 10.0

        with pytest.raises(ValueError, match="not unit-normalized"):
            create_last_position_steering_hook(direction, coefficient)

    def test_same_coefficient_same_magnitude_for_normalized_directions(self):
        """Same coefficient should produce same perturbation magnitude for normalized directions."""
        coefficient = 5.0

        # Both directions normalized to unit norm
        dir1 = normalize_direction(torch.tensor([1.0, 0.0]))
        dir2 = normalize_direction(torch.tensor([0.0, 1.0]))

        hook1 = create_last_position_steering_hook(dir1, coefficient)
        hook2 = create_last_position_steering_hook(dir2, coefficient)

        # Apply both hooks - residual must match d_model=2
        residual = torch.zeros(1, 2, 2)

        output1 = hook1(None, (residual.clone(),))
        output2 = hook2(None, (residual.clone(),))

        # Both should have same perturbation magnitude = coefficient
        perturbation1 = output1[0][0, -1, :].norm()
        perturbation2 = output2[0][0, -1, :].norm()

        assert perturbation1 == pytest.approx(coefficient, rel=1e-5)
        assert perturbation2 == pytest.approx(coefficient, rel=1e-5)

    def test_only_last_position_modified(self):
        """Hook should only modify the last position in sequence."""
        direction = normalize_direction(torch.tensor([1.0, 0.0]))
        coefficient = 2.0

        hook = create_last_position_steering_hook(direction, coefficient)

        # Sequence with 5 positions
        residual = torch.ones(1, 5, 2) * 0.5
        original_residual = residual.clone()

        output = hook(None, (residual,))
        modified = output[0]

        # First 4 positions should be unchanged
        assert torch.allclose(modified[0, :4, :], original_residual[0, :4, :])

        # Last position should be modified
        assert not torch.allclose(modified[0, -1, :], original_residual[0, -1, :])

    def test_preserves_other_tuple_elements(self):
        """Hook should preserve other elements in input tuple."""
        direction = normalize_direction(torch.tensor([1.0]))
        hook = create_last_position_steering_hook(direction, 1.0)

        residual = torch.zeros(1, 1, 1)
        extra_arg1 = "extra1"
        extra_arg2 = {"key": "value"}

        output = hook(None, (residual, extra_arg1, extra_arg2))

        assert len(output) == 3
        assert output[1] == extra_arg1
        assert output[2] == extra_arg2

    def test_rejects_zero_norm_direction(self):
        """Hook should reject zero-norm direction (cannot normalize)."""
        direction = torch.tensor([0.0, 0.0])
        coefficient = 10.0

        with pytest.raises(ValueError, match="not unit-normalized"):
            create_last_position_steering_hook(direction, coefficient)


# =============================================================================
# Missing baseline_passed KeyError Tests
# =============================================================================

class TestBaselinePassedRequired:
    """Test that missing 'baseline_passed' key raises KeyError.

    The steering metrics functions require 'baseline_passed' to determine
    which problems were initially correct/incorrect. If this key is missing
    (e.g., from malformed data), the functions should raise KeyError rather
    than silently returning incorrect results.
    """

    def test_correction_rate_raises_on_missing_baseline(self):
        """calculate_correction_rate should raise KeyError when baseline_passed is missing."""
        results = [
            {'steered_correct': True},   # Missing baseline_passed
            {'steered_correct': False},  # Missing baseline_passed
        ]

        with pytest.raises(KeyError):
            calculate_correction_rate(results)

    def test_corruption_rate_raises_on_missing_baseline(self):
        """calculate_corruption_rate should raise KeyError when baseline_passed is missing."""
        results = [
            {'steered_correct': True},   # Missing baseline_passed
            {'steered_correct': False},  # Missing baseline_passed
        ]

        with pytest.raises(KeyError):
            calculate_corruption_rate(results)

    def test_preservation_rate_raises_on_missing_baseline(self):
        """calculate_preservation_rate should raise KeyError when baseline_passed is missing.

        Preservation rate delegates to corruption rate, which requires baseline_passed.
        """
        results = [
            {'steered_correct': True},   # Missing baseline_passed
            {'steered_correct': False},  # Missing baseline_passed
        ]

        with pytest.raises(KeyError):
            calculate_preservation_rate(results)

    def test_correction_rate_raises_on_missing_baseline_dataframe(self):
        """calculate_correction_rate should raise KeyError for DataFrame without baseline_passed."""
        df = pd.DataFrame([
            {'steered_correct': True},
            {'steered_correct': False},
        ])

        with pytest.raises(KeyError):
            calculate_correction_rate(df)

    def test_corruption_rate_raises_on_missing_baseline_dataframe(self):
        """calculate_corruption_rate should raise KeyError for DataFrame without baseline_passed."""
        df = pd.DataFrame([
            {'steered_correct': True},
            {'steered_correct': False},
        ])

        with pytest.raises(KeyError):
            calculate_corruption_rate(df)
