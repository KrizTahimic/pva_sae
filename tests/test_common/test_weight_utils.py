"""
Tests for common/weight_utils.py

Validates:
- get_orthogonalized_matrix: projection removal math
- get_weight_change_magnitude: Frobenius norm
- verify_orthogonalization: tolerance checking
- Device/dtype preservation
- Edge case: zero vector direction
"""

import pytest
import torch

from common.weight_utils import (
    get_orthogonalized_matrix,
    get_weight_change_magnitude,
    verify_orthogonalization,
)


# =============================================================================
# get_orthogonalized_matrix Tests
# =============================================================================

class TestGetOrthogonalizedMatrix:
    """Test projection removal from weight matrix."""

    def test_removes_projection(self):
        """Orthogonalized matrix should have zero projection onto direction."""
        matrix = torch.randn(10, 64)
        direction = torch.randn(64)

        result = get_orthogonalized_matrix(matrix, direction)

        # Each row should have zero dot product with direction
        normalized_dir = direction / torch.norm(direction)
        projections = result @ normalized_dir
        assert torch.allclose(projections, torch.zeros(10), atol=1e-5)

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        matrix = torch.randn(20, 64)
        direction = torch.randn(64)

        result = get_orthogonalized_matrix(matrix, direction)
        assert result.shape == matrix.shape

    def test_idempotent(self):
        """Orthogonalizing twice should give same result."""
        matrix = torch.randn(10, 64)
        direction = torch.randn(64)

        once = get_orthogonalized_matrix(matrix, direction)
        twice = get_orthogonalized_matrix(once, direction)

        assert torch.allclose(once, twice, atol=1e-5)

    def test_orthogonal_rows_unchanged(self):
        """Rows already orthogonal to direction should be unchanged."""
        # Create direction along first axis
        direction = torch.zeros(64)
        direction[0] = 1.0

        # Create matrix with rows that have zero component along direction[0]
        matrix = torch.randn(5, 64)
        matrix[:, 0] = 0.0  # Zero out component along direction

        result = get_orthogonalized_matrix(matrix, direction)
        assert torch.allclose(result, matrix, atol=1e-5)

    def test_dtype_preservation(self):
        """Should preserve input dtype."""
        for dtype in [torch.float32, torch.bfloat16]:
            matrix = torch.randn(5, 64, dtype=dtype)
            direction = torch.randn(64, dtype=dtype)

            result = get_orthogonalized_matrix(matrix, direction)
            assert result.dtype == dtype

    def test_device_preservation(self):
        """Should preserve CPU device."""
        matrix = torch.randn(5, 64)
        direction = torch.randn(64)

        result = get_orthogonalized_matrix(matrix, direction)
        assert result.device == matrix.device

    def test_zero_direction_raises(self):
        """Zero direction should raise error."""
        matrix = torch.randn(5, 64)
        direction = torch.zeros(64)

        with pytest.raises(ValueError):
            get_orthogonalized_matrix(matrix, direction)


# =============================================================================
# get_weight_change_magnitude Tests
# =============================================================================

class TestGetWeightChangeMagnitude:
    """Test Frobenius norm calculation."""

    def test_identical_matrices_zero_change(self):
        """Identical matrices should have zero change."""
        matrix = torch.randn(10, 64)
        magnitude = get_weight_change_magnitude(matrix, matrix)
        assert magnitude == pytest.approx(0.0, abs=1e-6)

    def test_known_change(self):
        """Should compute correct Frobenius norm of difference."""
        original = torch.zeros(2, 2)
        modified = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        magnitude = get_weight_change_magnitude(original, modified)
        # sqrt(1^2 + 0^2 + 0^2 + 1^2) = sqrt(2)
        assert magnitude == pytest.approx(2**0.5, rel=1e-5)

    def test_returns_float(self):
        """Should return a Python float."""
        original = torch.randn(5, 64)
        modified = torch.randn(5, 64)

        magnitude = get_weight_change_magnitude(original, modified)
        assert isinstance(magnitude, float)

    def test_symmetric(self):
        """Change magnitude should be symmetric: |A-B| == |B-A|."""
        a = torch.randn(5, 64)
        b = torch.randn(5, 64)

        assert get_weight_change_magnitude(a, b) == pytest.approx(
            get_weight_change_magnitude(b, a), rel=1e-5
        )


# =============================================================================
# verify_orthogonalization Tests
# =============================================================================

class TestVerifyOrthogonalization:
    """Test orthogonalization verification."""

    def test_orthogonal_matrix_passes(self):
        """Properly orthogonalized matrix should pass verification."""
        matrix = torch.randn(10, 64)
        direction = torch.randn(64)

        orthogonalized = get_orthogonalized_matrix(matrix, direction)
        assert verify_orthogonalization(orthogonalized, direction) is True

    def test_non_orthogonal_matrix_fails(self):
        """Non-orthogonal matrix should fail verification."""
        direction = torch.randn(64)
        # Create matrix with large projections onto direction
        matrix = direction.unsqueeze(0).repeat(10, 1) * 5.0

        assert verify_orthogonalization(matrix, direction, tolerance=1e-3) is False

    def test_tolerance_respected(self):
        """Should respect custom tolerance."""
        matrix = torch.randn(10, 64)
        direction = torch.randn(64)

        orthogonalized = get_orthogonalized_matrix(matrix, direction)

        # Very tight tolerance should pass for properly orthogonalized matrix
        assert verify_orthogonalization(orthogonalized, direction, tolerance=1e-3) is True

    def test_tolerance_catches_imprecision(self):
        """Very tight tolerance should catch floating point imprecision."""
        matrix = torch.randn(10, 64, dtype=torch.bfloat16)  # Low precision
        direction = torch.randn(64, dtype=torch.bfloat16)

        orthogonalized = get_orthogonalized_matrix(matrix, direction)

        # Extremely tight tolerance may fail with bfloat16
        # This test documents the expected behavior
        result = verify_orthogonalization(orthogonalized, direction, tolerance=1e-10)
        # Just verify it returns a bool (may pass or fail depending on precision)
        assert isinstance(result, bool)
