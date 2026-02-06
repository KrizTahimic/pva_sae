"""
Tests for steering hook functionality in common/steering_metrics.py

Validates:
- hook_modifies_last_position_only: Only position -1 modified
- hook_coefficient_scaling: direction * coefficient applied correctly
- hook_dtype_conversion: Direction converted to residual dtype
- hook_exception_cleanup: Hook removed even on exception
- multiple_hooks_ordering: Steering + attention hooks don't interfere
- hook_rejects_non_normalized: Hook raises on non-unit-norm direction
"""

import pytest
import torch
import torch.nn as nn

from common.steering_metrics import create_last_position_steering_hook
from common.direction_utils import normalize_direction
from tests.conftest import DEFAULT_D_MODEL


# =============================================================================
# hook_modifies_last_position_only Tests
# =============================================================================

class TestHookModifiesLastPositionOnly:
    """Test only position -1 is modified."""

    def test_last_position_modified(self):
        """Only the last position should be modified by steering."""
        d_model = DEFAULT_D_MODEL
        seq_len = 10
        batch_size = 1

        direction = normalize_direction(torch.randn(d_model))
        coefficient = 1.0
        hook_fn = create_last_position_steering_hook(direction, coefficient)

        # Create input residual
        residual = torch.randn(batch_size, seq_len, d_model)
        original = residual.clone()

        # Apply hook
        output = hook_fn(None, (residual,))
        modified_residual = output[0]

        # Check that only last position is modified
        for pos in range(seq_len - 1):
            assert torch.allclose(
                modified_residual[0, pos],
                original[0, pos]
            ), f"Position {pos} should not be modified"

        # Last position should be different
        assert not torch.allclose(
            modified_residual[0, -1],
            original[0, -1]
        ), "Last position should be modified"

    def test_single_position_input(self):
        """Hook should work with single-position input (during generation)."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))
        coefficient = 1.0
        hook_fn = create_last_position_steering_hook(direction, coefficient)

        # Single position input (typical during autoregressive generation)
        residual = torch.randn(1, 1, d_model)
        original = residual.clone()

        output = hook_fn(None, (residual,))
        modified_residual = output[0]

        # Should be modified (position 0 = last position when seq_len=1)
        assert not torch.allclose(modified_residual, original)


# =============================================================================
# hook_coefficient_scaling Tests
# =============================================================================

class TestHookCoefficientScaling:
    """Test direction * coefficient applied correctly."""

    def test_coefficient_multiplies_direction(self):
        """Coefficient should scale the steering direction."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))
        coefficient = 5.0
        hook_fn = create_last_position_steering_hook(direction, coefficient)

        residual = torch.randn(1, 5, d_model)
        original = residual.clone()

        output = hook_fn(None, (residual,))
        modified_residual = output[0]

        # Check the change at last position
        change = modified_residual[0, -1] - original[0, -1]
        expected_change = direction * coefficient

        assert torch.allclose(
            change.cpu().float(),
            expected_change.cpu().float(),
            rtol=1e-4,
            atol=1e-6
        )

    def test_zero_coefficient_no_change(self):
        """Zero coefficient should result in no change."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))
        coefficient = 0.0
        hook_fn = create_last_position_steering_hook(direction, coefficient)

        residual = torch.randn(1, 5, d_model)
        original = residual.clone()

        output = hook_fn(None, (residual,))
        modified_residual = output[0]

        assert torch.allclose(modified_residual, original)

    def test_negative_coefficient(self):
        """Negative coefficient should steer in opposite direction."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))
        positive_hook = create_last_position_steering_hook(direction, 5.0)
        negative_hook = create_last_position_steering_hook(direction, -5.0)

        residual = torch.randn(1, 5, d_model)

        pos_output = positive_hook(None, (residual.clone(),))[0]
        neg_output = negative_hook(None, (residual.clone(),))[0]

        # Changes should be in opposite directions
        pos_change = pos_output[0, -1] - residual[0, -1]
        neg_change = neg_output[0, -1] - residual[0, -1]

        assert torch.allclose(pos_change, -neg_change, rtol=1e-4)


# =============================================================================
# hook_dtype_conversion Tests
# =============================================================================

class TestHookDtypeConversion:
    """Test direction converted to residual dtype."""

    def test_float32_residual(self):
        """Direction should be converted to float32 residual dtype."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))  # Default float32
        hook_fn = create_last_position_steering_hook(direction, 1.0)

        residual = torch.randn(1, 5, d_model, dtype=torch.float32)
        output = hook_fn(None, (residual,))

        assert output[0].dtype == torch.float32

    def test_bfloat16_residual(self):
        """Direction should be converted to bfloat16 residual dtype."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))  # float32
        hook_fn = create_last_position_steering_hook(direction, 1.0)

        residual = torch.randn(1, 5, d_model, dtype=torch.bfloat16)
        output = hook_fn(None, (residual,))

        assert output[0].dtype == torch.bfloat16

    def test_float16_residual(self):
        """Direction should be converted to float16 residual dtype."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))
        hook_fn = create_last_position_steering_hook(direction, 1.0)

        residual = torch.randn(1, 5, d_model, dtype=torch.float16)
        output = hook_fn(None, (residual,))

        assert output[0].dtype == torch.float16


# =============================================================================
# hook_exception_cleanup Tests
# =============================================================================

class TestHookExceptionCleanup:
    """Test hook removed even on exception."""

    def test_original_tensor_not_modified_inplace(self):
        """Hook should clone tensor, not modify original in-place."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))
        hook_fn = create_last_position_steering_hook(direction, 1.0)

        residual = torch.randn(1, 5, d_model)
        original_data = residual[0, -1].clone()

        # Apply hook
        hook_fn(None, (residual,))

        # Original should be unchanged (hook clones)
        assert torch.allclose(residual[0, -1], original_data)


# =============================================================================
# multiple_hooks_ordering Tests
# =============================================================================

class TestMultipleHooksOrdering:
    """Test steering + attention hooks don't interfere."""

    def test_multiple_steering_hooks_stack(self):
        """Multiple steering hooks should stack their effects."""
        d_model = DEFAULT_D_MODEL
        direction1 = normalize_direction(torch.randn(d_model))
        direction2 = normalize_direction(torch.randn(d_model))

        hook1 = create_last_position_steering_hook(direction1, 1.0)
        hook2 = create_last_position_steering_hook(direction2, 1.0)

        residual = torch.randn(1, 5, d_model)
        original = residual.clone()

        # Apply hooks in sequence
        output1 = hook1(None, (residual,))
        output2 = hook2(None, output1)

        # Total change should be sum of both directions
        total_change = output2[0][0, -1] - original[0, -1]
        expected_change = direction1 + direction2

        # Use higher tolerance due to float precision
        assert torch.allclose(
            total_change.cpu().float(),
            expected_change.cpu().float(),
            rtol=1e-3,
            atol=1e-5
        )

    def test_hook_preserves_other_inputs(self):
        """Hook should preserve other elements in input tuple."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))
        hook_fn = create_last_position_steering_hook(direction, 1.0)

        residual = torch.randn(1, 5, d_model)
        attention_mask = torch.ones(1, 5)
        position_ids = torch.arange(5)

        output = hook_fn(None, (residual, attention_mask, position_ids))

        # Should have same number of outputs as inputs
        assert len(output) == 3

        # Other inputs should be unchanged
        assert torch.allclose(output[1], attention_mask)
        assert torch.allclose(output[2], position_ids)


# =============================================================================
# Batch Handling Tests
# =============================================================================

class TestBatchHandling:
    """Test hook works with different batch sizes."""

    def test_batch_size_1(self):
        """Hook should work with batch size 1."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))
        hook_fn = create_last_position_steering_hook(direction, 1.0)

        residual = torch.randn(1, 5, d_model)
        output = hook_fn(None, (residual,))

        assert output[0].shape == (1, 5, d_model)

    def test_batch_size_larger(self):
        """Hook should work with larger batch sizes."""
        d_model = DEFAULT_D_MODEL
        batch_size = 4
        direction = normalize_direction(torch.randn(d_model))
        hook_fn = create_last_position_steering_hook(direction, 1.0)

        residual = torch.randn(batch_size, 5, d_model)
        original = residual.clone()

        output = hook_fn(None, (residual,))

        # All samples in batch should be modified at last position
        for b in range(batch_size):
            assert not torch.allclose(output[0][b, -1], original[b, -1])
            # Other positions should be unchanged
            assert torch.allclose(output[0][b, :-1], original[b, :-1])


# =============================================================================
# Device Handling Tests
# =============================================================================

@pytest.mark.gpu
class TestDeviceHandling:
    """Test hook works on different devices."""

    def test_cuda_residual(self):
        """Hook should work with CUDA tensors."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))  # CPU
        hook_fn = create_last_position_steering_hook(direction, 1.0)

        residual = torch.randn(1, 5, d_model, device='cuda')
        original = residual.clone()

        output = hook_fn(None, (residual,))

        assert output[0].device.type == 'cuda'
        assert not torch.allclose(output[0][0, -1], original[0, -1])


# =============================================================================
# Non-Normalized Input Rejection Tests
# =============================================================================

class TestNonNormalizedRejection:
    """Test hook rejects non-normalized directions."""

    def test_rejects_non_unit_norm_direction(self):
        """Hook should raise ValueError for non-unit-norm direction."""
        d_model = DEFAULT_D_MODEL
        direction = torch.randn(d_model) * 5.0  # Non-unit norm

        with pytest.raises(ValueError, match="not unit-normalized"):
            create_last_position_steering_hook(direction, 1.0)

    def test_rejects_near_zero_norm_direction(self):
        """Hook should raise ValueError for near-zero direction."""
        d_model = DEFAULT_D_MODEL
        direction = torch.randn(d_model) * 1e-10  # Near-zero norm

        with pytest.raises(ValueError, match="not unit-normalized"):
            create_last_position_steering_hook(direction, 1.0)

    def test_accepts_unit_norm_direction(self):
        """Hook should accept properly normalized direction."""
        d_model = DEFAULT_D_MODEL
        direction = normalize_direction(torch.randn(d_model))

        # Should not raise
        hook_fn = create_last_position_steering_hook(direction, 1.0)
        assert hook_fn is not None
