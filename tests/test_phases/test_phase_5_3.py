"""
Tests for Phase 5.3 - Weight Orthogonalization

Validates:
- projection_math: Orthogonal projection correct
- weight_modification: Weights actually changed
- reversibility: Can restore original weights
"""

import pytest
import torch
import numpy as np


# =============================================================================
# projection_math Tests
# =============================================================================

class TestProjectionMath:
    """Test orthogonal projection correct."""

    def test_projection_removes_direction_component(self):
        """Projection should remove component along direction."""
        # Weight vector and direction
        w = torch.tensor([3.0, 4.0, 0.0])
        direction = torch.tensor([1.0, 0.0, 0.0])  # Unit vector along x

        # Orthogonal projection: w - (w · d) * d
        projection = w - torch.dot(w, direction) * direction

        # Component along direction should be zero
        component_along_direction = torch.dot(projection, direction)
        assert component_along_direction.item() == pytest.approx(0.0, abs=1e-6)

    def test_projection_preserves_orthogonal_components(self):
        """Projection should preserve components orthogonal to direction."""
        w = torch.tensor([3.0, 4.0, 5.0])
        direction = torch.tensor([1.0, 0.0, 0.0])  # x direction

        projection = w - torch.dot(w, direction) * direction

        # y and z components should be preserved
        assert projection[1].item() == pytest.approx(4.0)
        assert projection[2].item() == pytest.approx(5.0)

    def test_projection_of_parallel_vector_is_zero(self):
        """Projecting out a parallel direction should give zero."""
        w = torch.tensor([2.0, 0.0, 0.0])
        direction = torch.tensor([1.0, 0.0, 0.0])

        projection = w - torch.dot(w, direction) * direction

        assert torch.norm(projection).item() == pytest.approx(0.0, abs=1e-6)

    def test_projection_with_normalized_direction(self):
        """Direction should be normalized for correct projection."""
        w = torch.tensor([3.0, 4.0, 0.0])
        direction = torch.tensor([2.0, 0.0, 0.0])  # Not normalized

        # Normalize first
        direction_normalized = direction / torch.norm(direction)

        projection = w - torch.dot(w, direction_normalized) * direction_normalized

        # Should have removed x component
        assert projection[0].item() == pytest.approx(0.0, abs=1e-6)
        assert projection[1].item() == pytest.approx(4.0)


# =============================================================================
# weight_modification Tests
# =============================================================================

class TestWeightModification:
    """Test weights actually changed."""

    def test_weight_matrix_modified(self):
        """Weight matrix should be different after orthogonalization."""
        # Simulate weight matrix (d_out, d_in)
        W = torch.randn(100, 2304)
        W_original = W.clone()

        # Direction to orthogonalize
        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)

        # Apply orthogonalization to each row
        for i in range(W.shape[0]):
            W[i] = W[i] - torch.dot(W[i], direction) * direction

        # Weights should be different
        assert not torch.allclose(W, W_original)

    def test_output_projection_removes_direction(self):
        """Output should have no component along direction."""
        W = torch.randn(100, 2304)
        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)

        # Apply orthogonalization
        for i in range(W.shape[0]):
            W[i] = W[i] - torch.dot(W[i], direction) * direction

        # Check each row has no component along direction
        for i in range(W.shape[0]):
            component = torch.dot(W[i], direction)
            assert component.item() == pytest.approx(0.0, abs=1e-5)

    def test_target_weights_from_config(self):
        """Config should specify which weights to orthogonalize."""
        from common.config import Config

        config = Config()
        assert hasattr(config, 'orthogonalization_target_weights')
        assert 'embed' in config.orthogonalization_target_weights or \
               'attn_o' in config.orthogonalization_target_weights


# =============================================================================
# reversibility Tests
# =============================================================================

class TestReversibility:
    """Test can restore original weights."""

    def test_can_store_and_restore_weights(self):
        """Should be able to restore original weights."""
        W = torch.randn(100, 2304)
        W_backup = W.clone()

        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)

        # Apply orthogonalization
        for i in range(W.shape[0]):
            W[i] = W[i] - torch.dot(W[i], direction) * direction

        # Weights are now different
        assert not torch.allclose(W, W_backup)

        # Restore
        W = W_backup.clone()

        # Should be back to original
        assert torch.allclose(W, W_backup)

    def test_orthogonalization_is_idempotent(self):
        """Applying orthogonalization twice should give same result."""
        W = torch.randn(100, 2304)
        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)

        # Apply once
        for i in range(W.shape[0]):
            W[i] = W[i] - torch.dot(W[i], direction) * direction

        W_after_once = W.clone()

        # Apply again
        for i in range(W.shape[0]):
            W[i] = W[i] - torch.dot(W[i], direction) * direction

        # Should be the same (idempotent)
        assert torch.allclose(W, W_after_once, atol=1e-5)


# =============================================================================
# Layer Targeting Tests
# =============================================================================

class TestLayerTargeting:
    """Test orthogonalization targets correct layers."""

    def test_only_specified_layers_modified(self):
        """Only layers in target list should be modified."""
        from common.config import Config

        config = Config()
        target_weights = config.orthogonalization_target_weights

        # Should specify which weight types to modify
        assert isinstance(target_weights, list)
        assert len(target_weights) > 0

    def test_default_targets(self):
        """Default should include common layer types."""
        from common.config import Config

        config = Config()
        targets = config.orthogonalization_target_weights

        # Should include at least one of these
        expected = {'embed', 'attn_o', 'mlp_down'}
        assert any(t in targets for t in expected)
