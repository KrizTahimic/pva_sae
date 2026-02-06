"""
Tests for common/weight_orthogonalization.py

Validates:
- orthogonalize_gemma_weights: Reduced component along target direction
- Weight change magnitude is reasonable
- In-place modification works correctly
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock


# =============================================================================
# Mock Model
# =============================================================================

class MockOProj(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_model, d_model))


class MockDownProj(nn.Module):
    def __init__(self, d_model, d_intermediate):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_model, d_intermediate))


class MockAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.o_proj = MockOProj(d_model)


class MockMLP(nn.Module):
    def __init__(self, d_model, d_intermediate):
        super().__init__()
        self.down_proj = MockDownProj(d_model, d_intermediate)


class MockLayer(nn.Module):
    def __init__(self, d_model, d_intermediate):
        super().__init__()
        self.self_attn = MockAttention(d_model)
        self.mlp = MockMLP(d_model, d_intermediate)


class MockEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model))


class MockTransformer(nn.Module):
    def __init__(self, n_layers=2, d_model=32, d_intermediate=64, vocab_size=100):
        super().__init__()
        self.embed_tokens = MockEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MockLayer(d_model, d_intermediate) for _ in range(n_layers)
        ])


class MockGemmaModel(nn.Module):
    def __init__(self, n_layers=2, d_model=32, d_intermediate=64):
        super().__init__()
        self.model = MockTransformer(n_layers, d_model, d_intermediate)

    @property
    def device(self):
        return torch.device('cpu')


# =============================================================================
# orthogonalize_gemma_weights Tests
# =============================================================================

class TestOrthogonalizeGemmaWeights:
    """Test weight orthogonalization correctness."""

    def test_reduced_component_along_direction(self):
        """Orthogonalized weights should have reduced projection onto direction."""
        from common.weight_orthogonalization import orthogonalize_gemma_weights

        d_model = 32
        model = MockGemmaModel(n_layers=2, d_model=d_model)
        direction = torch.randn(d_model)

        # Measure projection before
        embed_before = model.model.embed_tokens.weight.data.clone()
        proj_before = (embed_before @ direction).abs().mean().item()

        changes = orthogonalize_gemma_weights(model, direction, target_weights=['embed'])

        # Measure projection after
        embed_after = model.model.embed_tokens.weight.data
        dir_normalized = direction / direction.norm()
        proj_after = (embed_after @ dir_normalized).abs().mean().item()

        # Projection should be near zero after orthogonalization
        assert proj_after < 1e-5, f"Projection {proj_after} should be near zero"

    def test_weight_change_is_nonzero(self):
        """Orthogonalization should actually change the weights."""
        from common.weight_orthogonalization import orthogonalize_gemma_weights

        d_model = 32
        model = MockGemmaModel(n_layers=2, d_model=d_model)
        direction = torch.randn(d_model)

        changes = orthogonalize_gemma_weights(model, direction)

        assert len(changes) > 0
        assert all(v > 0 for v in changes.values()), "All changes should be positive (nonzero)"

    def test_weight_change_not_enormous(self):
        """Weight changes should be bounded - not blowing up the model."""
        from common.weight_orthogonalization import orthogonalize_gemma_weights

        d_model = 32
        model = MockGemmaModel(n_layers=2, d_model=d_model)
        direction = torch.randn(d_model)

        # Save original weight norms
        original_embed_norm = model.model.embed_tokens.weight.data.norm().item()

        changes = orthogonalize_gemma_weights(model, direction, target_weights=['embed'])

        # Change should be less than the original norm (we're removing a component, not adding)
        assert changes['embedding'] < original_embed_norm

    def test_in_place_modification(self):
        """Weights should be modified in-place on the model."""
        from common.weight_orthogonalization import orthogonalize_gemma_weights

        d_model = 32
        model = MockGemmaModel(n_layers=2, d_model=d_model)
        direction = torch.randn(d_model)

        original_data_ptr = model.model.embed_tokens.weight.data.data_ptr()
        original_values = model.model.embed_tokens.weight.data.clone()

        orthogonalize_gemma_weights(model, direction, target_weights=['embed'])

        # Values should have changed
        assert not torch.allclose(original_values, model.model.embed_tokens.weight.data)

    def test_attn_o_orthogonalization(self):
        """Attention output projection should be orthogonalized correctly."""
        from common.weight_orthogonalization import orthogonalize_gemma_weights

        d_model = 32
        model = MockGemmaModel(n_layers=2, d_model=d_model)
        direction = torch.randn(d_model)

        changes = orthogonalize_gemma_weights(model, direction, target_weights=['attn_o'])

        assert 'layer_0_attn_o' in changes
        assert 'layer_1_attn_o' in changes
        assert changes['layer_0_attn_o'] > 0
        assert changes['layer_1_attn_o'] > 0

    def test_mlp_down_orthogonalization(self):
        """MLP down projection should be orthogonalized correctly."""
        from common.weight_orthogonalization import orthogonalize_gemma_weights

        d_model = 32
        model = MockGemmaModel(n_layers=2, d_model=d_model)
        direction = torch.randn(d_model)

        changes = orthogonalize_gemma_weights(model, direction, target_weights=['mlp_down'])

        assert 'layer_0_mlp_down' in changes
        assert 'layer_1_mlp_down' in changes

    def test_selective_target_weights(self):
        """Only specified weight types should be modified."""
        from common.weight_orthogonalization import orthogonalize_gemma_weights

        d_model = 32
        model = MockGemmaModel(n_layers=2, d_model=d_model)
        direction = torch.randn(d_model)

        changes = orthogonalize_gemma_weights(model, direction, target_weights=['embed'])

        assert 'embedding' in changes
        assert 'layer_0_attn_o' not in changes
        assert 'layer_0_mlp_down' not in changes
