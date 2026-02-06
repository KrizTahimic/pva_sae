"""
Tests for common/activation_hooks.py

Validates:
- ActivationExtractor: Hook registration, extraction, cleanup
- AttentionExtractor: Hook registration, pattern capture, cleanup
"""

import pytest
import torch
import torch.nn as nn


# =============================================================================
# Mock Model
# =============================================================================

class MockAttention(nn.Module):
    """Mock attention module that returns dummy attention weights."""
    def forward(self, hidden_states, **kwargs):
        batch, seq_len, d_model = hidden_states.shape
        n_heads = 4
        attn_weights = torch.ones(batch, n_heads, seq_len, seq_len) / seq_len
        return hidden_states, attn_weights


class MockLayer(nn.Module):
    """Mock transformer layer with self_attn."""
    def __init__(self, d_model=64):
        super().__init__()
        self.self_attn = MockAttention()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)


class MockTransformer(nn.Module):
    """Mock transformer with .model.layers structure matching HuggingFace."""
    def __init__(self, n_layers=4, d_model=64):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer(d_model) for _ in range(n_layers)])

    def forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MockModel(nn.Module):
    """Top-level mock model with .model attribute."""
    def __init__(self, n_layers=4, d_model=64):
        super().__init__()
        self.model = MockTransformer(n_layers, d_model)

    def forward(self, input_ids, **kwargs):
        # Create embeddings from input_ids
        batch, seq_len = input_ids.shape
        hidden = torch.randn(batch, seq_len, 64)
        return self.model(hidden)


# =============================================================================
# ActivationExtractor Tests
# =============================================================================

class TestActivationExtractor:
    """Test ActivationExtractor hook registration and extraction."""

    def test_hook_registration_attaches_to_correct_layers(self):
        """Hooks should be registered on the specified layers."""
        from common.activation_hooks import ActivationExtractor

        model = MockModel(n_layers=4)
        layers = [0, 2]
        extractor = ActivationExtractor(model, layers)
        extractor.setup_hooks()

        assert len(extractor.hooks) == 2
        extractor.remove_hooks()

    def test_extracts_last_position_activation(self):
        """Default position=-1 should extract the last token's activation."""
        from common.activation_hooks import ActivationExtractor

        model = MockModel(n_layers=4, d_model=64)
        layers = [0, 1]
        extractor = ActivationExtractor(model, layers, position=-1)
        extractor.setup_hooks()

        input_ids = torch.randint(0, 100, (1, 10))
        activations = extractor.extract(input_ids)

        assert 0 in activations
        assert 1 in activations
        # Should be (batch, d_model) since we extract a single position
        assert activations[0].shape == (1, 64)
        extractor.remove_hooks()

    def test_hook_cleanup_on_context_manager_exit(self):
        """Context manager should clean up hooks even on normal exit."""
        from common.activation_hooks import ActivationExtractor

        model = MockModel(n_layers=4)
        layers = [0]

        with ActivationExtractor(model, layers) as extractor:
            assert len(extractor.hooks) == 1

        # After context manager exit, hooks should be cleared
        assert len(extractor.hooks) == 0

    def test_hook_cleanup_on_exception(self):
        """Context manager should clean up hooks on exception."""
        from common.activation_hooks import ActivationExtractor

        model = MockModel(n_layers=4)
        layers = [0]

        try:
            with ActivationExtractor(model, layers) as extractor:
                assert len(extractor.hooks) == 1
                raise ValueError("test error")
        except ValueError:
            pass

        assert len(extractor.hooks) == 0

    def test_multiple_layers_extraction(self):
        """Should extract activations from all specified layers."""
        from common.activation_hooks import ActivationExtractor

        model = MockModel(n_layers=4, d_model=64)
        layers = [0, 1, 2, 3]
        extractor = ActivationExtractor(model, layers)
        extractor.setup_hooks()

        input_ids = torch.randint(0, 100, (1, 5))
        activations = extractor.extract(input_ids)

        assert len(activations) == 4
        for layer_idx in layers:
            assert layer_idx in activations
            assert activations[layer_idx].shape == (1, 64)
        extractor.remove_hooks()

    def test_invalid_layer_raises(self):
        """Requesting a layer beyond model size should raise."""
        from common.activation_hooks import ActivationExtractor

        model = MockModel(n_layers=4)
        layers = [10]  # Only 4 layers exist
        extractor = ActivationExtractor(model, layers)

        with pytest.raises((IndexError, AttributeError)):
            extractor.setup_hooks()


# =============================================================================
# AttentionExtractor Tests
# =============================================================================

class TestAttentionExtractor:
    """Test AttentionExtractor hook registration and pattern capture."""

    def test_hook_registration(self):
        """Hooks should be registered on self_attn modules."""
        from common.activation_hooks import AttentionExtractor

        model = MockModel(n_layers=4)
        layers = [0, 2]
        extractor = AttentionExtractor(model, layers)
        extractor.setup_hooks()

        assert len(extractor.hooks) == 2
        extractor.remove_hooks()

    def test_context_manager_cleanup(self):
        """Context manager should clean up hooks."""
        from common.activation_hooks import AttentionExtractor

        model = MockModel(n_layers=4)
        layers = [0]

        with AttentionExtractor(model, layers) as extractor:
            assert len(extractor.hooks) == 1

        assert len(extractor.hooks) == 0

    def test_capture_only_once(self):
        """Should capture attention only once per layer (not during generation)."""
        from common.activation_hooks import AttentionExtractor

        model = MockModel(n_layers=4)
        layers = [0]
        extractor = AttentionExtractor(model, layers)
        extractor.setup_hooks()

        # Simulate two forward passes (prompt + generation)
        hidden = torch.randn(1, 5, 64)
        model.model.layers[0].self_attn(hidden)
        model.model.layers[0].self_attn(hidden)

        patterns = extractor.get_attention_patterns()
        assert 0 in patterns
        # Should have captured from the first forward pass only
        extractor.remove_hooks()

    def test_get_attention_patterns_clears_cache(self):
        """get_attention_patterns should clear captured patterns."""
        from common.activation_hooks import AttentionExtractor

        model = MockModel(n_layers=4)
        layers = [0]
        extractor = AttentionExtractor(model, layers)
        extractor.setup_hooks()

        hidden = torch.randn(1, 5, 64)
        model.model.layers[0].self_attn(hidden)

        patterns = extractor.get_attention_patterns()
        assert len(patterns) == 1

        # Second call should return empty (cache cleared)
        patterns2 = extractor.get_attention_patterns()
        assert len(patterns2) == 0

        extractor.remove_hooks()
