"""
Tests for Phase 2.2 - Pile Activation Hook

Validates:
- hook_signature: hook_fn matches register_forward_pre_hook pattern
- activation_extraction: Extracts from input[0] (resid_pre)
- position_bounds: Returns None for out-of-bounds position
"""

import pytest
import torch
import inspect

from phase2_2_pile_caching.pile_activation_hook import PileActivationHook


# =============================================================================
# hook_signature Tests
# =============================================================================

class TestHookSignature:
    """Test hook_fn has correct signature for register_forward_pre_hook."""

    def test_hook_fn_signature_is_pre_hook(self):
        """hook_fn should accept (module, input) - pre_hook pattern."""
        hook = PileActivationHook(position=0)
        sig = inspect.signature(hook.hook_fn)
        params = list(sig.parameters.keys())

        # Pre-hook pattern: (self, module, input) - no 'output' parameter
        assert 'module' in params
        assert 'input' in params
        assert 'output' not in params

    def test_hook_fn_param_count(self):
        """Pre-hook should have exactly 2 params (module, input) on bound method."""
        hook = PileActivationHook(position=0)
        sig = inspect.signature(hook.hook_fn)
        params = list(sig.parameters.keys())

        # Bound method: module, input (self is implicit)
        assert len(params) == 2


# =============================================================================
# activation_extraction Tests
# =============================================================================

class TestActivationExtraction:
    """Test activation extraction from input (pre-hook pattern)."""

    def test_extracts_from_input(self):
        """Should extract activation from input[0] at specified position."""
        hook = PileActivationHook(position=2)

        # Simulate input tuple: (hidden_state,) where hidden_state is [batch, seq, d_model]
        hidden_state = torch.randn(1, 5, 128)
        input_tuple = (hidden_state,)

        hook.hook_fn(None, input_tuple)

        assert hook.activation is not None
        assert hook.activation.shape == (128,)
        # Should match the value at position 2
        expected = hidden_state[0, 2, :]
        assert torch.allclose(hook.activation, expected)

    def test_extracts_correct_position(self):
        """Should extract the correct token position."""
        d_model = 64
        seq_len = 10

        for pos in [0, 3, 9]:
            hook = PileActivationHook(position=pos)
            hidden_state = torch.randn(1, seq_len, d_model)
            input_tuple = (hidden_state,)

            hook.hook_fn(None, input_tuple)

            expected = hidden_state[0, pos, :]
            assert torch.allclose(hook.activation, expected)

    def test_activation_is_detached(self):
        """Extracted activation should not require grad."""
        hook = PileActivationHook(position=0)
        hidden_state = torch.randn(1, 5, 128, requires_grad=True)
        input_tuple = (hidden_state,)

        hook.hook_fn(None, input_tuple)

        assert not hook.activation.requires_grad

    def test_activation_is_on_cpu(self):
        """Extracted activation should be on CPU."""
        hook = PileActivationHook(position=0)
        hidden_state = torch.randn(1, 5, 128)
        input_tuple = (hidden_state,)

        hook.hook_fn(None, input_tuple)

        assert hook.activation.device.type == 'cpu'


# =============================================================================
# position_bounds Tests
# =============================================================================

class TestPositionBounds:
    """Test position out-of-bounds handling."""

    def test_out_of_bounds_returns_none(self):
        """Position beyond sequence length should set activation to None."""
        hook = PileActivationHook(position=10)  # Beyond seq_len=5
        hidden_state = torch.randn(1, 5, 128)
        input_tuple = (hidden_state,)

        hook.hook_fn(None, input_tuple)

        assert hook.activation is None

    def test_exact_boundary_returns_none(self):
        """Position equal to sequence length should set activation to None."""
        hook = PileActivationHook(position=5)  # Equal to seq_len=5
        hidden_state = torch.randn(1, 5, 128)
        input_tuple = (hidden_state,)

        hook.hook_fn(None, input_tuple)

        assert hook.activation is None

    def test_last_valid_position(self):
        """Last valid position (seq_len - 1) should work."""
        hook = PileActivationHook(position=4)  # Last valid for seq_len=5
        hidden_state = torch.randn(1, 5, 128)
        input_tuple = (hidden_state,)

        hook.hook_fn(None, input_tuple)

        assert hook.activation is not None
        assert hook.activation.shape == (128,)
