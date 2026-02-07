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


# =============================================================================
# Runner Orchestration Tests
# =============================================================================

from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

from common.config import Config
from phase2_2_pile_caching.runner import run_phase2_2_caching


class TestRunnerOrchestration:
    """Test that run_phase2_2_caching calls the right methods in the right order."""

    @patch('common.phase_discovery.write_phase_output')
    @patch('phase2_2_pile_caching.runner.save_activation')
    @patch('phase2_2_pile_caching.runner.load_dataset')
    @patch('phase2_2_pile_caching.runner.load_model_and_tokenizer')
    def test_calls_model_load_before_dataset(self, mock_load_model, mock_load_dataset,
                                              mock_save_activation, mock_write_phase):
        """Model and tokenizer should be loaded, then dataset."""
        config = Config()
        config.pile_samples = 2
        config.activation_layers = [5]

        # Setup mock model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Setup mock dataset with minimal text
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=['hello world', 'foo bar'])
        mock_load_dataset.return_value = mock_ds

        # Make tokenizer return token IDs and decoded text
        mock_tokenizer.return_value = MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
            to=MagicMock(return_value=MagicMock(input_ids=torch.tensor([[1, 2, 3]])))
        )
        mock_tokenizer.decode.return_value = 'hello world'

        # Setup model forward pass with hook-friendly structure
        mock_model.model = MagicMock()
        mock_model.model.layers = [MagicMock() for _ in range(26)]
        mock_model.return_value = MagicMock()

        call_order = []
        mock_load_model.side_effect = lambda *a, **kw: (
            call_order.append('load_model'),
            (mock_model, mock_tokenizer)
        )[-1]
        mock_load_dataset.side_effect = lambda *a, **kw: (
            call_order.append('load_dataset'),
            mock_ds
        )[-1]

        try:
            run_phase2_2_caching(config, device="cpu")
        except Exception:
            pass  # We only care about call order

        assert 'load_model' in call_order
        assert 'load_dataset' in call_order
        assert call_order.index('load_model') < call_order.index('load_dataset')

    @patch('phase2_2_pile_caching.runner.load_dataset')
    @patch('phase2_2_pile_caching.runner.load_model_and_tokenizer')
    def test_model_set_to_eval_mode(self, mock_load_model, mock_load_dataset):
        """Model should be set to eval mode after loading."""
        config = Config()
        config.pile_samples = 1
        config.activation_layers = [5]

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=['hello'])
        mock_load_dataset.return_value = mock_ds

        try:
            run_phase2_2_caching(config, device="cpu")
        except Exception:
            pass

        mock_model.eval.assert_called_once()


class TestRunnerOutputDirectory:
    """Test that run_phase2_2_caching creates the expected output directory structure."""

    @patch('common.phase_discovery.write_phase_output')
    @patch('phase2_2_pile_caching.runner.save_activation')
    @patch('phase2_2_pile_caching.runner.load_dataset')
    @patch('phase2_2_pile_caching.runner.load_model_and_tokenizer')
    def test_creates_pile_activations_subdir(self, mock_load_model, mock_load_dataset,
                                             mock_save_activation, mock_write_phase,
                                             tmp_path):
        """Output directory should include a pile_activations subdirectory."""
        config = Config()
        config.pile_samples = 1
        config.activation_layers = [5]

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=['hello world'])
        mock_load_dataset.return_value = mock_ds

        mock_tokenizer.return_value = MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
            to=MagicMock(return_value=MagicMock(input_ids=torch.tensor([[1, 2, 3]])))
        )
        mock_tokenizer.decode.return_value = 'hello world'
        mock_model.model = MagicMock()
        mock_model.model.layers = [MagicMock() for _ in range(26)]

        # Patch get_phase_output_dir to use tmp_path
        with patch('phase2_2_pile_caching.runner.get_phase_output_dir',
                   return_value=str(tmp_path / "phase2_2")):
            try:
                run_phase2_2_caching(config, device="cpu")
            except Exception:
                pass

        pile_activations_dir = tmp_path / "phase2_2" / "pile_activations"
        assert pile_activations_dir.exists()

    @patch('common.phase_discovery.write_phase_output')
    @patch('phase2_2_pile_caching.runner.save_activation')
    @patch('phase2_2_pile_caching.runner.load_dataset')
    @patch('phase2_2_pile_caching.runner.load_model_and_tokenizer')
    def test_writes_phase_manifest_in_sequential_mode(self, mock_load_model,
                                                       mock_load_dataset,
                                                       mock_save_activation,
                                                       mock_write_phase, tmp_path):
        """In sequential mode (n_gpus=1), a phase_output.json manifest should be written."""
        config = Config()
        config.pile_samples = 0  # No samples to process
        config.activation_layers = [5]

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=[])
        mock_load_dataset.return_value = mock_ds

        with patch('phase2_2_pile_caching.runner.get_phase_output_dir',
                   return_value=str(tmp_path / "phase2_2")):
            try:
                run_phase2_2_caching(config, gpu_id=0, n_gpus=1, device="cpu")
            except Exception:
                pass

        # In sequential mode, write_phase_output should be called
        mock_write_phase.assert_called_once()
        call_kwargs = mock_write_phase.call_args
        assert call_kwargs[1].get('phase', call_kwargs[0][0] if call_kwargs[0] else None) == "2.2" or \
               (len(call_kwargs[0]) > 0 and call_kwargs[0][0] == "2.2")

    @patch('phase2_2_pile_caching.runner.save_activation')
    @patch('phase2_2_pile_caching.runner.load_dataset')
    @patch('phase2_2_pile_caching.runner.load_model_and_tokenizer')
    def test_skips_manifest_in_parallel_mode(self, mock_load_model, mock_load_dataset,
                                              mock_save_activation, tmp_path):
        """In parallel mode (n_gpus > 1), phase_output.json should NOT be written."""
        config = Config()
        config.pile_samples = 0
        config.activation_layers = [5]

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=[])
        mock_load_dataset.return_value = mock_ds

        with patch('phase2_2_pile_caching.runner.get_phase_output_dir',
                   return_value=str(tmp_path / "phase2_2")), \
             patch('common.phase_discovery.write_phase_output') as mock_write_phase:
            try:
                run_phase2_2_caching(config, gpu_id=0, n_gpus=4, device="cpu")
            except Exception:
                pass

        mock_write_phase.assert_not_called()


class TestRunnerMissingDependencies:
    """Test that run_phase2_2_caching handles missing input dependencies."""

    @patch('phase2_2_pile_caching.runner.load_dataset')
    @patch('phase2_2_pile_caching.runner.load_model_and_tokenizer')
    def test_handles_empty_pile_dataset(self, mock_load_model, mock_load_dataset, tmp_path):
        """Should handle an empty pile dataset gracefully (0 texts)."""
        config = Config()
        config.pile_samples = 0
        config.activation_layers = [5]

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Return empty list
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=[])
        mock_load_dataset.return_value = mock_ds

        with patch('phase2_2_pile_caching.runner.get_phase_output_dir',
                   return_value=str(tmp_path / "phase2_2")), \
             patch('common.phase_discovery.write_phase_output'):
            # Should not raise - just process zero items
            run_phase2_2_caching(config, device="cpu")

    @patch('phase2_2_pile_caching.runner.load_model_and_tokenizer')
    def test_model_load_failure_propagates(self, mock_load_model):
        """If model loading fails, error should propagate."""
        config = Config()
        config.pile_samples = 5
        config.activation_layers = [5]

        mock_load_model.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            run_phase2_2_caching(config, device="cuda")

    @patch('phase2_2_pile_caching.runner.load_dataset')
    @patch('phase2_2_pile_caching.runner.load_model_and_tokenizer')
    def test_dataset_load_failure_propagates(self, mock_load_model, mock_load_dataset):
        """If pile dataset loading fails, error should propagate."""
        config = Config()
        config.pile_samples = 5
        config.activation_layers = [5]

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        mock_load_dataset.side_effect = ConnectionError("Cannot reach HuggingFace")

        with pytest.raises(ConnectionError, match="Cannot reach HuggingFace"):
            run_phase2_2_caching(config, device="cpu")
