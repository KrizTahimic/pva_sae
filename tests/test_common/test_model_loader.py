"""
Tests for common/model_loader.py

Validates dtype/device selection logic and model loading interface.
Uses mocks to avoid requiring GPU or model downloads.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from common.model_loader import load_model_and_tokenizer, get_model_info


class TestDtypeSelection:
    """Test automatic dtype selection based on device type."""

    @patch('common.model_loader.AutoModelForCausalLM')
    @patch('common.model_loader.AutoTokenizer')
    @patch('common.model_loader.detect_device')
    def test_cuda_bf16_when_supported(self, mock_detect, mock_tok, mock_model):
        """CUDA device with bf16 support should use bfloat16."""
        mock_detect.return_value = torch.device('cuda')
        mock_model_instance = MagicMock()
        mock_model_instance.device = torch.device('cuda')
        mock_model_instance.config.architectures = ['TestModel']
        mock_model_instance.parameters.return_value = iter([torch.tensor([1.0], device='cpu')])
        # Make .to() return same instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tok.from_pretrained.return_value = MagicMock()

        with patch('torch.cuda.is_bf16_supported', return_value=True):
            model, tokenizer = load_model_and_tokenizer('test-model')

        # Verify bfloat16 was passed to from_pretrained
        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert call_kwargs['torch_dtype'] == torch.bfloat16

    @patch('common.model_loader.AutoModelForCausalLM')
    @patch('common.model_loader.AutoTokenizer')
    @patch('common.model_loader.detect_device')
    def test_cuda_fp16_when_bf16_unsupported(self, mock_detect, mock_tok, mock_model):
        """CUDA device without bf16 support should use float16."""
        mock_detect.return_value = torch.device('cuda')
        mock_model_instance = MagicMock()
        mock_model_instance.device = torch.device('cuda')
        mock_model_instance.config.architectures = ['TestModel']
        mock_model_instance.parameters.return_value = iter([torch.tensor([1.0], device='cpu')])
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tok.from_pretrained.return_value = MagicMock()

        with patch('torch.cuda.is_bf16_supported', return_value=False):
            model, tokenizer = load_model_and_tokenizer('test-model')

        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert call_kwargs['torch_dtype'] == torch.float16

    @patch('common.model_loader.AutoModelForCausalLM')
    @patch('common.model_loader.AutoTokenizer')
    def test_cpu_uses_float32(self, mock_tok, mock_model):
        """CPU device should use float32."""
        mock_model_instance = MagicMock()
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.config.architectures = ['TestModel']
        mock_model_instance.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tok.from_pretrained.return_value = MagicMock()

        model, tokenizer = load_model_and_tokenizer('test-model', device='cpu')

        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert call_kwargs['torch_dtype'] == torch.float32

    @patch('common.model_loader.AutoModelForCausalLM')
    @patch('common.model_loader.AutoTokenizer')
    def test_explicit_dtype_overrides_auto(self, mock_tok, mock_model):
        """Explicitly provided dtype should override auto-detection."""
        mock_model_instance = MagicMock()
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.config.architectures = ['TestModel']
        mock_model_instance.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tok.from_pretrained.return_value = MagicMock()

        model, tokenizer = load_model_and_tokenizer(
            'test-model', device='cpu', dtype=torch.float16
        )

        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert call_kwargs['torch_dtype'] == torch.float16


class TestEagerAttention:
    """Test eager attention flag for steering phases."""

    @patch('common.model_loader.AutoModelForCausalLM')
    @patch('common.model_loader.AutoTokenizer')
    def test_eager_attention_sets_implementation(self, mock_tok, mock_model):
        """use_eager_attention=True should set attn_implementation='eager'."""
        mock_model_instance = MagicMock()
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.config.architectures = ['TestModel']
        mock_model_instance.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tok.from_pretrained.return_value = MagicMock()

        load_model_and_tokenizer('test-model', device='cpu', use_eager_attention=True)

        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert call_kwargs['attn_implementation'] == 'eager'

    @patch('common.model_loader.AutoModelForCausalLM')
    @patch('common.model_loader.AutoTokenizer')
    def test_default_no_eager_attention(self, mock_tok, mock_model):
        """Default should not set attn_implementation."""
        mock_model_instance = MagicMock()
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.config.architectures = ['TestModel']
        mock_model_instance.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tok.from_pretrained.return_value = MagicMock()

        load_model_and_tokenizer('test-model', device='cpu')

        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert 'attn_implementation' not in call_kwargs


class TestDevicePlacement:
    """Test model device placement logic."""

    @patch('common.model_loader.AutoModelForCausalLM')
    @patch('common.model_loader.AutoTokenizer')
    def test_cpu_model_not_moved(self, mock_tok, mock_model):
        """CPU model should not call .to(device)."""
        mock_model_instance = MagicMock()
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.config.architectures = ['TestModel']
        mock_model_instance.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tok.from_pretrained.return_value = MagicMock()

        load_model_and_tokenizer('test-model', device='cpu')

        # .to() should NOT be called for CPU (device.type == "cpu" guard)
        mock_model_instance.to.assert_not_called()

    @patch('common.model_loader.AutoModelForCausalLM')
    @patch('common.model_loader.AutoTokenizer')
    def test_model_set_to_eval(self, mock_tok, mock_model):
        """Model should always be set to eval mode."""
        mock_model_instance = MagicMock()
        mock_model_instance.device = torch.device('cpu')
        mock_model_instance.config.architectures = ['TestModel']
        mock_model_instance.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tok.from_pretrained.return_value = MagicMock()

        load_model_and_tokenizer('test-model', device='cpu')

        mock_model_instance.eval.assert_called_once()


class TestGetModelInfo:
    """Test model info extraction."""

    def test_basic_info_extraction(self):
        """Should extract architecture, params, layers, hidden size, vocab size."""
        mock_model = MagicMock()
        mock_model.config.architectures = ['GemmaForCausalLM']
        mock_model.config.num_hidden_layers = 18
        mock_model.config.hidden_size = 2048
        mock_model.config.vocab_size = 256000
        mock_model.parameters.return_value = [torch.randn(10, 10)]

        info = get_model_info(mock_model)

        assert info['architecture'] == 'GemmaForCausalLM'
        assert info['num_parameters'] == 100
        assert info['num_layers'] == 18
        assert info['hidden_size'] == 2048
        assert info['vocab_size'] == 256000
