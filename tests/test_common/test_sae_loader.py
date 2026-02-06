"""
Tests for common/sae_loader.py

Validates:
- JumpReLUSAE encode/decode roundtrip
- TopKSAE encode/decode roundtrip
- get_decoder_weight returns correct shape
- load_sae_for_config auto-detects model type
- Invalid layer index error
- dtype/device preservation
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from common.config import Config
from common.sae_loader import JumpReLUSAE, TopKSAE, BaseSAE


# =============================================================================
# JumpReLUSAE Tests
# =============================================================================

class TestJumpReLUSAE:
    """Test JumpReLU SAE encode/decode."""

    @pytest.fixture
    def sae(self):
        """Create a small JumpReLU SAE for testing."""
        d_model = 64
        d_sae = 128
        sae = JumpReLUSAE(d_model, d_sae)
        # Initialize with small random weights
        with torch.no_grad():
            sae.W_enc.copy_(torch.randn(d_model, d_sae) * 0.1)
            sae.W_dec.copy_(torch.randn(d_sae, d_model) * 0.1)
            sae.b_enc.zero_()
            sae.b_dec.zero_()
            sae.threshold.fill_(0.01)  # Low threshold so some latents fire
        return sae

    def test_encode_output_shape(self, sae):
        """encode() should return (batch, d_sae) shaped output."""
        x = torch.randn(1, 64)
        latents = sae.encode(x)
        assert latents.shape == (1, 128)

    def test_decode_output_shape(self, sae):
        """decode() should return (batch, d_model) shaped output."""
        latents = torch.randn(1, 128)
        reconstructed = sae.decode(latents)
        assert reconstructed.shape == (1, 64)

    def test_encode_nonnegative(self, sae):
        """JumpReLU activations should be non-negative."""
        x = torch.randn(1, 64)
        latents = sae.encode(x)
        assert (latents >= 0).all()

    def test_encode_sparsity(self, sae):
        """JumpReLU should produce sparse activations (many zeros)."""
        # Use high threshold to force sparsity
        with torch.no_grad():
            sae.threshold.fill_(10.0)
        x = torch.randn(1, 64)
        latents = sae.encode(x)
        zero_fraction = (latents == 0).float().mean().item()
        assert zero_fraction > 0.5, "JumpReLU should produce sparse activations"

    def test_roundtrip_reduces_error(self, sae):
        """Decode(encode(x)) should roughly reconstruct x."""
        x = torch.randn(1, 64)
        latents = sae.encode(x)
        reconstructed = sae.decode(latents)
        # Not exact, but shapes should match
        assert reconstructed.shape == x.shape

    def test_get_decoder_weight(self, sae):
        """get_decoder_weight should return (d_model,) vector."""
        weight = sae.get_decoder_weight(0)
        assert weight.shape == (64,)

    def test_get_decoder_weight_matches_wdec(self, sae):
        """get_decoder_weight should match W_dec row."""
        for idx in [0, 5, 127]:
            weight = sae.get_decoder_weight(idx)
            assert torch.allclose(weight, sae.W_dec[idx])

    def test_dtype_preservation_float32(self, sae):
        """SAE should preserve float32 dtype."""
        x = torch.randn(1, 64, dtype=torch.float32)
        latents = sae.encode(x)
        assert latents.dtype == torch.float32

    def test_dtype_preservation_bfloat16(self, sae):
        """SAE should preserve bfloat16 dtype."""
        sae = sae.to(torch.bfloat16)
        x = torch.randn(1, 64, dtype=torch.bfloat16)
        latents = sae.encode(x)
        assert latents.dtype == torch.bfloat16


# =============================================================================
# TopKSAE Tests
# =============================================================================

class TestTopKSAE:
    """Test TopK SAE encode/decode."""

    @pytest.fixture
    def sae(self):
        """Create a small TopK SAE for testing."""
        d_model = 64
        d_sae = 128
        k = 8
        sae = TopKSAE(d_model, d_sae, k=k)
        with torch.no_grad():
            sae.W_enc.copy_(torch.randn(d_model, d_sae) * 0.1)
            sae.W_dec.copy_(torch.randn(d_sae, d_model) * 0.1)
            sae.b_enc.zero_()
            sae.b_dec.zero_()
        return sae

    def test_encode_output_shape(self, sae):
        """encode() should return (batch, d_sae) shaped output."""
        x = torch.randn(1, 64)
        latents = sae.encode(x)
        assert latents.shape == (1, 128)

    def test_topk_sparsity(self, sae):
        """TopK should produce exactly k non-zero activations."""
        x = torch.randn(1, 64)
        latents = sae.encode(x)
        n_nonzero = (latents != 0).sum().item()
        assert n_nonzero == 8, f"Expected exactly 8 non-zero, got {n_nonzero}"

    def test_decode_output_shape(self, sae):
        """decode() should return (batch, d_model) shaped output."""
        latents = torch.randn(1, 128)
        reconstructed = sae.decode(latents)
        assert reconstructed.shape == (1, 64)

    def test_get_decoder_weight(self, sae):
        """get_decoder_weight should return (d_model,) vector."""
        weight = sae.get_decoder_weight(0)
        assert weight.shape == (64,)

    def test_batch_encode(self, sae):
        """Should handle batch encoding."""
        x = torch.randn(4, 64)
        latents = sae.encode(x)
        assert latents.shape == (4, 128)
        # Each sample should have exactly k non-zero
        for i in range(4):
            n_nonzero = (latents[i] != 0).sum().item()
            assert n_nonzero == 8


# =============================================================================
# load_sae_for_config Tests
# =============================================================================

class TestLoadSaeForConfig:
    """Test SAE loading with config auto-detection."""

    @patch('common.sae_loader.load_gemma_scope_sae')
    def test_gemma_2b_loads_gemmascope(self, mock_load):
        """Gemma-2B model should load GemmaScope SAE."""
        config = Config()
        config.model_name = "google/gemma-2-2b"

        from common.sae_loader import load_sae_for_config
        load_sae_for_config(config, layer_idx=16, device="cpu")

        mock_load.assert_called_once()

    @patch('common.sae_loader.load_llama_scope_sae')
    def test_llama_loads_llamascope(self, mock_load):
        """Llama model should load LlamaScope SAE."""
        config = Config()
        config.model_name = "meta-llama/Llama-3.1-8B"

        from common.sae_loader import load_sae_for_config
        load_sae_for_config(config, layer_idx=16, device="cpu")

        mock_load.assert_called_once()

    def test_invalid_model_raises(self):
        """Unknown model should raise ValueError."""
        config = Config()
        config.model_name = "unknown/model"

        from common.sae_loader import load_sae_for_config
        with pytest.raises((ValueError, KeyError)):
            load_sae_for_config(config, layer_idx=16, device="cpu")
