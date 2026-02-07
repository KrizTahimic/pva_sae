"""Tests for score_activation() in common/steering_setup.py."""

import unittest
from unittest.mock import MagicMock

import torch

from common.steering_setup import score_activation


class TestScoreActivationProbeMode(unittest.TestCase):
    """Test probe mode: dot product + bias."""

    def test_basic_dot_product(self):
        activation = torch.tensor([1.0, 2.0, 3.0])
        direction = torch.tensor([0.5, 0.5, 0.5])
        # dot product = 0.5 + 1.0 + 1.5 = 3.0, bias = 0
        score = score_activation(
            activation, use_probe=True,
            predicting_direction=direction, predicting_bias=0.0,
        )
        self.assertAlmostEqual(score, 3.0, places=4)

    def test_dot_product_with_bias(self):
        activation = torch.tensor([1.0, 0.0, 0.0])
        direction = torch.tensor([2.0, 0.0, 0.0])
        # dot product = 2.0, bias = -1.5 → score = 0.5
        score = score_activation(
            activation, use_probe=True,
            predicting_direction=direction, predicting_bias=-1.5,
        )
        self.assertAlmostEqual(score, 0.5, places=4)

    def test_orthogonal_vectors_zero_score(self):
        activation = torch.tensor([1.0, 0.0])
        direction = torch.tensor([0.0, 1.0])
        score = score_activation(
            activation, use_probe=True,
            predicting_direction=direction, predicting_bias=0.0,
        )
        self.assertAlmostEqual(score, 0.0, places=4)


class TestScoreActivationSAEMode(unittest.TestCase):
    """Test SAE mode: encode + extract latent."""

    def _make_mock_sae(self, latent_value: float, d_model: int = 4, n_latents: int = 8):
        """Create a mock SAE that returns a known latent activation."""
        sae = MagicMock()
        sae.W_enc = torch.zeros(d_model, n_latents, dtype=torch.bfloat16)

        def fake_encode(x):
            batch = x.shape[0]
            result = torch.zeros(batch, n_latents, dtype=x.dtype, device=x.device)
            result[:, 3] = latent_value  # latent_idx=3 gets the value
            return result

        sae.encode = fake_encode
        return sae

    def test_sae_mode_1d_activation(self):
        sae = self._make_mock_sae(latent_value=42.0)
        activation = torch.randn(4)
        score = score_activation(
            activation, use_probe=False,
            predicting_sae=sae, latent_idx=3,
            device=torch.device("cpu"),
        )
        self.assertAlmostEqual(score, 42.0, places=2)

    def test_sae_mode_2d_activation(self):
        sae = self._make_mock_sae(latent_value=7.5)
        activation = torch.randn(1, 4)
        score = score_activation(
            activation, use_probe=False,
            predicting_sae=sae, latent_idx=3,
            device=torch.device("cpu"),
        )
        self.assertAlmostEqual(score, 7.5, places=2)

    def test_sae_mode_zero_latent(self):
        sae = self._make_mock_sae(latent_value=0.0)
        activation = torch.randn(4)
        score = score_activation(
            activation, use_probe=False,
            predicting_sae=sae, latent_idx=3,
            device=torch.device("cpu"),
        )
        self.assertAlmostEqual(score, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
