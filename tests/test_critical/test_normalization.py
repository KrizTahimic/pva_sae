"""
Tests for direction normalization consistency.

Validates:
- phase_4_5_normalization: Directions normalized before coefficient search
- phase_4_8_normalization: Same normalization as Phase 4.5
- coefficient_interpretation: Coefficient meaning consistent across phases
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from common.config import Config


# =============================================================================
# phase_4_5_normalization Tests
# =============================================================================

class TestPhase45Normalization:
    """Test directions normalized before coefficient search."""

    def test_sae_directions_normalized_to_unit_norm(self):
        """SAE directions should be L2 normalized to unit norm."""
        from common.steering_setup import load_sae_and_directions

        # Create mock SAE with non-unit norm W_dec
        mock_sae = MagicMock()
        unnormalized_direction = torch.randn(2304) * 5.0  # Non-unit norm
        mock_sae.W_dec = unnormalized_direction.unsqueeze(0).repeat(16384, 1)

        # Create mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])

        best_correct = {'layer': 16, 'latent_idx': 0}
        best_incorrect = {'layer': 16, 'latent_idx': 0}

        with patch('common.steering_setup.load_sae_for_config', return_value=mock_sae):
            result = load_sae_and_directions(
                Config(),
                torch.device('cpu'),
                mock_model,
                best_correct,
                best_incorrect
            )

        # Check unit L2 norm
        assert torch.norm(result.correct_direction).item() == pytest.approx(1.0, rel=1e-5)
        assert torch.norm(result.incorrect_direction).item() == pytest.approx(1.0, rel=1e-5)

    def test_normalization_preserves_direction(self):
        """Normalization should preserve direction, only change magnitude."""
        from common.steering_setup import load_sae_and_directions

        mock_sae = MagicMock()
        original_direction = torch.randn(2304)
        mock_sae.W_dec = original_direction.unsqueeze(0).repeat(16384, 1)

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])

        best_correct = {'layer': 16, 'latent_idx': 0}
        best_incorrect = {'layer': 16, 'latent_idx': 0}

        with patch('common.steering_setup.load_sae_for_config', return_value=mock_sae):
            result = load_sae_and_directions(
                Config(),
                torch.device('cpu'),
                mock_model,
                best_correct,
                best_incorrect
            )

        # Normalized direction should point in same direction
        expected_normalized = original_direction / torch.norm(original_direction)
        cosine_sim = torch.dot(result.correct_direction.float(), expected_normalized) / (
            torch.norm(result.correct_direction.float()) * torch.norm(expected_normalized)
        )
        assert cosine_sim.item() == pytest.approx(1.0, rel=1e-5)


# =============================================================================
# phase_4_8_normalization Tests
# =============================================================================

class TestPhase48Normalization:
    """Test same normalization as Phase 4.5."""

    def test_4_5_and_4_8_use_same_normalization(self):
        """Both phases should normalize directions the same way."""
        # Phase 4.5 and 4.8 both use load_sae_and_directions from steering_setup
        # This test verifies they use the same code path

        from common.steering_setup import load_sae_and_directions

        mock_sae = MagicMock()
        mock_sae.W_dec = torch.randn(16384, 2304)

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])

        best_correct = {'layer': 16, 'latent_idx': 100}
        best_incorrect = {'layer': 18, 'latent_idx': 200}

        with patch('common.steering_setup.load_sae_for_config', return_value=mock_sae):
            # First call (simulating Phase 4.5)
            result1 = load_sae_and_directions(
                Config(),
                torch.device('cpu'),
                mock_model,
                best_correct,
                best_incorrect
            )

            # Second call (simulating Phase 4.8) - reset iterator
            mock_model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])
            result2 = load_sae_and_directions(
                Config(),
                torch.device('cpu'),
                mock_model,
                best_correct,
                best_incorrect
            )

        # Both should produce unit norm directions
        assert torch.norm(result1.correct_direction).item() == pytest.approx(1.0, rel=1e-5)
        assert torch.norm(result2.correct_direction).item() == pytest.approx(1.0, rel=1e-5)


# =============================================================================
# coefficient_interpretation Tests
# =============================================================================

class TestCoefficientInterpretation:
    """Test coefficient meaning consistent across phases."""

    def test_coefficient_scales_steering_magnitude(self):
        """Coefficient should directly scale steering magnitude."""
        from common.steering_metrics import create_last_position_steering_hook

        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)  # Unit norm

        hook_coef_1 = create_last_position_steering_hook(direction, 1.0)
        hook_coef_10 = create_last_position_steering_hook(direction, 10.0)

        residual = torch.zeros(1, 5, 2304)

        output_1 = hook_coef_1(None, (residual.clone(),))[0]
        output_10 = hook_coef_10(None, (residual.clone(),))[0]

        change_1 = output_1[0, -1]
        change_10 = output_10[0, -1]

        # Change should scale proportionally with coefficient
        assert torch.allclose(change_10, change_1 * 10, rtol=1e-5)

    def test_unit_norm_direction_coefficient_is_magnitude(self):
        """With unit norm direction, coefficient equals steering magnitude."""
        from common.steering_metrics import create_last_position_steering_hook

        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)  # Unit norm

        coefficient = 42.0
        hook = create_last_position_steering_hook(direction, coefficient)

        residual = torch.zeros(1, 5, 2304)
        output = hook(None, (residual.clone(),))[0]

        change = output[0, -1]
        change_magnitude = torch.norm(change).item()

        # With unit norm direction, coefficient IS the magnitude
        assert change_magnitude == pytest.approx(coefficient, rel=1e-4)

    def test_coefficient_interpretation_same_for_sae_and_probe(self):
        """Coefficient should mean same thing for SAE and probe directions."""
        from common.steering_metrics import create_last_position_steering_hook

        # Both SAE and probe directions are normalized to unit norm
        sae_direction = torch.randn(2304)
        sae_direction = sae_direction / torch.norm(sae_direction)

        probe_direction = torch.randn(2304)
        probe_direction = probe_direction / torch.norm(probe_direction)

        coefficient = 30.0

        sae_hook = create_last_position_steering_hook(sae_direction, coefficient)
        probe_hook = create_last_position_steering_hook(probe_direction, coefficient)

        residual = torch.zeros(1, 5, 2304)

        sae_output = sae_hook(None, (residual.clone(),))[0]
        probe_output = probe_hook(None, (residual.clone(),))[0]

        sae_magnitude = torch.norm(sae_output[0, -1]).item()
        probe_magnitude = torch.norm(probe_output[0, -1]).item()

        # Same coefficient should produce same magnitude change
        assert sae_magnitude == pytest.approx(probe_magnitude, rel=1e-4)
        assert sae_magnitude == pytest.approx(coefficient, rel=1e-4)


# =============================================================================
# Probe Direction Normalization Tests
# =============================================================================

class TestProbeDirectionNormalization:
    """Test probe direction handling."""

    def test_probe_for_steering_matches_model_dtype(self):
        """Probe directions for steering should match model dtype."""
        from common.steering_setup import load_probe_directions_for_steering

        mock_direction = torch.randn(2304, dtype=torch.float32)

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])

        with patch('common.steering_setup._load_probe_base',
                  return_value=(mock_direction, 16, 0.0, "/mock/path")):
            result = load_probe_directions_for_steering(
                Config(),
                torch.device('cpu'),
                mock_model,
                method="mass_mean"
            )

        assert result.correct_direction.dtype == torch.bfloat16
        assert result.incorrect_direction.dtype == torch.bfloat16

    def test_probe_for_predicting_stays_float32(self):
        """Probe directions for predicting should stay float32."""
        from common.steering_setup import load_probe_directions_for_predicting

        mock_direction = torch.randn(2304, dtype=torch.float32)

        with patch('common.steering_setup._load_probe_base',
                  return_value=(mock_direction, 16, 0.0, "/mock/path")):
            result = load_probe_directions_for_predicting(
                Config(),
                torch.device('cpu'),
                method="logreg"
            )

        assert result.correct_direction.dtype == torch.float32
        assert result.incorrect_direction.dtype == torch.float32

    def test_incorrect_direction_is_negated(self):
        """Incorrect direction should be negated version of correct direction."""
        from common.steering_setup import load_probe_directions_for_predicting

        mock_direction = torch.randn(2304, dtype=torch.float32)

        with patch('common.steering_setup._load_probe_base',
                  return_value=(mock_direction.clone(), 16, 0.0, "/mock/path")):
            result = load_probe_directions_for_predicting(
                Config(),
                torch.device('cpu'),
                method="logreg"
            )

        # Incorrect should be negation of correct
        assert torch.allclose(result.incorrect_direction, -result.correct_direction)
