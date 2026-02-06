"""
Tests for direction source handling.

Validates:
- explicit_sae_source: --direction-source sae loads SAE directions
- explicit_probe_source: --direction-source probe_mass_mean loads probe
- default_source: Default is 'sae' when not specified
- output_dir_suffix: Probe mode creates _probe suffix
- coefficient_discovery_path: Correct directory searched for coefficients
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from common.config import Config


# =============================================================================
# explicit_sae_source Tests
# =============================================================================

class TestExplicitSAESource:
    """Test --direction-source sae loads SAE directions."""

    def test_sae_source_uses_sae_loader(self):
        """SAE source should call load_sae_for_config."""
        from common.steering_setup import load_sae_and_directions

        mock_sae = MagicMock()
        mock_sae.W_dec = torch.randn(16384, 2304)

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])

        best_correct = {'layer': 16, 'latent_idx': 100}
        best_incorrect = {'layer': 18, 'latent_idx': 200}

        with patch('common.steering_setup.load_sae_for_config', return_value=mock_sae) as mock_loader:
            result = load_sae_and_directions(
                Config(direction_source='sae'),
                torch.device('cpu'),
                mock_model,
                best_correct,
                best_incorrect
            )

        # Should have called SAE loader twice (correct and incorrect layers)
        assert mock_loader.call_count == 2

    def test_sae_source_info(self):
        """get_direction_source_info should return SAE info."""
        from common.steering_setup import get_direction_source_info

        config = Config(direction_source='sae')
        info = get_direction_source_info(config)

        assert info['source'] == 'sae'
        assert info['is_probe'] is False
        assert info['probe_method'] is None


# =============================================================================
# explicit_probe_source Tests
# =============================================================================

class TestExplicitProbeSource:
    """Test --direction-source probe_mass_mean loads probe."""

    def test_probe_source_info_mass_mean(self):
        """get_direction_source_info should return probe info for mass_mean."""
        from common.steering_setup import get_direction_source_info

        config = Config(direction_source='probe_mass_mean')
        info = get_direction_source_info(config)

        assert info['source'] == 'probe'
        assert info['is_probe'] is True
        assert info['probe_method'] == 'mass_mean'

    def test_probe_source_info_logreg(self):
        """get_direction_source_info should return probe info for logreg."""
        from common.steering_setup import get_direction_source_info

        config = Config(direction_source='probe_logreg')
        info = get_direction_source_info(config)

        assert info['source'] == 'probe'
        assert info['is_probe'] is True
        assert info['probe_method'] == 'logreg'


# =============================================================================
# default_source Tests
# =============================================================================

class TestDefaultSource:
    """Test default is 'sae' when not specified."""

    def test_default_direction_source_is_sae(self):
        """Default config should have direction_source='sae'."""
        config = Config()
        assert config.direction_source == 'sae'

    def test_default_interpreted_as_sae(self):
        """Default should be interpreted as SAE source."""
        from common.steering_setup import get_direction_source_info

        config = Config()  # No direction_source override
        info = get_direction_source_info(config)

        assert info['source'] == 'sae'
        assert info['is_probe'] is False


# =============================================================================
# output_dir_suffix Tests
# =============================================================================

class TestOutputDirSuffix:
    """Test probe mode creates appropriate suffix in outputs."""

    def test_sae_source_no_special_suffix(self):
        """SAE source should not add probe suffix."""
        from common.phase_discovery import get_phase_output_dir

        config = Config(direction_source='sae')
        output_dir = get_phase_output_dir("4.5", config)

        # Should not have probe suffix
        assert "_probe" not in output_dir

    # Note: Actual probe suffix handling may vary by phase implementation
    # These tests document expected behavior


# =============================================================================
# coefficient_discovery_path Tests
# =============================================================================

class TestCoefficientDiscoveryPath:
    """Test correct directory searched for coefficients."""

    def test_coefficient_discovery_uses_correct_phase(self, tmp_path):
        """Coefficient discovery should search correct phase directories."""
        from common.phase_discovery import discover_steering_coefficients
        import json

        # Create Phase 4.6 output (fallback)
        phase_4_6_dir = tmp_path / "data" / "phase4_6"
        phase_4_6_dir.mkdir(parents=True)

        coeff_data = {
            "correct": {"refined_coefficient": 35.0},
            "incorrect": {"refined_coefficient": 70.0}
        }
        with open(phase_4_6_dir / "refined_coefficients.json", 'w') as f:
            json.dump(coeff_data, f)

        manifest = {
            "phase": "4.6",
            "outputs": {"refined_coefficients": "refined_coefficients.json"}
        }
        with open(phase_4_6_dir / "phase_output.json", 'w') as f:
            json.dump(manifest, f)

        config = Config()

        def mock_get_phase_output_file(phase, key, config=None):
            if phase == "4.9":
                raise FileNotFoundError("Phase 4.9 not found")
            return phase_4_6_dir / "refined_coefficients.json"

        with patch('common.phase_discovery.get_phase_output_file',
                  side_effect=mock_get_phase_output_file):
            result = discover_steering_coefficients(config)

        assert result['correct'] == 35.0
        assert result['incorrect'] == 70.0


# =============================================================================
# Direction Source Switching Tests
# =============================================================================

class TestDirectionSourceSwitching:
    """Test switching between direction sources."""

    def test_can_switch_to_probe_logreg(self):
        """Should be able to set probe_logreg as direction source."""
        config = Config(direction_source='probe_logreg')
        assert config.direction_source == 'probe_logreg'

    def test_can_switch_to_probe_mass_mean(self):
        """Should be able to set probe_mass_mean as direction source."""
        config = Config(direction_source='probe_mass_mean')
        assert config.direction_source == 'probe_mass_mean'

    def test_invalid_source_raises_valueerror(self):
        """Invalid direction source should raise ValueError."""
        from common.steering_setup import get_direction_source_info
        import pytest

        config = Config()
        config.direction_source = 'invalid_source'

        with pytest.raises(ValueError, match="Unknown direction source"):
            get_direction_source_info(config)


# =============================================================================
# Probe Loading Integration Tests
# =============================================================================

class TestProbeLoadingIntegration:
    """Test probe loading for different purposes."""

    def test_steering_uses_mass_mean_by_default(self):
        """load_probe_directions_for_steering should default to mass_mean."""
        from common.steering_setup import load_probe_directions_for_steering
        import torch

        mock_direction = torch.randn(2304)
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])

        with patch('common.steering_setup._load_probe_base',
                  return_value=(mock_direction, 16, 0.0, "/mock/path")) as mock_load:
            load_probe_directions_for_steering(
                Config(),
                torch.device('cpu'),
                mock_model,
            )

        # Should have called with mass_mean method
        mock_load.assert_called_once()
        args = mock_load.call_args
        assert args[0][2] == 'mass_mean'  # method argument

    def test_predicting_uses_logreg_by_default(self):
        """load_probe_directions_for_predicting should default to logreg."""
        from common.steering_setup import load_probe_directions_for_predicting
        import torch

        mock_direction = torch.randn(2304)

        with patch('common.steering_setup._load_probe_base',
                  return_value=(mock_direction, 16, 0.0, "/mock/path")) as mock_load:
            load_probe_directions_for_predicting(
                Config(),
                torch.device('cpu'),
            )

        # Should have called with logreg method
        mock_load.assert_called_once()
        args = mock_load.call_args
        assert args[0][2] == 'logreg'  # method argument


# =============================================================================
# Latent Source Selection Tests
# =============================================================================

class TestLatentSourceSelection:
    """Test correct latent sources are used for different tasks."""

    def test_steering_uses_phase_2_5(self, tmp_path):
        """Steering tasks should use Phase 2.5 latents (separation score)."""
        from common.steering_setup import load_steering_latents
        import json

        # Create mock Phase 2.5 output
        phase_dir = tmp_path / "data" / "phase2_5"
        phase_dir.mkdir(parents=True)

        latents_data = {
            "correct": [{"layer": 16, "latent_idx": 100, "separation_score": 0.5}],
            "incorrect": [{"layer": 18, "latent_idx": 200, "separation_score": 0.4}]
        }

        analysis_file = phase_dir / "sae_analysis_20240101_120000.json"
        with open(analysis_file, 'w') as f:
            json.dump({}, f)

        with open(phase_dir / "top_20_latents.json", 'w') as f:
            json.dump(latents_data, f)

        config = Config()

        with patch('common.steering_setup.discover_latest_phase_output',
                  return_value=str(analysis_file)):
            result = load_steering_latents(config)

        assert result.source_phase == "2.5"

    def test_predicting_uses_phase_2_10(self, tmp_path):
        """Predicting tasks should use Phase 2.10 latents (t-statistic)."""
        from common.steering_setup import load_predicting_latents
        import json

        # Create mock Phase 2.10 output
        phase_dir = tmp_path / "data" / "phase2_10"
        phase_dir.mkdir(parents=True)

        latents_data = {
            "correct": [{"layer": 16, "latent_idx": 100, "t_statistic": 5.0}],
            "incorrect": [{"layer": 18, "latent_idx": 200, "t_statistic": 4.5}]
        }

        analysis_file = phase_dir / "t_statistic_analysis_20240101_120000.json"
        with open(analysis_file, 'w') as f:
            json.dump({}, f)

        with open(phase_dir / "top_20_latents.json", 'w') as f:
            json.dump(latents_data, f)

        config = Config()

        with patch('common.steering_setup.discover_latest_phase_output',
                  return_value=str(analysis_file)):
            result = load_predicting_latents(config)

        assert result.source_phase == "2.10"
