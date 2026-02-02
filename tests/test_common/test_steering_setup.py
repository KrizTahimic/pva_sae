"""
Tests for common/steering_setup.py

Validates:
- direction_normalization: Output is unit L2 norm
- dtype_matching_for_steering: Direction matches model dtype
- dtype_float32_for_predicting: Prediction directions stay float32
- split_by_correctness: Correct boolean masking
- sae_vs_probe_loading: Different paths produce expected formats
"""

import pytest
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

from common.config import Config


# =============================================================================
# direction_normalization Tests
# =============================================================================

class TestDirectionNormalization:
    """Test output is unit L2 norm."""

    def test_sae_direction_normalized(self):
        """SAE directions should be normalized to unit L2 norm."""
        from common.steering_setup import load_sae_and_directions

        # Create mock SAE with known W_dec
        mock_sae = MagicMock()
        mock_sae.W_dec = torch.randn(16384, 2304)

        # Create mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])

        best_correct = {'layer': 16, 'latent_idx': 100}
        best_incorrect = {'layer': 18, 'latent_idx': 200}

        with patch('common.steering_setup.load_sae_for_config', return_value=mock_sae):
            result = load_sae_and_directions(
                Config(),
                torch.device('cpu'),
                mock_model,
                best_correct,
                best_incorrect
            )

        # Check normalization
        correct_norm = torch.norm(result.correct_direction).item()
        incorrect_norm = torch.norm(result.incorrect_direction).item()

        assert correct_norm == pytest.approx(1.0, rel=1e-5)
        assert incorrect_norm == pytest.approx(1.0, rel=1e-5)

    def test_probe_direction_not_modified_for_prediction(self):
        """Probe directions for prediction should not be normalized (kept as loaded)."""
        from common.steering_setup import load_probe_directions_for_predicting

        # Mock the probe loading
        mock_direction = torch.randn(2304)
        mock_direction = mock_direction / torch.norm(mock_direction)  # Pre-normalized

        with patch('common.steering_setup._load_probe_base',
                  return_value=(mock_direction, 16, 0.0, "/mock/path")):
            result = load_probe_directions_for_predicting(
                Config(),
                torch.device('cpu'),
                method="logreg"
            )

        # Direction should be preserved as loaded
        assert torch.norm(result.correct_direction).item() == pytest.approx(1.0, rel=1e-5)


# =============================================================================
# dtype_matching_for_steering Tests
# =============================================================================

class TestDtypeMatchingForSteering:
    """Test direction matches model dtype for steering."""

    def test_bfloat16_model_gets_bfloat16_direction(self):
        """Steering direction should match bfloat16 model dtype."""
        from common.steering_setup import load_sae_and_directions

        mock_sae = MagicMock()
        mock_sae.W_dec = torch.randn(16384, 2304)

        # Create model with bfloat16 parameters
        mock_model = MagicMock()
        mock_param = torch.zeros(1, dtype=torch.bfloat16)
        mock_model.parameters.return_value = iter([mock_param])

        best_correct = {'layer': 16, 'latent_idx': 100}
        best_incorrect = {'layer': 18, 'latent_idx': 200}

        with patch('common.steering_setup.load_sae_for_config', return_value=mock_sae):
            result = load_sae_and_directions(
                Config(),
                torch.device('cpu'),
                mock_model,
                best_correct,
                best_incorrect
            )

        assert result.correct_direction.dtype == torch.bfloat16
        assert result.incorrect_direction.dtype == torch.bfloat16

    def test_float32_model_gets_float32_direction(self):
        """Steering direction should match float32 model dtype."""
        from common.steering_setup import load_sae_and_directions

        mock_sae = MagicMock()
        mock_sae.W_dec = torch.randn(16384, 2304)

        # Create model with float32 parameters
        mock_model = MagicMock()
        mock_param = torch.zeros(1, dtype=torch.float32)
        mock_model.parameters.return_value = iter([mock_param])

        best_correct = {'layer': 16, 'latent_idx': 100}
        best_incorrect = {'layer': 18, 'latent_idx': 200}

        with patch('common.steering_setup.load_sae_for_config', return_value=mock_sae):
            result = load_sae_and_directions(
                Config(),
                torch.device('cpu'),
                mock_model,
                best_correct,
                best_incorrect
            )

        assert result.correct_direction.dtype == torch.float32
        assert result.incorrect_direction.dtype == torch.float32


# =============================================================================
# dtype_float32_for_predicting Tests
# =============================================================================

class TestDtypeFloat32ForPredicting:
    """Test prediction directions stay float32."""

    def test_prediction_direction_stays_float32(self):
        """Prediction probe directions should remain float32."""
        from common.steering_setup import load_probe_directions_for_predicting

        # Mock direction in float32
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


# =============================================================================
# split_by_correctness Tests
# =============================================================================

class TestSplitByCorrectness:
    """Test correct boolean masking."""

    def test_basic_split(self):
        """Should correctly split by baseline_passed column."""
        from common.steering_setup import split_by_correctness

        df = pd.DataFrame({
            'task_id': ['t1', 't2', 't3', 't4'],
            'baseline_passed': [True, False, True, False]
        })

        correct, incorrect = split_by_correctness(df)

        assert len(correct) == 2
        assert len(incorrect) == 2
        assert set(correct['task_id']) == {'t1', 't3'}
        assert set(incorrect['task_id']) == {'t2', 't4'}

    def test_all_correct(self):
        """Should handle all correct case."""
        from common.steering_setup import split_by_correctness

        df = pd.DataFrame({
            'task_id': ['t1', 't2'],
            'baseline_passed': [True, True]
        })

        correct, incorrect = split_by_correctness(df)

        assert len(correct) == 2
        assert len(incorrect) == 0

    def test_all_incorrect(self):
        """Should handle all incorrect case."""
        from common.steering_setup import split_by_correctness

        df = pd.DataFrame({
            'task_id': ['t1', 't2'],
            'baseline_passed': [False, False]
        })

        correct, incorrect = split_by_correctness(df)

        assert len(correct) == 0
        assert len(incorrect) == 2

    def test_copies_are_independent(self):
        """Returned DataFrames should be copies, not views."""
        from common.steering_setup import split_by_correctness

        df = pd.DataFrame({
            'task_id': ['t1', 't2'],
            'baseline_passed': [True, False]
        })

        correct, incorrect = split_by_correctness(df)

        # Modify correct - should not affect original
        correct['new_col'] = 1

        assert 'new_col' not in df.columns


# =============================================================================
# sae_vs_probe_loading Tests
# =============================================================================

class TestSaeVsProbeLoading:
    """Test different paths produce expected formats."""

    def test_get_direction_source_info_sae(self):
        """SAE source should return correct info dict."""
        from common.steering_setup import get_direction_source_info

        config = Config(direction_source="sae")
        info = get_direction_source_info(config)

        assert info['source'] == 'sae'
        assert info['is_probe'] is False
        assert info['probe_method'] is None

    def test_get_direction_source_info_probe_logreg(self):
        """Probe logreg source should return correct info dict."""
        from common.steering_setup import get_direction_source_info

        config = Config(direction_source="probe_logreg")
        info = get_direction_source_info(config)

        assert info['source'] == 'probe'
        assert info['is_probe'] is True
        assert info['probe_method'] == 'logreg'

    def test_get_direction_source_info_probe_mass_mean(self):
        """Probe mass_mean source should return correct info dict."""
        from common.steering_setup import get_direction_source_info

        config = Config(direction_source="probe_mass_mean")
        info = get_direction_source_info(config)

        assert info['source'] == 'probe'
        assert info['is_probe'] is True
        assert info['probe_method'] == 'mass_mean'

    def test_unknown_source_defaults_to_sae(self):
        """Unknown direction source should default to SAE."""
        from common.steering_setup import get_direction_source_info

        config = Config()
        config.direction_source = "invalid_source"  # Bypass validation

        info = get_direction_source_info(config)

        assert info['source'] == 'sae'
        assert info['is_probe'] is False


# =============================================================================
# PVA Latents Loading Tests
# =============================================================================

class TestPVALatentsLoading:
    """Test latent loading from different phases."""

    def test_load_steering_latents_structure(self, tmp_path):
        """Steering latents should return correct structure."""
        from common.steering_setup import load_steering_latents

        # Create mock Phase 2.5 output
        phase_dir = tmp_path / "data" / "phase2_5"
        phase_dir.mkdir(parents=True)

        latents_data = {
            "correct": [
                {"layer": 16, "latent_idx": 100, "separation_score": 0.5},
            ],
            "incorrect": [
                {"layer": 18, "latent_idx": 200, "separation_score": 0.4},
            ]
        }

        # Create sae_analysis file (for discover_latest_phase_output)
        analysis_file = phase_dir / "sae_analysis_20240101_120000.json"
        with open(analysis_file, 'w') as f:
            json.dump({}, f)

        # Create top_20_latents.json
        with open(phase_dir / "top_20_latents.json", 'w') as f:
            json.dump(latents_data, f)

        config = Config()

        with patch('common.steering_setup.discover_latest_phase_output',
                  return_value=str(analysis_file)):
            result = load_steering_latents(config)

        assert result.source_phase == "2.5"
        assert result.best_correct_latent['layer'] == 16
        assert result.best_incorrect_latent['layer'] == 18

    def test_load_predicting_latents_structure(self, tmp_path):
        """Predicting latents should return correct structure."""
        from common.steering_setup import load_predicting_latents

        # Create mock Phase 2.10 output
        phase_dir = tmp_path / "data" / "phase2_10"
        phase_dir.mkdir(parents=True)

        latents_data = {
            "correct": [
                {"layer": 16, "latent_idx": 100, "t_statistic": 5.0},
            ],
            "incorrect": [
                {"layer": 18, "latent_idx": 200, "t_statistic": 4.5},
            ]
        }

        # Create output file
        analysis_file = phase_dir / "t_statistic_analysis_20240101_120000.json"
        with open(analysis_file, 'w') as f:
            json.dump({}, f)

        # Create top_20_latents.json
        with open(phase_dir / "top_20_latents.json", 'w') as f:
            json.dump(latents_data, f)

        config = Config()

        with patch('common.steering_setup.discover_latest_phase_output',
                  return_value=str(analysis_file)):
            result = load_predicting_latents(config)

        assert result.source_phase == "2.10"
        assert result.best_correct_latent['layer'] == 16
        assert result.best_incorrect_latent['layer'] == 18


# =============================================================================
# Baseline Data Loading Tests
# =============================================================================

class TestBaselineDataLoading:
    """Test baseline data loading from phases."""

    def test_applies_filter_by_range(self, tmp_path):
        """Should apply --start/--end filtering."""
        from common.steering_setup import load_baseline_data

        # Create mock Phase 3.5 output
        phase_dir = tmp_path / "data" / "phase3_5"
        phase_dir.mkdir(parents=True)

        # Create dataset
        df = pd.DataFrame({
            'task_id': [f't{i}' for i in range(100)],
            'baseline_passed': [i % 2 == 0 for i in range(100)]
        })
        df.to_parquet(phase_dir / "dataset_temp_0_0.parquet")

        config = Config(dataset_start_idx=10, dataset_end_idx=20)

        with patch('common.steering_setup.discover_latest_phase_output',
                  return_value=str(phase_dir / "dataset_temp_0_0.parquet")):
            result_df, _ = load_baseline_data(config, "3.5")

        # Should have filtered to 11 rows (indices 10-20 inclusive)
        assert len(result_df) == 11

    def test_merged_file_fallback(self, tmp_path):
        """Should fall back to merged file if primary not found."""
        from common.steering_setup import load_baseline_data

        # Create mock Phase 3.5 output with merged file only
        phase_dir = tmp_path / "data" / "phase3_5"
        phase_dir.mkdir(parents=True)

        df = pd.DataFrame({
            'task_id': ['t1', 't2'],
            'baseline_passed': [True, False]
        })
        df.to_parquet(phase_dir / "dataset_merged_20240101.parquet")

        config = Config()

        with patch('common.steering_setup.discover_latest_phase_output',
                  return_value=str(phase_dir / "dataset_merged_20240101.parquet")):
            result_df, _ = load_baseline_data(config, "3.5")

        assert len(result_df) == 2
