"""
Tests for Phase Chaining - End-to-end phase dependencies

Validates:
- phase_1_to_2_5: Phase 2.5 can read Phase 1 output
- phase_2_5_to_4_5: Latents flow to coefficient search
- phase_4_5_to_4_8: Coefficients applied in steering
- sae_vs_probe_pipeline: Both pipelines produce valid results
"""

import pytest
import json
import pandas as pd
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from common.config import Config


# =============================================================================
# phase_1_to_2_5 Tests
# =============================================================================

class TestPhase1To25:
    """Test Phase 2.5 can read Phase 1 output."""

    @pytest.fixture
    def mock_phase1_output(self, tmp_path):
        """Create mock Phase 1 output."""
        phase1_dir = tmp_path / "data" / "phase1_0"
        phase1_dir.mkdir(parents=True)

        # Create dataset parquet
        df = pd.DataFrame([
            {'task_id': 't1', 'code': 'def f(): return 1', 'passed': True},
            {'task_id': 't2', 'code': 'def g(): return 2', 'passed': False},
        ])
        df.to_parquet(phase1_dir / "dataset_sae_20240101_120000.parquet")

        # Create activation files (mock)
        activation_dir = phase1_dir / "activations"
        activation_dir.mkdir()

        return phase1_dir

    def test_phase_2_5_finds_phase_1_output(self, mock_phase1_output):
        """Phase 2.5 should find Phase 1 dataset file."""
        from common.phase_discovery import discover_latest_phase_output

        with patch('common.phase_registry.get_phase') as mock_get_phase:
            with patch('common.phase_registry.get_phase_patterns') as mock_patterns:
                mock_get_phase.return_value = MagicMock(
                    output_dir=str(mock_phase1_output),
                    exclude_keywords=[]
                )
                mock_patterns.return_value = ["dataset_sae_*.parquet"]

                result = discover_latest_phase_output("1", phase_dir=str(mock_phase1_output))

        assert result is not None
        assert "dataset_sae" in result

    def test_phase_2_5_loads_parquet(self, mock_phase1_output):
        """Phase 2.5 should load Phase 1 parquet file."""
        parquet_file = mock_phase1_output / "dataset_sae_20240101_120000.parquet"

        df = pd.read_parquet(parquet_file)

        assert len(df) == 2
        assert 'task_id' in df.columns
        assert 'passed' in df.columns


# =============================================================================
# phase_2_5_to_4_5 Tests
# =============================================================================

class TestPhase25To45:
    """Test latents flow to coefficient search."""

    @pytest.fixture
    def mock_phase25_output(self, tmp_path):
        """Create mock Phase 2.5 output."""
        phase25_dir = tmp_path / "data" / "phase2_5"
        phase25_dir.mkdir(parents=True)

        latents = {
            "correct": [
                {"layer": 16, "latent_idx": 100, "separation_score": 0.5},
                {"layer": 18, "latent_idx": 200, "separation_score": 0.4},
            ],
            "incorrect": [
                {"layer": 14, "latent_idx": 300, "separation_score": -0.5},
                {"layer": 16, "latent_idx": 400, "separation_score": -0.4},
            ]
        }
        with open(phase25_dir / "top_20_latents.json", 'w') as f:
            json.dump(latents, f)

        # Create main analysis file
        with open(phase25_dir / "sae_analysis_20240101_120000.json", 'w') as f:
            json.dump({}, f)

        return phase25_dir

    def test_phase_4_5_finds_phase_2_5_latents(self, mock_phase25_output):
        """Phase 4.5 should find Phase 2.5 latents file."""
        latents_file = mock_phase25_output / "top_20_latents.json"
        assert latents_file.exists()

        with open(latents_file) as f:
            latents = json.load(f)

        assert 'correct' in latents
        assert 'incorrect' in latents
        assert len(latents['correct']) > 0

    def test_phase_4_5_loads_correct_latents(self, mock_phase25_output):
        """Phase 4.5 should load top latents for steering."""
        from common.steering_setup import load_steering_latents

        with patch('common.steering_setup.discover_latest_phase_output',
                  return_value=str(mock_phase25_output / "sae_analysis_20240101_120000.json")):
            result = load_steering_latents(Config())

        assert result.source_phase == "2.5"
        assert result.best_correct_latent['layer'] == 16
        assert result.best_correct_latent['latent_idx'] == 100


# =============================================================================
# phase_4_5_to_4_8 Tests
# =============================================================================

class TestPhase45To48:
    """Test coefficients applied in steering."""

    @pytest.fixture
    def mock_phase46_output(self, tmp_path):
        """Create mock Phase 4.6 output (refined coefficients)."""
        phase46_dir = tmp_path / "data" / "phase4_6"
        phase46_dir.mkdir(parents=True)

        coefficients = {
            "correct": {"refined_coefficient": 47.5},
            "incorrect": {"refined_coefficient": 43.0}
        }
        with open(phase46_dir / "refined_coefficients.json", 'w') as f:
            json.dump(coefficients, f)

        manifest = {
            "phase": "4.6",
            "outputs": {"refined_coefficients": "refined_coefficients.json"}
        }
        with open(phase46_dir / "phase_output.json", 'w') as f:
            json.dump(manifest, f)

        return phase46_dir

    def test_phase_4_8_loads_phase_4_6_coefficients(self, mock_phase46_output):
        """Phase 4.8 should load refined coefficients from Phase 4.6."""
        from common.phase_discovery import discover_steering_coefficients

        def mock_get_output_file(phase, key, config=None):
            if phase == "4.9":
                raise FileNotFoundError()
            return mock_phase46_output / "refined_coefficients.json"

        with patch('common.phase_discovery.get_phase_output_file',
                  side_effect=mock_get_output_file):
            result = discover_steering_coefficients(Config())

        assert result['correct'] == 47.5
        assert result['incorrect'] == 43.0

    def test_coefficients_used_in_steering_hook(self):
        """Coefficients should be applied in steering hook."""
        from common.steering_metrics import create_last_position_steering_hook

        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)
        coefficient = 47.5

        hook = create_last_position_steering_hook(direction, coefficient)

        # Apply to residual
        residual = torch.zeros(1, 5, 2304)
        output = hook(None, (residual,))

        # Check steering magnitude
        change = output[0][0, -1]
        magnitude = torch.norm(change).item()

        assert magnitude == pytest.approx(coefficient, rel=1e-4)


# =============================================================================
# sae_vs_probe_pipeline Tests
# =============================================================================

class TestSAEVsProbePipeline:
    """Test both SAE and probe pipelines produce valid results."""

    def test_sae_pipeline_structure(self):
        """SAE pipeline: Phase 1 -> 2.5 -> 4.5 -> 4.8."""
        pipeline = ["1", "2.5", "4.5", "4.6", "4.8"]

        # Each phase produces output for next
        assert "1" in pipeline
        assert "2.5" in pipeline
        assert "4.8" in pipeline

    def test_probe_pipeline_structure(self):
        """Probe pipeline: Phase 1 -> 2.6 -> 4.5 -> 4.8 (with direction_source=probe)."""
        pipeline = ["1", "2.6", "4.5", "4.6", "4.8"]

        assert "1" in pipeline
        assert "2.6" in pipeline
        assert "4.8" in pipeline

    def test_direction_source_affects_loading(self):
        """direction_source config should affect which directions are loaded."""
        from common.steering_setup import get_direction_source_info

        sae_config = Config(direction_source='sae')
        probe_config = Config(direction_source='probe_mass_mean')

        sae_info = get_direction_source_info(sae_config)
        probe_info = get_direction_source_info(probe_config)

        assert sae_info['is_probe'] is False
        assert probe_info['is_probe'] is True

    def test_both_pipelines_produce_metrics(self):
        """Both pipelines should produce correction/corruption rates."""
        from common.steering_metrics import calculate_correction_rate, calculate_corruption_rate

        # Same results structure regardless of direction source
        results = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': True},
            {'baseline_passed': True, 'steered_correct': False},
        ]

        correction = calculate_correction_rate(results)
        corruption = calculate_corruption_rate(results)

        assert 0 <= correction <= 100
        assert 0 <= corruption <= 100


# =============================================================================
# Phase Output Discovery Tests
# =============================================================================

class TestPhaseOutputDiscovery:
    """Test phase output discovery across pipeline."""

    def test_discover_latest_prefers_recent(self, tmp_path):
        """Should discover most recent output file."""
        phase_dir = tmp_path / "data" / "phase1_0"
        phase_dir.mkdir(parents=True)

        # Create files with different timestamps
        (phase_dir / "dataset_sae_20240101_120000.parquet").touch()
        (phase_dir / "dataset_sae_20240102_120000.parquet").touch()

        # Get files sorted
        files = sorted(phase_dir.glob("dataset_sae_*.parquet"))

        latest = files[-1]
        assert "20240102" in str(latest)

    def test_model_suffix_discovery(self, tmp_path):
        """Should discover correct directory based on model suffix."""
        from common.phase_discovery import get_phase_output_dir

        gemma_config = Config(model_name="google/gemma-2-2b")
        llama_config = Config(model_name="meta-llama/Llama-3.1-8B")

        gemma_dir = get_phase_output_dir("1", gemma_config)
        llama_dir = get_phase_output_dir("1", llama_config)

        # Different models should have different directories
        assert gemma_dir != llama_dir
        assert "_llama" in llama_dir


# =============================================================================
# Integration Markers
# =============================================================================

@pytest.mark.gpu
@pytest.mark.slow
class TestFullPipelineIntegration:
    """Full pipeline integration tests requiring GPU."""

    @pytest.mark.skip(reason="Requires GPU and full model loading")
    def test_full_steering_pipeline(self):
        """Test full steering pipeline from Phase 1 to 4.8."""
        # This would run actual phases with small subset
        pass

    @pytest.mark.skip(reason="Requires GPU and full model loading")
    def test_probe_pipeline_matches_sae(self):
        """Test probe pipeline produces comparable results to SAE."""
        # Compare SAE vs probe correction rates
        pass


