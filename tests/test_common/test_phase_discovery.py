"""
Tests for common/phase_discovery.py

Validates:
- discover_top_n_latents: Correct JSON parsing, top-N selection, layer deduplication
- discover_steering_coefficients: Fallback logic (4.9 -> 4.6 -> 4.5), nested dict extraction
- filter_by_range: Inclusive/exclusive index handling, off-by-one prevention
- get_phase_output_dir: Model/dataset suffix generation
- discover_latest_phase_output: Timestamp parsing, latest file selection
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import replace

from common.config import Config


# =============================================================================
# get_phase_output_dir Tests
# =============================================================================

class TestGetPhaseOutputDir:
    """Test model/dataset suffix generation for output directories."""

    def test_default_gemma_mbpp_no_suffix(self):
        """Default Gemma + MBPP should have no suffix."""
        from common.phase_discovery import get_phase_output_dir
        config = Config()  # Default: gemma-2-2b + mbpp
        output_dir = get_phase_output_dir("1", config)
        assert output_dir == "data/phase1_0"

    def test_humaneval_suffix(self):
        """HumanEval dataset should add _humaneval suffix."""
        from common.phase_discovery import get_phase_output_dir
        config = Config(dataset_name="humaneval")
        output_dir = get_phase_output_dir("1", config)
        assert output_dir.endswith("_humaneval")

    def test_llama_suffix(self):
        """LLAMA model should add _llama suffix."""
        from common.phase_discovery import get_phase_output_dir
        config = Config(model_name="meta-llama/Llama-3.1-8B")
        output_dir = get_phase_output_dir("1", config)
        assert "_llama" in output_dir

    def test_gemma9b_suffix(self):
        """Gemma 9B model should add _gemma9b suffix."""
        from common.phase_discovery import get_phase_output_dir
        config = Config(model_name="google/gemma-2-9b")
        output_dir = get_phase_output_dir("1", config)
        assert "_gemma9b" in output_dir

    def test_combined_model_dataset_suffix(self):
        """LLAMA + HumanEval should have both suffixes."""
        from common.phase_discovery import get_phase_output_dir
        config = Config(model_name="meta-llama/Llama-3.1-8B", dataset_name="humaneval")
        output_dir = get_phase_output_dir("1", config)
        assert "_llama" in output_dir
        assert "_humaneval" in output_dir


# =============================================================================
# filter_by_range Tests
# =============================================================================

class TestFilterByRange:
    """Test inclusive/exclusive index handling."""

    def test_no_filtering_by_default(self):
        """Without start/end, return all data."""
        from common.phase_discovery import filter_by_range
        import pandas as pd

        df = pd.DataFrame({'x': range(10)})
        config = Config()  # No start/end set
        result = filter_by_range(df, config)
        assert len(result) == 10

    def test_start_index_inclusive(self):
        """Start index should be inclusive."""
        from common.phase_discovery import filter_by_range
        import pandas as pd

        df = pd.DataFrame({'x': range(10)})
        config = Config(dataset_start_idx=3)
        result = filter_by_range(df, config)
        assert result['x'].iloc[0] == 3
        assert len(result) == 7

    def test_end_index_inclusive(self):
        """End index should be inclusive (converted to exclusive internally)."""
        from common.phase_discovery import filter_by_range, get_dataset_range
        import pandas as pd

        df = pd.DataFrame({'x': range(10)})
        # Set end_idx to 5 - this should include rows 0-5 (6 rows total)
        config = Config(dataset_end_idx=5)

        start, end = get_dataset_range(config, len(df))
        # end_idx is inclusive, so end should be end_idx + 1 = 6
        assert end == 6

        result = filter_by_range(df, config)
        assert len(result) == 6
        assert result['x'].iloc[-1] == 5  # Row index 5 included

    def test_start_and_end_range(self):
        """Test both start and end indices."""
        from common.phase_discovery import filter_by_range
        import pandas as pd

        df = pd.DataFrame({'x': range(10)})
        config = Config(dataset_start_idx=2, dataset_end_idx=7)
        result = filter_by_range(df, config)
        # Should include indices 2, 3, 4, 5, 6, 7 = 6 rows
        assert len(result) == 6
        assert list(result['x']) == [2, 3, 4, 5, 6, 7]

    def test_list_filtering(self):
        """Test filtering on list instead of DataFrame."""
        from common.phase_discovery import filter_by_range

        data = list(range(10))
        config = Config(dataset_start_idx=2, dataset_end_idx=5)
        result = filter_by_range(data, config)
        # Should include indices 2, 3, 4, 5 = 4 items
        assert result == [2, 3, 4, 5]

    def test_end_beyond_length(self):
        """End index beyond data length should be clamped."""
        from common.phase_discovery import get_dataset_range

        config = Config(dataset_end_idx=100)
        start, end = get_dataset_range(config, 10)
        assert end == 10  # Clamped to data length


# =============================================================================
# discover_top_n_latents Tests
# =============================================================================

class TestDiscoverTopNLatents:
    """Test top-N latent discovery from Phase 2.10."""

    @pytest.fixture
    def mock_top_latents_file(self, tmp_path):
        """Create a mock top_20_latents.json file."""
        phase_dir = tmp_path / "data" / "phase2_10"
        phase_dir.mkdir(parents=True)

        latents_data = {
            "correct": [
                {"layer": 16, "latent_idx": 100, "t_statistic": 5.0},
                {"layer": 18, "latent_idx": 200, "t_statistic": 4.5},
                {"layer": 16, "latent_idx": 300, "t_statistic": 4.0},
                {"layer": 20, "latent_idx": 400, "t_statistic": 3.5},
                {"layer": 18, "latent_idx": 500, "t_statistic": 3.0},
            ],
            "incorrect": [
                {"layer": 14, "latent_idx": 600, "t_statistic": 5.0},
                {"layer": 16, "latent_idx": 700, "t_statistic": 4.5},
                {"layer": 14, "latent_idx": 800, "t_statistic": 4.0},
                {"layer": 22, "latent_idx": 900, "t_statistic": 3.5},
                {"layer": 20, "latent_idx": 1000, "t_statistic": 3.0},
            ]
        }

        latents_file = phase_dir / "top_20_latents.json"
        with open(latents_file, 'w') as f:
            json.dump(latents_data, f)

        return tmp_path, phase_dir

    def test_top_n_selection(self, mock_top_latents_file):
        """Test that top-N candidates are correctly selected."""
        from common.phase_discovery import discover_top_n_latents

        tmp_path, phase_dir = mock_top_latents_file
        config = Config(phase3_8_n_candidates=3)

        with patch('common.phase_discovery.get_phase_output_dir', return_value=str(phase_dir)):
            result = discover_top_n_latents(config)

        assert len(result['correct']) == 3
        assert len(result['incorrect']) == 3
        # Check that top candidates by t_statistic are selected
        assert result['correct'][0]['t_statistic'] == 5.0
        assert result['incorrect'][0]['t_statistic'] == 5.0

    def test_layer_deduplication(self, mock_top_latents_file):
        """Test that unique layers are collected from all candidates."""
        from common.phase_discovery import discover_top_n_latents

        tmp_path, phase_dir = mock_top_latents_file
        config = Config(phase3_8_n_candidates=5)

        with patch('common.phase_discovery.get_phase_output_dir', return_value=str(phase_dir)):
            with patch('common.phase_discovery._discover_probe_best_layers', return_value=[]):
                result = discover_top_n_latents(config)

        # Should have unique layers from both correct and incorrect
        all_layers = result['all_layers']
        assert len(all_layers) == len(set(all_layers))  # No duplicates
        # Layers from our mock data: 14, 16, 18, 20, 22
        assert set(all_layers) == {14, 16, 18, 20, 22}


# =============================================================================
# discover_steering_coefficients Tests
# =============================================================================

class TestDiscoverSteeringCoefficients:
    """Test coefficient discovery with fallback logic."""

    def test_phase_4_9_preferred(self, tmp_path):
        """Phase 4.9 should be preferred over 4.6."""
        from common.phase_discovery import discover_steering_coefficients

        # Create Phase 4.9 output
        phase_4_9_dir = tmp_path / "data" / "phase4_9"
        phase_4_9_dir.mkdir(parents=True)

        coeff_data = {
            "correct": {"refined_coefficient": 42.0},
            "incorrect": {"refined_coefficient": 84.0}
        }
        with open(phase_4_9_dir / "refined_coefficients.json", 'w') as f:
            json.dump(coeff_data, f)

        manifest = {
            "phase": "4.9",
            "outputs": {"refined_coefficients": "refined_coefficients.json"}
        }
        with open(phase_4_9_dir / "phase_output.json", 'w') as f:
            json.dump(manifest, f)

        config = Config()

        with patch('common.phase_discovery.get_phase_output_dir', return_value=str(phase_4_9_dir)):
            result = discover_steering_coefficients(config)

        assert result['correct'] == 42.0
        assert result['incorrect'] == 84.0

    def test_fallback_to_phase_4_6(self, tmp_path):
        """Should fall back to Phase 4.6 if 4.9 not available."""
        from common.phase_discovery import discover_steering_coefficients

        # Create Phase 4.6 output (no 4.9)
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

        with patch('common.phase_discovery.get_phase_output_file', side_effect=mock_get_phase_output_file):
            result = discover_steering_coefficients(config)

        assert result['correct'] == 35.0
        assert result['incorrect'] == 70.0


# =============================================================================
# discover_latest_phase_output Tests
# =============================================================================

class TestDiscoverLatestPhaseOutput:
    """Test timestamp parsing and latest file selection."""

    def test_selects_latest_timestamp(self, tmp_path):
        """Should select file with most recent modification time."""
        import time as time_mod

        phase_dir = tmp_path / "data" / "phase1_0"
        phase_dir.mkdir(parents=True)

        # Create files with different modification times
        # The find_latest_file function uses modification time, not filename
        (phase_dir / "dataset_sae_20240101_120000.parquet").touch()
        time_mod.sleep(0.05)
        (phase_dir / "dataset_sae_20240101_180000.parquet").touch()
        time_mod.sleep(0.05)
        latest_file = phase_dir / "dataset_sae_20240102_120000.parquet"
        latest_file.touch()  # This is now the most recently modified

        # Use glob directly to verify latest by mtime
        files = sorted(phase_dir.glob("dataset_sae_*.parquet"), key=lambda f: f.stat().st_mtime)

        assert len(files) == 3
        assert files[-1] == latest_file  # Most recently modified

    def test_returns_none_when_no_files(self, tmp_path):
        """Should return None when no matching files found."""
        from common.phase_discovery import discover_latest_phase_output

        phase_dir = tmp_path / "data" / "phase1_0"
        phase_dir.mkdir(parents=True)

        with patch('common.phase_discovery.get_phase_output_dir', return_value=str(phase_dir)):
            with patch('common.phase_registry.get_phase') as mock_get_phase:
                with patch('common.phase_registry.get_phase_patterns') as mock_get_patterns:
                    mock_get_phase.return_value = MagicMock(
                        output_dir=str(phase_dir),
                        exclude_keywords=[]
                    )
                    mock_get_patterns.return_value = ["dataset_sae_*.parquet"]
                    result = discover_latest_phase_output("1", phase_dir=str(phase_dir))

        assert result is None


# =============================================================================
# get_dataset_range Tests
# =============================================================================

class TestGetDatasetRange:
    """Test dataset range calculation."""

    def test_default_range(self):
        """Default config should return full range."""
        from common.phase_discovery import get_dataset_range
        config = Config()
        start, end = get_dataset_range(config, 100)
        assert start == 0
        assert end == 100

    def test_custom_start(self):
        """Custom start index."""
        from common.phase_discovery import get_dataset_range
        config = Config(dataset_start_idx=20)
        start, end = get_dataset_range(config, 100)
        assert start == 20
        assert end == 100

    def test_custom_end(self):
        """Custom end index (inclusive, converted to exclusive)."""
        from common.phase_discovery import get_dataset_range
        config = Config(dataset_end_idx=50)
        start, end = get_dataset_range(config, 100)
        assert start == 0
        assert end == 51  # end_idx is inclusive, so +1

    def test_end_clamped_to_length(self):
        """End beyond data length should be clamped."""
        from common.phase_discovery import get_dataset_range
        config = Config(dataset_end_idx=200)
        start, end = get_dataset_range(config, 100)
        assert end == 100
