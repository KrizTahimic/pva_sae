"""
Tests for common/pile_filter_utils.py

Validates:
- load_pile_frequencies: directory handling, layer loading, missing files
- apply_pile_filter: threshold filtering, max_features cap, missing pile data
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from common.config import Config
from common.pile_filter_utils import load_pile_frequencies, apply_pile_filter


# =============================================================================
# load_pile_frequencies Tests
# =============================================================================

class TestLoadPileFrequencies:
    """Test pile frequency loading from Phase 2.3 output."""

    def test_missing_directory_raises_error(self, tmp_path):
        """Should raise FileNotFoundError when Phase 2.3 dir doesn't exist."""
        config = Config()
        with patch('common.pile_filter_utils.get_phase_output_dir', return_value=str(tmp_path / "nonexistent")):
            with pytest.raises(FileNotFoundError, match="Pile frequencies not found"):
                load_pile_frequencies(config)

    def test_loads_existing_frequency_files(self, tmp_path):
        """Should load frequency tensors for each activation layer."""
        config = Config()
        freq_dir = tmp_path / "phase2_3"
        freq_dir.mkdir()

        # Create mock frequency files for configured layers
        test_layers = config.activation_layers[:2]  # Use first 2 layers
        for layer_idx in test_layers:
            freq_tensor = torch.rand(16384)  # 16k SAE width
            from common.tensor_utils import save_activation
            save_activation(freq_tensor, freq_dir / f"layer_{layer_idx}_frequencies.safetensors")

        with patch('common.pile_filter_utils.get_phase_output_dir', return_value=str(freq_dir)):
            frequencies = load_pile_frequencies(config)

        # Verify loaded layers
        for layer_idx in test_layers:
            assert layer_idx in frequencies
            assert frequencies[layer_idx] is not None
            assert frequencies[layer_idx].shape == (16384,)

    def test_missing_layer_file_returns_none(self, tmp_path):
        """Should set None for layers without frequency files."""
        config = Config()
        freq_dir = tmp_path / "phase2_3"
        freq_dir.mkdir()
        # Don't create any frequency files

        with patch('common.pile_filter_utils.get_phase_output_dir', return_value=str(freq_dir)):
            frequencies = load_pile_frequencies(config)

        # All layers should be None
        for layer_idx in config.activation_layers:
            assert frequencies[layer_idx] is None


# =============================================================================
# apply_pile_filter Tests
# =============================================================================

class TestApplyPileFilter:
    """Test pile-based feature filtering."""

    def _make_features(self, layer, indices):
        """Helper to create feature dicts."""
        return [{'layer': layer, 'latent_idx': idx, 'score': 1.0} for idx in indices]

    def test_filters_high_frequency_features(self):
        """Features above threshold should be filtered out."""
        features = {
            'correct': self._make_features(10, [0, 1, 2, 3, 4]),
            'incorrect': self._make_features(10, [5, 6, 7, 8, 9])
        }
        # Frequencies: features 0,1,5,6 are high-frequency (general language)
        freq_tensor = torch.zeros(100)
        freq_tensor[0] = 0.8
        freq_tensor[1] = 0.9
        freq_tensor[5] = 0.7
        freq_tensor[6] = 0.85
        # Features 2,3,4,7,8,9 are low-frequency (code-specific)
        freq_tensor[2] = 0.1
        freq_tensor[3] = 0.05
        freq_tensor[4] = 0.2
        freq_tensor[7] = 0.15
        freq_tensor[8] = 0.02
        freq_tensor[9] = 0.3

        pile_frequencies = {10: freq_tensor}

        filtered = apply_pile_filter(features, pile_frequencies, threshold=0.5)

        # High-frequency features should be removed
        correct_indices = [f['latent_idx'] for f in filtered['correct']]
        incorrect_indices = [f['latent_idx'] for f in filtered['incorrect']]

        assert 0 not in correct_indices  # 0.8 >= 0.5
        assert 1 not in correct_indices  # 0.9 >= 0.5
        assert 2 in correct_indices      # 0.1 < 0.5
        assert 3 in correct_indices      # 0.05 < 0.5
        assert 4 in correct_indices      # 0.2 < 0.5

        assert 5 not in incorrect_indices  # 0.7 >= 0.5
        assert 6 not in incorrect_indices  # 0.85 >= 0.5
        assert 7 in incorrect_indices      # 0.15 < 0.5

    def test_max_features_cap(self):
        """Should keep at most max_features per category."""
        features = {
            'correct': self._make_features(10, list(range(30))),
            'incorrect': self._make_features(10, list(range(30, 60)))
        }
        # All low frequency (pass filter)
        freq_tensor = torch.zeros(100) + 0.01
        pile_frequencies = {10: freq_tensor}

        filtered = apply_pile_filter(features, pile_frequencies, threshold=0.5, max_features=5)

        assert len(filtered['correct']) == 5
        assert len(filtered['incorrect']) == 5

    def test_missing_pile_data_keeps_features(self):
        """Features from layers without pile data should be kept."""
        features = {
            'correct': self._make_features(10, [0, 1, 2]),
            'incorrect': self._make_features(15, [3, 4, 5])
        }
        # Only layer 10 has data, layer 15 is missing
        pile_frequencies = {10: torch.zeros(100) + 0.01}

        filtered = apply_pile_filter(features, pile_frequencies, threshold=0.5)

        # Layer 15 features should be kept (no pile data = keep)
        assert len(filtered['incorrect']) == 3

    def test_none_pile_data_keeps_features(self):
        """Features from layers with None pile data should be kept."""
        features = {
            'correct': self._make_features(10, [0, 1]),
            'incorrect': []
        }
        pile_frequencies = {10: None}

        filtered = apply_pile_filter(features, pile_frequencies, threshold=0.5)

        assert len(filtered['correct']) == 2

    def test_empty_input(self):
        """Empty feature lists should return empty results."""
        features = {'correct': [], 'incorrect': []}
        pile_frequencies = {}

        filtered = apply_pile_filter(features, pile_frequencies, threshold=0.5)

        assert filtered == {'correct': [], 'incorrect': []}

    def test_threshold_boundary(self):
        """Feature exactly at threshold should be filtered (>= comparison)."""
        features = {
            'correct': self._make_features(10, [0]),
            'incorrect': []
        }
        freq_tensor = torch.zeros(100)
        freq_tensor[0] = 0.5  # Exactly at threshold
        pile_frequencies = {10: freq_tensor}

        filtered = apply_pile_filter(features, pile_frequencies, threshold=0.5)

        # Exactly at threshold: >= means filtered out
        assert len(filtered['correct']) == 0
