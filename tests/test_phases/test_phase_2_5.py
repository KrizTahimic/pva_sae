"""
Tests for Phase 2.5 - SAE Analysis

Validates:
- separation_score_formula: mean_correct - mean_incorrect calculation
- pile_filtering: Baseline comparison logic
- top_n_latent_selection: Correct ranking by separation score
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


# =============================================================================
# separation_score_formula Tests
# =============================================================================

class TestSeparationScoreFormula:
    """Test mean_correct - mean_incorrect calculation."""

    def test_positive_separation_score(self):
        """Correct-predicting latent should have positive separation score."""
        # Simulate activations: correct samples have higher mean
        correct_activations = np.array([1.0, 1.2, 0.9, 1.1, 1.0])
        incorrect_activations = np.array([0.2, 0.3, 0.1, 0.25, 0.15])

        mean_correct = np.mean(correct_activations)
        mean_incorrect = np.mean(incorrect_activations)
        separation_score = mean_correct - mean_incorrect

        assert separation_score > 0
        assert separation_score == pytest.approx(0.84, rel=0.01)

    def test_negative_separation_score(self):
        """Incorrect-predicting latent should have negative separation score."""
        # Simulate activations: incorrect samples have higher mean
        correct_activations = np.array([0.2, 0.3, 0.1, 0.25, 0.15])
        incorrect_activations = np.array([1.0, 1.2, 0.9, 1.1, 1.0])

        mean_correct = np.mean(correct_activations)
        mean_incorrect = np.mean(incorrect_activations)
        separation_score = mean_correct - mean_incorrect

        assert separation_score < 0
        assert separation_score == pytest.approx(-0.84, rel=0.01)

    def test_zero_separation_score(self):
        """No discrimination should give zero separation score."""
        activations = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        mean_correct = np.mean(activations)
        mean_incorrect = np.mean(activations)
        separation_score = mean_correct - mean_incorrect

        assert separation_score == 0.0

    def test_separation_score_with_variance(self):
        """Should work with high variance activations."""
        correct_activations = np.array([5.0, 0.1, 3.0, 2.5, 4.5])  # High variance
        incorrect_activations = np.array([0.5, 0.1, 0.3, 0.2, 0.4])

        mean_correct = np.mean(correct_activations)
        mean_incorrect = np.mean(incorrect_activations)
        separation_score = mean_correct - mean_incorrect

        assert separation_score > 0


# =============================================================================
# pile_filtering Tests
# =============================================================================

class TestPileFiltering:
    """Test baseline comparison logic."""

    def test_pile_filter_from_config(self):
        """Config should have pile filtering settings."""
        from common.config import Config

        config = Config()
        assert hasattr(config, 'pile_filter_enabled')
        assert hasattr(config, 'pile_threshold')
        assert hasattr(config, 'pile_samples')

    def test_pile_threshold_default(self):
        """Default pile threshold should be 0.02."""
        from common.config import Config

        config = Config()
        assert config.pile_threshold == 0.02

    def test_pile_samples_default(self):
        """Default pile samples should be 10000."""
        from common.config import Config

        config = Config()
        assert config.pile_samples == 10000

    def test_latent_passes_pile_filter(self):
        """Latent with low pile activation should pass filter."""
        # Simulate: code activates more than pile baseline
        code_mean_activation = 0.5
        pile_mean_activation = 0.01
        threshold = 0.02

        passes_filter = code_mean_activation > pile_mean_activation + threshold
        assert passes_filter is True

    def test_latent_fails_pile_filter(self):
        """Latent with high pile activation should fail filter."""
        # Simulate: code doesn't activate much more than pile
        code_mean_activation = 0.03
        pile_mean_activation = 0.02
        threshold = 0.02

        passes_filter = code_mean_activation > pile_mean_activation + threshold
        assert passes_filter is False


# =============================================================================
# top_n_latent_selection Tests
# =============================================================================

class TestTopNLatentSelection:
    """Test correct ranking by separation score."""

    def test_top_n_by_positive_separation(self):
        """Correct-predicting latents should be ranked by positive separation."""
        latents = [
            {'layer': 16, 'latent_idx': 100, 'separation_score': 0.3},
            {'layer': 18, 'latent_idx': 200, 'separation_score': 0.5},  # Best
            {'layer': 16, 'latent_idx': 300, 'separation_score': 0.1},
            {'layer': 20, 'latent_idx': 400, 'separation_score': 0.4},
        ]

        # Sort by separation score descending
        sorted_latents = sorted(latents, key=lambda x: x['separation_score'], reverse=True)
        top_3 = sorted_latents[:3]

        assert top_3[0]['latent_idx'] == 200
        assert top_3[0]['separation_score'] == 0.5

    def test_top_n_by_negative_separation(self):
        """Incorrect-predicting latents should be ranked by most negative separation."""
        latents = [
            {'layer': 16, 'latent_idx': 100, 'separation_score': -0.3},
            {'layer': 18, 'latent_idx': 200, 'separation_score': -0.5},  # Best (most negative)
            {'layer': 16, 'latent_idx': 300, 'separation_score': -0.1},
            {'layer': 20, 'latent_idx': 400, 'separation_score': -0.4},
        ]

        # Sort by separation score ascending (most negative first)
        sorted_latents = sorted(latents, key=lambda x: x['separation_score'])
        top_3 = sorted_latents[:3]

        assert top_3[0]['latent_idx'] == 200
        assert top_3[0]['separation_score'] == -0.5

    def test_n_candidates_from_config(self):
        """Config should specify number of candidates."""
        from common.config import Config

        config = Config()
        # Phase 4 uses phase4_n_candidates for steering
        assert hasattr(config, 'phase4_n_candidates')
        assert config.phase4_n_candidates == 5

    def test_top_20_output_format(self):
        """top_20_latents.json should have correct structure."""
        expected_structure = {
            'correct': [
                {'layer': 16, 'latent_idx': 100, 'separation_score': 0.5}
            ],
            'incorrect': [
                {'layer': 18, 'latent_idx': 200, 'separation_score': -0.5}
            ]
        }

        assert 'correct' in expected_structure
        assert 'incorrect' in expected_structure
        assert len(expected_structure['correct']) > 0
        assert 'layer' in expected_structure['correct'][0]
        assert 'latent_idx' in expected_structure['correct'][0]
        assert 'separation_score' in expected_structure['correct'][0]


# =============================================================================
# Layer Selection Tests
# =============================================================================

class TestLayerSelection:
    """Test unique layers extracted from candidates."""

    def test_unique_layers_from_candidates(self):
        """Should extract unique layers from all candidates."""
        correct = [
            {'layer': 16, 'latent_idx': 100},
            {'layer': 18, 'latent_idx': 200},
            {'layer': 16, 'latent_idx': 300},  # Duplicate layer
        ]
        incorrect = [
            {'layer': 14, 'latent_idx': 400},
            {'layer': 18, 'latent_idx': 500},  # Duplicate layer
        ]

        all_layers = sorted(set(
            c['layer'] for c in correct + incorrect
        ))

        assert all_layers == [14, 16, 18]

    def test_config_provides_activation_layers(self):
        """Config should provide activation layers list."""
        from common.config import Config

        config = Config()
        assert hasattr(config, 'activation_layers')
        assert len(config.activation_layers) > 0
