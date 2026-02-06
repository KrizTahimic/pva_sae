"""
Tests for Phase 2.5 - SAE Analysis

Validates:
- separation_score_formula: compute_separation_scores produces correct results
- top_n_latent_selection: select_top_k_latents_globally ranks correctly
- pile_filtering: Config provides pile filtering settings
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from common.config import Config
from phase2_5_separation_score_analysis.sae_analyzer import (
    SimplifiedSAEAnalyzer,
)


# =============================================================================
# compute_separation_scores Tests
# =============================================================================

class TestComputeSeparationScores:
    """Test compute_separation_scores using real production code."""

    @pytest.fixture
    def analyzer(self):
        """Create an analyzer with mocked dependencies (no filesystem needed)."""
        analyzer = object.__new__(SimplifiedSAEAnalyzer)
        analyzer.config = Config()
        analyzer.device = torch.device("cpu")
        return analyzer

    def test_positive_separation_for_correct_latent(self, analyzer):
        """Latent that fires more on correct samples should have positive s_correct."""
        n_correct, n_incorrect, n_latents = 20, 20, 5

        # Latent 0 fires on all correct, none on incorrect
        correct_acts = torch.zeros(n_correct, n_latents)
        correct_acts[:, 0] = 1.0  # All correct fire on latent 0
        incorrect_acts = torch.zeros(n_incorrect, n_latents)

        scores = analyzer.compute_separation_scores(correct_acts, incorrect_acts)

        assert scores['s_correct'][0].item() > 0
        assert scores['s_incorrect'][0].item() < 0

    def test_negative_separation_for_incorrect_latent(self, analyzer):
        """Latent that fires more on incorrect samples should have positive s_incorrect."""
        n_correct, n_incorrect, n_latents = 20, 20, 5

        correct_acts = torch.zeros(n_correct, n_latents)
        incorrect_acts = torch.zeros(n_incorrect, n_latents)
        incorrect_acts[:, 2] = 1.0  # All incorrect fire on latent 2

        scores = analyzer.compute_separation_scores(correct_acts, incorrect_acts)

        assert scores['s_incorrect'][2].item() > 0
        assert scores['s_correct'][2].item() < 0

    def test_zero_separation_for_uniform_latent(self, analyzer):
        """Latent that fires equally on both should have zero separation."""
        n_correct, n_incorrect, n_latents = 50, 50, 3

        # All fire equally
        correct_acts = torch.ones(n_correct, n_latents)
        incorrect_acts = torch.ones(n_incorrect, n_latents)

        scores = analyzer.compute_separation_scores(correct_acts, incorrect_acts)

        assert scores['s_correct'][0].item() == pytest.approx(0.0, abs=1e-6)
        assert scores['s_incorrect'][0].item() == pytest.approx(0.0, abs=1e-6)

    def test_returns_all_expected_keys(self, analyzer):
        """compute_separation_scores should return all required keys."""
        correct_acts = torch.randn(10, 4).abs()
        incorrect_acts = torch.randn(10, 4).abs()

        scores = analyzer.compute_separation_scores(correct_acts, incorrect_acts)

        expected_keys = ['f_correct', 'f_incorrect', 's_correct', 's_incorrect',
                         'mean_correct', 'mean_incorrect']
        for key in expected_keys:
            assert key in scores, f"Missing key: {key}"
            assert isinstance(scores[key], torch.Tensor)

    def test_s_correct_and_s_incorrect_are_negatives(self, analyzer):
        """s_correct and s_incorrect should sum to zero for each latent."""
        correct_acts = torch.randn(30, 8).abs()
        incorrect_acts = torch.randn(30, 8).abs()

        scores = analyzer.compute_separation_scores(correct_acts, incorrect_acts)

        sums = scores['s_correct'] + scores['s_incorrect']
        for i in range(sums.shape[0]):
            assert sums[i].item() == pytest.approx(0.0, abs=1e-5)

    def test_frequency_values_in_valid_range(self, analyzer):
        """f_correct and f_incorrect should be in [0, 1]."""
        correct_acts = torch.randn(20, 5)
        incorrect_acts = torch.randn(20, 5)

        scores = analyzer.compute_separation_scores(correct_acts, incorrect_acts)

        for key in ['f_correct', 'f_incorrect']:
            assert (scores[key] >= 0).all()
            assert (scores[key] <= 1).all()


# =============================================================================
# select_top_k_latents_globally Tests
# =============================================================================

class TestSelectTopKLatentsGlobally:
    """Test select_top_k_latents_globally using real production code."""

    @pytest.fixture
    def analyzer(self):
        """Create an analyzer with mocked dependencies."""
        analyzer = object.__new__(SimplifiedSAEAnalyzer)
        analyzer.config = Config()
        analyzer.device = torch.device("cpu")
        return analyzer

    def test_selects_correct_number(self, analyzer):
        """Should select exactly k latents per category."""
        all_results = {
            1: {
                'latents': {
                    'correct': [
                        {'latent_idx': i, 'separation_score': 0.1 * i,
                         'f_correct': 0.5, 'f_incorrect': 0.3, 'mean_activation': 0.1}
                        for i in range(10)
                    ],
                    'incorrect': [
                        {'latent_idx': i, 'separation_score': 0.05 * i,
                         'f_correct': 0.3, 'f_incorrect': 0.5, 'mean_activation': 0.1}
                        for i in range(10)
                    ],
                }
            }
        }

        result = analyzer.select_top_k_latents_globally(all_results, k=3)

        assert len(result['correct']) == 3
        assert len(result['incorrect']) == 3

    def test_ranks_by_separation_score(self, analyzer):
        """Top correct latents should have highest separation scores."""
        all_results = {
            1: {
                'latents': {
                    'correct': [
                        {'latent_idx': 0, 'separation_score': 0.3,
                         'f_correct': 0.5, 'f_incorrect': 0.2, 'mean_activation': 0.1},
                        {'latent_idx': 1, 'separation_score': 0.9,
                         'f_correct': 0.8, 'f_incorrect': 0.1, 'mean_activation': 0.2},
                        {'latent_idx': 2, 'separation_score': 0.1,
                         'f_correct': 0.4, 'f_incorrect': 0.3, 'mean_activation': 0.05},
                    ],
                    'incorrect': [
                        {'latent_idx': 0, 'separation_score': 0.2,
                         'f_correct': 0.2, 'f_incorrect': 0.5, 'mean_activation': 0.1},
                    ],
                }
            }
        }

        result = analyzer.select_top_k_latents_globally(all_results, k=2)

        # Best correct latent should be latent_idx=1 (score 0.9)
        assert result['correct'][0]['latent_idx'] == 1
        assert result['correct'][0]['separation_score'] == 0.9

    def test_includes_layer_in_results(self, analyzer):
        """Each selected latent should include its layer."""
        all_results = {
            16: {
                'latents': {
                    'correct': [
                        {'latent_idx': 100, 'separation_score': 0.5,
                         'f_correct': 0.7, 'f_incorrect': 0.2, 'mean_activation': 0.3},
                    ],
                    'incorrect': [
                        {'latent_idx': 200, 'separation_score': 0.4,
                         'f_correct': 0.2, 'f_incorrect': 0.6, 'mean_activation': 0.2},
                    ],
                }
            }
        }

        result = analyzer.select_top_k_latents_globally(all_results, k=1)

        assert result['correct'][0]['layer'] == 16
        assert result['incorrect'][0]['layer'] == 16

    def test_selects_across_layers(self, analyzer):
        """Should select best latents globally across multiple layers."""
        all_results = {
            10: {
                'latents': {
                    'correct': [
                        {'latent_idx': 0, 'separation_score': 0.3,
                         'f_correct': 0.5, 'f_incorrect': 0.2, 'mean_activation': 0.1},
                    ],
                    'incorrect': [
                        {'latent_idx': 0, 'separation_score': 0.1,
                         'f_correct': 0.3, 'f_incorrect': 0.4, 'mean_activation': 0.1},
                    ],
                }
            },
            20: {
                'latents': {
                    'correct': [
                        {'latent_idx': 0, 'separation_score': 0.8,
                         'f_correct': 0.9, 'f_incorrect': 0.1, 'mean_activation': 0.5},
                    ],
                    'incorrect': [
                        {'latent_idx': 0, 'separation_score': 0.6,
                         'f_correct': 0.1, 'f_incorrect': 0.7, 'mean_activation': 0.4},
                    ],
                }
            },
        }

        result = analyzer.select_top_k_latents_globally(all_results, k=1)

        # Best correct should come from layer 20 (score 0.8 > 0.3)
        assert result['correct'][0]['layer'] == 20
        assert result['incorrect'][0]['layer'] == 20


# =============================================================================
# Pile Filtering Config Tests
# =============================================================================

class TestPileFilteringConfig:
    """Test Config provides pile filtering settings."""

    def test_pile_filter_enabled(self):
        """Config should have pile_filter_enabled."""
        config = Config()
        assert hasattr(config, 'pile_filter_enabled')

    def test_pile_threshold_default(self):
        """Default pile threshold should be 0.02."""
        config = Config()
        assert config.pile_threshold == 0.02

    def test_pile_samples_default(self):
        """Default pile samples should be 10000."""
        config = Config()
        assert config.pile_samples == 10000

    def test_activation_layers(self):
        """Config should provide activation layers list."""
        config = Config()
        assert hasattr(config, 'activation_layers')
        assert len(config.activation_layers) > 0
