"""
Tests for Phase 6.3 attention analyzer.

Focuses on missing-data edge cases: when steered attention is unavailable
(e.g., Phase 4.8 multi-candidate SAE mode doesn't save attention), the
analyzer should produce valid plots with baseline-only data instead of
crashing on NaN/Inf axis limits.
"""

import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from phase6_3_attention_analysis.attention_analyzer import AttentionAnalyzer
from common.config import Config


def _make_analyzer(tmp_path, n_heads=8):
    """Create an AttentionAnalyzer without triggering __init__ side effects."""
    analyzer = AttentionAnalyzer.__new__(AttentionAnalyzer)
    analyzer.n_heads = n_heads
    analyzer.visualizations_dir = tmp_path / "visualizations"
    analyzer.visualizations_dir.mkdir(parents=True)
    return analyzer


class TestCalculateAverageDeltasEmptyInput:
    """_calculate_average_deltas must handle empty differences dict."""

    def test_empty_dict_returns_zeros(self, tmp_path):
        analyzer = _make_analyzer(tmp_path)
        result = analyzer._calculate_average_deltas({})
        assert result['means'] == [0.0, 0.0, 0.0]
        assert result['stds'] == [0.0, 0.0, 0.0]

    def test_nonempty_dict_still_works(self, tmp_path):
        """Sanity check: normal input still produces real values."""
        analyzer = _make_analyzer(tmp_path)
        differences = {
            'task_001': {
                'problem': [0.1, 0.2],
                'tests': [-0.1, -0.3],
                'solution_marker': [0.05, 0.15],
            }
        }
        result = analyzer._calculate_average_deltas(differences)
        assert len(result['means']) == 3
        assert all(np.isfinite(v) for v in result['means'])
        assert all(np.isfinite(v) for v in result['stds'])


class TestCreateAttentionDeltaPlotsNaN:
    """create_attention_delta_plots must not crash on all-zero or NaN values."""

    def test_both_empty_differences(self, tmp_path):
        """Both correct and incorrect differences empty → all-zero deltas."""
        analyzer = _make_analyzer(tmp_path)
        # Should not raise
        analyzer.create_attention_delta_plots({}, {})

        plot_path = analyzer.visualizations_dir / 'attention_delta_plots.png'
        assert plot_path.exists()

    def test_one_empty_one_populated(self, tmp_path):
        """One steering type has data, the other doesn't."""
        analyzer = _make_analyzer(tmp_path)
        differences = {
            'task_001': {
                'problem': [0.5, 0.3],
                'tests': [-0.2, 0.1],
                'solution_marker': [0.0, 0.1],
            }
        }
        analyzer.create_attention_delta_plots(differences, {})

        plot_path = analyzer.visualizations_dir / 'attention_delta_plots.png'
        assert plot_path.exists()


class TestCreateHeadAttentionChangeBarsNoSteered:
    """create_head_attention_change_bars must not crash when no task has steered data."""

    def test_baseline_only_no_steered_key(self, tmp_path):
        """Tasks have baseline but no steered_correct → y-limits default to (-1, 1)."""
        analyzer = _make_analyzer(tmp_path, n_heads=4)
        attention_data = {
            'task_001': {
                'baseline': {
                    'raw': {20: np.random.rand(4, 64).tolist()},
                    'boundaries': {'problem_end': 20, 'tests_end': 40},
                },
                # No 'steered_correct' key
            },
            'task_002': {
                'baseline': {
                    'raw': {20: np.random.rand(4, 64).tolist()},
                    'boundaries': {'problem_end': 20, 'tests_end': 40},
                },
            },
        }
        # Should not raise
        analyzer.create_head_attention_change_bars(attention_data, 'correct')

        plot_path = analyzer.visualizations_dir / 'head_attention_changes_correct.png'
        assert plot_path.exists()

    def test_empty_attention_data(self, tmp_path):
        """Completely empty attention_data dict."""
        analyzer = _make_analyzer(tmp_path, n_heads=4)
        analyzer.create_head_attention_change_bars({}, 'correct')

        plot_path = analyzer.visualizations_dir / 'head_attention_changes_correct.png'
        assert plot_path.exists()


class TestPhase63LoadsPvaFeatures:
    """SAE mode should call load_phase4_9_best_latent and set rank/layer from result."""

    def test_sae_mode_calls_phase4_9(self):
        """SAE mode should use load_phase4_9_best_latent and set best_correct_rank/best_incorrect_rank."""
        selection = {
            "correct": {"rank": 2, "layer": 15, "latent_idx": 12809, "refined_coefficient": 62},
            "incorrect": {"rank": 1, "layer": 18, "latent_idx": 4612, "refined_coefficient": 25},
        }

        analyzer = AttentionAnalyzer.__new__(AttentionAnalyzer)
        analyzer.config = Config()
        analyzer.device = "cpu"
        analyzer.use_probe = False
        analyzer.best_correct_rank = 0
        analyzer.best_incorrect_rank = 0

        with patch('common.steering_setup.load_phase4_9_best_latent',
                   return_value=selection) as mock_load:
            AttentionAnalyzer._load_pva_features(analyzer)
            mock_load.assert_called_once_with(analyzer.config)

        assert analyzer.best_correct_layer == 15
        assert analyzer.best_incorrect_layer == 18
        assert analyzer.best_correct_rank == 2
        assert analyzer.best_incorrect_rank == 1

    def test_sae_mode_missing_phase4_9_raises(self):
        """SAE mode should raise FileNotFoundError when Phase 4.9 not found."""
        analyzer = AttentionAnalyzer.__new__(AttentionAnalyzer)
        analyzer.config = Config()
        analyzer.device = "cpu"
        analyzer.use_probe = False
        analyzer.best_correct_rank = 0
        analyzer.best_incorrect_rank = 0

        with patch('common.steering_setup.load_phase4_9_best_latent',
                   side_effect=FileNotFoundError("Phase 4.9 output not found")):
            with pytest.raises(FileNotFoundError, match="Phase 4.9"):
                AttentionAnalyzer._load_pva_features(analyzer)
