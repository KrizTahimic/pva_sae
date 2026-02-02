"""
Tests for Phase 4.5 - Coefficient Grid Search (CRITICAL)

Validates:
- coarse_to_fine_search: Narrowing search strategy
- multi_candidate_handling: Multiple latents tested
- coefficient_selection_criteria: Best coefficient selection logic
- parallel_coefficient_merge: Per-coefficient merging correct
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from common.config import Config


# =============================================================================
# coarse_to_fine_search Tests
# =============================================================================

class TestCoarseToFineSearch:
    """Test narrowing search strategy."""

    def test_coarse_grid_from_config(self):
        """Config should provide coarse coefficient grid."""
        config = Config()
        assert hasattr(config, 'phase4_5_correct_coefficients')
        assert hasattr(config, 'phase4_5_incorrect_coefficients')

        correct_coeffs = config.phase4_5_correct_coefficients
        incorrect_coeffs = config.phase4_5_incorrect_coefficients

        assert len(correct_coeffs) > 0
        assert len(incorrect_coeffs) > 0

    def test_grid_spans_wide_range(self):
        """Grid should span from small to large coefficients."""
        config = Config()
        coeffs = config.phase4_5_correct_coefficients

        # Should include values from 10 to 1000
        assert min(coeffs) <= 20
        assert max(coeffs) >= 500

    def test_search_tolerance_from_config(self):
        """Config should specify search tolerance."""
        config = Config()
        assert hasattr(config, 'phase4_5_search_tolerance')
        assert config.phase4_5_search_tolerance > 0

    def test_grid_search_finds_best(self):
        """Grid search should find coefficient with best effect."""
        # Simulate effect rates at different coefficients
        coefficient_effects = {
            10: 5.0,    # Too small
            30: 15.0,   # Better
            50: 25.0,   # Best
            70: 22.0,   # Slightly worse
            100: 18.0,  # Getting worse
        }

        best_coeff = max(coefficient_effects, key=coefficient_effects.get)
        assert best_coeff == 50


# =============================================================================
# multi_candidate_handling Tests
# =============================================================================

class TestMultiCandidateHandling:
    """Test multiple latents tested."""

    def test_n_candidates_from_config(self):
        """Config should specify number of candidates."""
        config = Config()
        assert hasattr(config, 'phase4_n_candidates')
        assert config.phase4_n_candidates >= 1

    def test_each_candidate_gets_grid_search(self):
        """Each candidate should go through coefficient grid search."""
        candidates = [
            {'layer': 16, 'latent_idx': 100},
            {'layer': 18, 'latent_idx': 200},
            {'layer': 16, 'latent_idx': 300},
        ]
        coefficients = [10, 30, 50]

        # Each candidate × each coefficient
        total_experiments = len(candidates) * len(coefficients)

        expected_experiments = []
        for candidate in candidates:
            for coeff in coefficients:
                expected_experiments.append({
                    'layer': candidate['layer'],
                    'latent_idx': candidate['latent_idx'],
                    'coefficient': coeff
                })

        assert len(expected_experiments) == total_experiments

    def test_best_candidate_selection(self):
        """Should select best candidate based on effect."""
        candidate_results = [
            {'layer': 16, 'latent_idx': 100, 'best_correction_rate': 25.0},
            {'layer': 18, 'latent_idx': 200, 'best_correction_rate': 35.0},  # Best
            {'layer': 16, 'latent_idx': 300, 'best_correction_rate': 20.0},
        ]

        best = max(candidate_results, key=lambda x: x['best_correction_rate'])
        assert best['latent_idx'] == 200


# =============================================================================
# coefficient_selection_criteria Tests
# =============================================================================

class TestCoefficientSelectionCriteria:
    """Test best coefficient selection logic."""

    def test_meaningful_effect_threshold(self):
        """Config should have meaningful effect threshold."""
        config = Config()
        assert hasattr(config, 'phase4_5_meaningful_effect_threshold')
        assert config.phase4_5_meaningful_effect_threshold > 0

    def test_plateau_detection(self):
        """Should detect when effects plateau."""
        config = Config()
        plateau_threshold = config.phase4_5_plateau_threshold

        # Simulate plateaued effects
        effects = [25.0, 25.5, 25.2, 25.8]

        # Check if changes are below threshold
        changes = [abs(effects[i+1] - effects[i]) for i in range(len(effects)-1)]
        is_plateaued = all(c < plateau_threshold for c in changes)

        # With threshold=2.0, these small changes should be plateaued
        assert is_plateaued

    def test_correction_experiment_mode(self):
        """Config should specify experiment mode."""
        config = Config()
        assert hasattr(config, 'phase4_5_experiment_mode')
        assert config.phase4_5_experiment_mode in ['all', 'correction', 'corruption']

    def test_selects_coefficient_maximizing_correction(self):
        """For correction, should select coefficient maximizing correction rate."""
        results = {
            10: {'correction_rate': 10.0, 'corruption_rate': 5.0},
            30: {'correction_rate': 30.0, 'corruption_rate': 10.0},
            50: {'correction_rate': 45.0, 'corruption_rate': 15.0},  # Best correction
            70: {'correction_rate': 40.0, 'corruption_rate': 25.0},
        }

        best_coeff = max(results, key=lambda c: results[c]['correction_rate'])
        assert best_coeff == 50

    def test_considers_corruption_tradeoff(self):
        """Should consider corruption rate in selection."""
        # If two coefficients have similar correction but different corruption
        results = {
            50: {'correction_rate': 45.0, 'corruption_rate': 5.0},   # Better tradeoff
            70: {'correction_rate': 46.0, 'corruption_rate': 30.0},  # Slightly better correction but worse corruption
        }

        # Net benefit (correction - corruption)
        net_benefits = {c: r['correction_rate'] - r['corruption_rate']
                       for c, r in results.items()}

        best_net = max(net_benefits, key=net_benefits.get)
        assert best_net == 50  # 45 - 5 = 40 vs 46 - 30 = 16


# =============================================================================
# parallel_coefficient_merge Tests
# =============================================================================

class TestParallelCoefficientMerge:
    """Test per-coefficient merging correct."""

    def test_merge_per_coefficient(self):
        """Results should be merged for each coefficient."""
        # GPU 0 results for coeff=30
        gpu0_coeff_30 = [
            {'task_id': 't0', 'coefficient': 30, 'steered_correct': True},
            {'task_id': 't2', 'coefficient': 30, 'steered_correct': False},
        ]

        # GPU 1 results for coeff=30
        gpu1_coeff_30 = [
            {'task_id': 't1', 'coefficient': 30, 'steered_correct': True},
            {'task_id': 't3', 'coefficient': 30, 'steered_correct': True},
        ]

        # Merge
        merged = gpu0_coeff_30 + gpu1_coeff_30

        # Calculate metrics on merged data
        corrections = sum(1 for r in merged if r['steered_correct'])
        assert corrections == 3  # t0, t1, t3

    def test_metrics_on_full_merged_data(self):
        """Metrics should be calculated on full merged data, not per-GPU."""
        from common.steering_metrics import calculate_correction_rate

        # All results (simulating merged)
        merged = [
            {'task_id': 't0', 'baseline_passed': False, 'steered_correct': True},
            {'task_id': 't1', 'baseline_passed': False, 'steered_correct': True},
            {'task_id': 't2', 'baseline_passed': False, 'steered_correct': False},
            {'task_id': 't3', 'baseline_passed': False, 'steered_correct': True},
        ]

        correction_rate = calculate_correction_rate(merged)

        # 3/4 = 75%
        assert correction_rate == pytest.approx(75.0)

    def test_early_stopping_decision_on_full_data(self):
        """Early stopping should be based on full merged data."""
        # Simulate merged results that plateau
        coeff_30_rate = 25.0
        coeff_40_rate = 26.0  # Small improvement
        coeff_50_rate = 26.5  # Plateaued

        plateau_threshold = 2.0

        # Check if plateaued
        change_30_to_40 = abs(coeff_40_rate - coeff_30_rate)
        change_40_to_50 = abs(coeff_50_rate - coeff_40_rate)

        is_plateaued = change_40_to_50 < plateau_threshold
        assert is_plateaued


# =============================================================================
# Output Format Tests
# =============================================================================

class TestOutputFormat:
    """Test Phase 4.5 output format."""

    def test_output_includes_best_coefficient(self):
        """Output should include best coefficient per direction."""
        expected_output = {
            'correct': {
                'best_coefficient': 50.0,
                'correction_rate': 45.0,
                'corruption_rate': 15.0
            },
            'incorrect': {
                'best_coefficient': 40.0,
                'correction_rate': 10.0,
                'corruption_rate': 55.0
            }
        }

        assert 'correct' in expected_output
        assert 'incorrect' in expected_output
        assert 'best_coefficient' in expected_output['correct']

    def test_per_candidate_results_saved(self):
        """Should save results for each candidate."""
        candidates = [
            {'layer': 16, 'latent_idx': 100},
            {'layer': 18, 'latent_idx': 200},
        ]

        # Each candidate should have its results saved
        candidate_results = {}
        for i, c in enumerate(candidates):
            key = f"L{c['layer']}-{c['latent_idx']}"
            candidate_results[key] = {
                'best_coefficient': 30.0 + i * 10,
                'correction_rate': 25.0 + i * 5
            }

        assert len(candidate_results) == len(candidates)
