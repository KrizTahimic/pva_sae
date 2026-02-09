"""
Tests for Phase 5.6 - Zero-Discrimination Weight Orthogonalization (Multi-Feature)

Validates:
- Multi-feature iteration: All features from Phase 4.10 are iterated
- Per-feature checkpoint resume: Skip completed features
- Averaged metrics: Aggregation across features
- Direction caching: Pre-computed normalized directions
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from common.config import Config
from phase5_6_zero_disc_orthogonalization.zero_disc_weight_orthogonalizer import (
    ZeroDiscWeightOrthogonalizer,
)


# =============================================================================
# Multi-feature Iteration Tests
# =============================================================================

class TestMultiFeatureIteration:
    """Test that all features from Phase 4.10 are iterated."""

    def test_load_dependencies_loads_all_features(self):
        """_load_dependencies should load ALL zero-disc features, not select single best."""
        import inspect
        source = inspect.getsource(ZeroDiscWeightOrthogonalizer._load_dependencies)

        # Should NOT have single-best selection
        assert 'best_zero_disc' not in source
        assert "min(self.zero_disc_features" not in source

        # Should iterate/cache all features
        assert 'sae_cache' in source
        assert '_direction_cache' in source

    def test_run_iterates_all_features(self):
        """run() should iterate over all zero-disc features."""
        import inspect
        source = inspect.getsource(ZeroDiscWeightOrthogonalizer.run)

        # Should have multi-feature loop
        assert 'per_feature_results' in source
        assert 'apply_zero_disc_orthogonalization_for_feature' in source
        assert '_save_incremental_results' in source

    def test_per_feature_method_takes_feature_param(self):
        """apply_zero_disc_orthogonalization_for_feature should take a feature dict."""
        import inspect
        sig = inspect.signature(ZeroDiscWeightOrthogonalizer.apply_zero_disc_orthogonalization_for_feature)
        params = list(sig.parameters.keys())
        assert 'feature' in params


# =============================================================================
# Per-feature Checkpoint Resume Tests
# =============================================================================

class TestPerFeatureCheckpointResume:
    """Test per-feature checkpoint resume skips completed features."""

    def test_load_partial_results_returns_empty_dict(self):
        """_load_partial_results should return empty per_feature_results when no file exists."""
        ortho = object.__new__(ZeroDiscWeightOrthogonalizer)
        ortho.output_dir = MagicMock()
        ortho.output_dir.__truediv__ = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
        ortho.gpu_id = 0
        ortho.n_gpus = 1

        result = ortho._load_partial_results()
        assert result == {'per_feature_results': {}}

    def test_get_completed_feature_ids(self):
        """_get_completed_feature_ids should return feature IDs from partial results."""
        ortho = object.__new__(ZeroDiscWeightOrthogonalizer)

        partial = {
            'per_feature_results': {
                'L15F1234': {'metrics': {'correction_rate': 5.0}},
                'L15F5678': {'metrics': {'correction_rate': 3.0}},
            }
        }
        completed = ortho._get_completed_feature_ids(partial)
        assert completed == {'L15F1234', 'L15F5678'}

    def test_get_completed_feature_ids_empty(self):
        """_get_completed_feature_ids should return empty set for no results."""
        ortho = object.__new__(ZeroDiscWeightOrthogonalizer)
        completed = ortho._get_completed_feature_ids({})
        assert completed == set()


# =============================================================================
# Averaged Metrics Tests
# =============================================================================

class TestAveragedMetrics:
    """Test aggregation across features."""

    def test_compute_averaged_metrics(self):
        """_compute_averaged_metrics should compute mean and std across features."""
        ortho = object.__new__(ZeroDiscWeightOrthogonalizer)

        per_feature_results = {
            'L15F1': {'metrics': {'correction_rate': 10.0, 'corruption_rate': 5.0, 'preservation_rate': 95.0}},
            'L15F2': {'metrics': {'correction_rate': 20.0, 'corruption_rate': 15.0, 'preservation_rate': 85.0}},
            'L15F3': {'metrics': {'correction_rate': 0.0, 'corruption_rate': 10.0, 'preservation_rate': 90.0}},
        }

        averaged = ortho._compute_averaged_metrics(per_feature_results)

        assert averaged['correction_rate'] == pytest.approx(10.0)
        assert averaged['corruption_rate'] == pytest.approx(10.0)
        assert averaged['preservation_rate'] == pytest.approx(90.0)
        assert averaged['n_features'] == 3
        assert 'std_correction' in averaged
        assert 'std_corruption' in averaged
        assert 'std_preservation' in averaged

    def test_compute_averaged_metrics_empty(self):
        """_compute_averaged_metrics should return empty dict for no features."""
        ortho = object.__new__(ZeroDiscWeightOrthogonalizer)
        assert ortho._compute_averaged_metrics({}) == {}

    def test_compute_averaged_metrics_single_feature(self):
        """Single feature should have zero std."""
        ortho = object.__new__(ZeroDiscWeightOrthogonalizer)

        per_feature_results = {
            'L15F1': {'metrics': {'correction_rate': 10.0, 'corruption_rate': 5.0, 'preservation_rate': 95.0}},
        }

        averaged = ortho._compute_averaged_metrics(per_feature_results)
        assert averaged['std_correction'] == pytest.approx(0.0)


# =============================================================================
# Direction Caching Tests
# =============================================================================

class TestDirectionCaching:
    """Test pre-computed normalized directions."""

    def test_direction_cache_in_load_dependencies(self):
        """_load_dependencies should pre-compute and cache directions."""
        import inspect
        source = inspect.getsource(ZeroDiscWeightOrthogonalizer._load_dependencies)

        assert '_direction_cache' in source
        assert 'normalize_direction' in source

    def test_sae_cache_in_load_dependencies(self):
        """_load_dependencies should cache SAEs by layer."""
        import inspect
        source = inspect.getsource(ZeroDiscWeightOrthogonalizer._load_dependencies)

        assert 'sae_cache' in source
        assert 'load_sae_for_config' in source


# =============================================================================
# Feature-level Method Tests
# =============================================================================

class TestFeatureLevelMethod:
    """Test apply_zero_disc_orthogonalization_for_feature."""

    def test_method_loads_fresh_model(self):
        """Each feature should get a fresh model (orthogonalization is destructive)."""
        import inspect
        source = inspect.getsource(ZeroDiscWeightOrthogonalizer.apply_zero_disc_orthogonalization_for_feature)

        assert 'load_model_and_tokenizer' in source
        assert 'del model' in source

    def test_method_uses_direction_cache(self):
        """Should use _direction_cache for the feature direction."""
        import inspect
        source = inspect.getsource(ZeroDiscWeightOrthogonalizer.apply_zero_disc_orthogonalization_for_feature)

        assert '_direction_cache' in source

    def test_method_cleans_up_checkpoints(self):
        """Should clean up per-feature checkpoints after completion."""
        import inspect
        source = inspect.getsource(ZeroDiscWeightOrthogonalizer.apply_zero_disc_orthogonalization_for_feature)

        assert 'cleanup_all' in source
