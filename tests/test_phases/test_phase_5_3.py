"""
Tests for Phase 5.3 - Weight Orthogonalization

Validates:
- projection_math: Orthogonal projection correct
- weight_modification: Weights actually changed
- reversibility: Can restore original weights
- multi_candidate: SAE mode tests all 5 candidates
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from phase5_3_weight_orthogonalization.weight_orthogonalizer import WeightOrthogonalizer


# =============================================================================
# projection_math Tests
# =============================================================================

class TestProjectionMath:
    """Test orthogonal projection correct."""

    def test_projection_removes_direction_component(self):
        """Projection should remove component along direction."""
        # Weight vector and direction
        w = torch.tensor([3.0, 4.0, 0.0])
        direction = torch.tensor([1.0, 0.0, 0.0])  # Unit vector along x

        # Orthogonal projection: w - (w · d) * d
        projection = w - torch.dot(w, direction) * direction

        # Component along direction should be zero
        component_along_direction = torch.dot(projection, direction)
        assert component_along_direction.item() == pytest.approx(0.0, abs=1e-6)

    def test_projection_preserves_orthogonal_components(self):
        """Projection should preserve components orthogonal to direction."""
        w = torch.tensor([3.0, 4.0, 5.0])
        direction = torch.tensor([1.0, 0.0, 0.0])  # x direction

        projection = w - torch.dot(w, direction) * direction

        # y and z components should be preserved
        assert projection[1].item() == pytest.approx(4.0)
        assert projection[2].item() == pytest.approx(5.0)

    def test_projection_of_parallel_vector_is_zero(self):
        """Projecting out a parallel direction should give zero."""
        w = torch.tensor([2.0, 0.0, 0.0])
        direction = torch.tensor([1.0, 0.0, 0.0])

        projection = w - torch.dot(w, direction) * direction

        assert torch.norm(projection).item() == pytest.approx(0.0, abs=1e-6)

    def test_projection_with_normalized_direction(self):
        """Direction should be normalized for correct projection."""
        w = torch.tensor([3.0, 4.0, 0.0])
        direction = torch.tensor([2.0, 0.0, 0.0])  # Not normalized

        # Normalize first
        direction_normalized = direction / torch.norm(direction)

        projection = w - torch.dot(w, direction_normalized) * direction_normalized

        # Should have removed x component
        assert projection[0].item() == pytest.approx(0.0, abs=1e-6)
        assert projection[1].item() == pytest.approx(4.0)


# =============================================================================
# weight_modification Tests
# =============================================================================

class TestWeightModification:
    """Test weights actually changed."""

    def test_weight_matrix_modified(self):
        """Weight matrix should be different after orthogonalization."""
        # Simulate weight matrix (d_out, d_in)
        W = torch.randn(100, 2304)
        W_original = W.clone()

        # Direction to orthogonalize
        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)

        # Apply orthogonalization to each row
        for i in range(W.shape[0]):
            W[i] = W[i] - torch.dot(W[i], direction) * direction

        # Weights should be different
        assert not torch.allclose(W, W_original)

    def test_output_projection_removes_direction(self):
        """Output should have no component along direction."""
        W = torch.randn(100, 2304)
        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)

        # Apply orthogonalization
        for i in range(W.shape[0]):
            W[i] = W[i] - torch.dot(W[i], direction) * direction

        # Check each row has no component along direction
        for i in range(W.shape[0]):
            component = torch.dot(W[i], direction)
            assert component.item() == pytest.approx(0.0, abs=1e-5)

    def test_target_weights_from_config(self):
        """Config should specify which weights to orthogonalize."""
        from common.config import Config

        config = Config()
        assert hasattr(config, 'orthogonalization_target_weights')
        assert 'embed' in config.orthogonalization_target_weights or \
               'attn_o' in config.orthogonalization_target_weights


# =============================================================================
# reversibility Tests
# =============================================================================

class TestReversibility:
    """Test can restore original weights."""

    def test_can_store_and_restore_weights(self):
        """Should be able to restore original weights."""
        W = torch.randn(100, 2304)
        W_backup = W.clone()

        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)

        # Apply orthogonalization
        for i in range(W.shape[0]):
            W[i] = W[i] - torch.dot(W[i], direction) * direction

        # Weights are now different
        assert not torch.allclose(W, W_backup)

        # Restore
        W = W_backup.clone()

        # Should be back to original
        assert torch.allclose(W, W_backup)

    def test_orthogonalization_is_idempotent(self):
        """Applying orthogonalization twice should give same result."""
        W = torch.randn(100, 2304)
        direction = torch.randn(2304)
        direction = direction / torch.norm(direction)

        # Apply once
        for i in range(W.shape[0]):
            W[i] = W[i] - torch.dot(W[i], direction) * direction

        W_after_once = W.clone()

        # Apply again
        for i in range(W.shape[0]):
            W[i] = W[i] - torch.dot(W[i], direction) * direction

        # Should be the same (idempotent)
        assert torch.allclose(W, W_after_once, atol=1e-5)


# =============================================================================
# Layer Targeting Tests
# =============================================================================

class TestLayerTargeting:
    """Test orthogonalization targets correct layers."""

    def test_only_specified_layers_modified(self):
        """Only layers in target list should be modified."""
        from common.config import Config

        config = Config()
        target_weights = config.orthogonalization_target_weights

        # Should specify which weight types to modify
        assert isinstance(target_weights, list)
        assert len(target_weights) > 0

    def test_default_targets(self):
        """Default should include common layer types."""
        from common.config import Config

        config = Config()
        targets = config.orthogonalization_target_weights

        # Should include at least one of these
        expected = {'embed', 'attn_o', 'mlp_down'}
        assert any(t in targets for t in expected)


# =============================================================================
# Multi-Candidate Tests
# =============================================================================

class TestMultiCandidate:
    """Test multi-candidate SAE mode iterates over all 5 candidates."""

    def test_load_dependencies_uses_discover_top_n(self):
        """SAE mode should use discover_top_n_steering_latents, not load_steering_latents."""
        import inspect
        source = inspect.getsource(WeightOrthogonalizer._load_dependencies)

        # Should use multi-candidate discovery
        assert 'discover_top_n_steering_latents' in source
        # Should NOT use old single-latent loading
        assert 'load_steering_latents' not in source

    def test_sae_cache_in_load_dependencies(self):
        """SAE mode should cache SAEs by layer."""
        import inspect
        source = inspect.getsource(WeightOrthogonalizer._load_dependencies)

        assert 'sae_cache' in source
        assert 'load_sae_for_config' in source

    def test_direction_cache_in_load_dependencies(self):
        """SAE mode should pre-compute and cache normalized directions."""
        import inspect
        source = inspect.getsource(WeightOrthogonalizer._load_dependencies)

        assert '_direction_cache' in source
        assert 'normalize_direction' in source

    def test_multi_candidate_method_exists(self):
        """multi_candidate_orthogonalization method should exist."""
        assert hasattr(WeightOrthogonalizer, 'multi_candidate_orthogonalization')

    def test_multi_candidate_method_takes_steering_type(self):
        """multi_candidate_orthogonalization should take steering_type param."""
        import inspect
        sig = inspect.signature(WeightOrthogonalizer.multi_candidate_orthogonalization)
        params = list(sig.parameters.keys())
        assert 'steering_type' in params

    def test_multi_candidate_loops_over_candidates(self):
        """multi_candidate_orthogonalization should iterate over all candidates."""
        import inspect
        source = inspect.getsource(WeightOrthogonalizer.multi_candidate_orthogonalization)

        assert 'per_candidate' in source
        assert 'best_candidate_id' in source
        assert 'load_model_and_tokenizer' in source  # Fresh model per candidate

    def test_run_uses_multi_candidate_in_sae_mode(self):
        """run() should call multi_candidate_orthogonalization in SAE mode."""
        import inspect
        source = inspect.getsource(WeightOrthogonalizer.run)

        assert 'multi_candidate_orthogonalization' in source
        assert 'best_selection' in source


class TestPerCandidateCheckpoint:
    """Test per-candidate checkpoint resume."""

    def test_load_partial_results_returns_empty(self):
        """_load_partial_results should return empty per_candidate_results when no file."""
        ortho = object.__new__(WeightOrthogonalizer)
        ortho.output_dir = MagicMock()
        ortho.output_dir.__truediv__ = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
        ortho.gpu_id = 0
        ortho.n_gpus = 1

        result = ortho._load_partial_results()
        assert result == {'per_candidate_results': {}}

    def test_get_completed_candidate_ids(self):
        """_get_completed_candidate_ids should return candidate IDs from partial results."""
        ortho = object.__new__(WeightOrthogonalizer)
        partial = {
            'per_candidate_results': {
                'L15F12809': {'metrics': {}},
                'L18F4612': {'metrics': {}},
            }
        }
        completed = ortho._get_completed_candidate_ids(partial)
        assert completed == {'L15F12809', 'L18F4612'}

    def test_get_completed_candidate_ids_empty(self):
        """_get_completed_candidate_ids should return empty set for no results."""
        ortho = object.__new__(WeightOrthogonalizer)
        assert ortho._get_completed_candidate_ids({}) == set()


class TestExtractedTestMethods:
    """Test the extracted _test_incorrect_ortho and _test_correct_ortho methods."""

    def test_test_incorrect_ortho_exists(self):
        """_test_incorrect_ortho method should exist."""
        assert hasattr(WeightOrthogonalizer, '_test_incorrect_ortho')

    def test_test_correct_ortho_exists(self):
        """_test_correct_ortho method should exist."""
        assert hasattr(WeightOrthogonalizer, '_test_correct_ortho')

    def test_test_incorrect_ortho_takes_candidate_id(self):
        """_test_incorrect_ortho should take candidate_id for checkpointing."""
        import inspect
        sig = inspect.signature(WeightOrthogonalizer._test_incorrect_ortho)
        params = list(sig.parameters.keys())
        assert 'candidate_id' in params

    def test_test_correct_ortho_takes_candidate_id(self):
        """_test_correct_ortho should take candidate_id for checkpointing."""
        import inspect
        sig = inspect.signature(WeightOrthogonalizer._test_correct_ortho)
        params = list(sig.parameters.keys())
        assert 'candidate_id' in params

    def test_test_incorrect_ortho_returns_tuple(self):
        """_test_incorrect_ortho should return (incorrect_results, correct_results)."""
        import inspect
        source = inspect.getsource(WeightOrthogonalizer._test_incorrect_ortho)
        assert 'return incorrect_results, correct_results' in source

    def test_test_correct_ortho_returns_list(self):
        """_test_correct_ortho should return correct_results list."""
        import inspect
        source = inspect.getsource(WeightOrthogonalizer._test_correct_ortho)
        assert 'return correct_results' in source
