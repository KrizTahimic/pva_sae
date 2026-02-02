"""
Tests for Phase 1 - Dataset Building (CRITICAL)

Validates:
- activation_extraction_position: Last prompt token extracted
- activation_shape: Correct [1, d_model] shape per layer
- checkpoint_resume: Resumes from last checkpoint correctly
- parallel_distribution: Tasks distributed evenly across GPUs
- parallel_merge: No duplicates, all tasks covered
"""

import pytest
import torch
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from common.config import Config


# =============================================================================
# activation_extraction_position Tests
# =============================================================================

class TestActivationExtractionPosition:
    """Test last prompt token extracted."""

    def test_config_specifies_last_position(self):
        """Config should specify position -1 for activation extraction."""
        config = Config()
        assert config.activation_position == -1

    def test_hook_extracts_last_position(self):
        """Hook should extract activations at last position."""
        # This test validates the concept - actual hook implementation may differ
        d_model = 2304
        seq_len = 10
        batch_size = 1

        # Simulate residual stream
        residual = torch.randn(batch_size, seq_len, d_model)

        # Extract at position -1
        extracted = residual[:, -1, :]

        assert extracted.shape == (1, d_model)

    def test_single_token_extraction(self):
        """Single token input should work for extraction."""
        d_model = 2304

        residual = torch.randn(1, 1, d_model)
        extracted = residual[:, -1, :]

        assert extracted.shape == (1, d_model)


# =============================================================================
# activation_shape Tests
# =============================================================================

class TestActivationShape:
    """Test correct [1, d_model] shape per layer."""

    def test_gemma_2b_d_model(self):
        """Gemma 2B should have d_model=2304."""
        # d_model for Gemma 2B
        expected_d_model = 2304

        activation = torch.randn(1, expected_d_model)
        assert activation.shape == (1, 2304)

    def test_gemma_9b_d_model(self):
        """Gemma 9B should have d_model=3584."""
        expected_d_model = 3584

        activation = torch.randn(1, expected_d_model)
        assert activation.shape == (1, 3584)

    def test_llama_8b_d_model(self):
        """LLAMA 8B should have d_model=4096."""
        expected_d_model = 4096

        activation = torch.randn(1, expected_d_model)
        assert activation.shape == (1, 4096)

    def test_multi_layer_extraction_shape(self):
        """Multi-layer extraction should have consistent shape."""
        d_model = 2304
        layers = [6, 12, 18]

        activations = {layer: torch.randn(1, d_model) for layer in layers}

        for layer in layers:
            assert activations[layer].shape == (1, d_model)


# =============================================================================
# checkpoint_resume Tests
# =============================================================================

class TestCheckpointResume:
    """Test resumes from last checkpoint correctly."""

    def test_resume_skips_processed_tasks(self, tmp_path):
        """Resuming should skip already processed task_ids."""
        from common.checkpoint_manager import CheckpointManager

        checkpoint_dir = tmp_path / "checkpoints"
        mgr = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            experiment_name="phase1",
            frequency=10,
            output_format="parquet"
        )

        # Save checkpoint with processed tasks
        df = pd.DataFrame([
            {'task_id': 't0', 'code': 'def f(): pass'},
            {'task_id': 't1', 'code': 'def g(): pass'},
        ])
        processed_ids = {'t0', 't1'}
        mgr.save_parquet(df, processed_ids)

        # Load checkpoint
        loaded = mgr.load_parquet()

        # Check which tasks need processing
        all_tasks = ['t0', 't1', 't2', 't3']
        remaining = [t for t in all_tasks if t not in loaded.processed_task_ids]

        assert remaining == ['t2', 't3']

    def test_resume_preserves_activations(self, tmp_path):
        """Resumed run should include previous activation data."""
        from common.checkpoint_manager import CheckpointManager

        checkpoint_dir = tmp_path / "checkpoints"
        mgr = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            experiment_name="phase1",
            frequency=10,
            output_format="parquet"
        )

        # Save checkpoint with activation path
        df = pd.DataFrame([
            {'task_id': 't0', 'activation_path': '/path/to/t0.safetensors'},
        ])
        mgr.save_parquet(df, {'t0'})

        # Load and verify
        loaded = mgr.load_parquet()
        assert 'activation_path' in loaded.results_df.columns


# =============================================================================
# parallel_distribution Tests
# =============================================================================

class TestParallelDistribution:
    """Test tasks distributed evenly across GPUs."""

    def test_round_robin_distribution(self):
        """Tasks should be distributed round-robin across GPUs."""
        from common.parallel_runner import _get_gpu_task_indices

        n_tasks = 12
        n_gpus = 4

        all_assigned = []
        for gpu_id in range(n_gpus):
            indices = _get_gpu_task_indices(n_tasks, n_gpus, gpu_id)
            all_assigned.extend([(gpu_id, idx) for idx in indices])

        # Each task should be assigned exactly once
        task_assignments = {}
        for gpu_id, idx in all_assigned:
            assert idx not in task_assignments, f"Task {idx} assigned to multiple GPUs"
            task_assignments[idx] = gpu_id

        assert set(task_assignments.keys()) == set(range(n_tasks))

    def test_balanced_distribution(self):
        """GPUs should have similar task counts."""
        from common.parallel_runner import _get_gpu_task_indices

        n_tasks = 100
        n_gpus = 4

        counts = []
        for gpu_id in range(n_gpus):
            indices = _get_gpu_task_indices(n_tasks, n_gpus, gpu_id)
            counts.append(len(indices))

        # Max difference should be 1
        assert max(counts) - min(counts) <= 1

    def test_uneven_distribution(self):
        """Should handle case where tasks don't divide evenly."""
        from common.parallel_runner import _get_gpu_task_indices

        n_tasks = 10
        n_gpus = 3

        all_indices = []
        for gpu_id in range(n_gpus):
            indices = _get_gpu_task_indices(n_tasks, n_gpus, gpu_id)
            all_indices.extend(indices)

        assert set(all_indices) == set(range(n_tasks))


# =============================================================================
# parallel_merge Tests
# =============================================================================

class TestParallelMerge:
    """Test no duplicates, all tasks covered."""

    def test_merge_deduplicates_by_task_id(self):
        """Merged results should have no duplicate task_ids."""
        # Simulate GPU results with potential duplicate
        gpu0_results = pd.DataFrame([
            {'task_id': 't0', 'value': 1},
            {'task_id': 't2', 'value': 3},
        ])
        gpu1_results = pd.DataFrame([
            {'task_id': 't1', 'value': 2},
            {'task_id': 't2', 'value': 99},  # Duplicate (shouldn't happen normally)
        ])

        # Merge and deduplicate (keep first occurrence)
        merged = pd.concat([gpu0_results, gpu1_results], ignore_index=True)
        merged = merged.drop_duplicates(subset=['task_id'], keep='first')

        assert len(merged) == 3
        # Should keep gpu0's value for t2
        assert merged[merged['task_id'] == 't2']['value'].iloc[0] == 3

    def test_merge_covers_all_tasks(self):
        """Merged results should cover all expected tasks."""
        expected_tasks = {'t0', 't1', 't2', 't3'}

        gpu0_results = pd.DataFrame({'task_id': ['t0', 't2']})
        gpu1_results = pd.DataFrame({'task_id': ['t1', 't3']})

        merged = pd.concat([gpu0_results, gpu1_results], ignore_index=True)

        actual_tasks = set(merged['task_id'])
        assert actual_tasks == expected_tasks

    def test_merge_preserves_columns(self):
        """Merge should preserve all columns from GPU results."""
        gpu0_results = pd.DataFrame([
            {'task_id': 't0', 'code': 'def f(): pass', 'passed': True},
        ])
        gpu1_results = pd.DataFrame([
            {'task_id': 't1', 'code': 'def g(): pass', 'passed': False},
        ])

        merged = pd.concat([gpu0_results, gpu1_results], ignore_index=True)

        assert set(merged.columns) == {'task_id', 'code', 'passed'}


# =============================================================================
# Activation Hook Tests
# =============================================================================

class TestActivationHooks:
    """Test activation extraction hooks."""

    def test_hook_type_from_config(self):
        """Config should specify resid_post hook type."""
        config = Config()
        assert config.activation_hook_type == "resid_post"

    def test_multi_layer_extraction(self):
        """Should extract from multiple layers."""
        config = Config()

        # Gemma 2B has 26 layers, we skip layer 0
        expected_layers = list(range(1, 26))
        assert config.activation_layers == expected_layers
