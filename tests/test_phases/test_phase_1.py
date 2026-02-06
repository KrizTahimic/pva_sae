"""
Tests for Phase 1 - Dataset Building (CRITICAL)

Validates:
- activation_extraction_position: Config specifies last prompt token
- prompt_building: PromptBuilder produces valid prompts
- code_extraction: extract_code handles edge cases
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
from common.prompt_utils import PromptBuilder
from common.dataset_utils import extract_code
from common.activation_hooks import ActivationExtractor


# =============================================================================
# activation_extraction_position Tests
# =============================================================================

class TestActivationExtractionPosition:
    """Test last prompt token extracted - validates config and hook setup."""

    def test_config_specifies_last_position(self):
        """Config should specify position -1 for activation extraction."""
        config = Config()
        assert config.activation_position == -1

    def test_hook_type_from_config(self):
        """Config should specify resid_post hook type."""
        config = Config()
        assert config.activation_hook_type == "resid_post"

    def test_activation_extractor_stores_position(self):
        """ActivationExtractor should store the position from config."""
        mock_model = MagicMock()
        config = Config()
        extractor = ActivationExtractor(
            model=mock_model,
            layers=[1, 2],
            position=config.activation_position
        )
        assert extractor.position == -1

    def test_activation_layers_from_config(self):
        """Config should provide activation layers list (skipping layer 0)."""
        config = Config()
        assert hasattr(config, 'activation_layers')
        assert len(config.activation_layers) > 0
        # Gemma 2B has 26 layers, we skip layer 0
        expected_layers = list(range(1, 26))
        assert config.activation_layers == expected_layers


# =============================================================================
# prompt_building Tests
# =============================================================================

class TestPromptBuilding:
    """Test PromptBuilder produces valid prompts using real production code."""

    def test_prompt_includes_problem_description(self):
        """Prompt should include the problem description."""
        prompt = PromptBuilder.build_prompt(
            problem_description="Write a function to add two numbers.",
            test_cases="assert add(1, 2) == 3"
        )
        assert "Write a function to add two numbers." in prompt

    def test_prompt_includes_test_cases(self):
        """Prompt should include test cases."""
        prompt = PromptBuilder.build_prompt(
            problem_description="Write a function.",
            test_cases="assert func(1) == 1\nassert func(2) == 4"
        )
        assert "assert func(1) == 1" in prompt
        assert "assert func(2) == 4" in prompt

    def test_prompt_includes_code_initiator(self):
        """Prompt should include the code initiator."""
        prompt = PromptBuilder.build_prompt(
            problem_description="Write a function.",
            test_cases="assert func(1) == 1"
        )
        # Default initiator is "# Solution:"
        assert "# Solution:" in prompt

    def test_prompt_custom_code_initiator(self):
        """Custom code initiator should override default."""
        prompt = PromptBuilder.build_prompt(
            problem_description="Write a function.",
            test_cases="assert func(1) == 1",
            code_initiator="# Write your function definition here:"
        )
        assert "# Write your function definition here:" in prompt

    def test_prompt_template_ordering(self):
        """Problem description should come before test cases."""
        prompt = PromptBuilder.build_prompt(
            problem_description="PROBLEM_MARKER",
            test_cases="TEST_MARKER"
        )
        assert prompt.index("PROBLEM_MARKER") < prompt.index("TEST_MARKER")


# =============================================================================
# code_extraction Tests
# =============================================================================

class TestCodeExtraction:
    """Test extract_code handles edge cases using real production code."""

    def test_extracts_code_after_prompt(self):
        """Should extract code after the prompt."""
        prompt = "Write a function.\nassert func(1) == 1\n# Solution:"
        generated = prompt + "\ndef func(x):\n    return x"
        code = extract_code(generated, prompt)
        assert "def func(x)" in code

    def test_empty_generation(self):
        """Should handle empty generation gracefully."""
        prompt = "Write a function."
        code = extract_code(prompt, prompt)
        # Should return something (possibly empty string) without crashing
        assert isinstance(code, str)


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
        gpu0_results = pd.DataFrame([
            {'task_id': 't0', 'value': 1},
            {'task_id': 't2', 'value': 3},
        ])
        gpu1_results = pd.DataFrame([
            {'task_id': 't1', 'value': 2},
            {'task_id': 't2', 'value': 99},
        ])

        merged = pd.concat([gpu0_results, gpu1_results], ignore_index=True)
        merged = merged.drop_duplicates(subset=['task_id'], keep='first')

        assert len(merged) == 3
        assert merged[merged['task_id'] == 't2']['value'].iloc[0] == 3

    def test_merge_covers_all_tasks(self):
        """Merged results should cover all expected tasks."""
        expected_tasks = {'t0', 't1', 't2', 't3'}

        gpu0_results = pd.DataFrame({'task_id': ['t0', 't2']})
        gpu1_results = pd.DataFrame({'task_id': ['t1', 't3']})

        merged = pd.concat([gpu0_results, gpu1_results], ignore_index=True)
        assert set(merged['task_id']) == expected_tasks
