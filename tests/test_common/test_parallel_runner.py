"""
Tests for common/parallel_runner.py

Validates:
- filter_dataframe_for_gpu: Round-robin distribution
- Task distribution exhaustiveness and exclusivity
"""

import pytest
import pandas as pd

from common.parallel_runner import filter_dataframe_for_gpu


# =============================================================================
# filter_dataframe_for_gpu Tests
# =============================================================================

class TestFilterDataframeForGpu:
    """Test round-robin GPU task distribution."""

    def test_single_gpu_returns_full_df(self):
        """With n_gpus=1, should return the entire dataframe."""
        df = pd.DataFrame({'task_id': range(10)})
        result = filter_dataframe_for_gpu(df, gpu_id=0, n_gpus=1)
        assert len(result) == 10

    def test_two_gpus_even_split(self):
        """With 10 tasks and 2 GPUs, each GPU gets 5 tasks."""
        df = pd.DataFrame({'task_id': range(10)})
        gpu0 = filter_dataframe_for_gpu(df, gpu_id=0, n_gpus=2)
        gpu1 = filter_dataframe_for_gpu(df, gpu_id=1, n_gpus=2)
        assert len(gpu0) == 5
        assert len(gpu1) == 5

    def test_round_robin_assignment(self):
        """GPU 0 gets indices 0,2,4,... and GPU 1 gets indices 1,3,5,..."""
        df = pd.DataFrame({'task_id': range(8)})
        gpu0 = filter_dataframe_for_gpu(df, gpu_id=0, n_gpus=2)
        gpu1 = filter_dataframe_for_gpu(df, gpu_id=1, n_gpus=2)

        assert list(gpu0['task_id']) == [0, 2, 4, 6]
        assert list(gpu1['task_id']) == [1, 3, 5, 7]

    def test_exhaustive_no_tasks_dropped(self):
        """All tasks must be assigned to exactly one GPU."""
        df = pd.DataFrame({'task_id': range(13)})
        n_gpus = 4

        all_tasks = set()
        for gpu_id in range(n_gpus):
            gpu_df = filter_dataframe_for_gpu(df, gpu_id=gpu_id, n_gpus=n_gpus)
            all_tasks.update(gpu_df['task_id'].tolist())

        assert all_tasks == set(range(13))

    def test_exclusive_no_duplicates(self):
        """No task should be assigned to more than one GPU."""
        df = pd.DataFrame({'task_id': range(13)})
        n_gpus = 4

        all_tasks = []
        for gpu_id in range(n_gpus):
            gpu_df = filter_dataframe_for_gpu(df, gpu_id=gpu_id, n_gpus=n_gpus)
            all_tasks.extend(gpu_df['task_id'].tolist())

        assert len(all_tasks) == len(set(all_tasks))

    def test_uneven_split(self):
        """With 7 tasks and 3 GPUs, distribution should be 3, 2, 2."""
        df = pd.DataFrame({'task_id': range(7)})
        gpu0 = filter_dataframe_for_gpu(df, gpu_id=0, n_gpus=3)
        gpu1 = filter_dataframe_for_gpu(df, gpu_id=1, n_gpus=3)
        gpu2 = filter_dataframe_for_gpu(df, gpu_id=2, n_gpus=3)

        assert len(gpu0) == 3  # indices 0, 3, 6
        assert len(gpu1) == 2  # indices 1, 4
        assert len(gpu2) == 2  # indices 2, 5

    def test_empty_dataframe(self):
        """Empty dataframe should return empty for any GPU."""
        df = pd.DataFrame({'task_id': []})
        result = filter_dataframe_for_gpu(df, gpu_id=0, n_gpus=4)
        assert len(result) == 0

    def test_returns_copy(self):
        """Should return a copy, not a view of the original dataframe."""
        df = pd.DataFrame({'task_id': range(10), 'value': range(10)})
        result = filter_dataframe_for_gpu(df, gpu_id=0, n_gpus=2)

        # Modifying the result should not affect the original
        result.iloc[0, result.columns.get_loc('value')] = 999
        assert df.iloc[0]['value'] == 0
