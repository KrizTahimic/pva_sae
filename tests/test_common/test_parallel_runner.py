"""
Tests for common/parallel_runner.py

Validates:
- filter_dataframe_for_gpu: Round-robin distribution
- Task distribution exhaustiveness and exclusivity
- H1 regression: Phase 8.3 merge uses 'was_steered' key, not 'steered'
- R3: Helper functions _load_gpu_json_files and _cleanup_gpu_files
"""

import pytest
import json
import pandas as pd
from pathlib import Path

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


# =============================================================================
# H1 Regression: Phase 8.3 merge uses 'was_steered' key
# =============================================================================

class TestPhase83MergeUsesWasSteered:
    """Regression test: Phase 8.3 records use 'was_steered' key, not 'steered'.

    The merge function in parallel_runner.py was reading r.get('steered', False)
    but Phase 8.2/8.3 records use the key 'was_steered'. This caused steering
    trigger rates to always be reported as 0%.
    """

    def test_merge_code_references_was_steered(self):
        """Source code should reference 'was_steered', not 'steered' for Phase 8.3 merge."""
        import inspect
        import common.parallel_runner as mod
        source = inspect.getsource(mod)

        # The old bug: using r.get('steered', False) in the merge function
        # After fix: r.get('was_steered', False)
        # Check that 'was_steered' is used in the merge context
        assert "r.get('was_steered'" in source
        # The old pattern should NOT exist (except possibly in other contexts)
        assert "r.get('steered', False)" not in source


# =============================================================================
# R3: Helper function tests
# =============================================================================

class TestLoadGpuJsonFiles:
    """Test _load_gpu_json_files helper."""

    def test_loads_matching_files(self, tmp_path):
        """Should load all files matching the glob pattern."""
        from common.parallel_runner import _load_gpu_json_files

        for i in range(3):
            (tmp_path / f"results_gpu{i}.json").write_text(
                json.dumps({"gpu_id": i, "data": [i]})
            )

        results = _load_gpu_json_files(tmp_path, "results_gpu*.json")
        assert len(results) == 3
        assert results[0]["gpu_id"] == 0
        assert results[2]["gpu_id"] == 2

    def test_raises_on_no_files(self, tmp_path):
        """Should raise RuntimeError when no files match."""
        from common.parallel_runner import _load_gpu_json_files

        with pytest.raises(RuntimeError, match="No results_gpu"):
            _load_gpu_json_files(tmp_path, "results_gpu*.json")


class TestCleanupGpuFiles:
    """Test _cleanup_gpu_files helper."""

    def test_removes_matching_files(self, tmp_path):
        """Should remove all files matching the patterns."""
        from common.parallel_runner import _cleanup_gpu_files

        for i in range(3):
            (tmp_path / f"results_gpu{i}.json").write_text("{}")
            (tmp_path / f"summary_gpu{i}.json").write_text("{}")
        (tmp_path / "final_results.json").write_text("{}")

        _cleanup_gpu_files(tmp_path, ["results_gpu*.json", "summary_gpu*.json"])

        remaining = list(tmp_path.glob("*.json"))
        assert len(remaining) == 1
        assert remaining[0].name == "final_results.json"

    def test_no_error_on_missing_pattern(self, tmp_path):
        """Should not raise if no files match a pattern."""
        from common.parallel_runner import _cleanup_gpu_files
        _cleanup_gpu_files(tmp_path, ["nonexistent_*.json"])
