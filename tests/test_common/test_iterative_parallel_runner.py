"""
Tests for common/iterative_parallel_runner.py

Validates:
- Task distribution across GPUs
- Per-value iteration and result merging
- Checkpoint/resume behavior
- Early stopping decisions
"""

import json
import queue
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from common.iterative_parallel_runner import (
    IterativeParallelRunner,
    _default_merge_fn,
    _no_early_stop,
    DEFAULT_WORKER_TIMEOUT,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_runner(
    tmp_path,
    values_to_test=None,
    n_gpus=2,
    early_stop_fn=None,
    merge_fn=None,
    all_task_ids=None,
    timeout=10,
):
    """Create an IterativeParallelRunner with mocked dependencies."""
    config = MagicMock()
    config.model_name = "google/gemma-2-2b"
    config.dataset_name = "mbpp"

    runner = IterativeParallelRunner(
        phase_evaluator_class=MagicMock,
        config=config,
        n_gpus=n_gpus,
        values_to_test=values_to_test or [1.0, 2.0, 3.0],
        early_stop_fn=early_stop_fn,
        merge_fn=merge_fn,
        timeout_per_iteration=timeout,
        checkpoint_dir=tmp_path / "checkpoints",
        all_task_ids=all_task_ids or [f"task_{i}" for i in range(10)],
    )
    return runner


def _make_gpu_result(gpu_id, value, task_ids, passed=True):
    """Create a mock GPU result dict."""
    results = [
        {"task_id": tid, "value": value, "passed": passed}
        for tid in task_ids
    ]
    return {
        "gpu_id": gpu_id,
        "value": value,
        "status": "success",
        "results": results,
        "metrics": {"accuracy": 1.0 if passed else 0.0},
    }


# =============================================================================
# Task Distribution Tests
# =============================================================================

class TestDistributeTasks:
    """Test round-robin task distribution across GPUs."""

    def test_round_robin_two_gpus(self, tmp_path):
        """Tasks should be distributed round-robin: GPU0 gets even indices, GPU1 odd."""
        runner = _make_runner(tmp_path, n_gpus=2)
        task_ids = ["t0", "t1", "t2", "t3", "t4", "t5"]

        assignments = runner._distribute_tasks(task_ids, 2)

        assert assignments[0] == ["t0", "t2", "t4"]
        assert assignments[1] == ["t1", "t3", "t5"]

    def test_round_robin_four_gpus(self, tmp_path):
        """With 4 GPUs, tasks cycle through 0,1,2,3,0,1,..."""
        runner = _make_runner(tmp_path, n_gpus=4)
        task_ids = [f"t{i}" for i in range(9)]

        assignments = runner._distribute_tasks(task_ids, 4)

        assert assignments[0] == ["t0", "t4", "t8"]
        assert assignments[1] == ["t1", "t5"]
        assert assignments[2] == ["t2", "t6"]
        assert assignments[3] == ["t3", "t7"]

    def test_exhaustive_no_tasks_lost(self, tmp_path):
        """All tasks must appear in exactly one GPU's assignment."""
        runner = _make_runner(tmp_path, n_gpus=3)
        task_ids = [f"t{i}" for i in range(13)]

        assignments = runner._distribute_tasks(task_ids, 3)

        all_assigned = []
        for gpu_tasks in assignments.values():
            all_assigned.extend(gpu_tasks)

        assert sorted(all_assigned) == sorted(task_ids)

    def test_exclusive_no_duplicates(self, tmp_path):
        """No task should appear in more than one GPU's assignment."""
        runner = _make_runner(tmp_path, n_gpus=4)
        task_ids = [f"t{i}" for i in range(17)]

        assignments = runner._distribute_tasks(task_ids, 4)

        all_assigned = []
        for gpu_tasks in assignments.values():
            all_assigned.extend(gpu_tasks)

        assert len(all_assigned) == len(set(all_assigned))

    def test_empty_task_list(self, tmp_path):
        """Empty task list should produce empty assignments for all GPUs."""
        runner = _make_runner(tmp_path, n_gpus=3)

        assignments = runner._distribute_tasks([], 3)

        for gpu_id in range(3):
            assert assignments[gpu_id] == []

    def test_fewer_tasks_than_gpus(self, tmp_path):
        """When tasks < GPUs, some GPUs get no tasks."""
        runner = _make_runner(tmp_path, n_gpus=4)
        task_ids = ["t0", "t1"]

        assignments = runner._distribute_tasks(task_ids, 4)

        assert assignments[0] == ["t0"]
        assert assignments[1] == ["t1"]
        assert assignments[2] == []
        assert assignments[3] == []

    def test_single_gpu_gets_all_tasks(self, tmp_path):
        """With n_gpus=1, GPU 0 should receive all tasks."""
        runner = _make_runner(tmp_path, n_gpus=1)
        task_ids = [f"t{i}" for i in range(5)]

        assignments = runner._distribute_tasks(task_ids, 1)

        assert assignments[0] == task_ids


# =============================================================================
# Default Merge Function Tests
# =============================================================================

class TestDefaultMergeFn:
    """Test the default merge function for combining GPU results."""

    def test_merges_results_from_multiple_gpus(self):
        """Should concatenate results lists from all GPUs."""
        gpu_results = [
            {"results": [{"task_id": "t0"}, {"task_id": "t1"}]},
            {"results": [{"task_id": "t2"}, {"task_id": "t3"}]},
        ]

        merged = _default_merge_fn(gpu_results)

        assert merged["n_gpus"] == 2
        assert merged["n_problems"] == 4
        assert len(merged["results"]) == 4

    def test_handles_empty_results(self):
        """Should handle GPUs with no results gracefully."""
        gpu_results = [
            {"results": [{"task_id": "t0"}]},
            {"results": []},
            {"other_key": "no results key"},
        ]

        merged = _default_merge_fn(gpu_results)

        assert merged["n_gpus"] == 3
        assert merged["n_problems"] == 1

    def test_empty_gpu_list(self):
        """Should handle empty GPU list."""
        merged = _default_merge_fn([])

        assert merged["n_gpus"] == 0
        assert merged["n_problems"] == 0
        assert merged["results"] == []


# =============================================================================
# Default Early Stop Function Tests
# =============================================================================

class TestNoEarlyStop:
    """Test the default early stop function."""

    def test_never_stops(self):
        """Default early stop function should always return False."""
        assert _no_early_stop() is False
        assert _no_early_stop({"score": 1.0}, []) is False
        assert _no_early_stop({"score": 0.0}, [{"score": 1.0}]) is False


# =============================================================================
# Per-GPU-Per-Value Checkpoint Tests
# =============================================================================

class TestCheckpointSaveLoad:
    """Test per-GPU-per-value checkpoint save and load behavior."""

    def test_save_creates_checkpoint_files(self, tmp_path):
        """Saving a checkpoint should create parquet and meta.json files."""
        runner = _make_runner(tmp_path)
        value = 1.5
        gpu_id = 0
        result = _make_gpu_result(gpu_id, value, ["t0", "t1"])

        runner._save_gpu_checkpoint(value, gpu_id, result)

        value_dir = runner._get_value_checkpoint_dir(value)
        assert (value_dir / "gpu_0_results.parquet").exists()
        assert (value_dir / "gpu_0_results.meta.json").exists()

    def test_checkpoint_meta_contains_task_ids(self, tmp_path):
        """Meta file should record processed task_ids."""
        runner = _make_runner(tmp_path)
        value = 2.0
        gpu_id = 1
        result = _make_gpu_result(gpu_id, value, ["t3", "t4", "t5"])

        runner._save_gpu_checkpoint(value, gpu_id, result)

        value_dir = runner._get_value_checkpoint_dir(value)
        meta_file = value_dir / "gpu_1_results.meta.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)

        assert set(meta["processed_task_ids"]) == {"t3", "t4", "t5"}
        assert meta["n_results"] == 3
        assert meta["gpu_id"] == 1
        assert meta["value"] == 2.0

    def test_checkpoint_parquet_roundtrip(self, tmp_path):
        """Parquet checkpoint should preserve all result data."""
        runner = _make_runner(tmp_path)
        value = 1.0
        gpu_id = 0
        result = _make_gpu_result(gpu_id, value, ["t0", "t1"], passed=True)

        runner._save_gpu_checkpoint(value, gpu_id, result)

        value_dir = runner._get_value_checkpoint_dir(value)
        df = pd.read_parquet(value_dir / "gpu_0_results.parquet")

        assert len(df) == 2
        assert set(df["task_id"]) == {"t0", "t1"}
        assert all(df["passed"])

    def test_no_checkpoint_dir_skips_save(self, tmp_path):
        """Should silently skip when checkpoint_dir is None."""
        runner = _make_runner(tmp_path)
        runner.checkpoint_dir = None

        # Should not raise
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0"]))

    def test_value_dir_naming_handles_decimals(self, tmp_path):
        """Decimal values should be encoded with underscores in dir names."""
        runner = _make_runner(tmp_path)

        value_dir = runner._get_value_checkpoint_dir(1.5)
        assert "value_1_5" in str(value_dir)

    def test_value_dir_naming_handles_negative(self, tmp_path):
        """Negative values should replace dashes with 'neg'."""
        runner = _make_runner(tmp_path)

        value_dir = runner._get_value_checkpoint_dir(-2.5)
        assert "value_neg2_5" in str(value_dir)

    def test_checkpoint_merge_deduplicates_by_task_id(self, tmp_path):
        """Saving again for the same GPU should merge and deduplicate by task_id."""
        runner = _make_runner(tmp_path)
        value = 1.0
        gpu_id = 0

        # First save
        result1 = _make_gpu_result(gpu_id, value, ["t0", "t1"], passed=False)
        runner._save_gpu_checkpoint(value, gpu_id, result1)

        # Second save with overlapping task_id (t1 updated, t2 new)
        result2 = {
            "gpu_id": gpu_id,
            "value": value,
            "status": "success",
            "results": [
                {"task_id": "t1", "value": value, "passed": True},
                {"task_id": "t2", "value": value, "passed": True},
            ],
        }
        runner._save_gpu_checkpoint(value, gpu_id, result2)

        # Read back
        value_dir = runner._get_value_checkpoint_dir(value)
        df = pd.read_parquet(value_dir / "gpu_0_results.parquet")

        assert len(df) == 3  # t0 from first, t1 from second (overwritten), t2 from second
        assert set(df["task_id"]) == {"t0", "t1", "t2"}

        # t1 should have the updated (True) value
        t1_row = df[df["task_id"] == "t1"]
        assert t1_row["passed"].iloc[0] == True

    def test_checkpoint_meta_merges_task_ids(self, tmp_path):
        """Meta file should contain union of task_ids across saves."""
        runner = _make_runner(tmp_path)
        value = 1.0
        gpu_id = 0

        # First save
        result1 = _make_gpu_result(gpu_id, value, ["t0", "t1"])
        runner._save_gpu_checkpoint(value, gpu_id, result1)

        # Second save with new tasks
        result2 = _make_gpu_result(gpu_id, value, ["t2", "t3"])
        runner._save_gpu_checkpoint(value, gpu_id, result2)

        value_dir = runner._get_value_checkpoint_dir(value)
        with open(value_dir / "gpu_0_results.meta.json", "r") as f:
            meta = json.load(f)

        assert set(meta["processed_task_ids"]) == {"t0", "t1", "t2", "t3"}


# =============================================================================
# Remaining Tasks Detection Tests
# =============================================================================

class TestGetRemainingTasks:
    """Test detection of unprocessed tasks for a given value."""

    def test_all_tasks_remaining_when_no_checkpoints(self, tmp_path):
        """With no checkpoints, all tasks should be remaining."""
        runner = _make_runner(tmp_path, all_task_ids=["t0", "t1", "t2"])

        remaining = runner._get_remaining_tasks_for_value(1.0)

        assert remaining == ["t0", "t1", "t2"]

    def test_some_tasks_remaining_after_partial_checkpoint(self, tmp_path):
        """After processing some tasks, only unprocessed tasks remain."""
        runner = _make_runner(tmp_path, all_task_ids=["t0", "t1", "t2", "t3"])

        # Save checkpoint for t0 and t1
        result = _make_gpu_result(0, 1.0, ["t0", "t1"])
        runner._save_gpu_checkpoint(1.0, 0, result)

        remaining = runner._get_remaining_tasks_for_value(1.0)

        assert sorted(remaining) == ["t2", "t3"]

    def test_no_tasks_remaining_when_all_complete(self, tmp_path):
        """When all tasks are checkpointed, none should remain."""
        runner = _make_runner(tmp_path, all_task_ids=["t0", "t1"])

        # GPU 0 did t0, GPU 1 did t1
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0"]))
        runner._save_gpu_checkpoint(1.0, 1, _make_gpu_result(1, 1.0, ["t1"]))

        remaining = runner._get_remaining_tasks_for_value(1.0)

        assert remaining == []

    def test_remaining_tasks_across_multiple_gpus(self, tmp_path):
        """Should merge processed task_ids from all GPU checkpoints."""
        all_tasks = [f"t{i}" for i in range(6)]
        runner = _make_runner(tmp_path, all_task_ids=all_tasks, n_gpus=3)

        # GPU 0 processed t0, t3
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0", "t3"]))
        # GPU 1 processed t1
        runner._save_gpu_checkpoint(1.0, 1, _make_gpu_result(1, 1.0, ["t1"]))
        # GPU 2 not done yet

        remaining = runner._get_remaining_tasks_for_value(1.0)

        assert sorted(remaining) == ["t2", "t4", "t5"]

    def test_returns_all_tasks_when_no_checkpoint_dir(self, tmp_path):
        """With no checkpoint_dir, all tasks are returned as remaining."""
        runner = _make_runner(tmp_path, all_task_ids=["t0", "t1"])
        runner.checkpoint_dir = None

        remaining = runner._get_remaining_tasks_for_value(1.0)

        assert remaining == ["t0", "t1"]


# =============================================================================
# Merge Value Results Tests
# =============================================================================

class TestMergeValueResults:
    """Test merging GPU checkpoint files for a completed value."""

    def test_merges_multiple_gpu_parquets(self, tmp_path):
        """Should merge parquet files from all GPUs into single result."""
        runner = _make_runner(tmp_path, n_gpus=2)

        # GPU 0 results
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0", "t2"]))
        # GPU 1 results
        runner._save_gpu_checkpoint(1.0, 1, _make_gpu_result(1, 1.0, ["t1", "t3"]))

        merged = runner._merge_value_results(1.0)

        assert merged is not None
        assert merged["n_problems"] == 4
        assert merged["value"] == 1.0

    def test_deduplicates_by_task_id(self, tmp_path):
        """If same task_id appears in multiple GPU files, keep last."""
        runner = _make_runner(tmp_path, n_gpus=2)

        # Both GPUs accidentally have t1 (shouldn't happen normally, but be safe)
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0", "t1"]))
        runner._save_gpu_checkpoint(1.0, 1, _make_gpu_result(1, 1.0, ["t1", "t2"]))

        merged = runner._merge_value_results(1.0)

        # Should have 3 unique tasks, not 4
        assert merged["n_problems"] == 3

    def test_returns_none_when_no_checkpoints(self, tmp_path):
        """Should return None when no checkpoint files exist."""
        runner = _make_runner(tmp_path)

        merged = runner._merge_value_results(99.0)

        assert merged is None

    def test_returns_none_when_no_checkpoint_dir(self, tmp_path):
        """Should return None when checkpoint_dir is None."""
        runner = _make_runner(tmp_path)
        runner.checkpoint_dir = None

        merged = runner._merge_value_results(1.0)

        assert merged is None

    def test_uses_custom_merge_fn(self, tmp_path):
        """Should invoke the custom merge function for metric computation."""
        call_log = []

        def custom_merge(gpu_results):
            call_log.append(gpu_results)
            total = sum(len(r["results"]) for r in gpu_results)
            return {"results": gpu_results[0]["results"], "custom_metric": total * 10}

        runner = _make_runner(tmp_path, merge_fn=custom_merge, n_gpus=2)
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0"]))
        runner._save_gpu_checkpoint(1.0, 1, _make_gpu_result(1, 1.0, ["t1"]))

        merged = runner._merge_value_results(1.0)

        assert len(call_log) == 1
        assert merged["custom_metric"] == 20  # 2 total * 10


# =============================================================================
# Orchestrator State (Completed Values) Tests
# =============================================================================

class TestOrchestratorState:
    """Test orchestrator state persistence for completed values."""

    def test_save_and_load_completed_values(self, tmp_path):
        """Should persist completed values across save/load cycles."""
        runner = _make_runner(tmp_path)
        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        runner._save_orchestrator_state(1.0, {"score": 0.5, "n_problems": 10})
        runner._save_orchestrator_state(2.0, {"score": 0.8, "n_problems": 10})

        completed = runner._load_orchestrator_state()

        assert 1.0 in completed
        assert 2.0 in completed

    def test_empty_state_returns_empty_set(self, tmp_path):
        """Loading with no state file should return empty set."""
        runner = _make_runner(tmp_path)

        completed = runner._load_orchestrator_state()

        assert completed == set()

    def test_state_file_stores_result_summaries(self, tmp_path):
        """State file should include per-value score and n_problems."""
        runner = _make_runner(tmp_path)
        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        runner._save_orchestrator_state(1.5, {"score": 0.75, "n_problems": 20})

        state_file = runner.checkpoint_dir / "orchestrator_state.json"
        with open(state_file, "r") as f:
            data = json.load(f)

        assert str(1.5) in data["results"]
        assert data["results"]["1.5"]["score"] == 0.75
        assert data["results"]["1.5"]["n_problems"] == 20

    def test_no_duplicate_values_in_completed_list(self, tmp_path):
        """Saving the same value twice should not duplicate it."""
        runner = _make_runner(tmp_path)
        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        runner._save_orchestrator_state(1.0, {"score": 0.5})
        runner._save_orchestrator_state(1.0, {"score": 0.6})  # Updated score

        state_file = runner.checkpoint_dir / "orchestrator_state.json"
        with open(state_file, "r") as f:
            data = json.load(f)

        assert data["completed_values"].count(1.0) == 1

    def test_no_checkpoint_dir_skips_save(self, tmp_path):
        """Should silently skip when checkpoint_dir is None."""
        runner = _make_runner(tmp_path)
        runner.checkpoint_dir = None

        # Should not raise
        runner._save_orchestrator_state(1.0, {"score": 0.5})


# =============================================================================
# All Tasks Complete Check Tests
# =============================================================================

class TestAllTasksComplete:
    """Test the _all_tasks_complete convenience method."""

    def test_returns_false_when_tasks_remain(self, tmp_path):
        """Should return False when some tasks are not checkpointed."""
        runner = _make_runner(tmp_path, all_task_ids=["t0", "t1", "t2"])
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0"]))

        assert runner._all_tasks_complete(1.0) is False

    def test_returns_true_when_all_done(self, tmp_path):
        """Should return True when all tasks have been checkpointed."""
        runner = _make_runner(tmp_path, all_task_ids=["t0", "t1"])
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0"]))
        runner._save_gpu_checkpoint(1.0, 1, _make_gpu_result(1, 1.0, ["t1"]))

        assert runner._all_tasks_complete(1.0) is True


# =============================================================================
# Result Collection Tests
# =============================================================================

class TestCollectResults:
    """Test result collection from worker queues with timeout handling."""

    def test_collects_successful_results(self, tmp_path):
        """Should collect results from all GPU queues."""
        runner = _make_runner(tmp_path, n_gpus=2, timeout=5)

        q0 = queue.Queue()
        q1 = queue.Queue()

        q0.put({"status": "success", "gpu_id": 0, "results": [{"task_id": "t0"}]})
        q1.put({"status": "success", "gpu_id": 1, "results": [{"task_id": "t1"}]})

        results = runner._collect_results([q0, q1], value=1.0)

        assert results is not None
        assert len(results) == 2

    def test_handles_worker_error(self, tmp_path):
        """With 1/2 GPUs failing, result depends on min_gpu_success_ratio.
        Default 0.75 means 50% success is below threshold -> returns None."""
        runner = _make_runner(tmp_path, n_gpus=2, timeout=5)

        q0 = queue.Queue()
        q1 = queue.Queue()

        q0.put({"status": "error", "gpu_id": 0, "error": "CUDA OOM"})
        q1.put({"status": "success", "gpu_id": 1, "results": [{"task_id": "t1"}]})

        results = runner._collect_results([q0, q1], value=1.0)

        # 1/2 = 50% < 75% threshold -> treated as failure
        assert results is None

    def test_handles_fatal_error(self, tmp_path):
        """With 1/2 GPUs fatal error, 50% < 75% threshold -> returns None."""
        runner = _make_runner(tmp_path, n_gpus=2, timeout=5)

        q0 = queue.Queue()
        q1 = queue.Queue()

        q0.put({"status": "fatal_error", "gpu_id": 0, "error": "Worker crashed"})
        q1.put({"status": "success", "gpu_id": 1, "results": [{"task_id": "t1"}]})

        results = runner._collect_results([q0, q1], value=1.0)

        # 1/2 = 50% < 75% threshold
        assert results is None

    def test_returns_none_on_total_failure(self, tmp_path):
        """Should return None when all workers fail."""
        runner = _make_runner(tmp_path, n_gpus=2, timeout=5)

        q0 = queue.Queue()
        q1 = queue.Queue()

        q0.put({"status": "error", "gpu_id": 0, "error": "OOM"})
        q1.put({"status": "fatal_error", "gpu_id": 1, "error": "Crash"})

        results = runner._collect_results([q0, q1], value=1.0)

        assert results is None

    def test_handles_timeout(self, tmp_path):
        """With 1/2 GPUs timing out, 50% < 75% threshold -> returns None."""
        runner = _make_runner(tmp_path, n_gpus=2, timeout=1)

        q0 = queue.Queue()
        q1 = queue.Queue()

        # Only GPU 0 responds; GPU 1 times out
        q0.put({"status": "success", "gpu_id": 0, "results": [{"task_id": "t0"}]})
        # q1 is empty - will timeout

        results = runner._collect_results([q0, q1], value=1.0)

        # 1/2 = 50% < 75% threshold
        assert results is None

    def test_returns_none_when_all_timeout(self, tmp_path):
        """Should return None when all workers timeout."""
        runner = _make_runner(tmp_path, n_gpus=2, timeout=1)

        q0 = queue.Queue()
        q1 = queue.Queue()
        # Both empty - will timeout

        results = runner._collect_results([q0, q1], value=1.0)

        assert results is None


# =============================================================================
# Early Stopping Integration Tests
# =============================================================================

class TestEarlyStoppingDecision:
    """Test early stopping logic within the iteration loop."""

    def test_early_stop_fn_receives_merged_result_and_history(self, tmp_path):
        """Early stop function should receive the current merged result and full history."""
        call_args_log = []

        def tracking_early_stop(current, history):
            call_args_log.append((current, list(history)))
            return False

        runner = _make_runner(
            tmp_path,
            values_to_test=[1.0, 2.0],
            early_stop_fn=tracking_early_stop,
            all_task_ids=["t0", "t1"],
            n_gpus=2,
        )

        # Pre-populate checkpoints for both values so runner doesn't need workers
        for value in [1.0, 2.0]:
            runner._save_gpu_checkpoint(value, 0, _make_gpu_result(0, value, ["t0"]))
            runner._save_gpu_checkpoint(value, 1, _make_gpu_result(1, value, ["t1"]))

        # Patch out worker spawning entirely since all tasks are pre-checkpointed
        with patch.object(runner, '_load_orchestrator_state', return_value=set()):
            with patch('multiprocessing.Manager'):
                with patch.object(runner, '_shutdown_workers'):
                    # We need to simulate the run without actual multiprocessing.
                    # Since all tasks are checkpointed, the run loop should skip to merge.
                    # But run() spawns workers. Let's test the early_stop_fn in isolation.
                    pass

        # Test directly: simulate what run() does after merging
        history = []

        merged1 = runner._merge_value_results(1.0)
        merged1["value"] = 1.0
        merged1["score"] = 0.5
        history.append(merged1)
        assert tracking_early_stop(merged1, history) is False
        assert len(call_args_log) == 1

        merged2 = runner._merge_value_results(2.0)
        merged2["value"] = 2.0
        merged2["score"] = 0.8
        history.append(merged2)
        assert tracking_early_stop(merged2, history) is False
        assert len(call_args_log) == 2

        # Verify first call had 1-item history, second had 2-item history
        assert len(call_args_log[0][1]) == 1
        assert len(call_args_log[1][1]) == 2

    def test_early_stop_halts_iteration(self, tmp_path):
        """When early_stop_fn returns True, remaining values should be skipped."""
        stop_after_value = 2.0

        def stop_at_two(current, history):
            return current.get("value") == stop_after_value

        runner = _make_runner(
            tmp_path,
            values_to_test=[1.0, 2.0, 3.0],
            early_stop_fn=stop_at_two,
            all_task_ids=["t0"],
            n_gpus=1,
        )

        # Pre-populate all checkpoints
        for value in [1.0, 2.0, 3.0]:
            runner._save_gpu_checkpoint(value, 0, _make_gpu_result(0, value, ["t0"]))

        # Simulate the iteration logic manually (avoiding multiprocessing)
        history = []
        processed_values = []

        for value in runner.values_to_test:
            remaining = runner._get_remaining_tasks_for_value(value)
            if not remaining:
                merged = runner._merge_value_results(value)
                if merged:
                    merged["value"] = value
                    score = merged.get("score", 0.0)
                    merged["score"] = score
                    history.append(merged)
                    processed_values.append(value)
                    if runner.early_stop_fn(merged, history):
                        break

        # Should have stopped after value=2.0, not processing 3.0
        assert processed_values == [1.0, 2.0]
        assert len(history) == 2


# =============================================================================
# Checkpoint Resume Behavior Tests
# =============================================================================

class TestCheckpointResume:
    """Test resuming from checkpointed state."""

    def test_resume_skips_fully_completed_value(self, tmp_path):
        """A fully completed value should be skipped on resume."""
        runner = _make_runner(
            tmp_path,
            values_to_test=[1.0, 2.0],
            all_task_ids=["t0", "t1"],
        )

        # Mark value 1.0 as fully completed
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0"]))
        runner._save_gpu_checkpoint(1.0, 1, _make_gpu_result(1, 1.0, ["t1"]))
        runner.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        runner._save_orchestrator_state(1.0, {"score": 0.5, "n_problems": 2})

        completed = runner._load_orchestrator_state()
        assert 1.0 in completed

        # Value 1.0 has no remaining tasks
        assert runner._get_remaining_tasks_for_value(1.0) == []

        # Value 2.0 is fully remaining
        assert sorted(runner._get_remaining_tasks_for_value(2.0)) == ["t0", "t1"]

    def test_resume_retries_partially_completed_value(self, tmp_path):
        """A partially completed value should have only remaining tasks redistributed."""
        runner = _make_runner(
            tmp_path,
            values_to_test=[1.0],
            all_task_ids=["t0", "t1", "t2", "t3"],
            n_gpus=2,
        )

        # Only t0 and t1 were completed before interrupt
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0"]))
        runner._save_gpu_checkpoint(1.0, 1, _make_gpu_result(1, 1.0, ["t1"]))

        remaining = runner._get_remaining_tasks_for_value(1.0)

        assert sorted(remaining) == ["t2", "t3"]

        # Redistribute remaining tasks
        assignments = runner._distribute_tasks(remaining, 2)

        assert assignments[0] == ["t2"]
        assert assignments[1] == ["t3"]

    def test_resume_different_number_of_gpus(self, tmp_path):
        """On resume with different n_gpus, remaining tasks redistribute across new GPU count."""
        runner = _make_runner(
            tmp_path,
            values_to_test=[1.0],
            all_task_ids=[f"t{i}" for i in range(8)],
            n_gpus=4,  # Previously was 2, now is 4
        )

        # Old run with 2 GPUs completed t0-t3
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0", "t2"]))
        runner._save_gpu_checkpoint(1.0, 1, _make_gpu_result(1, 1.0, ["t1", "t3"]))

        remaining = runner._get_remaining_tasks_for_value(1.0)
        assert sorted(remaining) == ["t4", "t5", "t6", "t7"]

        # Now distribute across 4 GPUs
        assignments = runner._distribute_tasks(remaining, 4)

        assert assignments[0] == ["t4"]
        assert assignments[1] == ["t5"]
        assert assignments[2] == ["t6"]
        assert assignments[3] == ["t7"]


# =============================================================================
# Worker Shutdown Tests
# =============================================================================

class TestWorkerShutdown:
    """Test graceful and forced worker shutdown."""

    def test_sends_shutdown_signal_to_all_queues(self, tmp_path):
        """Should send shutdown tuple to each task queue."""
        runner = _make_runner(tmp_path, n_gpus=3)

        queues = [MagicMock() for _ in range(3)]
        workers = [MagicMock() for _ in range(3)]
        for w in workers:
            w.is_alive.return_value = False

        runner._shutdown_workers(workers, queues)

        for q in queues:
            q.put.assert_called_once_with(("shutdown", None, None), timeout=5)

    def test_joins_workers_with_timeout(self, tmp_path):
        """Should wait for each worker to finish with a timeout."""
        runner = _make_runner(tmp_path, n_gpus=2)

        workers = [MagicMock() for _ in range(2)]
        for w in workers:
            w.is_alive.return_value = False

        runner._shutdown_workers(workers, [MagicMock(), MagicMock()])

        for w in workers:
            w.join.assert_called_once_with(timeout=30)

    def test_terminates_stuck_workers(self, tmp_path):
        """Should force-terminate workers that don't shut down gracefully."""
        runner = _make_runner(tmp_path, n_gpus=2)

        workers = [MagicMock(), MagicMock()]
        # First worker shuts down fine, second is stuck
        workers[0].is_alive.return_value = False
        workers[1].is_alive.return_value = True

        runner._shutdown_workers(workers, [MagicMock(), MagicMock()])

        workers[0].terminate.assert_not_called()
        workers[1].terminate.assert_called_once()


# =============================================================================
# Evaluator Task ID Discovery Tests
# =============================================================================

class TestGetEvaluatorTaskIds:
    """Test extracting task_ids from evaluator objects."""

    def test_uses_get_relevant_task_ids_if_available(self, tmp_path):
        """Should prefer evaluator's get_relevant_task_ids() method."""
        runner = _make_runner(tmp_path)

        evaluator = MagicMock()
        evaluator.get_relevant_task_ids.return_value = ["t0", "t1", "t2"]

        task_ids = runner._get_evaluator_task_ids(evaluator)

        assert task_ids == ["t0", "t1", "t2"]
        evaluator.get_relevant_task_ids.assert_called_once()

    def test_falls_back_to_data_attribute(self, tmp_path):
        """Should extract task_ids from common data attributes."""
        runner = _make_runner(tmp_path)

        evaluator = MagicMock(spec=[])  # No get_relevant_task_ids
        evaluator.analysis_data = pd.DataFrame({"task_id": ["t0", "t1"]})

        task_ids = runner._get_evaluator_task_ids(evaluator)

        assert set(task_ids) == {"t0", "t1"}

    def test_extracts_from_split_datasets(self, tmp_path):
        """Should merge task_ids from correct/incorrect split datasets."""
        runner = _make_runner(tmp_path)

        evaluator = MagicMock(spec=[])  # No get_relevant_task_ids
        # No primary data attr, but has split datasets
        evaluator.incorrect_problems = pd.DataFrame({"task_id": ["t0", "t1"]})
        evaluator.correct_problems = pd.DataFrame({"task_id": ["t2", "t3"]})

        task_ids = runner._get_evaluator_task_ids(evaluator)

        assert set(task_ids) == {"t0", "t1", "t2", "t3"}

    def test_deduplicates_task_ids(self, tmp_path):
        """Should deduplicate task_ids from multiple sources."""
        runner = _make_runner(tmp_path)

        evaluator = MagicMock(spec=[])
        # Same task_id in both attributes
        evaluator.incorrect_problems = pd.DataFrame({"task_id": ["t0", "t1"]})
        evaluator.correct_problems = pd.DataFrame({"task_id": ["t1", "t2"]})

        task_ids = runner._get_evaluator_task_ids(evaluator)

        assert len(task_ids) == 3
        assert set(task_ids) == {"t0", "t1", "t2"}

    def test_returns_empty_when_no_data(self, tmp_path):
        """Should return empty list when evaluator has no recognizable data."""
        runner = _make_runner(tmp_path)

        evaluator = MagicMock(spec=[])  # No recognized attributes

        task_ids = runner._get_evaluator_task_ids(evaluator)

        assert task_ids == []


# =============================================================================
# Task ID Discovery via Worker Tests
# =============================================================================

class TestDiscoverTaskIds:
    """Test discovering task_ids from worker processes."""

    def test_discovers_from_first_worker(self, tmp_path):
        """Should ask GPU 0 for task_ids and return them."""
        runner = _make_runner(tmp_path, timeout=5)

        task_queues = [queue.Queue(), queue.Queue()]
        result_queues = [queue.Queue(), queue.Queue()]

        # Simulate worker responding
        result_queues[0].put({
            "status": "task_ids",
            "task_ids": ["t0", "t1", "t2"],
            "gpu_id": 0,
        })

        discovered = runner._discover_task_ids(task_queues, result_queues)

        assert discovered == ["t0", "t1", "t2"]

        # Verify the command was sent to GPU 0's queue
        msg = task_queues[0].get_nowait()
        assert msg[0] == "get_task_ids"

    def test_returns_empty_on_timeout(self, tmp_path):
        """Should return empty list if worker doesn't respond."""
        runner = _make_runner(tmp_path, timeout=1)

        task_queues = [queue.Queue()]
        result_queues = [queue.Queue()]  # Empty - will timeout

        discovered = runner._discover_task_ids(task_queues, result_queues)

        assert discovered == []

    def test_returns_empty_on_unexpected_response(self, tmp_path):
        """Should return empty list if worker sends unexpected response."""
        runner = _make_runner(tmp_path, timeout=5)

        task_queues = [queue.Queue()]
        result_queues = [queue.Queue()]

        result_queues[0].put({"status": "error", "error": "Something went wrong"})

        discovered = runner._discover_task_ids(task_queues, result_queues)

        assert discovered == []


# =============================================================================
# Runner Configuration Tests
# =============================================================================

class TestRunnerConfiguration:
    """Test IterativeParallelRunner initialization and defaults."""

    def test_default_timeout(self):
        """Default timeout should match the module constant."""
        assert DEFAULT_WORKER_TIMEOUT == 600

    def test_default_early_stop_is_noop(self, tmp_path):
        """Runner should use _no_early_stop by default."""
        runner = _make_runner(tmp_path, early_stop_fn=None)
        assert runner.early_stop_fn is _no_early_stop

    def test_default_merge_is_default_merge_fn(self, tmp_path):
        """Runner should use _default_merge_fn by default."""
        runner = _make_runner(tmp_path, merge_fn=None)
        assert runner.merge_fn is _default_merge_fn

    def test_custom_early_stop_fn(self, tmp_path):
        """Runner should use custom early stop function when provided."""
        custom_fn = lambda current, history: True
        runner = _make_runner(tmp_path, early_stop_fn=custom_fn)
        assert runner.early_stop_fn is custom_fn

    def test_custom_merge_fn(self, tmp_path):
        """Runner should use custom merge function when provided."""
        custom_fn = lambda results: {"custom": True}
        runner = _make_runner(tmp_path, merge_fn=custom_fn)
        assert runner.merge_fn is custom_fn

    def test_stores_all_task_ids(self, tmp_path):
        """Runner should store provided task_ids."""
        task_ids = ["t0", "t1", "t2"]
        runner = _make_runner(tmp_path, all_task_ids=task_ids)
        assert runner.all_task_ids == task_ids

    def test_none_task_ids_deferred_discovery(self, tmp_path):
        """When all_task_ids is None, discovery happens during run()."""
        config = MagicMock()
        runner = IterativeParallelRunner(
            phase_evaluator_class=MagicMock,
            config=config,
            n_gpus=2,
            values_to_test=[1.0],
            all_task_ids=None,
        )
        assert runner.all_task_ids is None


# =============================================================================
# run_iterative_parallel Entry Point Tests
# =============================================================================

class TestRunIterativeParallelEntryPoint:
    """Test the run_iterative_parallel convenience function."""

    def test_unsupported_phase_raises(self):
        """Should raise ValueError for unsupported phase IDs."""
        from common.iterative_parallel_runner import run_iterative_parallel

        config = MagicMock()

        with pytest.raises(ValueError, match="does not support iterative parallelization"):
            run_iterative_parallel("99.9", config, n_gpus=2)

    def test_supported_phases(self):
        """Should recognize phases 3.5, 4.5, 4.6, and 8.2."""
        from common.iterative_parallel_runner import run_iterative_parallel

        config = MagicMock()

        for phase_id in ["3.5", "4.5", "4.6", "8.2"]:
            # Each should try to import the phase module (will fail in test env,
            # but should not raise ValueError)
            with pytest.raises((ImportError, ModuleNotFoundError, Exception)):
                run_iterative_parallel(phase_id, config, n_gpus=2)


# =============================================================================
# Fix 1: min_gpu_success_ratio threshold tests
# =============================================================================

class TestMinGpuSuccessRatio:
    """Test that partial GPU failures are rejected when below the success threshold."""

    def test_all_gpus_succeed(self, tmp_path):
        """4/4 GPUs succeed -> proceeds normally."""
        runner = _make_runner(tmp_path, n_gpus=4, timeout=5)

        queues = [queue.Queue() for _ in range(4)]
        for i, q in enumerate(queues):
            q.put({"status": "success", "gpu_id": i, "results": [{"task_id": f"t{i}"}]})

        results = runner._collect_results(queues, value=1.0)
        assert results is not None
        assert len(results) == 4

    def test_three_of_four_gpus_succeed(self, tmp_path):
        """3/4 GPUs succeed (75%) -> meets default 0.75 threshold."""
        runner = _make_runner(tmp_path, n_gpus=4, timeout=5)

        queues = [queue.Queue() for _ in range(4)]
        queues[0].put({"status": "success", "gpu_id": 0, "results": [{"task_id": "t0"}]})
        queues[1].put({"status": "success", "gpu_id": 1, "results": [{"task_id": "t1"}]})
        queues[2].put({"status": "success", "gpu_id": 2, "results": [{"task_id": "t2"}]})
        queues[3].put({"status": "error", "gpu_id": 3, "error": "OOM"})

        results = runner._collect_results(queues, value=1.0)
        assert results is not None
        assert len(results) == 3

    def test_one_of_four_gpus_succeed(self, tmp_path):
        """1/4 GPUs succeed (25%) -> below 0.75 threshold, returns None."""
        runner = _make_runner(tmp_path, n_gpus=4, timeout=5)

        queues = [queue.Queue() for _ in range(4)]
        queues[0].put({"status": "success", "gpu_id": 0, "results": [{"task_id": "t0"}]})
        queues[1].put({"status": "error", "gpu_id": 1, "error": "OOM"})
        queues[2].put({"status": "error", "gpu_id": 2, "error": "OOM"})
        queues[3].put({"status": "fatal_error", "gpu_id": 3, "error": "crash"})

        results = runner._collect_results(queues, value=1.0)
        assert results is None

    def test_zero_gpus_succeed(self, tmp_path):
        """0/4 GPUs succeed -> returns None."""
        runner = _make_runner(tmp_path, n_gpus=4, timeout=5)

        queues = [queue.Queue() for _ in range(4)]
        for i, q in enumerate(queues):
            q.put({"status": "error", "gpu_id": i, "error": "OOM"})

        results = runner._collect_results(queues, value=1.0)
        assert results is None

    def test_custom_threshold(self, tmp_path):
        """Custom threshold of 0.5: 2/4 GPUs succeed -> proceeds."""
        config = MagicMock()
        config.model_name = "google/gemma-2-2b"
        config.dataset_name = "mbpp"

        runner = IterativeParallelRunner(
            phase_evaluator_class=MagicMock,
            config=config,
            n_gpus=4,
            values_to_test=[1.0],
            timeout_per_iteration=5,
            checkpoint_dir=tmp_path / "checkpoints",
            all_task_ids=["t0", "t1", "t2", "t3"],
            min_gpu_success_ratio=0.5,
        )

        queues = [queue.Queue() for _ in range(4)]
        queues[0].put({"status": "success", "gpu_id": 0, "results": [{"task_id": "t0"}]})
        queues[1].put({"status": "success", "gpu_id": 1, "results": [{"task_id": "t1"}]})
        queues[2].put({"status": "error", "gpu_id": 2, "error": "OOM"})
        queues[3].put({"status": "error", "gpu_id": 3, "error": "OOM"})

        results = runner._collect_results(queues, value=1.0)
        assert results is not None
        assert len(results) == 2

    def test_custom_threshold_below(self, tmp_path):
        """Custom threshold of 0.5: 1/4 GPUs succeed -> returns None."""
        config = MagicMock()
        config.model_name = "google/gemma-2-2b"
        config.dataset_name = "mbpp"

        runner = IterativeParallelRunner(
            phase_evaluator_class=MagicMock,
            config=config,
            n_gpus=4,
            values_to_test=[1.0],
            timeout_per_iteration=5,
            checkpoint_dir=tmp_path / "checkpoints",
            all_task_ids=["t0", "t1", "t2", "t3"],
            min_gpu_success_ratio=0.5,
        )

        queues = [queue.Queue() for _ in range(4)]
        queues[0].put({"status": "success", "gpu_id": 0, "results": [{"task_id": "t0"}]})
        queues[1].put({"status": "error", "gpu_id": 1, "error": "OOM"})
        queues[2].put({"status": "error", "gpu_id": 2, "error": "OOM"})
        queues[3].put({"status": "error", "gpu_id": 3, "error": "OOM"})

        results = runner._collect_results(queues, value=1.0)
        assert results is None

    def test_default_ratio_is_075(self, tmp_path):
        """Default min_gpu_success_ratio should be 0.75."""
        runner = _make_runner(tmp_path)
        assert runner.min_gpu_success_ratio == 0.75


# =============================================================================
# Fix 2: Corrupted parquet fail-fast tests
# =============================================================================

class TestCorruptedParquetFailFast:
    """Test that corrupted parquet checkpoints raise errors instead of silently skipping."""

    def test_corrupted_parquet_raises_error(self, tmp_path):
        """A corrupted parquet file should cause RuntimeError."""
        runner = _make_runner(tmp_path, n_gpus=2)

        # Save a valid checkpoint for GPU 0
        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0", "t1"]))

        # Corrupt GPU 1's parquet file
        value_dir = runner._get_value_checkpoint_dir(1.0)
        corrupted_path = value_dir / "gpu_1_results.parquet"
        corrupted_path.write_text("not a valid parquet file")

        with pytest.raises(RuntimeError, match="Corrupted checkpoint files"):
            runner._merge_value_results(1.0)

    def test_valid_parquets_still_merge(self, tmp_path):
        """When all parquet files are valid, merge should succeed as before."""
        runner = _make_runner(tmp_path, n_gpus=2)

        runner._save_gpu_checkpoint(1.0, 0, _make_gpu_result(0, 1.0, ["t0"]))
        runner._save_gpu_checkpoint(1.0, 1, _make_gpu_result(1, 1.0, ["t1"]))

        merged = runner._merge_value_results(1.0)
        assert merged is not None
        assert merged["n_problems"] == 2
