"""
Tests for common/parallel_runner.py

Validates:
- filter_dataframe_for_gpu: Round-robin distribution
- Task distribution exhaustiveness and exclusivity
- H1 regression: Phase 8.3 merge uses 'was_steered' key, not 'steered'
- R3: Helper functions _load_gpu_json_files and _cleanup_gpu_files
- Merge function behavioral tests: Phase 4.5, Phase 8.3
- GPU failure prevention (Fix 5)
- Missing field handling (Fix 2)
- Subprocess orchestration: worker exception handling, timeout behavior
"""

import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from concurrent.futures import Future

from common.parallel_runner import (
    filter_dataframe_for_gpu,
    _merge_phase4_5_json_results,
    _merge_phase4_8_results,
    _merge_phase8_3_results,
    _merge_parallel_results,
    run_phase_parallel,
)
from common.config import Config


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


# =============================================================================
# Merge Function Behavioral Tests: Phase 4.5
# =============================================================================

class TestMergePhase45JsonResults:
    """Test _merge_phase4_5_json_results with fixture data.

    Verifies that per-GPU coefficient analysis JSON files are merged correctly,
    metrics are recalculated from combined data, and corrupted files are handled.
    """

    def _make_gpu_json(self, steering_type, coefficient, results):
        """Build a per-GPU JSON structure matching Phase 4.5 output format."""
        steering_key = f"{steering_type}_steering"
        return {
            steering_key: {
                "optimal_coefficient": coefficient,
                "best_result": None,
                "search_history": [
                    {
                        "coefficient": coefficient,
                        "metrics": {},
                        "results": results,
                    }
                ],
            }
        }

    @patch("common.parallel_runner.write_phase_output")
    def test_correction_metrics_recalculated(self, mock_write, tmp_path):
        """Correction rate should be recalculated from merged per-problem results."""
        config = Config()

        # GPU 0: 2 problems, 1 correction (incorrect baseline -> correct steered)
        gpu0_data = self._make_gpu_json("correct", 10, [
            {"task_id": "t1", "baseline_passed": False, "steered_correct": True,
             "code_similarity": 0.8},
            {"task_id": "t2", "baseline_passed": False, "steered_correct": False,
             "code_similarity": 0.5},
        ])

        # GPU 1: 2 problems, 1 correction
        gpu1_data = self._make_gpu_json("correct", 10, [
            {"task_id": "t3", "baseline_passed": False, "steered_correct": True,
             "code_similarity": 0.9},
            {"task_id": "t4", "baseline_passed": False, "steered_correct": False,
             "code_similarity": 0.6},
        ])

        (tmp_path / "coefficient_analysis_gpu0.json").write_text(json.dumps(gpu0_data))
        (tmp_path / "coefficient_analysis_gpu1.json").write_text(json.dumps(gpu1_data))

        result = _merge_phase4_5_json_results(tmp_path, n_gpus=2, config=config, phase_id="4.5")

        # Should have merged file info
        assert "merged_file" in result
        assert result["n_gpus"] == 2

        # Load the merged analysis output
        merged = json.loads((tmp_path / "coefficient_analysis.json").read_text())
        assert "correct_steering" in merged
        correct = merged["correct_steering"]
        assert correct["optimal_coefficient"] == 10

        # 2 out of 4 incorrect baselines were corrected => 50%
        history_entry = correct["search_history"][0]
        assert history_entry["metrics"]["correction_rate"] == 50.0
        assert history_entry["n_problems"] == 4

    @patch("common.parallel_runner.write_phase_output")
    def test_corruption_metrics_recalculated(self, mock_write, tmp_path):
        """Corruption rate and composite score should be recalculated from merged data."""
        config = Config()

        # GPU 0: 1 correct baseline that gets corrupted
        gpu0_data = self._make_gpu_json("incorrect", 20, [
            {"task_id": "t1", "baseline_passed": True, "steered_correct": False,
             "code_similarity": 0.3},
        ])

        # GPU 1: 1 correct baseline that stays correct
        gpu1_data = self._make_gpu_json("incorrect", 20, [
            {"task_id": "t2", "baseline_passed": True, "steered_correct": True,
             "code_similarity": 0.9},
        ])

        (tmp_path / "coefficient_analysis_gpu0.json").write_text(json.dumps(gpu0_data))
        (tmp_path / "coefficient_analysis_gpu1.json").write_text(json.dumps(gpu1_data))

        _merge_phase4_5_json_results(tmp_path, n_gpus=2, config=config, phase_id="4.5")

        merged = json.loads((tmp_path / "coefficient_analysis.json").read_text())
        incorrect = merged["incorrect_steering"]
        history_entry = incorrect["search_history"][0]

        # 1 out of 2 correct baselines corrupted => 50%
        assert history_entry["metrics"]["corruption_rate"] == 50.0
        # avg_similarity = (0.3 + 0.9) / 2 * 100 = 60.0
        assert history_entry["metrics"]["avg_similarity"] == pytest.approx(60.0, abs=0.1)
        # composite = 50 * 0.5 + 60 * 0.5 = 55.0
        assert history_entry["metrics"]["composite_score"] == pytest.approx(55.0, abs=0.1)

    @patch("common.parallel_runner.write_phase_output")
    def test_all_required_fields_present(self, mock_write, tmp_path):
        """Merged results should contain all expected top-level keys."""
        config = Config()

        gpu_data = {
            "correct_steering": {
                "optimal_coefficient": 5,
                "best_result": None,
                "search_history": [
                    {
                        "coefficient": 5,
                        "results": [
                            {"task_id": "t1", "baseline_passed": False,
                             "steered_correct": True, "code_similarity": 0.7},
                        ],
                    }
                ],
            },
            "incorrect_steering": {
                "optimal_coefficient": 15,
                "best_result": None,
                "search_history": [
                    {
                        "coefficient": 15,
                        "results": [
                            {"task_id": "t2", "baseline_passed": True,
                             "steered_correct": False, "code_similarity": 0.4},
                        ],
                    }
                ],
            },
        }

        (tmp_path / "coefficient_analysis_gpu0.json").write_text(json.dumps(gpu_data))

        _merge_phase4_5_json_results(tmp_path, n_gpus=1, config=config, phase_id="4.5")

        merged = json.loads((tmp_path / "coefficient_analysis.json").read_text())
        assert "correct_steering" in merged
        assert "incorrect_steering" in merged

        for key in ["correct_steering", "incorrect_steering"]:
            section = merged[key]
            assert "optimal_coefficient" in section
            assert "best_result" in section
            assert "search_history" in section
            for entry in section["search_history"]:
                assert "coefficient" in entry
                assert "metrics" in entry
                assert "n_problems" in entry
                assert "results" in entry

    @patch("common.parallel_runner.write_phase_output")
    def test_corrupted_json_handled_gracefully(self, mock_write, tmp_path):
        """Corrupted GPU JSON files should be skipped; valid ones still merged (Fix 9)."""
        config = Config()

        # Valid GPU file
        gpu0_data = self._make_gpu_json("correct", 10, [
            {"task_id": "t1", "baseline_passed": False, "steered_correct": True,
             "code_similarity": 0.8},
        ])
        (tmp_path / "coefficient_analysis_gpu0.json").write_text(json.dumps(gpu0_data))

        # Corrupted GPU file
        (tmp_path / "coefficient_analysis_gpu1.json").write_text("{ invalid json !!!")

        # Should not raise; the valid file is still processed
        result = _merge_phase4_5_json_results(tmp_path, n_gpus=2, config=config, phase_id="4.5")
        assert result["n_gpus"] == 2

        # Merged file should exist and contain results from GPU 0 only
        merged = json.loads((tmp_path / "coefficient_analysis.json").read_text())
        correct = merged["correct_steering"]
        assert correct["search_history"][0]["n_problems"] == 1

    @patch("common.parallel_runner.write_phase_output")
    def test_all_files_corrupted_raises(self, mock_write, tmp_path):
        """If ALL GPU files are corrupted, should raise RuntimeError."""
        config = Config()

        (tmp_path / "coefficient_analysis_gpu0.json").write_text("not json")
        (tmp_path / "coefficient_analysis_gpu1.json").write_text("{broken")

        with pytest.raises(RuntimeError, match="corrupted"):
            _merge_phase4_5_json_results(tmp_path, n_gpus=2, config=config, phase_id="4.5")


# =============================================================================
# Merge Function Behavioral Tests: Phase 8.3
# =============================================================================

class TestMergePhase83Results:
    """Test _merge_phase8_3_results behavioral merge with parquet data.

    Verifies that per-GPU parquet files are merged correctly, metrics are
    recalculated, and deduplication works by task_id + experiment_type.
    """

    def _make_gpu_parquet(self, path, records):
        """Write a list of record dicts as a parquet file."""
        df = pd.DataFrame(records)
        df.to_parquet(path, index=False)

    @patch("common.parallel_runner.write_phase_output")
    def test_correction_rate_calculated_correctly(self, mock_write, tmp_path):
        """correction_rate = n_corrected / n_valid_correction."""
        config = Config()

        # GPU 0: 1 correction experiment, baseline_passed=False, steered_correct=True
        self._make_gpu_parquet(tmp_path / "results_gpu0.parquet", [
            {"task_id": "t1", "baseline_passed": False, "was_steered": True,
             "steered_correct": True, "experiment_type": "correction",
             "incorrect_pred_activation": 0.5},
        ])
        # GPU 1: 1 correction experiment, baseline_passed=False, steered_correct=False
        self._make_gpu_parquet(tmp_path / "results_gpu1.parquet", [
            {"task_id": "t2", "baseline_passed": False, "was_steered": True,
             "steered_correct": False, "experiment_type": "correction",
             "incorrect_pred_activation": 0.3},
        ])

        result = _merge_phase8_3_results(tmp_path, n_gpus=2, config=config)

        # 1 corrected out of 2 => 0.5
        assert result["correction_rate"] == pytest.approx(0.5, abs=0.01)

    @patch("common.parallel_runner.write_phase_output")
    def test_preservation_and_corruption_rates(self, mock_write, tmp_path):
        """preservation_rate and corruption_rate calculated from preservation experiments."""
        config = Config()

        self._make_gpu_parquet(tmp_path / "results_gpu0.parquet", [
            # Preservation: baseline correct, steered correct -> preserved
            {"task_id": "t1", "baseline_passed": True, "was_steered": False,
             "steered_correct": True, "experiment_type": "preservation",
             "incorrect_pred_activation": 0.1},
            # Preservation: baseline correct, steered incorrect -> corrupted
            {"task_id": "t2", "baseline_passed": True, "was_steered": True,
             "steered_correct": False, "experiment_type": "preservation",
             "incorrect_pred_activation": 0.8},
        ])
        self._make_gpu_parquet(tmp_path / "results_gpu1.parquet", [
            # Preservation: baseline correct, steered correct -> preserved
            {"task_id": "t3", "baseline_passed": True, "was_steered": False,
             "steered_correct": True, "experiment_type": "preservation",
             "incorrect_pred_activation": 0.05},
        ])

        result = _merge_phase8_3_results(tmp_path, n_gpus=2, config=config)

        # 2 preserved out of 3 valid => 2/3
        assert result["preservation_rate"] == pytest.approx(2 / 3, abs=0.01)
        # 1 corrupted out of 3 valid => 1/3
        assert result["corruption_rate"] == pytest.approx(1 / 3, abs=0.01)

    @patch("common.parallel_runner.write_phase_output")
    def test_deduplication_by_task_id_and_experiment_type(self, mock_write, tmp_path):
        """Duplicate task_id + experiment_type combos should be deduplicated (keep last)."""
        config = Config()

        # Both GPUs have task_id "t1" for "correction" (e.g., cross-run checkpoint overlap)
        self._make_gpu_parquet(tmp_path / "results_gpu0.parquet", [
            {"task_id": "t1", "baseline_passed": False, "was_steered": True,
             "steered_correct": False, "experiment_type": "correction",
             "incorrect_pred_activation": 0.5},
        ])
        self._make_gpu_parquet(tmp_path / "results_gpu1.parquet", [
            {"task_id": "t1", "baseline_passed": False, "was_steered": True,
             "steered_correct": True, "experiment_type": "correction",
             "incorrect_pred_activation": 0.5},
        ])

        result = _merge_phase8_3_results(tmp_path, n_gpus=2, config=config)

        # After dedup, only 1 correction record should remain
        # The last occurrence (from GPU 1) should be kept: steered_correct=True
        assert result["correction_rate"] == pytest.approx(1.0, abs=0.01)

    @patch("common.parallel_runner.write_phase_output")
    def test_same_task_different_experiment_types_not_deduped(self, mock_write, tmp_path):
        """Same task_id but different experiment_type should both be kept."""
        config = Config()

        self._make_gpu_parquet(tmp_path / "results_gpu0.parquet", [
            {"task_id": "t1", "baseline_passed": False, "was_steered": True,
             "steered_correct": True, "experiment_type": "correction",
             "incorrect_pred_activation": 0.5},
            {"task_id": "t1", "baseline_passed": True, "was_steered": False,
             "steered_correct": True, "experiment_type": "preservation",
             "incorrect_pred_activation": 0.1},
        ])

        result = _merge_phase8_3_results(tmp_path, n_gpus=1, config=config)

        # Both records should be kept (different experiment_type)
        assert result["total_results"] == 2
        assert result["correction_rate"] == pytest.approx(1.0, abs=0.01)
        assert result["preservation_rate"] == pytest.approx(1.0, abs=0.01)

    @patch("common.parallel_runner.write_phase_output")
    def test_field_access_uses_direct_keys(self, mock_write, tmp_path):
        """Merge should use r['baseline_passed'] and r['steered_correct'] directly."""
        config = Config()

        # Records with exact required fields, no extras
        self._make_gpu_parquet(tmp_path / "results_gpu0.parquet", [
            {"task_id": "t1", "baseline_passed": False, "was_steered": True,
             "steered_correct": True, "experiment_type": "correction",
             "incorrect_pred_activation": 0.5},
            {"task_id": "t2", "baseline_passed": True, "was_steered": False,
             "steered_correct": True, "experiment_type": "preservation",
             "incorrect_pred_activation": 0.05},
        ])

        # Should not raise KeyError — fields are accessed directly
        result = _merge_phase8_3_results(tmp_path, n_gpus=1, config=config)
        assert result["correction_rate"] == pytest.approx(1.0, abs=0.01)
        assert result["preservation_rate"] == pytest.approx(1.0, abs=0.01)


# =============================================================================
# GPU Failure Prevention (Fix 5)
# =============================================================================

class TestFailedGpuMergePrevention:
    """Test that GPU failures prevent merge.

    When any GPU worker fails, run_phase_parallel should raise RuntimeError
    instead of silently merging incomplete data (Fix 5).
    """

    def test_failed_gpu_raises_runtime_error(self, tmp_path):
        """run_phase_parallel should raise RuntimeError when any GPU fails."""
        from unittest.mock import MagicMock
        from common.parallel_runner import run_phase_parallel

        config = Config()

        # Mock dependencies to avoid actual GPU/subprocess work
        with patch("common.phase_discovery.get_phase_output_dir", return_value=str(tmp_path)), \
             patch("common.parallel_runner.mp.get_context") as mock_ctx, \
             patch("common.parallel_runner.ProcessPoolExecutor") as mock_executor_cls, \
             patch("common.parallel_runner._merge_parallel_results"):

            mock_manager = mock_ctx.return_value.Manager.return_value
            mock_manager.Semaphore.return_value = None

            # Create mock futures: one success, one failure
            mock_future_success = MagicMock()
            mock_future_success.exception.return_value = None
            mock_future_success.result.return_value = {
                "gpu_id": 0, "status": "success", "result": {}
            }

            mock_future_fail = MagicMock()
            mock_future_fail.exception.return_value = None
            mock_future_fail.result.return_value = {
                "gpu_id": 1, "status": "error", "error": "CUDA OOM",
                "traceback": "fake traceback"
            }

            # Set up executor to submit futures mapped to gpu_ids
            mock_executor = MagicMock()
            mock_executor_cls.return_value.__enter__ = MagicMock(return_value=mock_executor)
            mock_executor_cls.return_value.__exit__ = MagicMock(return_value=False)

            # submit() returns futures; as_completed yields them
            mock_executor.submit.side_effect = [mock_future_success, mock_future_fail]

            with patch("common.parallel_runner.as_completed",
                       return_value=iter([mock_future_success, mock_future_fail])) as mock_ac:
                with pytest.raises(RuntimeError, match="GPU workers failed"):
                    run_phase_parallel("4.8", config, n_gpus=2)

    def test_source_code_checks_failed_gpus(self):
        """run_phase_parallel source should check failed_gpus and raise RuntimeError."""
        import inspect
        from common.parallel_runner import run_phase_parallel
        source = inspect.getsource(run_phase_parallel)

        # The function must check for failures and refuse to merge
        assert "failed_gpus" in source
        assert "raise RuntimeError" in source


# =============================================================================
# Missing Field Handling (Fix 2)
# =============================================================================

class TestMergeWithMissingFields:
    """Test behavior when records have missing fields.

    Records missing 'baseline_passed' or 'steered_correct' should be skipped
    during metric calculation; valid records should still be processed (Fix 2).
    """

    @patch("common.parallel_runner.write_phase_output")
    def test_phase45_skips_records_missing_baseline_passed(self, mock_write, tmp_path):
        """Records without 'baseline_passed' should be skipped in metric calculation."""
        config = Config()

        gpu_data = {
            "correct_steering": {
                "optimal_coefficient": 10,
                "best_result": None,
                "search_history": [
                    {
                        "coefficient": 10,
                        "results": [
                            # Valid record
                            {"task_id": "t1", "baseline_passed": False,
                             "steered_correct": True, "code_similarity": 0.8},
                            # Missing baseline_passed
                            {"task_id": "t2", "steered_correct": True,
                             "code_similarity": 0.9},
                            # Valid record
                            {"task_id": "t3", "baseline_passed": False,
                             "steered_correct": False, "code_similarity": 0.4},
                        ],
                    }
                ],
            }
        }

        (tmp_path / "coefficient_analysis_gpu0.json").write_text(json.dumps(gpu_data))

        _merge_phase4_5_json_results(tmp_path, n_gpus=1, config=config, phase_id="4.5")

        merged = json.loads((tmp_path / "coefficient_analysis.json").read_text())
        correct = merged["correct_steering"]
        history = correct["search_history"][0]

        # n_problems should include all 3 (total count)
        assert history["n_problems"] == 3
        # But correction_rate should be calculated from 2 valid records only
        # 1 correction out of 2 valid incorrect baselines => 50%
        assert history["metrics"]["correction_rate"] == 50.0

    @patch("common.parallel_runner.write_phase_output")
    def test_phase45_skips_records_missing_steered_correct(self, mock_write, tmp_path):
        """Records without 'steered_correct' should be skipped in metric calculation."""
        config = Config()

        gpu_data = {
            "correct_steering": {
                "optimal_coefficient": 10,
                "best_result": None,
                "search_history": [
                    {
                        "coefficient": 10,
                        "results": [
                            # Valid record
                            {"task_id": "t1", "baseline_passed": False,
                             "steered_correct": True, "code_similarity": 0.8},
                            # Missing steered_correct
                            {"task_id": "t2", "baseline_passed": False,
                             "code_similarity": 0.5},
                        ],
                    }
                ],
            }
        }

        (tmp_path / "coefficient_analysis_gpu0.json").write_text(json.dumps(gpu_data))

        _merge_phase4_5_json_results(tmp_path, n_gpus=1, config=config, phase_id="4.5")

        merged = json.loads((tmp_path / "coefficient_analysis.json").read_text())
        correct = merged["correct_steering"]
        history = correct["search_history"][0]

        # correction_rate from 1 valid record: 1/1 = 100%
        assert history["metrics"]["correction_rate"] == 100.0

    @patch("common.parallel_runner.write_phase_output")
    def test_phase45_valid_records_still_processed_with_mixed_data(self, mock_write, tmp_path):
        """Valid records should be correctly processed even when mixed with incomplete ones."""
        config = Config()

        gpu_data = {
            "incorrect_steering": {
                "optimal_coefficient": 20,
                "best_result": None,
                "search_history": [
                    {
                        "coefficient": 20,
                        "results": [
                            # Valid: baseline correct, steered incorrect -> corruption
                            {"task_id": "t1", "baseline_passed": True,
                             "steered_correct": False, "code_similarity": 0.3},
                            # Incomplete: missing both fields
                            {"task_id": "t2", "code_similarity": 0.5},
                            # Valid: baseline correct, steered correct -> no corruption
                            {"task_id": "t3", "baseline_passed": True,
                             "steered_correct": True, "code_similarity": 0.9},
                            # Incomplete: missing steered_correct
                            {"task_id": "t4", "baseline_passed": True,
                             "code_similarity": 0.7},
                        ],
                    }
                ],
            }
        }

        (tmp_path / "coefficient_analysis_gpu0.json").write_text(json.dumps(gpu_data))

        _merge_phase4_5_json_results(tmp_path, n_gpus=1, config=config, phase_id="4.5")

        merged = json.loads((tmp_path / "coefficient_analysis.json").read_text())
        incorrect = merged["incorrect_steering"]
        history = incorrect["search_history"][0]

        # 2 valid records with baseline_passed=True: t1 (corrupted), t3 (not corrupted)
        # corruption_rate = 1/2 = 50%
        assert history["metrics"]["corruption_rate"] == 50.0
        # n_problems should be total (all 4)
        assert history["n_problems"] == 4


# =============================================================================
# Fix 3: Multi-candidate merge by (layer, latent_idx) key
# =============================================================================

class TestMultiCandidateMergeByKey:
    """Test that multi-candidate merge matches candidates by (layer, latent_idx)
    rather than positional index, making it order-independent."""

    def test_same_order_merges_correctly(self):
        """Candidates in same order across GPUs should merge correctly."""
        from common.parallel_runner import _merge_phase4_8_results

        candidates = [
            {"layer": 10, "latent_idx": 100, "n_total": 5},
            {"layer": 12, "latent_idx": 200, "n_total": 3},
        ]

        gpu0 = {
            "correct": [
                {"layer": 10, "latent_idx": 100, "n_total": 5},
                {"layer": 12, "latent_idx": 200, "n_total": 3},
            ],
            "incorrect": [],
            "direction_source": "sae",
            "coefficients": {},
            "correction": [{"task_id": "t0", "baseline_passed": False, "steered_correct": True}],
            "corruption": [],
            "preservation": [],
            "n_problems": {"initially_correct": 0, "initially_incorrect": 1, "total": 1},
        }

        gpu1 = {
            "correct": [
                {"layer": 10, "latent_idx": 100, "n_total": 7},
                {"layer": 12, "latent_idx": 200, "n_total": 4},
            ],
            "incorrect": [],
            "direction_source": "sae",
            "coefficients": {},
            "correction": [{"task_id": "t1", "baseline_passed": False, "steered_correct": False}],
            "corruption": [],
            "preservation": [],
            "n_problems": {"initially_correct": 0, "initially_incorrect": 1, "total": 1},
        }

        # Write GPU files
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for i, data in enumerate([gpu0, gpu1]):
                (tmpdir / f"steering_effect_analysis_gpu{i}.json").write_text(json.dumps(data))

            with patch("common.parallel_runner.write_phase_output"):
                result = _merge_phase4_8_results(tmpdir, n_gpus=2, config=Config(), phase_id="4.8")

            merged = json.loads((tmpdir / "steering_effect_analysis.json").read_text())

            # n_total should be summed: 5+7=12 and 3+4=7
            assert merged["correct"][0]["n_total"] == 12
            assert merged["correct"][1]["n_total"] == 7

    def test_different_order_merges_by_key(self):
        """Candidates in different order across GPUs should still merge correctly by key."""
        from common.parallel_runner import _merge_phase4_8_results

        gpu0 = {
            "correct": [
                {"layer": 10, "latent_idx": 100, "n_total": 5},
                {"layer": 12, "latent_idx": 200, "n_total": 3},
            ],
            "incorrect": [],
            "direction_source": "sae",
            "coefficients": {},
            "correction": [{"task_id": "t0", "baseline_passed": False, "steered_correct": True}],
            "corruption": [],
            "preservation": [],
            "n_problems": {"initially_correct": 0, "initially_incorrect": 1, "total": 1},
        }

        # GPU 1 has candidates in REVERSED order
        gpu1 = {
            "correct": [
                {"layer": 12, "latent_idx": 200, "n_total": 4},
                {"layer": 10, "latent_idx": 100, "n_total": 7},
            ],
            "incorrect": [],
            "direction_source": "sae",
            "coefficients": {},
            "correction": [{"task_id": "t1", "baseline_passed": False, "steered_correct": False}],
            "corruption": [],
            "preservation": [],
            "n_problems": {"initially_correct": 0, "initially_incorrect": 1, "total": 1},
        }

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for i, data in enumerate([gpu0, gpu1]):
                (tmpdir / f"steering_effect_analysis_gpu{i}.json").write_text(json.dumps(data))

            with patch("common.parallel_runner.write_phase_output"):
                result = _merge_phase4_8_results(tmpdir, n_gpus=2, config=Config(), phase_id="4.8")

            merged = json.loads((tmpdir / "steering_effect_analysis.json").read_text())

            # Reference order is from GPU 0: L10_100 first, L12_200 second
            # Even though GPU 1 has reversed order, should merge by key
            assert merged["correct"][0]["layer"] == 10
            assert merged["correct"][0]["latent_idx"] == 100
            assert merged["correct"][0]["n_total"] == 12  # 5 + 7

            assert merged["correct"][1]["layer"] == 12
            assert merged["correct"][1]["latent_idx"] == 200
            assert merged["correct"][1]["n_total"] == 7  # 3 + 4


# =============================================================================
# Subprocess Orchestration Tests (M3)
# =============================================================================

class TestRunPhaseParallel:
    """Test run_phase_parallel orchestration logic."""

    def test_non_parallelizable_phase_raises(self):
        """Phases not in PARALLELIZABLE_PHASES should raise ValueError."""
        config = Config()
        with pytest.raises(ValueError, match="does not support parallel"):
            run_phase_parallel("99.99", config, n_gpus=2)

    @patch('common.parallel_runner._merge_parallel_results')
    @patch('common.parallel_runner.ProcessPoolExecutor')
    @patch('common.parallel_runner.mp')
    @patch('common.phase_discovery.get_phase_output_dir', return_value='/tmp/test_phase')
    def test_worker_exception_causes_failure(self, mock_get_dir, mock_mp, mock_executor_cls, mock_merge):
        """If any GPU worker fails, run_phase_parallel should raise RuntimeError."""
        # Setup mock context
        mock_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_ctx
        mock_manager = MagicMock()
        mock_ctx.Manager.return_value = mock_manager

        # Create futures that simulate one success and one failure
        future_success = Future()
        future_success.set_result({'gpu_id': 0, 'status': 'success', 'result': {}})

        future_fail = Future()
        future_fail.set_result({'gpu_id': 1, 'status': 'error', 'error': 'OOM'})

        # Setup executor mock
        mock_executor = MagicMock()
        mock_executor_cls.return_value.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.side_effect = [future_success, future_fail]

        config = Config()
        with patch('common.parallel_runner.Path.mkdir'):
            with pytest.raises(RuntimeError, match="GPU workers failed"):
                run_phase_parallel("1", config, n_gpus=2)

    @patch('common.parallel_runner._merge_parallel_results')
    @patch('common.parallel_runner.ProcessPoolExecutor')
    @patch('common.parallel_runner.mp')
    @patch('common.phase_discovery.get_phase_output_dir', return_value='/tmp/test_phase')
    def test_all_workers_succeed_merges(self, mock_get_dir, mock_mp, mock_executor_cls, mock_merge):
        """If all workers succeed, should proceed to merge."""
        mock_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_ctx
        mock_manager = MagicMock()
        mock_ctx.Manager.return_value = mock_manager

        future0 = Future()
        future0.set_result({'gpu_id': 0, 'status': 'success', 'result': {}})
        future1 = Future()
        future1.set_result({'gpu_id': 1, 'status': 'success', 'result': {}})

        mock_executor = MagicMock()
        mock_executor_cls.return_value.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.side_effect = [future0, future1]

        mock_merge.return_value = {'merged': True}

        config = Config()
        with patch('common.parallel_runner.Path.mkdir'):
            result = run_phase_parallel("1", config, n_gpus=2)

        mock_merge.assert_called_once()
        assert result == {'merged': True}


# =============================================================================
# General Parquet Merge Router Tests (M3 from code review)
# =============================================================================

class TestMergeParallelResultsParquetPath:
    """Test _merge_parallel_results general parquet merge edge cases."""

    def test_missing_gpu_files_raises_error(self, tmp_path):
        """If fewer GPU files than expected, should raise RuntimeError."""
        # Create only 2 of 4 expected GPU files
        for i in range(2):
            df = pd.DataFrame({'task_id': [f't{i}_0', f't{i}_1'], 'score': [0.5, 0.6]})
            df.to_parquet(tmp_path / f"results_gpu{i}.parquet", index=False)

        config = Config()
        with pytest.raises(RuntimeError, match="Only found 2/4 GPU result files"):
            _merge_parallel_results("3.6", str(tmp_path), n_gpus=4, config=config)

    def test_all_gpu_files_merge_correctly(self, tmp_path):
        """All GPU files present should merge and deduplicate."""
        for i in range(2):
            df = pd.DataFrame({
                'task_id': [f't{i}_0', f't{i}_1'],
                'baseline_passed': [True, False],
            })
            df.to_parquet(tmp_path / f"results_gpu{i}.parquet", index=False)

        config = Config()
        with patch('common.parallel_runner.write_phase_output'):
            result = _merge_parallel_results("3.6", str(tmp_path), n_gpus=2, config=config)

        assert result['total_rows'] == 4

    def test_old_merged_files_cleaned_up(self, tmp_path):
        """Old dataset_merged_*.parquet files should be removed before new merge."""
        # Create old merged file
        old_merged = tmp_path / "dataset_merged_20260101_000000.parquet"
        pd.DataFrame({'task_id': ['old']}).to_parquet(old_merged, index=False)

        # Create GPU result files
        for i in range(2):
            df = pd.DataFrame({'task_id': [f't{i}'], 'score': [0.5]})
            df.to_parquet(tmp_path / f"results_gpu{i}.parquet", index=False)

        config = Config()
        with patch('common.parallel_runner.write_phase_output'):
            _merge_parallel_results("3.6", str(tmp_path), n_gpus=2, config=config)

        # Old merged file should be gone
        assert not old_merged.exists()
        # New merged file should exist
        new_merged = list(tmp_path.glob("dataset_merged_*.parquet"))
        assert len(new_merged) == 1

    def test_deduplication_by_task_id(self, tmp_path):
        """Overlapping task_ids across GPUs should be deduplicated (keep last)."""
        # Both GPUs have task t_overlap (simulates checkpoint overlap)
        df0 = pd.DataFrame({'task_id': ['t_overlap', 't_0'], 'score': [0.1, 0.2]})
        df1 = pd.DataFrame({'task_id': ['t_overlap', 't_1'], 'score': [0.9, 0.8]})
        df0.to_parquet(tmp_path / "results_gpu0.parquet", index=False)
        df1.to_parquet(tmp_path / "results_gpu1.parquet", index=False)

        config = Config()
        with patch('common.parallel_runner.write_phase_output'):
            result = _merge_parallel_results("3.6", str(tmp_path), n_gpus=2, config=config)

        # Should have 3 unique tasks (t_overlap deduplicated)
        assert result['total_rows'] == 3
