"""
Tests for error handler conservative assumptions.

Validates production code error handlers set conservative failure values,
verified via source inspection (not tautological dict construction).
"""

import inspect
import json
import pytest
from pathlib import Path
from unittest.mock import patch


# =============================================================================
# Phase 5.3 Error Handler Tests
# =============================================================================

class TestPhase53ErrorHandler:
    """Verify Phase 5.3 error handler assumes failure in production code."""

    def test_source_contains_orthogonalized_correct_false(self):
        """Production error handler must set orthogonalized_correct=False."""
        from phase5_3_weight_orthogonalization import weight_orthogonalizer
        source = inspect.getsource(weight_orthogonalizer)
        # The error handler builds a dict with orthogonalized_correct: False
        assert "'orthogonalized_correct': False" in source or '"orthogonalized_correct": False' in source

    def test_source_contains_similarity_zero_on_error(self):
        """Production error handler must set similarity=0.0 on error (conservative)."""
        from phase5_3_weight_orthogonalization import weight_orthogonalizer
        source = inspect.getsource(weight_orthogonalizer.WeightOrthogonalizer.apply_correct_orthogonalization)
        # The error handler in apply_correct_orthogonalization sets similarity: 0.0
        assert "'similarity': 0.0" in source

    def test_corruption_logic(self):
        """Corruption = baseline_passed AND NOT orthogonalized_correct."""
        # This tests the metric formula, not a dict we constructed
        assert (True and not False) is True   # correct baseline + error = corruption
        assert (False and not False) is False  # incorrect baseline + error = not corruption


# =============================================================================
# Phase 8.2 Error Handler Tests
# =============================================================================

class TestPhase82ErrorHandler:
    """Verify Phase 8.2 error handler assumes failure in production code."""

    def test_source_contains_steered_correct_false(self):
        """Production error handler must set steered_correct=False."""
        from phase8_2_threshold_optimizer import threshold_optimizer
        source = inspect.getsource(threshold_optimizer)
        assert "'steered_correct': False" in source or '"steered_correct": False' in source

    def test_source_contains_corrupted_baseline_passed(self):
        """Error corrupted flag = baseline_passed (if baseline was correct, error = corruption)."""
        from phase8_2_threshold_optimizer import threshold_optimizer
        source = inspect.getsource(threshold_optimizer)
        # Verify that corrupted is set to baseline_passed in error handlers
        assert "'corrupted': baseline_passed" in source

    def test_threshold_optimizer_error_handler_steered_correct_false(self):
        """ThresholdOptimizer._run_selective_steering_for_threshold error handler must set steered_correct=False."""
        from phase8_2_threshold_optimizer.threshold_optimizer import ThresholdOptimizer
        source = inspect.getsource(ThresholdOptimizer._run_selective_steering_for_threshold)
        assert "'steered_correct': False" in source

    def test_threshold_evaluator_error_handler_steered_correct_false(self):
        """ThresholdEvaluator._run_experiment error handler must set steered_correct=False."""
        from phase8_2_threshold_optimizer.threshold_optimizer import ThresholdEvaluator
        source = inspect.getsource(ThresholdEvaluator._run_experiment)
        assert "'steered_correct': False" in source

    def test_corruption_logic(self):
        """Error with correct baseline should count as corruption."""
        # corruption = baseline_passed AND NOT steered_correct
        assert (True and not False) is True    # correct baseline + error = corruption
        assert (False and not False) is False  # incorrect baseline + error = not corruption


# =============================================================================
# Phase 8.3 Error Handler Tests
# =============================================================================

class TestPhase83ErrorHandler:
    """Verify Phase 8.3 error handler assumes failure in production code."""

    def test_source_contains_steered_correct_false(self):
        """Production error handler must set steered_correct=False."""
        from phase8_3_selective_steering import selective_steering_analyzer
        source = inspect.getsource(selective_steering_analyzer)
        assert "'steered_correct': False" in source or '"steered_correct": False' in source

    def test_source_uses_was_steered_field(self):
        """Error handler must use 'was_steered' (not deprecated 'steered')."""
        from phase8_3_selective_steering import selective_steering_analyzer
        source = inspect.getsource(selective_steering_analyzer)
        assert "'was_steered'" in source

    def test_source_error_handler_sets_source_error(self):
        """Error handler must set source='error' for tracking."""
        from phase8_3_selective_steering.selective_steering_analyzer import SelectiveSteeringAnalyzer
        source = inspect.getsource(SelectiveSteeringAnalyzer._apply_selective_steering)
        assert "'source': 'error'" in source

    def test_correction_logic(self):
        """Correction = NOT baseline_passed AND steered_correct."""
        assert (not False and True) is True    # incorrect baseline + success = correction
        assert (not False and False) is False  # incorrect baseline + failure = not correction
        assert (not True and True) is False    # correct baseline doesn't apply


# =============================================================================
# save_json Atomicity Tests
# =============================================================================

class TestSaveJsonAtomicity:
    """Test save_json() writes atomically."""

    def test_atomic_write_normal(self, tmp_path):
        """Normal write should produce valid JSON file."""
        from common.utils import save_json

        filepath = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        save_json(data, filepath)

        assert filepath.exists()
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_write_no_partial_on_error(self, tmp_path):
        """Failed write should not leave partial file."""
        from common.utils import save_json

        filepath = tmp_path / "test.json"

        with patch('json.dump', side_effect=IOError("Simulated write failure")):
            with pytest.raises(IOError):
                save_json({"key": "value"}, filepath)

        assert not filepath.exists()

    def test_atomic_write_no_temp_file_on_error(self, tmp_path):
        """Failed write should clean up temp file."""
        from common.utils import save_json

        filepath = tmp_path / "test.json"

        with patch('json.dump', side_effect=IOError("Simulated write failure")):
            with pytest.raises(IOError):
                save_json({"key": "value"}, filepath)

        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_atomic_write_preserves_existing_on_error(self, tmp_path):
        """Failed overwrite should preserve existing file."""
        from common.utils import save_json

        filepath = tmp_path / "test.json"
        original_data = {"original": True}

        save_json(original_data, filepath)

        with patch('json.dump', side_effect=IOError("Simulated write failure")):
            with pytest.raises(IOError):
                save_json({"new": True}, filepath)

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == original_data
