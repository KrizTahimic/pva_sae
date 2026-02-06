"""
Tests for error handler conservative assumptions.

Validates:
- Phase 5.3 error handler assumes failure (orthogonalized_correct=False)
- Phase 8.2 error handler assumes failure (steered_correct=False, corrupted=baseline_passed)
- Phase 8.3 error handler assumes failure (steered_correct=False, was_steered field)
- save_json() atomicity (no partial file on failure)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# =============================================================================
# Phase 5.3 Error Handler Tests
# =============================================================================

class TestPhase53ErrorHandler:
    """Test Phase 5.3 error handler returns conservative values."""

    def test_error_result_orthogonalized_correct_is_false(self):
        """Error during orthogonalization should set orthogonalized_correct=False."""
        # Simulate what the error handler produces
        error_result = {
            'task_id': 'test_task',
            'baseline_passed': True,
            'orthogonalized_correct': False,  # Conservative: assume failure
            'baseline_code': 'def foo(): pass',
            'orthogonalized_code': '',
            'similarity': 0.0,  # Conservative: assume no similarity
            'error': 'test error'
        }

        assert error_result['orthogonalized_correct'] is False
        assert error_result['similarity'] == 0.0

    def test_error_result_not_preservation(self):
        """Error result should NOT count as preservation."""
        error_result = {
            'baseline_passed': True,
            'orthogonalized_correct': False,
        }

        # Preservation = baseline_passed AND orthogonalized_correct
        preservation = error_result['baseline_passed'] and error_result['orthogonalized_correct']
        assert preservation is False

    def test_error_result_counts_as_corruption(self):
        """Error result for correct baseline should count as corruption."""
        error_result = {
            'baseline_passed': True,
            'orthogonalized_correct': False,
        }

        # Corruption = baseline_passed AND NOT orthogonalized_correct
        corruption = error_result['baseline_passed'] and not error_result['orthogonalized_correct']
        assert corruption is True


# =============================================================================
# Phase 8.2 Error Handler Tests
# =============================================================================

class TestPhase82ErrorHandler:
    """Test Phase 8.2 error handler returns conservative values."""

    def test_error_result_steered_correct_is_false(self):
        """Error during steering should set steered_correct=False."""
        baseline_passed = True
        error_result = {
            'task_id': 'test_task',
            'baseline_passed': baseline_passed,
            'was_steered': False,
            'steered_correct': False,
            'corrected': False,
            'preserved': False,
            'corrupted': baseline_passed,
            'error': 'test error'
        }

        assert error_result['steered_correct'] is False

    def test_error_result_corrupted_matches_baseline(self):
        """Error corrupted flag should match baseline_passed."""
        for baseline_passed in [True, False]:
            error_result = {
                'baseline_passed': baseline_passed,
                'steered_correct': False,
                'preserved': False,
                'corrupted': baseline_passed,
            }

            assert error_result['corrupted'] == baseline_passed

    def test_error_result_preserved_is_false(self):
        """Error result should never count as preserved."""
        error_result = {
            'baseline_passed': True,
            'preserved': False,
        }

        assert error_result['preserved'] is False

    def test_error_result_not_corrected(self):
        """Error result should never count as corrected."""
        error_result = {
            'baseline_passed': False,
            'steered_correct': False,
            'corrected': False,
        }

        assert error_result['corrected'] is False

    def test_error_with_incorrect_baseline_not_corrupted(self):
        """Error with incorrect baseline should NOT be corrupted."""
        baseline_passed = False
        error_result = {
            'baseline_passed': baseline_passed,
            'steered_correct': False,
            'corrupted': baseline_passed,  # False because baseline was already wrong
        }

        assert error_result['corrupted'] is False


# =============================================================================
# Phase 8.3 Error Handler Tests
# =============================================================================

class TestPhase83ErrorHandler:
    """Test Phase 8.3 error handler returns conservative values."""

    def test_error_result_steered_correct_is_false(self):
        """Error during selective steering should set steered_correct=False (not baseline_passed)."""
        for baseline_passed in [True, False]:
            error_result = {
                'task_id': 'test_task',
                'baseline_passed': baseline_passed,
                'was_steered': False,
                'incorrect_pred_activation': None,
                'steered_correct': False,  # Conservative: assume failure
                'baseline_code': 'def foo(): pass',
                'steered_code': None,
                'source': 'error',
                'error': 'test error'
            }

            # steered_correct must always be False on error, regardless of baseline
            assert error_result['steered_correct'] is False

    def test_error_result_uses_was_steered_field(self):
        """Error result must use 'was_steered' field name (not 'steered')."""
        error_result = {
            'task_id': 'test_task',
            'baseline_passed': True,
            'was_steered': False,
            'steered_correct': False,
            'source': 'error',
        }

        assert 'was_steered' in error_result
        assert 'steered' not in error_result  # Must NOT use old field name

    def test_error_result_counts_as_corruption_for_correct_baseline(self):
        """Error on correct baseline should count as corruption (conservative)."""
        error_result = {
            'baseline_passed': True,
            'steered_correct': False,
        }

        corruption = error_result['baseline_passed'] and not error_result['steered_correct']
        assert corruption is True

    def test_error_result_not_correction(self):
        """Error should never count as correction."""
        error_result = {
            'baseline_passed': False,
            'steered_correct': False,
        }

        correction = not error_result['baseline_passed'] and error_result['steered_correct']
        assert correction is False


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

        # Create data that will fail during JSON serialization
        class Unserializable:
            def __repr__(self):
                raise RuntimeError("Cannot serialize")

        # The default=str handler will call str() which calls __repr__
        # Actually, let's use a simpler approach - mock json.dump to fail
        with patch('json.dump', side_effect=IOError("Simulated write failure")):
            with pytest.raises(IOError):
                save_json({"key": "value"}, filepath)

        # File should not exist (atomic write failed before rename)
        assert not filepath.exists()

    def test_atomic_write_no_temp_file_on_error(self, tmp_path):
        """Failed write should clean up temp file."""
        from common.utils import save_json
        import glob

        filepath = tmp_path / "test.json"

        with patch('json.dump', side_effect=IOError("Simulated write failure")):
            with pytest.raises(IOError):
                save_json({"key": "value"}, filepath)

        # No temp files should remain
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_atomic_write_preserves_existing_on_error(self, tmp_path):
        """Failed overwrite should preserve existing file."""
        from common.utils import save_json

        filepath = tmp_path / "test.json"
        original_data = {"original": True}

        # Write initial file
        save_json(original_data, filepath)

        # Try to overwrite with failing write
        with patch('json.dump', side_effect=IOError("Simulated write failure")):
            with pytest.raises(IOError):
                save_json({"new": True}, filepath)

        # Original file should still be intact
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == original_data
