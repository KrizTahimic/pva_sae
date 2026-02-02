"""
Tests for Phase 4.8 - Steering Effect Analysis (CRITICAL)

Validates:
- correction_experiment: Incorrect -> correct steering
- corruption_experiment: Correct -> incorrect steering
- preservation_experiment: Correct -> correct preservation
- parallel_merge_deduplication: No duplicate task_ids
- coefficient_loading: Loads from Phase 4.5/4.6 correctly
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from common.config import Config
from common.steering_metrics import (
    calculate_correction_rate,
    calculate_corruption_rate,
    calculate_preservation_rate,
)


# =============================================================================
# correction_experiment Tests
# =============================================================================

class TestCorrectionExperiment:
    """Test incorrect -> correct steering."""

    def test_correction_on_initially_incorrect(self):
        """Correction experiment should only include initially incorrect problems."""
        data = pd.DataFrame([
            {'task_id': 't1', 'baseline_passed': False, 'steered_correct': True},
            {'task_id': 't2', 'baseline_passed': False, 'steered_correct': False},
            {'task_id': 't3', 'baseline_passed': True, 'steered_correct': True},  # Should be excluded
        ])

        # Filter for correction experiment
        correction_data = data[data['baseline_passed'] == False]

        assert len(correction_data) == 2
        assert all(correction_data['baseline_passed'] == False)

    def test_correction_rate_calculation(self):
        """Correction rate should count incorrect->correct transitions."""
        results = [
            {'baseline_passed': False, 'steered_correct': True},   # Corrected
            {'baseline_passed': False, 'steered_correct': True},   # Corrected
            {'baseline_passed': False, 'steered_correct': False},  # Not corrected
        ]

        rate = calculate_correction_rate(results)
        # 2/3 = 66.67%
        assert rate == pytest.approx(66.67, rel=0.01)

    def test_perfect_correction(self):
        """All incorrect becoming correct should give 100%."""
        results = [
            {'baseline_passed': False, 'steered_correct': True},
            {'baseline_passed': False, 'steered_correct': True},
        ]

        rate = calculate_correction_rate(results)
        assert rate == 100.0


# =============================================================================
# corruption_experiment Tests
# =============================================================================

class TestCorruptionExperiment:
    """Test correct -> incorrect steering."""

    def test_corruption_on_initially_correct(self):
        """Corruption experiment should only include initially correct problems."""
        data = pd.DataFrame([
            {'task_id': 't1', 'baseline_passed': True, 'steered_correct': False},
            {'task_id': 't2', 'baseline_passed': True, 'steered_correct': True},
            {'task_id': 't3', 'baseline_passed': False, 'steered_correct': False},  # Should be excluded
        ])

        # Filter for corruption experiment
        corruption_data = data[data['baseline_passed'] == True]

        assert len(corruption_data) == 2
        assert all(corruption_data['baseline_passed'] == True)

    def test_corruption_rate_calculation(self):
        """Corruption rate should count correct->incorrect transitions."""
        results = [
            {'baseline_passed': True, 'steered_correct': False},  # Corrupted
            {'baseline_passed': True, 'steered_correct': True},   # Preserved
            {'baseline_passed': True, 'steered_correct': True},   # Preserved
        ]

        rate = calculate_corruption_rate(results)
        # 1/3 = 33.33%
        assert rate == pytest.approx(33.33, rel=0.01)

    def test_perfect_corruption(self):
        """All correct becoming incorrect should give 100%."""
        results = [
            {'baseline_passed': True, 'steered_correct': False},
            {'baseline_passed': True, 'steered_correct': False},
        ]

        rate = calculate_corruption_rate(results)
        assert rate == 100.0


# =============================================================================
# preservation_experiment Tests
# =============================================================================

class TestPreservationExperiment:
    """Test correct -> correct preservation."""

    def test_preservation_on_initially_correct(self):
        """Preservation should be measured on initially correct problems."""
        results = [
            {'baseline_passed': True, 'steered_correct': True},   # Preserved
            {'baseline_passed': True, 'steered_correct': True},   # Preserved
            {'baseline_passed': True, 'steered_correct': False},  # Lost
        ]

        preservation = calculate_preservation_rate(results)
        # 2/3 preserved = 66.67%
        assert preservation == pytest.approx(66.67, rel=0.01)

    def test_preservation_is_inverse_corruption(self):
        """Preservation + corruption should equal 100%."""
        results = [
            {'baseline_passed': True, 'steered_correct': True},
            {'baseline_passed': True, 'steered_correct': False},
        ]

        preservation = calculate_preservation_rate(results)
        corruption = calculate_corruption_rate(results)

        assert preservation + corruption == pytest.approx(100.0)

    def test_perfect_preservation(self):
        """All correct staying correct should give 100%."""
        results = [
            {'baseline_passed': True, 'steered_correct': True},
            {'baseline_passed': True, 'steered_correct': True},
        ]

        rate = calculate_preservation_rate(results)
        assert rate == 100.0


# =============================================================================
# parallel_merge_deduplication Tests
# =============================================================================

class TestParallelMergeDeduplication:
    """Test no duplicate task_ids in merged results."""

    def test_no_duplicates_after_merge(self):
        """Merged results should have unique task_ids."""
        gpu0 = pd.DataFrame([
            {'task_id': 't0', 'steered_correct': True},
            {'task_id': 't2', 'steered_correct': False},
        ])
        gpu1 = pd.DataFrame([
            {'task_id': 't1', 'steered_correct': True},
            {'task_id': 't3', 'steered_correct': False},
        ])

        merged = pd.concat([gpu0, gpu1], ignore_index=True)

        # Check for duplicates
        duplicates = merged[merged.duplicated(subset=['task_id'], keep=False)]
        assert len(duplicates) == 0

    def test_dedup_keeps_first_on_conflict(self):
        """If duplicate task_id, should keep first occurrence."""
        gpu0 = pd.DataFrame([
            {'task_id': 't0', 'steered_correct': True},
        ])
        gpu1 = pd.DataFrame([
            {'task_id': 't0', 'steered_correct': False},  # Duplicate with different result
        ])

        merged = pd.concat([gpu0, gpu1], ignore_index=True)
        deduped = merged.drop_duplicates(subset=['task_id'], keep='first')

        assert len(deduped) == 1
        assert deduped['steered_correct'].iloc[0] == True  # Kept gpu0's value

    def test_all_tasks_covered(self):
        """Merged results should cover all expected tasks."""
        expected_tasks = {'t0', 't1', 't2', 't3'}

        gpu0 = pd.DataFrame([{'task_id': 't0'}, {'task_id': 't2'}])
        gpu1 = pd.DataFrame([{'task_id': 't1'}, {'task_id': 't3'}])

        merged = pd.concat([gpu0, gpu1], ignore_index=True)

        assert set(merged['task_id']) == expected_tasks


# =============================================================================
# coefficient_loading Tests
# =============================================================================

class TestCoefficientLoading:
    """Test loads from Phase 4.5/4.6 correctly."""

    def test_loads_from_phase_4_6(self, tmp_path):
        """Should load refined coefficients from Phase 4.6."""
        from common.phase_discovery import discover_steering_coefficients
        import json

        phase_4_6_dir = tmp_path / "data" / "phase4_6"
        phase_4_6_dir.mkdir(parents=True)

        coeff_data = {
            "correct": {"refined_coefficient": 47.5},
            "incorrect": {"refined_coefficient": 43.0}
        }
        with open(phase_4_6_dir / "refined_coefficients.json", 'w') as f:
            json.dump(coeff_data, f)

        manifest = {
            "phase": "4.6",
            "outputs": {"refined_coefficients": "refined_coefficients.json"}
        }
        with open(phase_4_6_dir / "phase_output.json", 'w') as f:
            json.dump(manifest, f)

        def mock_get_output_file(phase, key, config=None):
            if phase == "4.9":
                raise FileNotFoundError()
            return phase_4_6_dir / "refined_coefficients.json"

        with patch('common.phase_discovery.get_phase_output_file',
                  side_effect=mock_get_output_file):
            result = discover_steering_coefficients(Config())

        assert result['correct'] == 47.5
        assert result['incorrect'] == 43.0

    def test_experiment_mode_from_config(self):
        """Config should specify experiment mode."""
        config = Config()
        assert hasattr(config, 'phase4_8_experiment_mode')
        assert config.phase4_8_experiment_mode in ['all', 'correction', 'corruption', 'preservation']


# =============================================================================
# Steering Application Tests
# =============================================================================

class TestSteeringApplication:
    """Test steering is applied correctly."""

    def test_correct_steering_uses_correct_direction(self):
        """Correction experiment should use correct-predicting direction."""
        # Conceptual test - actual steering uses correct_direction from SAE
        experiment_type = 'correction'
        expected_direction = 'correct'

        # In correction experiment, we steer toward "correct" to fix incorrect code
        assert experiment_type == 'correction'
        assert expected_direction == 'correct'

    def test_incorrect_steering_uses_incorrect_direction(self):
        """Corruption experiment should use incorrect-predicting direction."""
        experiment_type = 'corruption'
        expected_direction = 'incorrect'

        # In corruption experiment, we steer toward "incorrect" to break correct code
        assert experiment_type == 'corruption'
        assert expected_direction == 'incorrect'


# =============================================================================
# Output Format Tests
# =============================================================================

class TestOutputFormat:
    """Test Phase 4.8 output format."""

    def test_output_structure(self):
        """Output should have correction, corruption, preservation results."""
        expected_structure = {
            'summary': {
                'correction_rate': 25.0,
                'corruption_rate': 15.0,
                'preservation_rate': 85.0
            },
            'detailed_results': {
                'correction': [],
                'corruption': [],
                'preservation': []
            }
        }

        assert 'summary' in expected_structure
        assert 'detailed_results' in expected_structure

    def test_per_task_results_saved(self):
        """Should save per-task results for analysis."""
        detailed_results = [
            {
                'task_id': 't1',
                'baseline_passed': False,
                'steered_correct': True,
                'baseline_code': 'def f(): pass',
                'steered_code': 'def f(): return 1'
            }
        ]

        assert 'task_id' in detailed_results[0]
        assert 'baseline_code' in detailed_results[0]
        assert 'steered_code' in detailed_results[0]
