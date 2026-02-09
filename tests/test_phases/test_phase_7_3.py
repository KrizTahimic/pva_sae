"""
Tests for Phase 7.3 - Instruct Baseline Runner

Validates:
- _discover_best_layers uses Phase 4.9 (load_phase4_9_best_latent)
- Error propagation when Phase 4.9 missing
"""

import pytest
from unittest.mock import patch, MagicMock
from common.config import Config
from phase7_3_instruct_baseline.instruct_baseline_runner import InstructBaselineRunner


class TestDiscoverBestLayers:
    """Test _discover_best_layers uses Phase 4.9."""

    def test_loads_from_phase4_9(self):
        """Should call load_phase4_9_best_latent and return correct dict."""
        selection = {
            "correct": {"rank": 2, "layer": 15, "latent_idx": 12809, "refined_coefficient": 62},
            "incorrect": {"rank": 0, "layer": 18, "latent_idx": 4612, "refined_coefficient": 25},
        }

        runner = object.__new__(InstructBaselineRunner)
        runner.config = Config()

        with patch('common.steering_setup.load_phase4_9_best_latent',
                   return_value=selection):
            result = runner._discover_best_layers()

        assert result['correct'] == 15
        assert result['correct_latent_idx'] == 12809
        assert result['incorrect'] == 18
        assert result['incorrect_latent_idx'] == 4612

    def test_missing_phase4_9_raises(self):
        """Should propagate FileNotFoundError when Phase 4.9 not found."""
        runner = object.__new__(InstructBaselineRunner)
        runner.config = Config()

        with patch('common.steering_setup.load_phase4_9_best_latent',
                   side_effect=FileNotFoundError("Phase 4.9 output not found")):
            with pytest.raises(FileNotFoundError, match="Phase 4.9"):
                runner._discover_best_layers()
