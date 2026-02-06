"""
Tests for common/selective_steering.py

Validates:
- R1: SteeringState is importable from common.selective_steering
- SteeringState initializes correctly
- Phase 8.2 and 8.3 use the common version (not local definitions)
"""

import pytest
import inspect


class TestSteeringStateImport:
    """Verify SteeringState is properly shared between phases."""

    def test_importable_from_common(self):
        """SteeringState should be importable from common.selective_steering."""
        from common.selective_steering import SteeringState
        assert SteeringState is not None

    def test_initialization(self):
        """SteeringState should initialize with expected defaults."""
        from common.selective_steering import SteeringState
        state = SteeringState(prompt_length=42)
        assert state.prompt_length == 42
        assert state.first_token_checked is False
        assert state.incorrect_pred_activation is None
        assert state.should_steer is False

    def test_phase_8_2_uses_common_version(self):
        """Phase 8.2 should import SteeringState from common, not define locally."""
        import phase8_2_threshold_optimizer.threshold_optimizer as mod
        source = inspect.getsource(mod)
        assert "from common.selective_steering import SteeringState" in source
        assert "class SteeringState:" not in source

    def test_phase_8_3_uses_common_version(self):
        """Phase 8.3 should import SteeringState from common, not define locally."""
        import phase8_3_selective_steering.selective_steering_analyzer as mod
        source = inspect.getsource(mod)
        assert "from common.selective_steering import SteeringState" in source
        assert "class SteeringState:" not in source

    def test_same_class_used_in_both_phases(self):
        """Both phases should reference the exact same class object."""
        from phase8_2_threshold_optimizer.threshold_optimizer import SteeringState as SS82
        from phase8_3_selective_steering.selective_steering_analyzer import SteeringState as SS83
        from common.selective_steering import SteeringState as SSCommon
        assert SS82 is SSCommon
        assert SS83 is SSCommon
