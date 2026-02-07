"""
Tests for common/phase_runner.py

Validates:
- run_phase: Correct routing to class-based and function-based runners
- run_phase: SPECIAL_PHASES raise ValueError
- run_phase: ImportError handling when module doesn't exist
- run_phase: AttributeError handling when runner class doesn't exist
- can_use_generic_runner: Correct result for special and non-special phases
- get_special_phases: Returns a copy, not the original dict
"""

import pytest
from unittest.mock import MagicMock, patch

from common.config import Config
from common.phase_registry import PhaseInfo
from common.phase_runner import (
    run_phase,
    can_use_generic_runner,
    get_special_phases,
    SPECIAL_PHASES,
)


# =============================================================================
# Helper fixtures
# =============================================================================

@pytest.fixture
def config():
    """Minimal Config for testing."""
    return Config()


@pytest.fixture
def class_phase_info():
    """PhaseInfo with runner_type='class'."""
    return PhaseInfo(
        id="99.1",
        name="Test Class Phase",
        output_dir="data/phase99_1",
        module="fake_module.runner",
        runner="FakeClassRunner",
        runner_type="class",
        category="test",
    )


@pytest.fixture
def function_phase_info():
    """PhaseInfo with runner_type='function'."""
    return PhaseInfo(
        id="99.2",
        name="Test Function Phase",
        output_dir="data/phase99_2",
        module="fake_module.func_runner",
        runner="run_fake_phase",
        runner_type="function",
        category="test",
    )


# =============================================================================
# run_phase: Class-based runner routing
# =============================================================================

class TestRunPhaseClassRunner:
    """Test that run_phase correctly routes to class-based runners."""

    @patch("common.phase_runner.get_logger")
    @patch("common.phase_runner.importlib.import_module")
    @patch("common.phase_runner.get_phase")
    def test_class_runner_instantiates_and_calls_run(
        self, mock_get_phase, mock_import, mock_logger, config, class_phase_info
    ):
        """Class-based runner: instantiate with config, call .run()."""
        mock_get_phase.return_value = class_phase_info
        mock_logger.return_value = MagicMock()

        # Create mock module with a mock runner class
        mock_runner_class = MagicMock()
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = {"status": "ok"}
        mock_runner_class.return_value = mock_runner_instance

        mock_module = MagicMock()
        mock_module.FakeClassRunner = mock_runner_class
        mock_import.return_value = mock_module

        result = run_phase("99.1", config, "cpu")

        mock_runner_class.assert_called_once_with(config)
        mock_runner_instance.run.assert_called_once()
        assert result == {"status": "ok"}

    @patch("common.phase_runner.get_logger")
    @patch("common.phase_runner.importlib.import_module")
    @patch("common.phase_runner.get_phase")
    def test_class_runner_returns_result(
        self, mock_get_phase, mock_import, mock_logger, config, class_phase_info
    ):
        """Class-based runner should forward the return value from .run()."""
        mock_get_phase.return_value = class_phase_info
        mock_logger.return_value = MagicMock()

        mock_runner_class = MagicMock()
        mock_runner_class.return_value.run.return_value = 42
        mock_module = MagicMock()
        mock_module.FakeClassRunner = mock_runner_class
        mock_import.return_value = mock_module

        result = run_phase("99.1", config, "cpu")
        assert result == 42


# =============================================================================
# run_phase: Function-based runner routing
# =============================================================================

class TestRunPhaseFunctionRunner:
    """Test that run_phase correctly routes to function-based runners."""

    @patch("common.phase_runner.get_logger")
    @patch("common.phase_runner.importlib.import_module")
    @patch("common.phase_runner.get_phase")
    def test_function_runner_called_with_config(
        self, mock_get_phase, mock_import, mock_logger, config, function_phase_info
    ):
        """Function-based runner: called directly with config."""
        mock_get_phase.return_value = function_phase_info
        mock_logger.return_value = MagicMock()

        mock_func = MagicMock(return_value="done")
        mock_module = MagicMock()
        mock_module.run_fake_phase = mock_func
        mock_import.return_value = mock_module

        result = run_phase("99.2", config, "cpu")

        mock_func.assert_called_once_with(config)
        assert result == "done"

    @patch("common.phase_runner.get_logger")
    @patch("common.phase_runner.importlib.import_module")
    @patch("common.phase_runner.get_phase")
    def test_function_runner_returns_none(
        self, mock_get_phase, mock_import, mock_logger, config, function_phase_info
    ):
        """Function-based runner returning None is fine."""
        mock_get_phase.return_value = function_phase_info
        mock_logger.return_value = MagicMock()

        mock_func = MagicMock(return_value=None)
        mock_module = MagicMock()
        mock_module.run_fake_phase = mock_func
        mock_import.return_value = mock_module

        result = run_phase("99.2", config, "cpu")
        assert result is None


# =============================================================================
# run_phase: SPECIAL_PHASES raise ValueError
# =============================================================================

class TestRunPhaseSpecialPhases:
    """Test that SPECIAL_PHASES raise ValueError."""

    def test_special_phase_raises_valueerror(self, config):
        """Running a special phase should raise ValueError."""
        for phase_id in SPECIAL_PHASES:
            with pytest.raises(ValueError, match="requires special handling"):
                run_phase(phase_id, config, "cpu")

    def test_special_phase_error_includes_phase_id(self, config):
        """Error message should include the phase ID."""
        for phase_id in SPECIAL_PHASES:
            with pytest.raises(ValueError, match=phase_id):
                run_phase(phase_id, config, "cpu")


# =============================================================================
# run_phase: ImportError handling
# =============================================================================

class TestRunPhaseImportError:
    """Test ImportError handling when module doesn't exist."""

    @patch("common.phase_runner.get_logger")
    @patch("common.phase_runner.importlib.import_module")
    @patch("common.phase_runner.get_phase")
    def test_import_error_raised(
        self, mock_get_phase, mock_import, mock_logger, config, class_phase_info
    ):
        """ImportError should be raised when module cannot be imported."""
        mock_get_phase.return_value = class_phase_info
        mock_logger.return_value = MagicMock()
        mock_import.side_effect = ImportError("No module named 'fake_module'")

        with pytest.raises(ImportError, match="Cannot import"):
            run_phase("99.1", config, "cpu")

    @patch("common.phase_runner.get_logger")
    @patch("common.phase_runner.importlib.import_module")
    @patch("common.phase_runner.get_phase")
    def test_import_error_includes_module_name(
        self, mock_get_phase, mock_import, mock_logger, config, class_phase_info
    ):
        """ImportError message should reference the module name."""
        mock_get_phase.return_value = class_phase_info
        mock_logger.return_value = MagicMock()
        mock_import.side_effect = ImportError("No module named 'fake_module'")

        with pytest.raises(ImportError, match="fake_module.runner"):
            run_phase("99.1", config, "cpu")


# =============================================================================
# run_phase: AttributeError handling
# =============================================================================

class TestRunPhaseAttributeError:
    """Test AttributeError handling when runner class doesn't exist."""

    @patch("common.phase_runner.get_logger")
    @patch("common.phase_runner.importlib.import_module")
    @patch("common.phase_runner.get_phase")
    def test_attribute_error_raised(
        self, mock_get_phase, mock_import, mock_logger, config, class_phase_info
    ):
        """AttributeError should be raised when runner not found in module."""
        mock_get_phase.return_value = class_phase_info
        mock_logger.return_value = MagicMock()

        mock_module = MagicMock(spec=[])  # Empty spec = no attributes
        mock_import.return_value = mock_module

        with pytest.raises(AttributeError, match="Cannot find"):
            run_phase("99.1", config, "cpu")

    @patch("common.phase_runner.get_logger")
    @patch("common.phase_runner.importlib.import_module")
    @patch("common.phase_runner.get_phase")
    def test_attribute_error_includes_runner_name(
        self, mock_get_phase, mock_import, mock_logger, config, class_phase_info
    ):
        """AttributeError message should reference the runner name."""
        mock_get_phase.return_value = class_phase_info
        mock_logger.return_value = MagicMock()

        mock_module = MagicMock(spec=[])
        mock_import.return_value = mock_module

        with pytest.raises(AttributeError, match="FakeClassRunner"):
            run_phase("99.1", config, "cpu")


# =============================================================================
# run_phase: Unknown runner_type
# =============================================================================

class TestRunPhaseUnknownRunnerType:
    """Test that unknown runner_type raises ValueError."""

    @patch("common.phase_runner.get_logger")
    @patch("common.phase_runner.importlib.import_module")
    @patch("common.phase_runner.get_phase")
    def test_unknown_runner_type_raises(
        self, mock_get_phase, mock_import, mock_logger, config
    ):
        """Unknown runner_type should raise ValueError."""
        bad_phase = PhaseInfo(
            id="99.9",
            name="Bad Runner Type",
            output_dir="data/phase99_9",
            module="fake_module.bad",
            runner="BadRunner",
            runner_type="unknown_type",
            category="test",
        )
        mock_get_phase.return_value = bad_phase
        mock_logger.return_value = MagicMock()

        mock_module = MagicMock()
        mock_import.return_value = mock_module

        with pytest.raises(ValueError, match="Unknown runner_type"):
            run_phase("99.9", config, "cpu")


# =============================================================================
# can_use_generic_runner Tests
# =============================================================================

class TestCanUseGenericRunner:
    """Test with special and non-special phases."""

    def test_special_phase_returns_false(self):
        """Special phases should return False."""
        for phase_id in SPECIAL_PHASES:
            assert can_use_generic_runner(phase_id) is False

    def test_non_special_phase_returns_true(self):
        """Non-special phases should return True."""
        assert can_use_generic_runner("0") is True
        assert can_use_generic_runner("1") is True
        assert can_use_generic_runner("4.8") is True

    def test_nonexistent_phase_returns_true(self):
        """Phases not in SPECIAL_PHASES return True (registry check is separate)."""
        assert can_use_generic_runner("999") is True


# =============================================================================
# get_special_phases Tests
# =============================================================================

class TestGetSpecialPhases:
    """Test it returns a copy."""

    def test_returns_dict(self):
        """Should return a dict."""
        result = get_special_phases()
        assert isinstance(result, dict)

    def test_returns_copy_not_original(self):
        """Modifying the returned dict should not affect SPECIAL_PHASES."""
        result = get_special_phases()
        original_len = len(SPECIAL_PHASES)
        result["999"] = "injected"
        assert len(SPECIAL_PHASES) == original_len
        assert "999" not in SPECIAL_PHASES

    def test_contains_expected_entries(self):
        """Returned dict should match SPECIAL_PHASES content."""
        result = get_special_phases()
        assert result == SPECIAL_PHASES

    def test_values_are_strings(self):
        """Each value should be a reason string."""
        result = get_special_phases()
        for phase_id, reason in result.items():
            assert isinstance(phase_id, str)
            assert isinstance(reason, str)
            assert len(reason) > 0
