"""
Tests for common/logging.py

Validates logging infrastructure: LoggingManager, get_logger, tqdm_with_logging.
"""

import pytest
import logging
import os
import tempfile
from unittest.mock import patch, MagicMock

from common.logging import LoggingManager, get_logger, tqdm_with_logging, set_logging_phase


class TestLoggingManager:
    """Test LoggingManager initialization and handler setup."""

    def test_init_defaults(self):
        """Should initialize with sensible defaults."""
        mgr = LoggingManager()
        assert mgr.phase is None
        assert mgr.gpu_id is None
        assert mgr.log_level == logging.INFO
        assert mgr.log_to_file is True
        assert mgr.log_to_console is True

    def test_init_with_phase(self):
        """Should accept phase and gpu_id."""
        mgr = LoggingManager(phase="3.5", gpu_id=2)
        assert mgr.phase == "3.5"
        assert mgr.gpu_id == 2

    def test_no_file_handler_without_phase(self):
        """Should skip file creation when phase is None."""
        mgr = LoggingManager(phase=None, log_to_console=False)
        mgr._initialize_handlers()
        assert mgr.file_handler is None

    def test_file_handler_with_phase(self):
        """Should create file handler when phase is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = LoggingManager(phase="1.0", log_dir=tmpdir, log_to_console=False)
            mgr._initialize_handlers()
            assert mgr.file_handler is not None
            assert mgr.log_file is not None
            assert "phase1_0" in mgr.log_file

    def test_gpu_id_in_filename(self):
        """GPU ID should appear in log filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = LoggingManager(phase="4.8", gpu_id=3, log_dir=tmpdir, log_to_console=False)
            mgr._initialize_handlers()
            assert "gpu3" in mgr.log_file

    def test_handlers_initialized_once(self):
        """_initialize_handlers should be idempotent."""
        mgr = LoggingManager(phase=None, log_to_console=False)
        mgr._initialize_handlers()
        mgr._initialize_handlers()  # Second call should be no-op
        assert mgr._initialized is True


class TestSetupLogging:
    """Test logger creation via setup_logging."""

    def test_returns_logger(self):
        """setup_logging should return a logging.Logger instance."""
        mgr = LoggingManager(phase=None, log_to_console=False, log_to_file=False)
        logger = mgr.setup_logging("test_module")
        assert isinstance(logger, logging.Logger)
        assert "test_module" in logger.name

    def test_module_name_filter(self):
        """Logger should have ModuleNameFilter that sets module_name on records."""
        mgr = LoggingManager(phase=None, log_to_console=False, log_to_file=False)
        logger = mgr.setup_logging("my_module")
        # Create a log record and check filter adds module_name
        record = logging.LogRecord(
            name=logger.name, level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None
        )
        for f in logger.filters:
            f.filter(record)
        assert hasattr(record, 'module_name')
        assert record.module_name == "my_module"


class TestGetLogger:
    """Test the get_logger convenience function."""

    def test_returns_logger(self):
        """get_logger should return a working logger."""
        logger = get_logger("test_get_logger")
        assert isinstance(logger, logging.Logger)

    def test_phase_context(self):
        """set_logging_phase should affect subsequent get_logger calls."""
        # Reset global state
        set_logging_phase(None)
        logger1 = get_logger("module_a")

        set_logging_phase("5.3")
        logger2 = get_logger("module_b")

        # Both should be valid loggers
        assert isinstance(logger1, logging.Logger)
        assert isinstance(logger2, logging.Logger)

        # Cleanup
        set_logging_phase(None)

    def test_invalid_log_level_raises(self):
        """Invalid LOG_LEVEL env var should raise ValueError."""
        # Clear cached managers to force re-creation
        import common.logging as log_module
        original_managers = log_module._phase_managers.copy()
        # Use a unique cache key to force creation
        with patch.dict(os.environ, {'LOG_LEVEL': 'INVALID'}):
            log_module._phase_managers.clear()
            with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
                get_logger("test_invalid", phase="unique_test_phase")
        # Restore
        log_module._phase_managers = original_managers


class TestTqdmWithLogging:
    """Test tqdm wrapper with milestone logging."""

    def test_yields_all_items(self):
        """Should yield all items from the iterable."""
        logger = get_logger("test_tqdm")
        items = list(range(10))
        result = list(tqdm_with_logging(items, logger, desc="Test", disable=True))
        assert result == items

    def test_empty_iterable(self):
        """Should handle empty iterables."""
        logger = get_logger("test_tqdm_empty")
        result = list(tqdm_with_logging([], logger, desc="Empty", disable=True))
        assert result == []

    def test_custom_milestones(self):
        """Should accept custom milestone percentages."""
        logger = get_logger("test_tqdm_milestones")
        items = list(range(100))
        result = list(tqdm_with_logging(
            items, logger, desc="Custom", milestones=[50, 100], disable=True
        ))
        assert len(result) == 100


class TestLogPhaseEvents:
    """Test phase lifecycle logging methods."""

    def test_log_experiment_info(self):
        """log_experiment_info should not raise."""
        mgr = LoggingManager(phase=None, log_to_console=False, log_to_file=False)
        mgr.logger = mgr.setup_logging("test")
        mgr.log_experiment_info({"model": "gemma-2b", "dataset": "mbpp"})

    def test_log_phase_start(self):
        """log_phase_start should not raise."""
        mgr = LoggingManager(phase=None, log_to_console=False, log_to_file=False)
        mgr.logger = mgr.setup_logging("test")
        mgr.log_phase_start("Test Phase", total_items=100)

    def test_log_phase_end(self):
        """log_phase_end should not raise."""
        mgr = LoggingManager(phase=None, log_to_console=False, log_to_file=False)
        mgr.logger = mgr.setup_logging("test")
        mgr.log_phase_end("Test Phase", duration=10.5, success_count=90, error_count=10)
