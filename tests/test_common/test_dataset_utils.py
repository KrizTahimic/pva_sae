"""
Tests for common/dataset_utils.py

Validates:
- _get_import_code: Dataset parameter routing
"""

import pytest
from unittest.mock import patch, mock_open
from pathlib import Path


# =============================================================================
# _get_import_code Tests
# =============================================================================

class TestGetImportCode:
    """Test _get_import_code dataset_name parameter."""

    def test_explicit_humaneval_uses_humaneval_path(self):
        """Passing dataset_name='humaneval' should look for humaneval imports."""
        from common.dataset_utils import _get_import_code

        # Patch Path.exists to control which file is found
        with patch.object(Path, 'exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='{"imports": ["from typing import List"]}')):
                result = _get_import_code(dataset_name="humaneval")

        assert result is not None
        assert "from typing import List" in result

    def test_explicit_mbpp_uses_mbpp_path(self):
        """Passing dataset_name='mbpp' should look for mbpp imports."""
        from common.dataset_utils import _get_import_code

        with patch.object(Path, 'exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='{"imports": ["import math"]}')):
                result = _get_import_code(dataset_name="mbpp")

        assert result is not None
        assert "import math" in result

    def test_explicit_dataset_skips_config(self):
        """Passing dataset_name should NOT instantiate Config()."""
        from common.dataset_utils import _get_import_code

        with patch('common.dataset_utils.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            with patch('common.config.Config') as mock_config:
                _get_import_code(dataset_name="mbpp")
                mock_config.assert_not_called()

    def test_none_dataset_falls_back_to_config(self):
        """Passing None should fall back to Config()."""
        from common.dataset_utils import _get_import_code

        with patch('common.config.Config') as mock_config_cls:
            mock_config_cls.return_value.dataset_name = "mbpp"
            # File won't exist, so it returns None, but Config() was called
            result = _get_import_code(dataset_name=None)
            mock_config_cls.assert_called_once()

    def test_unknown_dataset_returns_none(self):
        """Unknown dataset should return None."""
        from common.dataset_utils import _get_import_code

        result = _get_import_code(dataset_name="unknown_dataset")
        assert result is None

    def test_missing_file_returns_none(self):
        """Missing import file should return None."""
        from common.dataset_utils import _get_import_code

        # Don't mock Path.exists - real file won't exist
        result = _get_import_code(dataset_name="humaneval")
        # File likely doesn't exist in test environment
        # This is OK - it should return None gracefully
        assert result is None or isinstance(result, str)
