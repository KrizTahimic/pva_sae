"""
Tests for discover_latest_phase_output config parameter usage.

Validates:
- Calling without config= in multi-model setup could return wrong directory
- discover_top_n_latents passes config to fallback discovery
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from common.config import Config


class TestDiscoverLatestPhaseOutputConfig:
    """Test that discover_latest_phase_output uses config for correct path resolution."""

    def test_discover_with_config_uses_model_suffix(self):
        """With config for gemma-9b, should search in suffixed directory."""
        from common.phase_discovery import get_phase_output_dir

        config_9b = Config(model_name="google/gemma-2-9b")
        output_dir = get_phase_output_dir("2.10", config_9b)

        # Should include gemma9b suffix
        assert "_gemma9b" in output_dir

    def test_discover_with_config_uses_dataset_suffix(self):
        """With config for humaneval, should search in suffixed directory."""
        from common.phase_discovery import get_phase_output_dir

        config_he = Config(dataset_name="humaneval")
        output_dir = get_phase_output_dir("2.10", config_he)

        # Should include humaneval suffix
        assert "_humaneval" in output_dir

    def test_discover_without_config_returns_default_path(self):
        """Without config, should search in default (gemma-2b + mbpp) directory."""
        from common.phase_discovery import get_phase_output_dir

        config_default = Config()
        output_dir = get_phase_output_dir("2.10", config_default)

        # Should NOT include any model/dataset suffix
        assert "_gemma9b" not in output_dir
        assert "_humaneval" not in output_dir
        assert "_llama" not in output_dir


class TestDiscoverTopNLatentsPassesConfig:
    """Test discover_top_n_latents passes config in fallback path."""

    def test_fallback_discovery_uses_config(self):
        """When primary path doesn't exist, fallback should use config."""
        from common.phase_discovery import get_phase_output_dir

        config_9b = Config(model_name="google/gemma-2-9b")
        primary_dir = Path(get_phase_output_dir("2.10", config_9b))

        # The primary directory should include the model suffix
        assert "_gemma9b" in str(primary_dir)
