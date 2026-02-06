"""
Tests for common/utils.py

Validates:
- save_json: Atomic write behavior
- load_json: Round-trip consistency
"""

import json
import pytest
from pathlib import Path


# =============================================================================
# save_json Tests
# =============================================================================

class TestSaveJson:
    """Test save_json atomic write behavior."""

    def test_basic_save(self, tmp_path):
        """Should save valid JSON file."""
        from common.utils import save_json

        filepath = tmp_path / "output.json"
        data = {"hello": "world", "count": 123, "nested": {"a": 1}}

        save_json(data, filepath)

        assert filepath.exists()
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        """Should create parent directories if needed."""
        from common.utils import save_json

        filepath = tmp_path / "sub" / "dir" / "output.json"
        save_json({"key": "value"}, filepath)

        assert filepath.exists()

    def test_overwrites_existing(self, tmp_path):
        """Should overwrite existing file."""
        from common.utils import save_json

        filepath = tmp_path / "output.json"
        save_json({"version": 1}, filepath)
        save_json({"version": 2}, filepath)

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == {"version": 2}

    def test_accepts_path_string(self, tmp_path):
        """Should accept string path as well as Path object."""
        from common.utils import save_json

        filepath = str(tmp_path / "output.json")
        save_json({"key": "value"}, filepath)

        assert Path(filepath).exists()

    def test_default_str_serializer(self, tmp_path):
        """Should use default=str for non-serializable types."""
        from common.utils import save_json
        from pathlib import PurePosixPath

        filepath = tmp_path / "output.json"
        data = {"path": PurePosixPath("/some/path")}

        save_json(data, filepath)

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == {"path": "/some/path"}


# =============================================================================
# load_json Tests
# =============================================================================

class TestLoadJson:
    """Test load_json behavior."""

    def test_basic_load(self, tmp_path):
        """Should load JSON file."""
        from common.utils import load_json

        filepath = tmp_path / "input.json"
        data = {"hello": "world"}
        with open(filepath, 'w') as f:
            json.dump(data, f)

        loaded = load_json(filepath)
        assert loaded == data

    def test_round_trip(self, tmp_path):
        """save_json then load_json should preserve data."""
        from common.utils import save_json, load_json

        filepath = tmp_path / "roundtrip.json"
        data = {
            "string": "hello",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": {"b": "c"}}
        }

        save_json(data, filepath)
        loaded = load_json(filepath)

        assert loaded == data

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading non-existent file should raise."""
        from common.utils import load_json

        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")
