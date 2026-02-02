"""
Tests for common/tensor_utils.py

Validates:
- dtype_preservation: bfloat16 survives save/load
- safetensors_roundtrip: Activation data integrity
- to_numpy_conversion: Correct dtype in numpy output
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from common.tensor_utils import (
    save_activation,
    load_activation,
    save_activations,
    load_activations,
    to_numpy,
)


# =============================================================================
# dtype_preservation Tests
# =============================================================================

class TestDtypePreservation:
    """Test bfloat16 survives save/load cycle."""

    def test_bfloat16_roundtrip(self, tmp_path):
        """bfloat16 tensor should preserve dtype through save/load."""
        tensor = torch.randn(10, 2304, dtype=torch.bfloat16)
        path = tmp_path / "activation.safetensors"

        save_activation(tensor, path)
        loaded = load_activation(path)

        assert loaded.dtype == torch.bfloat16

    def test_float32_roundtrip(self, tmp_path):
        """float32 tensor should preserve dtype through save/load."""
        tensor = torch.randn(10, 2304, dtype=torch.float32)
        path = tmp_path / "activation.safetensors"

        save_activation(tensor, path)
        loaded = load_activation(path)

        assert loaded.dtype == torch.float32

    def test_float16_roundtrip(self, tmp_path):
        """float16 tensor should preserve dtype through save/load."""
        tensor = torch.randn(10, 2304, dtype=torch.float16)
        path = tmp_path / "activation.safetensors"

        save_activation(tensor, path)
        loaded = load_activation(path)

        assert loaded.dtype == torch.float16


# =============================================================================
# safetensors_roundtrip Tests
# =============================================================================

class TestSafetensorsRoundtrip:
    """Test activation data integrity through save/load."""

    def test_single_activation_values(self, tmp_path):
        """Single activation values should be preserved exactly."""
        tensor = torch.randn(1, 2304, dtype=torch.float32)
        path = tmp_path / "activation.safetensors"

        save_activation(tensor, path)
        loaded = load_activation(path)

        assert torch.allclose(tensor, loaded)

    def test_multi_layer_activations(self, tmp_path):
        """Multi-layer activations should be preserved."""
        activations = {
            6: torch.randn(1, 2304, dtype=torch.bfloat16),
            12: torch.randn(1, 2304, dtype=torch.bfloat16),
            18: torch.randn(1, 2304, dtype=torch.bfloat16),
        }
        path = tmp_path / "multi_layer.safetensors"

        save_activations(activations, path)
        loaded = load_activations(path)

        assert set(loaded.keys()) == {6, 12, 18}
        for layer in [6, 12, 18]:
            assert torch.allclose(activations[layer], loaded[layer])

    def test_shape_preserved(self, tmp_path):
        """Tensor shape should be preserved through save/load."""
        shapes = [
            (1, 2304),
            (1, 1, 2304),
            (4, 10, 2304),
        ]

        for i, shape in enumerate(shapes):
            tensor = torch.randn(shape)
            path = tmp_path / f"activation_{i}.safetensors"

            save_activation(tensor, path)
            loaded = load_activation(path)

            assert loaded.shape == tensor.shape

    def test_device_transfer(self, tmp_path):
        """Should load to specified device."""
        tensor = torch.randn(1, 2304)
        path = tmp_path / "activation.safetensors"

        save_activation(tensor, path)
        loaded_cpu = load_activation(path, device="cpu")

        assert loaded_cpu.device.type == "cpu"

    @pytest.mark.gpu
    def test_device_transfer_to_cuda(self, tmp_path):
        """Should load to CUDA device when specified."""
        tensor = torch.randn(1, 2304)
        path = tmp_path / "activation.safetensors"

        save_activation(tensor, path)
        loaded_cuda = load_activation(path, device="cuda")

        assert loaded_cuda.device.type == "cuda"


# =============================================================================
# to_numpy_conversion Tests
# =============================================================================

class TestToNumpyConversion:
    """Test correct dtype in numpy output."""

    def test_bfloat16_to_float32(self):
        """bfloat16 should convert to float32 numpy array."""
        tensor = torch.randn(10, 2304, dtype=torch.bfloat16)
        arr = to_numpy(tensor)

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32

    def test_float32_stays_float32(self):
        """float32 tensor should stay float32 in numpy."""
        tensor = torch.randn(10, 2304, dtype=torch.float32)
        arr = to_numpy(tensor)

        assert arr.dtype == np.float32

    def test_float16_to_float32(self):
        """float16 should convert to float32 numpy array."""
        tensor = torch.randn(10, 2304, dtype=torch.float16)
        arr = to_numpy(tensor)

        assert arr.dtype == np.float32

    def test_values_preserved(self):
        """Values should be preserved in conversion (within float32 precision)."""
        # Use float32 to avoid bfloat16 precision issues
        tensor = torch.randn(10, 2304, dtype=torch.float32)
        arr = to_numpy(tensor)

        # Convert back and compare
        back = torch.from_numpy(arr)
        assert torch.allclose(tensor, back)

    def test_gradient_detached(self):
        """Should handle tensors with gradients."""
        tensor = torch.randn(10, 2304, requires_grad=True)
        tensor = tensor * 2  # Create gradient

        # Should not raise
        arr = to_numpy(tensor)
        assert isinstance(arr, np.ndarray)

    @pytest.mark.gpu
    def test_cuda_tensor_to_numpy(self):
        """CUDA tensor should be moved to CPU for numpy conversion."""
        tensor = torch.randn(10, 2304, dtype=torch.float32, device="cuda")
        arr = to_numpy(tensor)

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32


# =============================================================================
# Edge Cases
# =============================================================================

class TestTensorEdgeCases:
    """Test edge cases in tensor operations."""

    def test_empty_tensor(self, tmp_path):
        """Empty tensor should save and load correctly."""
        tensor = torch.empty(0, 2304)
        path = tmp_path / "empty.safetensors"

        save_activation(tensor, path)
        loaded = load_activation(path)

        assert loaded.shape == (0, 2304)

    def test_scalar_like_tensor(self, tmp_path):
        """1D tensor should save and load correctly."""
        tensor = torch.randn(2304)
        path = tmp_path / "1d.safetensors"

        save_activation(tensor, path)
        loaded = load_activation(path)

        assert torch.allclose(tensor, loaded)

    def test_path_as_string(self, tmp_path):
        """Should accept path as string."""
        tensor = torch.randn(1, 2304)
        path = str(tmp_path / "activation.safetensors")

        save_activation(tensor, path)
        loaded = load_activation(path)

        assert torch.allclose(tensor, loaded)

    def test_path_as_pathlib(self, tmp_path):
        """Should accept path as Path object."""
        tensor = torch.randn(1, 2304)
        path = tmp_path / "activation.safetensors"

        save_activation(tensor, path)
        loaded = load_activation(path)

        assert torch.allclose(tensor, loaded)
