# Testing Strategy for SAE-Code-Correctness

## Overview

Tests are organized into **4 tiers** based on scope and priority:

1. **Tier 1: Common Infrastructure Tests** (`test_common/`) - Shared utilities used by all phases
2. **Tier 2: Critical Path Tests** (`test_critical/`) - Code that affects experiment correctness
3. **Tier 3: Phase-Specific Tests** (`test_phases/`) - Per-phase logic validation
4. **Tier 4: Integration Tests** (`test_integration/`) - End-to-end phase chaining

---

## Running Tests

### Quick Start

```bash
# Activate conda first
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc

# All unit tests (no GPU needed)
pytest tests/test_common/ tests/test_critical/ -v

# All tests (skips GPU tests if no GPU)
pytest tests/ -v

# Specific tier
pytest tests/test_common/ -v           # Tier 1
pytest tests/test_critical/ -v         # Tier 2
pytest tests/test_phases/ -v           # Tier 3
pytest tests/test_integration/ -v      # Tier 4

# Specific phase
pytest tests/test_phases/test_phase_4_8.py -v

# By marker
pytest tests/ -v -m "unit"             # Only unit tests
pytest tests/ -v -m "gpu"              # Only GPU tests
pytest tests/ -v -m "not gpu"          # Exclude GPU tests
pytest tests/ -v -m "slow"             # Only slow tests
pytest tests/ -v -m "integration"      # Only integration tests
```

### Verify Existing Outputs

Check already-generated parallel outputs without re-running phases:

```bash
# Check a specific file
python tests/verify_parallel_output.py data/phase4_8/steering_effect_analysis.json

# Check all outputs for a phase
python tests/verify_parallel_output.py --phase 4.8

# Verbose mode
python tests/verify_parallel_output.py --phase 4.8 -v
```

---

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | No GPU required, fast |
| `@pytest.mark.gpu` | Requires 1+ GPU |
| `@pytest.mark.multi_gpu` | Requires 2+ GPUs |
| `@pytest.mark.slow` | Takes > 30 seconds |
| `@pytest.mark.integration` | End-to-end tests |

---

## Directory Structure

```
tests/
├── conftest.py                    # Shared fixtures and pytest config
├── pytest.ini                     # Pytest settings
├── README.md                      # This file
├── verify_parallel_output.py      # CLI verification utility
├── test_parallel_equivalence.py   # Legacy (moved to test_integration/)
│
├── test_common/                   # Tier 1: Common utilities
│   ├── test_steering_metrics.py   # Metric formulas
│   ├── test_phase_discovery.py    # Auto-discovery, output paths
│   ├── test_checkpoint_manager.py # Resume logic
│   ├── test_tensor_utils.py       # Tensor save/load
│   ├── test_config.py             # Config validation
│   └── test_steering_setup.py     # Direction loading
│
├── test_critical/                 # Tier 2: Critical paths
│   ├── test_steering_hook.py      # Hook correctness
│   ├── test_metric_consistency.py # Cross-phase validation
│   ├── test_normalization.py      # Direction normalization
│   └── test_direction_source.py   # SAE vs probe handling
│
├── test_phases/                   # Tier 3: Phase-specific
│   ├── test_phase_0.py            # Difficulty analysis
│   ├── test_phase_0_1.py          # Problem splitting
│   ├── test_phase_1.py            # Dataset building (CRITICAL)
│   ├── test_phase_2_5.py          # SAE analysis
│   ├── test_phase_2_6.py          # Probe directions
│   ├── test_phase_3_5.py          # Temperature robustness
│   ├── test_phase_3_8.py          # AUROC/F1 evaluation
│   ├── test_phase_4_5.py          # Coefficient search (CRITICAL)
│   ├── test_phase_4_6.py          # Golden section refinement
│   ├── test_phase_4_8.py          # Steering analysis (CRITICAL)
│   ├── test_phase_5_3.py          # Weight orthogonalization
│   ├── test_phase_7_6.py          # Instruct steering
│   ├── test_phase_8_2.py          # Threshold optimizer
│   └── test_phase_8_3.py          # Selective steering
│
└── test_integration/              # Tier 4: End-to-end
    ├── test_parallel_equivalence.py  # Parallel vs sequential
    └── test_phase_chaining.py        # Phase dependencies
```

---

## Key Tests by Priority

### Critical (must pass before any experiment)

| Test | File | What it validates |
|------|------|-------------------|
| Correction rate formula | `test_common/test_steering_metrics.py` | `(incorrect→correct) / total_incorrect * 100` |
| Corruption rate formula | `test_common/test_steering_metrics.py` | `(correct→incorrect) / total_correct * 100` |
| Hook position | `test_critical/test_steering_hook.py` | Only last position modified |
| Direction normalization | `test_critical/test_normalization.py` | Unit L2 norm |
| Checkpoint resume | `test_common/test_checkpoint_manager.py` | Resume after interrupt |

### High Priority (validates experiment correctness)

| Test | File | What it validates |
|------|------|-------------------|
| Metric consistency | `test_critical/test_metric_consistency.py` | Same formula across phases |
| Direction source | `test_critical/test_direction_source.py` | SAE vs probe loading |
| Phase discovery | `test_common/test_phase_discovery.py` | Auto-discovery, latest file |
| Parallel merge | `test_integration/test_parallel_equivalence.py` | No duplicates |

---

## Fixtures

### Common Fixtures (from `conftest.py`)

```python
# Configuration
base_config         # Fresh Config instance
small_subset_config # Config with --start 0 --end 10

# Directories
temp_output_dir     # Temporary output directory
mock_phase_dirs     # Mock phase directory structure

# Mock Objects (for unit tests)
mock_model          # Mock model with bfloat16 dtype
mock_sae            # Mock SAE with random W_dec

# Sample Data
sample_steering_results  # List of result dicts
sample_steering_df       # DataFrame version
sample_latents           # Top latents structure

# Integration (GPU required)
real_model          # Actual Gemma model (session-scoped)
real_sae            # Actual GemmaScope SAE (session-scoped)
```

---

## Writing New Tests

### Unit Test Template

```python
"""Tests for module_name."""

import pytest
from common.config import Config


class TestFeatureName:
    """Test description."""

    def test_basic_case(self):
        """Test basic expected behavior."""
        result = function_under_test(input)
        assert result == expected

    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            function_under_test(bad_input)

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
    ])
    def test_parametrized(self, input, expected):
        """Test with multiple inputs."""
        assert function(input) == expected
```

### GPU Test Template

```python
@pytest.mark.gpu
class TestGPUFeature:
    """GPU-required tests."""

    def test_cuda_tensor(self):
        """Test with CUDA tensors."""
        tensor = torch.randn(100, device='cuda')
        # ... test logic
```
