# Common Utilities

This directory contains shared utilities used across all phases of the PVA-SAE project.

## Module Overview

| Module | Purpose |
|--------|---------|
| `config.py` | Centralized configuration with model/dataset settings |
| `utils.py` | Core utilities: device detection, auto-discovery, memory management |
| `logging.py` | Phase-aware logging with progress tracking |
| `helpers.py` | File I/O, JSON handling, code extraction/evaluation |
| `model_loader.py` | Model/tokenizer loading from HuggingFace |
| `activation_hooks.py` | PyTorch hooks for activation extraction |
| `sae_loader.py` | Universal SAE loading (Gemma & LLAMA) |
| `steering_metrics.py` | Correction/corruption rate calculations |
| `weight_orthogonalization.py` | Permanent weight modifications |
| `prompt_utils.py` | Prompt building for code generation |
| `dataset_utils.py` | Dataset splitting and task ID discovery |
| `initialization.py` | Deterministic generation setup |
| `statistics_utils.py` | Binomial significance testing |
| `metrics_utils.py` | Classification metrics and activation encoding |

## Key Functions

### Output Directory Management

```python
from common.utils import get_phase_output_dir

# Automatically adds dataset/model suffix
output_dir = Path(get_phase_output_dir('4.8', config))
# Results in: data/phase4_8 (MBPP) or data/phase4_8_humaneval (HumanEval)
```

### Dataset Utilities

```python
from common.dataset_utils import split_by_correctness, discover_task_ids

# Split dataset by correctness
correct_df, incorrect_df = split_by_correctness(baseline_data)

# Discover task IDs from activation files
task_ids = discover_task_ids(Path("data/phase1_0/activations/correct"))
```

### Deterministic Generation

```python
from common.initialization import setup_deterministic_generation

# Set all seeds for reproducibility
setup_deterministic_generation(seed=42)
```

### Statistical Testing

```python
from common.statistics_utils import binomial_significance_test

# Test steering improvement
result = binomial_significance_test(
    n_successes=15,
    n_trials=100,
    expected_rate=0.10,
    alternative='greater'
)
if result['significant']:
    print(f"Significant improvement: p={result['p_value']:.4f}")
```

### Classification Metrics

```python
from common.metrics_utils import calculate_classification_metrics

metrics = calculate_classification_metrics(y_true, scores, threshold=0.5)
print(f"AUROC: {metrics['auroc']:.3f}, F1: {metrics['f1']:.3f}")
```

### Activation Loading

```python
from common.metrics_utils import load_and_encode_activation

# Load activation and encode through SAE
value = load_and_encode_activation(
    task_id="42",
    layer=16,
    feature_idx=1234,
    sae=my_sae,
    device=torch.device("cuda"),
    activation_dir=Path("data/phase1_0/activations/task_activations")
)
```

## Best Practices

1. **Use `get_phase_output_dir`** instead of manually building dataset suffixes
2. **Use `setup_deterministic_generation`** at the start of generation phases
3. **Use `split_by_correctness`** instead of inline DataFrame filtering
4. **Use `binomial_significance_test`** for statistical validation
