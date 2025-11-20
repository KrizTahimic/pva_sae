# Phase 8.2: Percentile Threshold Optimizer

## Overview

**Goal**: Find the optimal percentile threshold that maximizes net benefit (correction_rate - corruption_rate) on the hyperparameter dataset.

**Problem**: Phase 8.3 initial implementation steered 97.9% of cases (nearly universal steering). Using 90th percentile improved preservation but reduced correction effectiveness. We need to find the sweet spot.

**Solution**: Grid search across percentile thresholds calculated in Phase 8.1, evaluate on Phase 3.6 hyperparameter dataset, select optimal threshold based on net benefit.

## Architecture and Data Flow

### Phase Dependencies

```
Phase 3.6 (hyperparams dataset) → Phase 8.1 (percentile thresholds) → Phase 8.2 (optimization) → Phase 8.3 (final evaluation)
                                                                                                      ↓
                                                                                        Phase 3.5 (validation dataset)
```

### Data Sources

1. **Phase 0.1 Output** (MBPP Problem Specifications):
   - `hyperparams_mbpp.parquet`: Original MBPP problems for hyperparameter tuning set
   - Contains: `task_id`, `text` (problem description), `test_list` (test cases), `code` (reference solution)
   - **This is the source of prompts for code generation**

2. **Phase 3.6 Output** (Baseline Results):
   - `dataset_hyperparams_temp_0_0.parquet`: Baseline generated code + correctness labels (no steering)
   - Contains: `task_id`, `generated_code`, `is_correct`, `execution_result`
   - **Used ONLY for correctness labels to split into correction/preservation experiments**
   - **NOTE**: We do NOT reuse Phase 3.6's generated code or activations

3. **Phase 8.1 Output** (Percentile Thresholds):
   - `percentile_thresholds.json`: Pre-calculated thresholds for percentiles [50, 75, 80, 85, 90, 95]
   - These thresholds were calculated from Phase 3.6's pre-computed raw activations (NPZ files)

4. **Phase 3.8 Output** (Feature Information):
   - `evaluation_results.json`: Incorrect-predicting feature info
   - Feature: Layer 19, Feature 5441
   - Reference threshold: 15.5086 (optimized for classification, not steering)

5. **Phase 2.5 Output** (Steering Direction):
   - `top_20_features.json`: Correct-predicting features for steering direction
   - Used to construct steering vector via SAE decoder weights

### Key Architectural Decision

**Question**: What data do we reuse vs regenerate?

**Answer**:

| Data Type | Source | Usage in Phase 8.2 |
|-----------|--------|-------------------|
| **Problem prompts + tests** | Phase 0.1 | ✓ Reused to regenerate code |
| **Baseline correctness labels** | Phase 3.6 | ✓ Reused to split datasets |
| **Baseline generated code** | Phase 3.6 | ✗ NOT reused, generate fresh with steering |
| **Baseline activations** | Phase 3.6 NPZ | ✗ NOT reused (except Phase 8.1 used them for thresholds) |
| **Fresh activations** | Generated in 8.2 | ✓ Captured during steering via hooks |

**Rationale**:
- Phase 3.6 activations are from baseline generation (no steering applied)
- When we apply steering during Phase 8.2, the model's internal state changes
- We need to capture activations WITH steering applied to measure real steering effects
- This mirrors Phase 8.3's architecture exactly

**Complete Data Flow**:
```
Phase 0.1 (problem prompts) ──┐
                              ├──→ Phase 8.2 (merge, generate WITH steering) ──→ Optimal threshold
Phase 3.6 (correctness) ──────┘                        ↑
                                                        │
Phase 3.6 (NPZ activations) ──→ Phase 8.1 (percentile thresholds) ──┘


Phase 0.1 (validation prompts) ──┐
                                  ├──→ Phase 8.3 (final evaluation WITH steering)
Phase 3.5 (correctness) ──────────┘          ↑
                                              │
Phase 8.2 (optimal threshold) ────────────────┘
```

**Implementation Code Pattern**:
```python
# Load Phase 0.1 prompts
hyperparams_problems = pd.read_parquet("data/phase0_1/hyperparams_mbpp.parquet")

# Load Phase 3.6 baseline correctness
phase3_6_baseline = pd.read_parquet("data/phase3_6/.../dataset_hyperparams_temp_0_0.parquet")

# Merge: Get prompts + correctness labels
dataset = hyperparams_problems.merge(
    phase3_6_baseline[['task_id', 'is_correct']],
    on='task_id'
)

# Split by correctness
incorrect_problems = dataset[dataset['is_correct'] == False]  # Correction experiment
correct_problems = dataset[dataset['is_correct'] == True]      # Preservation experiment

# For each problem, generate FRESH code with selective steering
for problem in incorrect_problems:
    # Build prompt from Phase 0.1 data
    prompt = build_prompt(problem['text'], problem['test_list'])

    # Generate with steering (fresh activations captured via hooks)
    code = generate_with_selective_steering(prompt, threshold)

    # Evaluate using Phase 0.1 test cases
    is_correct = evaluate(code, problem['test_list'])
```

## Grid Search Strategy

### Percentiles to Test

Test all percentiles calculated in Phase 8.1:
- 50th percentile (steer top 50%)
- 75th percentile (steer top 25%)
- 80th percentile (steer top 20%)
- 85th percentile (steer top 15%)
- 90th percentile (steer top 10%)
- 95th percentile (steer top 5%)

### Optimization Metric

**Net Benefit** = `correction_rate - corruption_rate`

This metric balances:
- **Correction rate**: % of initially incorrect problems fixed by steering
- **Corruption rate**: % of initially correct problems broken by steering

We want to maximize corrections while minimizing corruptions.

### Split Testing Approach

Following Phase 4.8's pattern, we run two separate experiments per percentile:

1. **Correction Experiment**:
   - Dataset: Initially incorrect problems only
   - Apply: Steering toward correctness
   - Measure: Correction rate, steering rate

2. **Preservation Experiment**:
   - Dataset: Initially correct problems only
   - Apply: Same steering (tests if it breaks correct solutions)
   - Measure: Preservation rate (1 - corruption_rate), steering rate

This split testing ensures clean measurements without confounding variables.

## Implementation Details

### Main Components

#### 1. ThresholdOptimizer Class (`threshold_optimizer.py`)

**Responsibilities**:
- Load dependencies:
  - Phase 0.1: MBPP problem prompts and test cases (hyperparams set)
  - Phase 3.6: Baseline correctness labels (to split into correct/incorrect)
  - Phase 8.1: Percentile thresholds to test
  - Phase 3.8: Incorrect-predicting feature info (L19-F5441)
  - Phase 2.5: Steering direction features
  - Phase 4.8: Optimal steering coefficient (-26.0)
- Run grid search across all percentiles
- For each percentile: run correction + preservation experiments
- Calculate metrics and select optimal threshold
- Save results with full comparison data

**Key Methods**:

```python
class ThresholdOptimizer:
    def __init__(self, config: Config):
        """Initialize optimizer and load dependencies."""

    def _load_dependencies(self):
        """
        Load all dependencies for threshold optimization.

        Loads:
        - Phase 0.1: MBPP problem specifications (prompts + tests)
        - Phase 3.6: Baseline correctness labels
        - Phase 8.1: Percentile thresholds
        - Phase 3.8: Feature info (L19-F5441)
        - Phase 2.5: Steering features
        - Phase 4.8: Optimal coefficient
        """

    def _run_selective_steering_for_threshold(
        self,
        threshold: float,
        percentile: int,
        dataset_type: str
    ) -> Dict:
        """
        Run selective steering experiment for a single threshold.

        Args:
            threshold: Threshold value to test
            percentile: Percentile this threshold represents
            dataset_type: 'correction' or 'preservation'

        Returns:
            Dict with metrics (correction_rate, corruption_rate, etc.)
        """

    def optimize_threshold(self) -> Dict:
        """
        Main optimization loop.

        For each percentile:
            1. Run correction experiment
            2. Run preservation experiment
            3. Calculate net benefit

        Select percentile with highest net benefit.
        """

    def save_results(self, results: Dict):
        """Save optimization results and comparison data."""
```

### 2. Selective Steering Logic (Reused from Phase 8.3)

**Core Pattern** (from Phase 8.3's `_run_steering_experiment`):

```python
def _run_selective_steering_for_threshold(self, threshold, percentile, dataset_type):
    """Run steering with selective activation based on threshold."""

    # 1. Load appropriate dataset (correct or incorrect problems)
    if dataset_type == 'correction':
        dataset = self.incorrect_problems
    else:  # preservation
        dataset = self.correct_problems

    # 2. For each problem:
    for problem in dataset:
        # Generate code WITH steering hooks
        result = self._generate_with_selective_steering(
            problem=problem,
            threshold=threshold
        )

        # Track metrics
        was_steered = result['was_steered']
        is_correct = result['is_correct']

        # Update counters...

    # 3. Calculate metrics
    if dataset_type == 'correction':
        correction_rate = corrected / total_incorrect
    else:
        preservation_rate = preserved / total_correct
        corruption_rate = 1 - preservation_rate

    return metrics
```

**Selective Steering Decision** (from Phase 8.3):

```python
def _generate_with_selective_steering(self, problem, threshold):
    """Generate code with conditional steering based on feature activation."""

    # Hook to capture activation and conditionally apply steering
    def selective_steering_hook(module, input, output):
        # 1. Capture raw activation
        raw_activation = output[0][:, -1, :]

        # 2. Decompose via SAE to get features
        sae_features = self.sae.encode(raw_activation)

        # 3. Check incorrect-predicting feature activation
        incorrect_feature_activation = sae_features[0, self.incorrect_pred_feature]

        # 4. Decide whether to steer
        if incorrect_feature_activation > threshold:
            # Apply steering: add correction direction
            steering_vector = self.steering_direction * self.steering_coefficient
            output[0][:, -1, :] += steering_vector
            self.was_steered = True
        else:
            self.was_steered = False

        return output

    # Register hook and generate
    with hooks_registered:
        generated_code = model.generate(...)

    return {
        'code': generated_code,
        'was_steered': self.was_steered,
        'is_correct': evaluate_code(generated_code)
    }
```

### 3. Code Reuse Strategy

We can heavily reuse Phase 8.3's code:

**From Phase 8.3** → **To Phase 8.2**:
- `_split_baseline_by_correctness()` → Use directly
- `_generate_with_selective_steering()` → Use directly (parameterize threshold)
- `_run_steering_experiment()` → Adapt to `_run_selective_steering_for_threshold()`
- Hook creation logic → Use directly

**New Logic for Phase 8.2**:
- Grid search loop over percentiles
- Net benefit calculation and comparison
- Optimal threshold selection
- Comparison table generation

## Reusable Functions Reference

### From `common/` (Common Utilities)

**File: `common/utils.py`**
```python
from common.utils import (
    ensure_directory_exists,      # Create output directories
    discover_latest_phase_output,  # Auto-discover phase outputs
    get_timestamp,                 # Timestamp for filenames
    detect_device                  # GPU/CPU detection
)
```

**File: `common/config.py`**
```python
from common.config import Config  # Global configuration object
```

**File: `common/logging.py`**
```python
from common.logging import get_logger  # Phase-aware logging
```

**File: `common/prompt_utils.py`**
```python
from common.prompt_utils import build_prompt  # Build MBPP prompts from problem data

# Usage:
prompt = build_prompt(
    problem_text=problem['text'],
    test_cases=problem['test_list']
)
```

**File: `common/gpu_utils.py`**
```python
from common.gpu_utils import (
    cleanup_cache,         # Clear CUDA cache
    get_gpu_memory_info    # Check GPU memory usage
)
```

### From `common_simplified/` (Simplified Helpers)

**File: `common_simplified/helpers.py`**
```python
from common_simplified.helpers import (
    load_json,    # Load JSON files
    save_json     # Save JSON files with pretty formatting
)

# Usage:
data = load_json(Path("data/phase8_1/percentile_thresholds.json"))
save_json(results, Path("data/phase8_2/optimal_percentile.json"))
```

**File: `common_simplified/code_executor.py`**
```python
from common_simplified.code_executor import evaluate_code

# Usage:
is_correct, execution_result = evaluate_code(
    generated_code=code,
    test_cases=problem['test_list'],
    timeout=5
)
```

### From `phase2_5_simplified/` (SAE Loading)

**File: `phase2_5_simplified/sae_analyzer.py`**
```python
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

# Usage:
sae = load_gemma_scope_sae(layer=19, device=torch.device('cuda'))
sae_features = sae.encode(raw_activations)  # (batch, 2304) → (batch, 16384)
decoder_weights = sae.W_dec  # (16384, 2304) for constructing steering vectors
```

### From `phase8_3_selective_steering/` (Phase 8.3 Code)

**File: `phase8_3_selective_steering/selective_steering_analyzer.py`**

#### Methods to Reuse Directly:

**1. `_split_baseline_by_correctness()`**
```python
def _split_baseline_by_correctness(self):
    """Split baseline data into initially correct and initially incorrect problems."""
    # Returns: self.correct_problems, self.incorrect_problems

# Usage in Phase 8.2:
# Copy this method directly - splits dataset by Phase 3.6 correctness labels
```

**2. `_load_model_and_tokenizer()`**
```python
def _load_model_and_tokenizer(self):
    """Load Gemma model and tokenizer."""
    # Returns: self.model, self.tokenizer

# Usage in Phase 8.2:
# Copy this method directly - loads model for generation
```

**3. `_load_steering_features()`**
```python
def _load_steering_features(self):
    """Load Phase 2.5 top features and construct steering direction."""
    # Returns: self.steering_direction, self.steering_features

# Usage in Phase 8.2:
# Copy this method directly - builds steering vector from SAE decoder weights
```

**4. `_construct_steering_vector()`**
```python
def _construct_steering_vector(self, features: List[Dict]) -> torch.Tensor:
    """Construct steering vector from feature indices using SAE decoder weights."""
    # Returns: steering_direction (1, 2304)

# Usage in Phase 8.2:
# Copy this method directly - creates steering direction
```

#### Methods to Adapt:

**5. `_generate_with_selective_steering()` → Parameterize threshold**
```python
# Phase 8.3 version (hardcoded threshold):
def _generate_with_selective_steering(self, problem):
    if activation > self.threshold:  # Uses self.threshold
        # Apply steering

# Phase 8.2 version (parameterized threshold):
def _generate_with_selective_steering(self, problem, threshold):
    if activation > threshold:  # Pass threshold as parameter
        # Apply steering
```

**6. `_run_steering_experiment()` → Adapt to `_run_selective_steering_for_threshold()`**
```python
# Phase 8.3 version:
def _run_steering_experiment(self, experiment_type):
    # Runs single experiment with self.threshold

# Phase 8.2 version:
def _run_selective_steering_for_threshold(self, threshold, percentile, dataset_type):
    # Runs experiment with specific threshold
    # Returns metrics dict for this threshold
```

#### Helper Methods to Reuse:

**7. Checkpoint Management**
```python
# From Phase 8.3:
self._save_checkpoint(data, checkpoint_file)
self._load_checkpoint(checkpoint_file)
self._get_latest_checkpoint_index(checkpoint_dir)

# Usage in Phase 8.2:
# Copy checkpoint logic to enable resuming grid search
# Format: checkpoint_p{percentile}_{dataset_type}_{idx}.json
```

**8. Code Evaluation**
```python
# From Phase 8.3:
is_correct, exec_result = evaluate_code(
    generated_code,
    problem['test_list'],
    timeout=5
)

# Usage in Phase 8.2:
# Use common_simplified.code_executor.evaluate_code directly
```

**9. Hook Registration Pattern**
```python
# From Phase 8.3:
hook_handle = self.model.model.layers[self.steering_layer].register_forward_hook(
    selective_steering_hook
)
try:
    # Generate with hook active
finally:
    hook_handle.remove()

# Usage in Phase 8.2:
# Copy this pattern for safe hook registration/cleanup
```

### Summary of Code Reuse

| Component | Source | Reuse Type | Notes |
|-----------|--------|------------|-------|
| Directory utils | `common.utils` | Direct | `ensure_directory_exists`, `discover_latest_phase_output` |
| JSON I/O | `common_simplified.helpers` | Direct | `load_json`, `save_json` |
| Logging | `common.logging` | Direct | `get_logger` |
| Prompt building | `common.prompt_utils` | Direct | `build_prompt` |
| Code execution | `common_simplified.code_executor` | Direct | `evaluate_code` |
| SAE loading | `phase2_5_simplified.sae_analyzer` | Direct | `load_gemma_scope_sae` |
| Dataset splitting | Phase 8.3 | Direct | `_split_baseline_by_correctness()` |
| Model loading | Phase 8.3 | Direct | `_load_model_and_tokenizer()` |
| Steering vector | Phase 8.3 | Direct | `_construct_steering_vector()` |
| Selective steering | Phase 8.3 | Adapt | Add threshold parameter |
| Experiment runner | Phase 8.3 | Adapt | Add threshold + percentile tracking |
| Checkpointing | Phase 8.3 | Adapt | Modify for grid search structure |

**Estimated Code Reuse**: ~70% from Phase 8.3, ~20% from common utilities, ~10% new grid search logic

## Checkpointing and Resume Logic

Phase 8.2's grid search can take 4-6 hours. Robust checkpointing is critical for resuming interrupted runs.

### Checkpoint Structure

**Directory Layout**:
```
data/phase8_2/
├── checkpoints/
│   ├── p50_correction/
│   │   ├── checkpoint_0.json
│   │   ├── checkpoint_50.json
│   │   └── checkpoint_100.json
│   ├── p50_preservation/
│   │   └── checkpoint_0.json
│   ├── p75_correction/
│   │   └── checkpoint_0.json
│   └── ...
├── optimal_percentile.json      (final output)
├── threshold_comparison.json    (final output)
└── threshold_summary.txt         (final output)
```

**Checkpoint Naming**: `checkpoint_{problem_index}.json`

### Checkpoint File Format

```json
{
  "percentile": 75,
  "threshold": 16.2345,
  "dataset_type": "correction",
  "last_completed_index": 49,
  "total_problems": 242,
  "results": [
    {
      "problem_index": 0,
      "task_id": "task_123",
      "was_steered": true,
      "is_correct": false,
      "initial_correct": false
    },
    ...
  ],
  "cumulative_metrics": {
    "n_steered": 12,
    "n_corrected": 3,
    "n_problems_processed": 50
  },
  "timestamp": "2025-01-19T10:30:45"
}
```

### Checkpointing Logic

**When to Save Checkpoints**:
- Every 50 problems (following Phase 8.3 pattern)
- After completing each percentile's correction experiment
- After completing each percentile's preservation experiment
- Before starting a new percentile

**Implementation Pattern**:

```python
class ThresholdOptimizer:
    def __init__(self, config: Config):
        self.checkpoint_dir = self.output_dir / "checkpoints"
        ensure_directory_exists(self.checkpoint_dir)

        # Track grid search progress
        self.grid_search_state = {
            'completed_percentiles': [],
            'current_percentile': None,
            'current_dataset_type': None
        }

    def _get_checkpoint_dir(self, percentile: int, dataset_type: str) -> Path:
        """Get checkpoint directory for specific percentile + dataset type."""
        return self.checkpoint_dir / f"p{percentile}_{dataset_type}"

    def _save_checkpoint(
        self,
        percentile: int,
        threshold: float,
        dataset_type: str,
        last_index: int,
        results: List[Dict]
    ):
        """Save checkpoint for current grid search iteration."""
        checkpoint_dir = self._get_checkpoint_dir(percentile, dataset_type)
        ensure_directory_exists(checkpoint_dir)

        checkpoint_file = checkpoint_dir / f"checkpoint_{last_index}.json"

        checkpoint_data = {
            'percentile': percentile,
            'threshold': threshold,
            'dataset_type': dataset_type,
            'last_completed_index': last_index,
            'total_problems': len(self.correct_problems if dataset_type == 'preservation' else self.incorrect_problems),
            'results': results,
            'cumulative_metrics': self._calculate_cumulative_metrics(results, dataset_type),
            'timestamp': datetime.now().isoformat()
        }

        save_json(checkpoint_data, checkpoint_file)
        logger.info(f"✓ Checkpoint saved: {checkpoint_file.name}")

    def _load_checkpoint(self, percentile: int, dataset_type: str) -> Optional[Dict]:
        """Load most recent checkpoint for percentile + dataset type."""
        checkpoint_dir = self._get_checkpoint_dir(percentile, dataset_type)

        if not checkpoint_dir.exists():
            return None

        # Find latest checkpoint
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoints:
            return None

        latest_checkpoint = checkpoints[-1]
        logger.info(f"Found checkpoint: {latest_checkpoint}")

        checkpoint_data = load_json(latest_checkpoint)

        logger.info(f"Resuming from index {checkpoint_data['last_completed_index']}")
        logger.info(f"Progress: {checkpoint_data['last_completed_index'] + 1}/{checkpoint_data['total_problems']} problems")

        return checkpoint_data

    def _is_percentile_completed(self, percentile: int) -> bool:
        """Check if both correction and preservation are complete for percentile."""
        correction_dir = self._get_checkpoint_dir(percentile, 'correction')
        preservation_dir = self._get_checkpoint_dir(percentile, 'preservation')

        # Check if both experiments have completed checkpoints
        correction_complete = self._is_experiment_complete(correction_dir, len(self.incorrect_problems))
        preservation_complete = self._is_experiment_complete(preservation_dir, len(self.correct_problems))

        return correction_complete and preservation_complete

    def _is_experiment_complete(self, checkpoint_dir: Path, total_problems: int) -> bool:
        """Check if experiment has checkpoint for all problems."""
        if not checkpoint_dir.exists():
            return False

        checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))

        if not checkpoints:
            return False

        # Load latest checkpoint
        latest = sorted(checkpoints)[-1]
        data = load_json(latest)

        # Experiment is complete if last_completed_index == total_problems - 1
        return data['last_completed_index'] >= total_problems - 1
```

### Resume Logic

**Grid Search Resume Flow**:

```python
def optimize_threshold(self) -> Dict:
    """Main optimization loop with resume support."""

    # Load Phase 8.1 percentile thresholds
    percentile_thresholds = self._load_percentile_thresholds()

    results = {}

    for pct in [50, 75, 80, 85, 90, 95]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {pct}th Percentile")
        logger.info(f"{'='*60}")

        threshold = percentile_thresholds[f'p{pct}']['threshold']

        # Check if this percentile is already completed
        if self._is_percentile_completed(pct):
            logger.info(f"✓ Percentile {pct} already completed, loading results...")
            results[f'p{pct}'] = self._load_percentile_results(pct, threshold)
            continue

        # === CORRECTION EXPERIMENT ===
        logger.info(f"\n--- Correction Experiment (p{pct}) ---")

        # Try to resume from checkpoint
        correction_checkpoint = self._load_checkpoint(pct, 'correction')

        if correction_checkpoint:
            logger.info(f"Resuming correction experiment from checkpoint...")
            correction_results = correction_checkpoint['results']
            start_idx = correction_checkpoint['last_completed_index'] + 1
        else:
            logger.info(f"Starting correction experiment from beginning...")
            correction_results = []
            start_idx = 0

        # Run correction experiment (with resume)
        correction_metrics = self._run_selective_steering_for_threshold(
            threshold=threshold,
            percentile=pct,
            dataset_type='correction',
            start_idx=start_idx,
            previous_results=correction_results
        )

        # === PRESERVATION EXPERIMENT ===
        logger.info(f"\n--- Preservation Experiment (p{pct}) ---")

        # Try to resume from checkpoint
        preservation_checkpoint = self._load_checkpoint(pct, 'preservation')

        if preservation_checkpoint:
            logger.info(f"Resuming preservation experiment from checkpoint...")
            preservation_results = preservation_checkpoint['results']
            start_idx = preservation_checkpoint['last_completed_index'] + 1
        else:
            logger.info(f"Starting preservation experiment from beginning...")
            preservation_results = []
            start_idx = 0

        # Run preservation experiment (with resume)
        preservation_metrics = self._run_selective_steering_for_threshold(
            threshold=threshold,
            percentile=pct,
            dataset_type='preservation',
            start_idx=start_idx,
            previous_results=preservation_results
        )

        # Calculate net benefit
        net_benefit = correction_metrics['correction_rate'] - preservation_metrics['corruption_rate']

        results[f'p{pct}'] = {
            'percentile': pct,
            'threshold': threshold,
            'correction_experiment': correction_metrics,
            'preservation_experiment': preservation_metrics,
            'net_benefit': net_benefit
        }

        logger.info(f"✓ Percentile {pct} complete: Net benefit = {net_benefit:.4f}")

    # Select optimal percentile
    optimal_key = max(results.keys(), key=lambda k: results[k]['net_benefit'])

    return {
        'optimal_percentile': results[optimal_key]['percentile'],
        'optimal_threshold': results[optimal_key]['threshold'],
        'results': results
    }
```

### Experiment Runner with Checkpointing

```python
def _run_selective_steering_for_threshold(
    self,
    threshold: float,
    percentile: int,
    dataset_type: str,
    start_idx: int = 0,
    previous_results: List[Dict] = None
) -> Dict:
    """Run steering experiment with checkpoint support."""

    # Select dataset
    if dataset_type == 'correction':
        dataset = self.incorrect_problems
    else:
        dataset = self.correct_problems

    # Initialize results
    results = previous_results if previous_results else []

    # Process problems (starting from start_idx for resume)
    for idx in range(start_idx, len(dataset)):
        problem = dataset.iloc[idx]

        # Generate with selective steering
        result = self._generate_with_selective_steering(problem, threshold)

        results.append({
            'problem_index': idx,
            'task_id': problem['task_id'],
            'was_steered': result['was_steered'],
            'is_correct': result['is_correct'],
            'initial_correct': problem['is_correct']
        })

        # Save checkpoint every 50 problems
        if (idx + 1) % 50 == 0:
            self._save_checkpoint(
                percentile=percentile,
                threshold=threshold,
                dataset_type=dataset_type,
                last_index=idx,
                results=results
            )

    # Final checkpoint
    self._save_checkpoint(
        percentile=percentile,
        threshold=threshold,
        dataset_type=dataset_type,
        last_index=len(dataset) - 1,
        results=results
    )

    # Calculate final metrics
    metrics = self._calculate_metrics(results, dataset_type)

    return metrics
```

### Cleanup After Successful Completion

```python
def _cleanup_checkpoints(self):
    """Remove checkpoint files after successful completion."""
    if self.checkpoint_dir.exists():
        logger.info("Cleaning up checkpoint files...")

        for checkpoint_file in self.checkpoint_dir.rglob("checkpoint_*.json"):
            checkpoint_file.unlink()

        # Remove empty directories
        for subdir in self.checkpoint_dir.iterdir():
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()

        logger.info("✓ Checkpoint files cleaned up")
```

### User Instructions for Resume

**To Resume Interrupted Run**:
```bash
# Simply re-run the same command
python3 run.py phase 8.2

# Phase 8.2 will:
# 1. Detect existing checkpoints
# 2. Skip completed percentiles
# 3. Resume incomplete percentiles from last checkpoint
# 4. Continue until all percentiles complete
```

**To Start Fresh (Ignore Checkpoints)**:
```bash
# Delete checkpoint directory
rm -rf data/phase8_2/checkpoints/

# Then run normally
python3 run.py phase 8.2
```

**To Resume Specific Percentile Only** (Advanced):
```bash
# Keep only the checkpoints you want to resume
rm -rf data/phase8_2/checkpoints/p50_*
rm -rf data/phase8_2/checkpoints/p75_*
# Keep p80, p85, p90, p95 checkpoints

python3 run.py phase 8.2  # Will recompute p50, p75 only
```

### Checkpointing with `--start` and `--end` Flags

**Question**: Does checkpointing work when testing with `--start 0 --end 5`?

**Answer**: Yes, but with important considerations:

**Problem**: When using `--start 0 --end 5` for testing:
- The dataset is filtered to 5 problems
- Checkpoint logic needs to handle this correctly
- Resume logic must account for partial dataset runs

**Solution**: Track both filtered index and original problem index

```python
def _run_selective_steering_for_threshold(
    self,
    threshold: float,
    percentile: int,
    dataset_type: str,
    start_idx: int = 0,
    previous_results: List[Dict] = None
) -> Dict:
    """Run steering experiment with checkpoint support."""

    # Select dataset (already filtered by --start/--end in _load_dependencies)
    if dataset_type == 'correction':
        dataset = self.incorrect_problems
    else:
        dataset = self.correct_problems

    results = previous_results if previous_results else []

    # Process problems
    for filtered_idx in range(start_idx, len(dataset)):
        problem = dataset.iloc[filtered_idx]

        # Get original problem index (for proper checkpoint tracking)
        # This ensures checkpoints work correctly even with filtered datasets
        original_idx = problem.name  # pandas DataFrame index is the original position

        result = self._generate_with_selective_steering(problem, threshold)

        results.append({
            'problem_index': original_idx,  # Use ORIGINAL index
            'filtered_index': filtered_idx,  # Also track filtered index
            'task_id': problem['task_id'],
            'was_steered': result['was_steered'],
            'is_correct': result['is_correct'],
            'initial_correct': problem['is_correct']
        })

        # Save checkpoint every 50 problems (using filtered index)
        if (filtered_idx + 1) % 50 == 0:
            self._save_checkpoint(
                percentile=percentile,
                threshold=threshold,
                dataset_type=dataset_type,
                last_index=filtered_idx,  # Checkpoint at filtered position
                results=results
            )

    # Final checkpoint
    self._save_checkpoint(
        percentile=percentile,
        threshold=threshold,
        dataset_type=dataset_type,
        last_index=len(dataset) - 1,  # Use filtered dataset length
        results=results
    )

    metrics = self._calculate_metrics(results, dataset_type)
    return metrics
```

**Checkpoint Behavior Examples**:

**Example 1: Testing with `--start 0 --end 5`**
```bash
python3 run.py phase 8.2 --start 0 --end 5
```

- Dataset filtered to 5 problems
- Checkpoint saved after processing all 5 (no intermediate checkpoints, since < 50)
- Checkpoint saved as: `checkpoint_4.json` (last filtered index)
- If interrupted and resumed: picks up from checkpoint, processes remaining (if any)

**Example 2: Testing with `--start 0 --end 100`**
```bash
python3 run.py phase 8.2 --start 0 --end 100
```

- Dataset filtered to 100 problems
- Checkpoint saved at index 49: `checkpoint_49.json`
- Checkpoint saved at index 99 (final): `checkpoint_99.json`
- If interrupted at problem 60 and resumed: loads `checkpoint_49.json`, continues from index 50

**Example 3: Full Run (No `--start`/`--end`)**
```bash
python3 run.py phase 8.2
```

- Full dataset (~487 problems)
- Checkpoints every 50: `checkpoint_49.json`, `checkpoint_99.json`, etc.
- Final checkpoint: `checkpoint_486.json`

**Important Notes**:
1. **Checkpoint frequency**: Checkpoints save every 50 problems based on filtered dataset
2. **Original indices**: Results track original problem indices for proper analysis
3. **Resume behavior**: When resuming with different `--start`/`--end`, old checkpoints are still valid
4. **Testing workflow**: Use `--end 5` to test checkpointing logic without waiting for full run

### Key Differences from Phase 8.3 Checkpointing

| Aspect | Phase 8.3 | Phase 8.2 |
|--------|-----------|-----------|
| **Checkpoint Structure** | Single experiment type | Grid search: multiple percentiles × 2 experiments |
| **Resume Granularity** | Problem-level | Percentile + experiment + problem level |
| **Completion Check** | Single dataset completion | 6 percentiles × 2 experiments = 12 completions |
| **Directory Structure** | `checkpoints/correction/`, `checkpoints/preservation/` | `checkpoints/p{N}_correction/`, `checkpoints/p{N}_preservation/` |
| **Progress Tracking** | Single progress bar | Nested: percentile → experiment → problem |
| **Range Filtering** | Checkpoints relative to filtered dataset | Checkpoints relative to filtered dataset |

## Expected Outputs

### 1. `optimal_percentile.json`

```json
{
  "phase": "8.2",
  "timestamp": "2025-01-19T...",
  "optimization_summary": {
    "metric": "net_benefit",
    "formula": "correction_rate - corruption_rate",
    "optimal_percentile": 80,
    "optimal_threshold": 18.3456,
    "optimal_net_benefit": 0.0892,
    "optimal_metrics": {
      "correction_rate": 0.1234,
      "corruption_rate": 0.0342,
      "preservation_rate": 0.9658,
      "steering_rate_correction": 0.2145,
      "steering_rate_preservation": 0.1987
    }
  },
  "source_dataset": {
    "phase": "3.6",
    "dataset": "hyperparams",
    "n_correct_problems": 245,
    "n_incorrect_problems": 242
  },
  "feature_info": {
    "layer": 19,
    "feature_idx": 5441,
    "description": "Incorrect-predicting feature"
  },
  "steering_info": {
    "layer": 19,
    "coefficient": -26.0,
    "description": "From Phase 4.8 optimal steering"
  }
}
```

### 2. `threshold_comparison.json`

Full grid search results:

```json
{
  "percentiles_tested": [50, 75, 80, 85, 90, 95],
  "results": {
    "p50": {
      "percentile": 50,
      "threshold": 12.3456,
      "steer_percentage": 50.0,
      "correction_experiment": {
        "correction_rate": 0.0987,
        "steering_rate": 0.5123,
        "n_problems": 242,
        "n_steered": 124,
        "n_corrected": 24
      },
      "preservation_experiment": {
        "preservation_rate": 0.9234,
        "corruption_rate": 0.0766,
        "steering_rate": 0.4987,
        "n_problems": 245,
        "n_steered": 122,
        "n_corrupted": 19
      },
      "net_benefit": 0.0221
    },
    "p75": { ... },
    "p80": { ... },
    "p85": { ... },
    "p90": {
      "percentile": 90,
      "threshold": 22.2769,
      "steer_percentage": 10.0,
      "correction_experiment": {
        "correction_rate": 0.0372,
        "steering_rate": 0.1034,
        "n_problems": 242,
        "n_steered": 25,
        "n_corrected": 9
      },
      "preservation_experiment": {
        "preservation_rate": 0.9959,
        "corruption_rate": 0.0041,
        "steering_rate": 0.0980,
        "n_problems": 245,
        "n_steered": 24,
        "n_corrupted": 1
      },
      "net_benefit": 0.0331
    },
    "p95": { ... }
  },
  "optimal_percentile": "p80"
}
```

### 3. `threshold_summary.txt`

Human-readable comparison table:

```
=============================================================================
PHASE 8.2: PERCENTILE THRESHOLD OPTIMIZATION
=============================================================================

Dataset: Phase 3.6 (hyperparams, 242 incorrect, 245 correct)
Feature: Layer 19, Feature 5441 (incorrect-predicting)
Steering: Layer 19, Coefficient -26.0

THRESHOLD COMPARISON (sorted by net benefit)
-----------------------------------------------------------------------------
Percentile  Threshold   Steer%   Correction%  Corruption%  Net Benefit
-----------------------------------------------------------------------------
80th        18.3456     20.0%    12.34%       3.42%        +8.92%  ← OPTIMAL
85th        19.8765     15.0%    10.12%       2.11%        +8.01%
75th        16.2345     25.0%    13.45%       5.67%        +7.78%
90th        22.2769     10.0%    3.72%        0.41%        +3.31%
95th        28.4567     5.0%     1.24%        0.08%        +1.16%
50th        12.3456     50.0%    9.87%        7.66%        +2.21%

OPTIMAL THRESHOLD SELECTED
-----------------------------------------------------------------------------
Percentile:         80th
Threshold:          18.3456
Net Benefit:        +8.92%
Correction Rate:    12.34% (30 / 242 initially incorrect)
Corruption Rate:    3.42% (8 / 245 initially correct)
Preservation Rate:  96.58%
Steering Rate:      ~20%

INTERPRETATION
-----------------------------------------------------------------------------
- Steering top 20% of cases (80th percentile) provides best balance
- Corrects 12.34% of incorrect solutions
- Only breaks 3.42% of correct solutions
- Net improvement: +8.92 percentage points
- More selective than Phase 3.8 threshold (15.5086), less than 90th percentile

NEXT STEPS
-----------------------------------------------------------------------------
Run Phase 8.3 with optimal threshold (18.3456) on validation set (Phase 3.5)
to measure final performance.
```

## Configuration Changes

### Add to `common/config.py`

```python
# Phase 8.2: Threshold Optimization
phase8_2_output_dir: str = "data/phase8_2"
```

### Update `common/utils.py`

Add to PHASE_CONFIGS:

```python
"8.2": {
    "dir": "data/phase8_2",
    "patterns": ["optimal_percentile.json", "threshold_comparison.json"],
    "exclude_keywords": None
}
```

### Update Phase 8.3 to Use Phase 8.2 Results

Modify Phase 8.3's `_load_dependencies()` to check for Phase 8.2 first:

```python
# In phase8_3_selective_steering/selective_steering_analyzer.py

if self.config.phase8_3_use_percentile_threshold:
    # Try Phase 8.2 first (optimized threshold)
    phase8_2_output = discover_latest_phase_output("8.2")

    if phase8_2_output:
        phase8_2_results = load_json(Path(phase8_2_output).parent / "optimal_percentile.json")
        optimal = phase8_2_results['optimization_summary']
        self.threshold = optimal['optimal_threshold']

        logger.info(f"✓ Using Phase 8.2 optimal threshold: {self.threshold:.4f}")
        logger.info(f"  (Percentile: {optimal['optimal_percentile']}, Net benefit: {optimal['optimal_net_benefit']:.4f})")
    else:
        # Fall back to Phase 8.1 percentile
        phase8_1_output = discover_latest_phase_output("8.1")
        # ... existing Phase 8.1 fallback logic
```

## Implementation Checklist

### Phase 8.2 Files to Create

- [ ] `phase8_2_threshold_optimizer/__init__.py`
- [ ] `phase8_2_threshold_optimizer/threshold_optimizer.py`
- [ ] `phase8_2_threshold_optimizer/runner.py`
- [ ] `phase8_2_threshold_optimizer/PHASE_8_2_DESIGN.md` (this file)

### Files to Modify

- [ ] `run.py`: Add Phase 8.2 handler
- [ ] `common/config.py`: Add `phase8_2_output_dir`
- [ ] `common/utils.py`: Add Phase 8.2 to PHASE_CONFIGS
- [ ] `phase8_3_selective_steering/selective_steering_analyzer.py`: Update to check Phase 8.2 first

### Testing Plan

1. **Phase 8.1 Test** (already completed):
   ```bash
   python3 run.py phase 8.1
   ```
   - Verify percentile thresholds calculated
   - Check output: `data/phase8_1/percentile_thresholds.json`

2. **Phase 8.2 Test** (small sample):
   ```bash
   python3 run.py phase 8.2 --start 0 --end 50
   ```
   - Test on first 50 problems only
   - Verify grid search runs for all percentiles
   - Check optimal threshold selection logic
   - Review comparison output

3. **Phase 8.2 Full Run**:
   ```bash
   python3 run.py phase 8.2
   ```
   - Run on full hyperparams dataset (~487 problems)
   - Expected runtime: ~4-6 hours (6 percentiles × ~40 min each)
   - Checkpointing will enable resuming if interrupted

4. **Phase 8.3 Integration Test**:
   ```bash
   python3 run.py phase 8.3 --start 0 --end 20
   ```
   - Verify Phase 8.3 loads Phase 8.2 optimal threshold
   - Check logs show correct threshold being used
   - Validate on small validation sample

5. **Phase 8.3 Final Evaluation**:
   ```bash
   python3 run.py phase 8.3
   ```
   - Run with optimal threshold on full validation set
   - Compare metrics to Phase 8.2 hyperparams results
   - Document final performance for ICLR response

## Expected Timeline

- **Phase 8.2 Implementation**: 1-2 hours
- **Phase 8.2 Testing (small)**: 30 minutes
- **Phase 8.2 Full Run**: 4-6 hours
- **Phase 8.3 Integration**: 30 minutes
- **Phase 8.3 Final Evaluation**: 2-3 hours

**Total**: ~8-12 hours for complete pipeline

## Success Criteria

1. **Phase 8.2 completes successfully**:
   - All 6 percentiles tested
   - Optimal threshold selected based on net benefit
   - Results saved with full comparison data

2. **Phase 8.3 integration works**:
   - Automatically loads Phase 8.2 optimal threshold
   - Falls back to Phase 8.1 if Phase 8.2 not found
   - Logs show correct threshold being used

3. **Performance improvement demonstrated**:
   - Net benefit (correction - corruption) is positive
   - Better than universal steering (original Phase 8.3)
   - Steering rate is reasonable (not 97.9%, not too conservative)

4. **Documentation complete**:
   - Clear explanation of threshold selection methodology
   - Comparison table showing all percentiles tested
   - Interpretation section explaining optimal choice

## Addressing ICLR Reviewer Feedback

The reviewer stated:

> "The 4.04% correction is overshadowed by the 14.66% corruption rate. This suggests that the feature identified may be correlated with correctness but not causally related in a way that benefits model performance when intervened upon."

**Phase 8.2 Response**:

1. **Selective Application**: We now demonstrate that universal steering (97.9% of cases) was the problem, not the feature itself.

2. **Threshold Optimization**: By optimizing the threshold on held-out hyperparameter data, we show a principled approach to selective steering.

3. **Net Benefit Metric**: We explicitly optimize for `correction_rate - corruption_rate`, directly addressing the reviewer's concern.

4. **Validation**: Final Phase 8.3 evaluation on validation set demonstrates generalization of the selective steering approach.

5. **Documentation**: Clear comparison showing performance across different selectivity levels (50% to 5% steering rates).

This directly addresses the reviewer's request: *"combining the detection mechanism with the steering intervention to demonstrate actual improvement in model performance."*
