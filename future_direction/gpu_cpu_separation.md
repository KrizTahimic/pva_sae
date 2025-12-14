# GPU/CPU Separation Architecture

**Date:** 2025-12-14
**Status:** Planning (to be implemented)

---

## Problem Statement

### Current Architecture (Mixed GPU/CPU)

All code-generating phases currently interleave GPU and CPU work:

```python
for task in tasks:
    output = model.generate(prompt)      # GPU
    code = extract_code(output)          # CPU
    passed = evaluate_code(code, tests)  # CPU (blocking!)
    save_activation(...)                 # CPU/GPU transfer
```

**Issues:**
1. GPU sits idle during code evaluation
2. No parallelization of CPU work
3. Context switching overhead between GPU↔CPU
4. If evaluation crashes, lose generation progress

### Affected Phases

| Phase | Purpose | Current Behavior |
|-------|---------|------------------|
| 1 | Baseline (SAE split) | Generate + Activate + Eval |
| 3.5 | Temperature robustness | Generate + Eval |
| 4.8 | Steering effect | Generate + Steer + Eval |
| 7.3 | Instruct baseline | Generate + Eval |
| 7.6 | Instruct steering | Generate + Steer + Eval |
| 8.3 | Selective steering | Generate + Steer + Eval |

---

## Proposed Architecture

### Two-Pass Pattern (Within Each Phase)

```python
def run(self):
    # ═══════════════════════════════════════════
    # PASS 1: GENERATION (GPU-ONLY)
    # ═══════════════════════════════════════════
    if not self.config.eval_only:
        raw_outputs = []
        for task in tqdm(tasks, desc="Generating"):
            output = self.generate(task)  # GPU stays hot
            raw_outputs.append({
                'task_id': task['task_id'],
                'prompt': task['prompt'],
                'raw_generated_text': output,
                'activations': activations,  # if applicable
            })

        # Checkpoint: survives crashes
        save_parquet(raw_outputs, "generations_raw.parquet")
    else:
        # --eval-only flag: skip generation
        raw_outputs = load_parquet("generations_raw.parquet")

    # ═══════════════════════════════════════════
    # PASS 2: EVALUATION (CPU-ONLY, PARALLEL)
    # ═══════════════════════════════════════════
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(evaluate_single, raw_outputs))

    save_parquet(results, "results.parquet")
```

### Visual Comparison

**Current (interleaved):**
```
GPU: gen1 ──────────────────────── gen2 ──────────────────────── gen3
CPU:      ↓ eval1 (GPU idle) ↓         ↓ eval2 (GPU idle) ↓
          [====waiting====]            [====waiting====]
```

**Proposed (batched):**
```
GPU: gen1 → gen2 → gen3 → gen4 → ... → genN  (continuous)
CPU:                                          eval1 ∥ eval2 ∥ eval3 (parallel)
```

---

## Benefits

| Aspect | Current | Proposed |
|--------|---------|----------|
| GPU utilization | Interrupted by CPU eval | 100% continuous |
| CPU evaluation | Sequential, blocking | Parallel (8 workers) |
| GPU memory | Reallocated frequently | Stays allocated |
| Crash recovery | Lose all progress | Checkpoint after gen |
| Re-run evaluation | Must regenerate | `--eval-only` flag |
| Estimated speedup | Baseline | 20-40% overall |

### Why Faster?

1. **GPU stays hot** - No cold starts between generations
2. **GPU memory stays allocated** - No reallocation overhead
3. **CPU parallelization** - 8 cores = up to 8x eval throughput
4. **Less context switching** - Reduced GPU↔CPU transfer overhead
5. **Better batching** - GPU optimized for continuous work

---

## Import Discovery Integration

### The Problem

Code evaluation requires standard library imports pre-loaded:
```python
# Without imports: NameError
sqrt(16)  # NameError: name 'sqrt' is not defined

# With imports pre-loaded: Works
from math import sqrt
sqrt(16)  # 4.0
```

### Solution: Discover Imports from Generated Code

After Phase 1 generation, before evaluation:

```python
# Phase 1, between Pass 1 and Pass 2:
if not Path("data/test_imports/metadata.json").exists():
    imports = discover_imports_ast(raw_outputs)
    save_imports("data/test_imports/metadata.json")
```

### Import Discovery Flow

```
Phase 1 Pass 1 (generate)
    ↓
Import Discovery (AST scan raw_generated_text)
    ↓
Save data/test_imports/metadata.json
    ↓
Phase 1 Pass 2 (evaluate with imports)
    ↓
Phase 3.5, 4.8, etc. (reuse imports file)
```

---

## Implementation Plan

### New Shared Utilities

```
common/
├── generation_utils.py    # GPU: batched generation helpers
├── evaluation_utils.py    # CPU: parallel evaluation with imports
└── import_discovery.py    # CPU: AST import scanning
```

### evaluation_utils.py

```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

def load_imports() -> list[str]:
    """Load pre-discovered imports for code evaluation."""
    imports_file = Path("data/test_imports/metadata.json")
    if imports_file.exists():
        with open(imports_file) as f:
            return json.load(f)['imports']
    return []

def evaluate_single(task: dict) -> dict:
    """Evaluate a single task (designed for parallel execution)."""
    code = extract_code(task['raw_generated_text'], task['prompt'])
    passed = evaluate_code_with_imports(code, task['test_list'])
    return {
        **task,
        'generated_code': code,
        'baseline_passed': passed,
    }

def evaluate_all_parallel(tasks: list[dict], max_workers: int = 8) -> list[dict]:
    """Evaluate all tasks in parallel."""
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(evaluate_single, tasks))
```

### import_discovery.py

```python
import ast
from collections import Counter

def extract_imports_ast(code: str) -> set[str]:
    """Extract import statements using AST parsing."""
    imports = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.add(f"from {module} import {alias.name}")
    except SyntaxError:
        # Fallback: regex for unparseable code
        pass
    return imports

def discover_imports(raw_outputs: list[dict]) -> list[str]:
    """Scan all generated code and consolidate imports."""
    all_imports = Counter()

    for output in raw_outputs:
        imports = extract_imports_ast(output['raw_generated_text'])
        all_imports.update(imports)

    # Filter: keep standard library only
    valid_imports = filter_standard_library(all_imports)

    return sorted(valid_imports)
```

### CLI Flag Addition

```python
# run.py
parser.add_argument(
    '--eval-only',
    action='store_true',
    help='Skip generation, only run evaluation on existing outputs'
)
```

### Phases to Refactor

| Phase | File | Changes |
|-------|------|---------|
| 1 | `phase1_latent_selection_dataset/runner.py` | Two-pass + import discovery |
| 3.5 | `phase3_5_temperature_robustness/temperature_generator.py` | Two-pass |
| 4.8 | `phase4_8_steering_analysis/steering_effect_analyzer.py` | Two-pass |
| 7.3 | `phase7_3_instruct_baseline/instruct_baseline_generator.py` | Two-pass |
| 7.6 | `phase7_6_instruct_steering/instruct_steering_analyzer.py` | Two-pass |
| 8.3 | `phase8_3_selective_steering/selective_steering_analyzer.py` | Two-pass |

---

## Output File Structure

### Phase 1 Example

**After Pass 1 (generation):**
```
data/phase1_0/
├── generations_raw.parquet      # Checkpoint
│   ├── task_id
│   ├── prompt
│   ├── raw_generated_text       # Full model output
│   ├── generation_time
│   └── test_list
├── activations/
│   └── task_activations/*.safetensors
└── phase_output.json
```

**After Pass 2 (evaluation):**
```
data/phase1_0/
├── generations_raw.parquet      # Kept for re-runs
├── results.parquet              # Final labeled data
│   ├── task_id
│   ├── prompt
│   ├── raw_generated_text
│   ├── generated_code           # Extracted
│   ├── baseline_passed          # True/False
│   ├── error_message
│   └── ...
├── activations/
└── phase_output.json
```

---

## Migration Path

### Step 1: Create Shared Utilities
- `common/evaluation_utils.py`
- `common/import_discovery.py`

### Step 2: Refactor Phase 1
- Implement two-pass pattern
- Add import discovery between passes
- Test with `--start 0 --end 50`

### Step 3: Refactor Remaining Phases
- Apply same pattern to 3.5, 4.8, 7.3, 7.6, 8.3
- Each uses `evaluate_all_parallel()` from shared utils

### Step 4: Add CLI Support
- `--eval-only` flag in run.py
- Update CLAUDE.md documentation

---

## Questions to Resolve

1. **Checkpoint format:** Parquet or separate files per task?
2. **Activation saving:** During Pass 1 or separate?
3. **Memory management:** Clear GPU memory between Pass 1 and Pass 2?
4. **Error handling:** What if some generations fail? Continue or stop?

---

## Related Files

- `docs/mbpp_import_analysis.md` - Original import analysis
- `docs/test_mbpp_with_imports.py` - Import test script with hardcoded list
- `common/dataset_utils.py:evaluate_code()` - Current evaluation function

---

## Notes

- This refactor addresses MBPP imports systematically
- Same pattern applies to HumanEval (already has imports file)
- Could extend to other datasets in future (APPS, CodeContests)
