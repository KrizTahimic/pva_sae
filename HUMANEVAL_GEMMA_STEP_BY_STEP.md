# HumanEval + Gemma-2-2B: Step-by-Step Implementation Guide

## Goal
Run full mechanistic analysis pipeline on HumanEval dataset using Gemma-2-2B model, reusing features from MBPP+Gemma.

**Priority**: 🎯 First implementation (lowest risk, validates architecture)

---

## 🔍 Critical Understanding

### MBPP Split Structure (What we already have)
- **50% SAE Split** (`sae_mbpp.parquet`) → Used by **Phase 1** → For feature **discovery**
- **10% Hyperparams** (`hyperparams_mbpp.parquet`) → Used by **Phase 3.6** → For tuning
- **40% Validation** (`validation_mbpp.parquet`) → Used by **Phase 3.5** → For feature **validation**

### For HumanEval (What we need)
- **All 164 problems** = Validation set (no discovery needed!)
- We're **reusing MBPP features** → No Phase 1, Phase 2.5, etc.
- We need **Phase 3.5 equivalent** at temp=0 to generate code + capture activations

## Overview

**What we're doing**:
- Extend validation pipeline (Phase 3.5+) to work with HumanEval
- Keep Gemma-2-2B model (no model changes)
- Reuse features from Phase 2.5, 2.10, 4.5, 4.6 (MBPP+Gemma)
- Output to `data/phase*_humaneval/` directories

**What we're NOT doing**:
- ❌ No Phase 1 (that's for SAE discovery split - not needed!)
- ❌ No feature selection (Phases 2.5, 2.10, 4.5, 4.6)
- ❌ No model changes
- ❌ No Phase 0/0.1 (HumanEval has no difficulty splits)

---

## 🤔 Open Planning Questions

### Phase 0.2 Architecture Decision

**User Proposal**: Create Phase 0.2 to pre-convert HumanEval dataset to MBPP format

**Questions**:

1. **Phase 0.2 output format**: Should Phase 0.2 produce a single `humaneval.parquet` file with the same schema as MBPP (with columns like `task_id`, `text` (problem description), `test_list`, etc.)? This way all downstream phases just see "MBPP-like" data?

2. **Field mapping**: For the conversion, I'm thinking:
   - HumanEval `task_id` → MBPP `task_id`
   - HumanEval `prompt` → MBPP `text` (problem description)
   - HumanEval `test` + `entry_point` → MBPP `test_list` (convert to assertion format)
   - HumanEval `canonical_solution` → Keep as reference but not used

   Does this mapping make sense?

3. **Test format conversion**: HumanEval uses a single `test` string with `check()` function. MBPP uses a list of assertion strings. Should we:
   - **Option A**: Parse the HumanEval test function and extract individual assertions
   - **Option B**: Keep the whole test function as-is in a single-item list
   - **Option C**: Use HumanEval's test format directly and update evaluation logic

4. **Downstream changes**: If we create Phase 0.2, then:
   - Phase 3.5 just loads `humaneval.parquet` (no dataset_loader needed)
   - All phases use the same code paths as MBPP
   - Only difference: `--dataset humaneval` loads from Phase 0.2 instead of Phase 0.1

   Is this the vision?

**Decision**: ✅ **Confirmed - Create Phase 0.2**

### Confirmed Decisions

1. **Test format conversion**: ✅ **Option A (Parse and extract assertions)**
   - Parse `check(candidate)` function and extract individual assertions
   - Replace `candidate` with actual function name from `entry_point`
   - Reason: Consistency in prompting for attention analysis phase

2. **Output schema**: ✅ **Match Phase 0.1 validation_mbpp.parquet exactly**
   ```python
   {
       "task_id": 0,  # int (0-163), not string
       "text": "from typing import List\n\ndef has_close_elements...",  # Full prompt
       "code": "    for idx, elem in enumerate...",  # canonical_solution
       "test_list": ["assert has_close_elements([1.0, 2.0], 0.3) == True", ...],  # Parsed assertions
       "cyclomatic_complexity": 0  # Calculate or set to 0
   }
   ```
   - Reason: Existing downstream code will work without changes

3. **Evaluation strategy**: ✅ **Use existing MBPP evaluation logic**
   - Since we're converting to MBPP format, evaluation "just works"
   - No need for special HumanEval evaluation code

4. **Output location**: ✅ **`data/phase0_2_humaneval/humaneval.parquet`**
   - Matches pattern of other phases (e.g., `phase3_5_humaneval`)

---

## Step 0.2: Create Phase 0.2 (HumanEval Preprocessing)

**NEW PHASE**: Convert HumanEval to MBPP format

**File**: `phase0_2_humaneval_preprocessing/converter.py` (NEW)

**Task**: Load HumanEval from HuggingFace and convert to MBPP schema

### Implementation Plan

```python
from datasets import load_dataset
import pandas as pd
import ast
import re

def parse_humaneval_test(test_code: str, entry_point: str) -> list:
    """
    Parse HumanEval test function and extract assertions.

    Input:
        def check(candidate):
            assert candidate([1.0, 2.0], 0.3) == True
            assert candidate([1.0, 2.0], 0.05) == False

    Output:
        [
            "assert has_close_elements([1.0, 2.0], 0.3) == True",
            "assert has_close_elements([1.0, 2.0], 0.05) == False"
        ]
    """
    # Extract assertions from check function
    # Replace 'candidate' with actual function name
    assertions = []
    for line in test_code.split('\n'):
        if 'assert candidate' in line:
            # Replace candidate with entry_point
            assertion = line.strip().replace('candidate', entry_point)
            assertions.append(assertion)

    return assertions

def convert_humaneval_to_mbpp():
    """Convert HumanEval dataset to MBPP format."""
    # Load HumanEval
    dataset = load_dataset("openai_humaneval", split="test")

    records = []
    for idx, problem in enumerate(dataset):
        # Parse test assertions
        test_list = parse_humaneval_test(problem['test'], problem['entry_point'])

        record = {
            'task_id': idx,  # 0-163
            'text': problem['prompt'],  # Full function signature + docstring
            'code': problem['canonical_solution'],  # Reference solution
            'test_list': test_list,  # Parsed assertions
            'cyclomatic_complexity': 0  # Set to 0 or calculate if needed
        }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to parquet
    output_dir = Path("data/phase0_2_humaneval")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "humaneval.parquet"
    df.to_parquet(output_file, index=False)

    print(f"Converted {len(df)} HumanEval problems")
    print(f"Saved to {output_file}")

    return df
```

### Testing Plan

```bash
# Run conversion
python3 run.py phase 0.2

# Verify output
python3 -c "
import pandas as pd
df = pd.read_parquet('data/phase0_2_humaneval/humaneval.parquet')
print(f'Total: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print(f'Sample task_id: {df.iloc[0].task_id}')
print(f'Sample text (first 100 chars): {df.iloc[0].text[:100]}')
print(f'Sample test_list: {df.iloc[0].test_list[:2]}')
"
```

**Checklist**:
- [ ] Create `phase0_2_humaneval_preprocessing/` directory
- [ ] Implement `converter.py` with `parse_humaneval_test()` function
- [ ] Add Phase 0.2 handler to `run.py`
- [ ] Test: Run Phase 0.2
- [ ] Verify: 164 problems converted
- [ ] Verify: Schema matches `validation_mbpp.parquet`
- [ ] Verify: Assertions properly parsed (candidate → function name)
- [ ] Commit: `git add phase0_2_humaneval_preprocessing/ && git commit -m "Add Phase 0.2: HumanEval preprocessing"`

---

## Step 1: Understand HumanEval Format

### Current State: MBPP Format
```python
{
    "task_id": 1,
    "text": "Write a function to find the similar elements from the given two tuple lists.",
    "code": "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) ",
    "test_list": [
        "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
        "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
        "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
    ]
}
```

**MBPP Prompt Template** (from `common/prompt_utils.py`):
```
{text}
{test_list[0]}
{test_list[1]}
{test_list[2]}
# Solution:
```

### Target State: HumanEval Format
```python
{
    "task_id": "HumanEval/0",
    "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
    "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    ...\n",
    "entry_point": "has_close_elements"
}
```

**Key Differences**:
1. HumanEval `prompt` = function signature + docstring (NOT problem description)
2. HumanEval `test` = function that calls `candidate()` (NOT assertions)
3. HumanEval expects function body completion (NOT full function)
4. Task IDs are strings ("HumanEval/0") not integers

### Decision: Prompt Transformation Strategy

**Option A: Keep HumanEval format, minimal changes**
- Prompt = `prompt` field + "# Solution:\n    "
- Evaluation = execute `test` function with generated code
- Pros: Preserves HumanEval format, easier
- Cons: Different evaluation logic

**Option B: Transform to MBPP-like format**
- Extract docstring as problem description
- Convert test function to assertions
- Pros: Reuses existing MBPP evaluation
- Cons: More complex transformation

**✅ Chosen: Option A (minimal changes)**

---

## Step 2: Update Phase 3.5 to Load from Phase 0.2

**File**: `phase3_5_temperature_robustness/temperature_runner.py`

**Task**: Make Phase 3.5 load HumanEval from Phase 0.2 (instead of Phase 0.1)

### Why This is Simple Now

With Phase 0.2 converting HumanEval to MBPP format:
- ✅ No need for `dataset_loader.py`
- ✅ No need for `HumanEvalPromptBuilder`
- ✅ No need for special evaluation logic
- ✅ Just load from Phase 0.2 instead of Phase 0.1!

### Implementation Plan

Update `_load_validation_data()` method:

**Before**:
```python
def _load_validation_data(self) -> pd.DataFrame:
    """Load validation data from Phase 0.1."""
    validation_file = Path(self.config.phase0_1_output_dir) / "validation_mbpp.parquet"
    if not validation_file.exists():
        raise FileNotFoundError(f"Validation data not found at {validation_file}")
    return pd.read_parquet(validation_file)
```

**After**:
```python
def _load_validation_data(self) -> pd.DataFrame:
    """Load validation data from Phase 0.1 (MBPP) or Phase 0.2 (HumanEval)."""
    if self.config.dataset_name == "mbpp":
        validation_file = Path(self.config.phase0_1_output_dir) / "validation_mbpp.parquet"
    elif self.config.dataset_name == "humaneval":
        validation_file = Path("data/phase0_2_humaneval") / "humaneval.parquet"
    else:
        raise ValueError(f"Unknown dataset: {self.config.dataset_name}")

    if not validation_file.exists():
        raise FileNotFoundError(f"Validation data not found at {validation_file}")

    return pd.read_parquet(validation_file)
```

**That's it!** Everything else works unchanged because HumanEval is now in MBPP format.

### OLD APPROACH (No Longer Needed)

~~The sections below (Steps 2-5) are NO LONGER NEEDED with Phase 0.2 approach.~~

<details>
<summary>Click to see old approach (for reference)</summary>

```python
class HumanEvalPromptBuilder:
    """
    Build prompts for HumanEval dataset.

    HumanEval provides function signature + docstring in 'prompt' field.
    We just need to add a code initiator for completion.
    """

    TEMPLATE = """{prompt}    # Solution:
"""

    @classmethod
    def build_prompt(cls, prompt: str, **kwargs) -> str:
        """
        Build HumanEval prompt.

        Args:
            prompt: HumanEval 'prompt' field (function signature + docstring)
            **kwargs: Ignored (for compatibility)

        Returns:
            Formatted prompt ready for model
        """
        return cls.TEMPLATE.format(prompt=prompt)
```

**Factory function** (update existing):
```python
def get_prompt_builder(dataset_name: str = "mbpp"):
    """Get prompt builder for dataset."""
    builders = {
        "mbpp": PromptBuilder,
        "humaneval": HumanEvalPromptBuilder,
    }
    return builders.get(dataset_name, PromptBuilder)
```

### Testing

Create test script `test_humaneval_prompt.py`:
```python
from common.prompt_utils import HumanEvalPromptBuilder

# Sample HumanEval prompt
humaneval_prompt = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

# Build prompt
builder = HumanEvalPromptBuilder()
formatted = builder.build_prompt(humaneval_prompt)

print("="*80)
print("FORMATTED PROMPT:")
print("="*80)
print(formatted)
print("="*80)

# Expected output:
# from typing import List
#
#
# def has_close_elements(numbers: List[float], threshold: float) -> bool:
#     """ Check if in given list of numbers, are any two numbers closer to each other than
#     given threshold.
#     >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
#     False
#     >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
#     True
#     """
#     # Solution:
```

**Checklist**:
- [ ] Implement `HumanEvalPromptBuilder` class
- [ ] Add `get_prompt_builder()` factory function
- [ ] Create test script
- [ ] Run test script, verify output format
- [ ] Commit changes: `git add common/prompt_utils.py && git commit -m "Add HumanEval prompt builder"`

---

## Step 3: Create HumanEval Dataset Loader

**File**: `common/dataset_loader.py` (NEW)

**Task**: Create abstraction for loading MBPP and HumanEval datasets

### Implementation Plan

```python
from abc import ABC, abstractmethod
from typing import List, Dict
from datasets import load_dataset


class DatasetLoader(ABC):
    """Base class for dataset loaders."""

    @abstractmethod
    def load(self, split: str = None) -> List[Dict]:
        """Load dataset split."""
        pass

    @abstractmethod
    def get_problem_description(self, problem: Dict) -> str:
        """Extract problem description."""
        pass

    @abstractmethod
    def get_tests(self, problem: Dict) -> List[str]:
        """Extract test cases."""
        pass

    @abstractmethod
    def evaluate_code(self, code: str, problem: Dict) -> bool:
        """Evaluate if generated code passes tests."""
        pass


class MBPPDatasetLoader(DatasetLoader):
    """Loader for MBPP dataset."""

    def __init__(self, split_file: str = None):
        """
        Args:
            split_file: Path to Phase 0.1 split file (e.g., validation_mbpp.parquet)
        """
        self.split_file = split_file

    def load(self, split: str = None) -> List[Dict]:
        """Load MBPP dataset."""
        if self.split_file:
            # Load from Phase 0.1 split
            import pandas as pd
            df = pd.read_parquet(self.split_file)
            return df.to_dict('records')
        else:
            # Load from HuggingFace
            dataset = load_dataset("Muennighoff/mbpp", split=split or "test")
            return list(dataset)

    def get_problem_description(self, problem: Dict) -> str:
        return problem["text"]

    def get_tests(self, problem: Dict) -> List[str]:
        return problem["test_list"]

    def evaluate_code(self, code: str, problem: Dict) -> bool:
        """Evaluate by running assertions."""
        # Use existing evaluation logic
        from common_simplified.helpers import evaluate_code
        return evaluate_code(code, problem["test_list"])


class HumanEvalDatasetLoader(DatasetLoader):
    """Loader for HumanEval dataset."""

    def __init__(self):
        pass

    def load(self, split: str = None) -> List[Dict]:
        """Load HumanEval dataset from HuggingFace."""
        dataset = load_dataset("openai_humaneval", split=split or "test")
        return list(dataset)

    def get_problem_description(self, problem: Dict) -> str:
        """Return the prompt field (function signature + docstring)."""
        return problem["prompt"]

    def get_tests(self, problem: Dict) -> List[str]:
        """Return test function as single-item list."""
        return [problem["test"]]

    def evaluate_code(self, code: str, problem: Dict) -> bool:
        """
        Evaluate HumanEval code by running test function.

        HumanEval test format:
        ```python
        def check(candidate):
            assert candidate([1.0, 2.0], 0.3) == True
            assert candidate([1.0, 2.0], 0.05) == False
        ```

        We need to:
        1. Extract function from generated code
        2. Run check(extracted_function)
        """
        import re
        from io import StringIO
        import sys

        try:
            # Extract function name from prompt
            entry_point = problem["entry_point"]

            # Create execution environment
            exec_globals = {}

            # Execute generated code to define the function
            exec(code, exec_globals)

            # Check if function exists
            if entry_point not in exec_globals:
                return False

            # Get the function
            candidate = exec_globals[entry_point]

            # Execute test function
            test_code = problem["test"]
            test_globals = {"candidate": candidate}
            exec(test_code, test_globals)

            # If no assertion error, code is correct
            return True

        except Exception as e:
            # Any error means code failed
            return False


def get_dataset_loader(dataset_name: str = "mbpp", **kwargs) -> DatasetLoader:
    """
    Factory function to get dataset loader.

    Args:
        dataset_name: "mbpp" or "humaneval"
        **kwargs: Passed to loader constructor

    Returns:
        DatasetLoader instance
    """
    loaders = {
        "mbpp": MBPPDatasetLoader,
        "humaneval": HumanEvalDatasetLoader,
    }

    loader_class = loaders.get(dataset_name)
    if loader_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return loader_class(**kwargs)
```

### Testing

Create test script `test_humaneval_loader.py`:
```python
from common.dataset_loader import get_dataset_loader

# Load HumanEval dataset
loader = get_dataset_loader("humaneval")
problems = loader.load()

print(f"Loaded {len(problems)} HumanEval problems")
print()

# Test first problem
problem = problems[0]
print("="*80)
print(f"Task ID: {problem['task_id']}")
print("="*80)
print("PROMPT:")
print(loader.get_problem_description(problem))
print("="*80)
print("CANONICAL SOLUTION:")
print(problem['canonical_solution'])
print("="*80)

# Test evaluation with correct solution
full_code = problem['prompt'] + problem['canonical_solution']
result = loader.evaluate_code(full_code, problem)
print(f"Evaluation (should be True): {result}")
print("="*80)

# Test evaluation with wrong solution
wrong_code = problem['prompt'] + "    return False\n"
result = loader.evaluate_code(wrong_code, problem)
print(f"Evaluation (should be False): {result}")
print("="*80)
```

**Checklist**:
- [ ] Create `common/dataset_loader.py`
- [ ] Implement `DatasetLoader` base class
- [ ] Implement `MBPPDatasetLoader` (refactor existing code)
- [ ] Implement `HumanEvalDatasetLoader`
- [ ] Implement `get_dataset_loader()` factory
- [ ] Create test script
- [ ] Run test: `python test_humaneval_loader.py`
- [ ] Verify: 164 problems loaded
- [ ] Verify: Evaluation works for correct/incorrect code
- [ ] Commit: `git add common/dataset_loader.py && git commit -m "Add HumanEval dataset loader"`

---

## Step 4: Update Config for HumanEval

**File**: `common/config.py`

**Task**: Add `dataset_name` field and HumanEval output directories

### Implementation Plan

Add to `Config` dataclass:
```python
@dataclass
class Config:
    # ... existing fields ...

    # Dataset configuration
    dataset_name: str = "mbpp"  # or "humaneval"

    # ... existing fields ...
```

Add helper function:
```python
def get_phase_output_dir(phase: str, dataset: str = None, model: str = None) -> str:
    """
    Get output directory for phase with dataset/model suffix.

    Args:
        phase: Phase number (e.g., "1", "3.8", "4.8")
        dataset: Dataset name ("mbpp", "humaneval", None for default)
        model: Model variant ("gemma", "llama", None for default)

    Returns:
        Directory path with suffix if needed

    Examples:
        get_phase_output_dir("1") -> "data/phase1_0"
        get_phase_output_dir("1", "humaneval") -> "data/phase1_0_humaneval"
        get_phase_output_dir("1", "mbpp", "llama") -> "data/phase1_0_llama"
        get_phase_output_dir("1", "humaneval", "llama") -> "data/phase1_0_humaneval_llama"
    """
    # Load config instance
    config = Config()

    # Use defaults if not specified
    if dataset is None:
        dataset = config.dataset_name
    if model is None:
        model = config.model_variant  # Will add this field later for LLAMA

    # Base directory
    base_dir = f"data/phase{phase.replace('.', '_')}"

    # Build suffix
    suffix_parts = []
    if dataset != "mbpp":
        suffix_parts.append(dataset)
    if model != "gemma":
        suffix_parts.append(model)

    if suffix_parts:
        suffix = "_" + "_".join(suffix_parts)
    else:
        suffix = ""

    return base_dir + suffix
```

### Testing

Create test script `test_config.py`:
```python
from common.config import get_phase_output_dir

# Test cases
tests = [
    (("1",), {}, "data/phase1_0"),
    (("1",), {"dataset": "humaneval"}, "data/phase1_0_humaneval"),
    (("3.8",), {"dataset": "humaneval"}, "data/phase3_8_humaneval"),
    (("4.8",), {"dataset": "humaneval"}, "data/phase4_8_humaneval"),
]

print("Testing get_phase_output_dir():")
print("="*80)

for args, kwargs, expected in tests:
    result = get_phase_output_dir(*args, **kwargs)
    status = "✓" if result == expected else "✗"
    print(f"{status} get_phase_output_dir{args, kwargs}")
    print(f"   Expected: {expected}")
    print(f"   Got:      {result}")
    print()
```

**Checklist**:
- [ ] Add `dataset_name` field to `Config`
- [ ] Implement `get_phase_output_dir()` helper
- [ ] Create test script
- [ ] Run test: `python test_config.py`
- [ ] Verify all test cases pass
- [ ] Commit: `git add common/config.py && git commit -m "Add dataset_name config and output directory helper"`

---

## Step 5: Update CLI Arguments

**File**: `run.py`

**Task**: Add `--dataset` argument

### Implementation Plan

Add argparse argument:
```python
parser.add_argument(
    '--dataset',
    type=str,
    default='mbpp',
    choices=['mbpp', 'humaneval'],
    help='Dataset to use (default: mbpp)'
)
```

Pass to config:
```python
# After parsing args
config = Config()
if args.dataset:
    config.dataset_name = args.dataset
```

### Testing

```bash
# Test help
python3 run.py --help
# Should show --dataset option

# Test invalid dataset
python3 run.py phase 1 --dataset invalid
# Should show error: invalid choice: 'invalid'

# Test valid dataset
python3 run.py phase 1 --dataset humaneval --start 0 --end 1
# Should accept (will fail at phase execution, but that's OK for now)
```

**Checklist**:
- [ ] Add `--dataset` argument to argparse
- [ ] Pass dataset to config
- [ ] Test `--help` shows new option
- [ ] Test invalid dataset shows error
- [ ] Commit: `git add run.py && git commit -m "Add --dataset CLI argument"`

---

## Step 6: Update Phase 3.5 for HumanEval

**File**: `phase3_5_temperature_robustness/temperature_runner.py`

**Task**: Make Phase 3.5 work with HumanEval dataset

### Current Phase 3.5 Flow

1. Load MBPP **validation split** from Phase 0.1
2. For each problem at each temperature:
   - Build MBPP prompt
   - Generate code (capture activations only at temp=0)
   - Evaluate with MBPP assertions
3. Save to `data/phase3_5/`

### Why Phase 3.5, not Phase 1?

**Phase 1** = Uses SAE split (50% of MBPP) for **feature discovery**
**Phase 3.5** = Uses **validation split** (40% of MBPP) for **feature validation**

For HumanEval:
- We're **reusing MBPP features** (no discovery needed)
- All 164 HumanEval problems = our **validation set**
- We need to run Phase 3.5 at **temp=0** to generate code + capture activations

### Required Changes

1. **Dataset loading**: Use `get_dataset_loader()` instead of loading `validation_mbpp.parquet`
2. **Prompt building**: Use `get_prompt_builder()`
3. **Evaluation**: Use loader's `evaluate_code()` method
4. **Output directory**: Use `get_phase_output_dir("3.5", config.dataset_name)`

### Implementation Plan

Update `_load_validation_data()` method:

**Before** (current code):
```python
def _load_validation_data(self) -> pd.DataFrame:
    """Load validation data from Phase 0.1."""
    validation_file = Path(self.config.phase0_1_output_dir) / "validation_mbpp.parquet"

    if not validation_file.exists():
        raise FileNotFoundError(
            f"Validation data not found at {validation_file}. "
            "Please run Phase 0.1 first."
        )

    return pd.read_parquet(validation_file)
```

**After** (updated code):
```python
def _load_validation_data(self) -> pd.DataFrame:
    """Load validation data - either from Phase 0.1 (MBPP) or HuggingFace (HumanEval)."""
    from common.dataset_loader import get_dataset_loader

    if self.config.dataset_name == "mbpp":
        # Load from Phase 0.1 validation split
        validation_file = Path(self.config.phase0_1_output_dir) / "validation_mbpp.parquet"
        if not validation_file.exists():
            raise FileNotFoundError(f"Validation data not found at {validation_file}")
        return pd.read_parquet(validation_file)

    elif self.config.dataset_name == "humaneval":
        # Load all 164 problems from HuggingFace
        loader = get_dataset_loader("humaneval")
        problems = loader.load()
        # Convert to DataFrame
        return pd.DataFrame(problems)

    else:
        raise ValueError(f"Unknown dataset: {self.config.dataset_name}")
```

Similarly update prompt building and evaluation throughout the file.

### Testing Plan

**Test 1: Dry run on 1 problem, temp=0 only**
```bash
python3 run.py phase 3.5 --dataset humaneval --start 0 --end 1
```

**Expected**:
- Loads HumanEval problem 0
- Generates code at temp=0 (and other temps if configured)
- Captures activations at temp=0
- Saves to `data/phase3_5_humaneval/`

**Test 2: Run on 10 problems**
```bash
python3 run.py phase 3.5 --dataset humaneval --start 0 --end 10
```

**Test 3: Verify MBPP still works**
```bash
python3 run.py phase 3.5 --dataset mbpp --start 0 --end 10
```

**Checklist**:
- [ ] Update `_load_validation_data()` in Phase 3.5
- [ ] Update prompt building
- [ ] Update evaluation
- [ ] Update output directory
- [ ] Test: Run on HumanEval problem 0
- [ ] Verify: Prompt correct
- [ ] Verify: Code generated
- [ ] Verify: Activations captured (temp=0)
- [ ] Verify: Output saved to `data/phase3_5_humaneval/`
- [ ] Test: Run on 10 problems
- [ ] Test: MBPP still works
- [ ] Commit: `git add phase3_5_temperature_robustness/ && git commit -m "Add HumanEval support to Phase 3.5"`

---

## Step 7: Run Full Phase 3.5 on HumanEval (temp=0 only)

**Task**: Generate HumanEval dataset with activations at temperature=0

### Command

```bash
# Start screen session
screen -S phase3_5_humaneval

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pva_sae

# Run Phase 3.5 on all 164 HumanEval problems
# Note: Only temp=0 will capture activations
python3 run.py phase 3.5 --dataset humaneval

# Detach: Ctrl+A, D
```

### Expected Output

```
data/phase3_5_humaneval/
├── temperature_0.0/
│   ├── dataset_*.parquet  # 164 rows
│   ├── activations/
│   │   ├── correct/
│   │   │   ├── HumanEval_0_layer_16.npz  # Only best layers
│   │   │   └── ...
│   │   └── incorrect/
│   │       └── ...
├── temperature_0.3/  # If configured
├── temperature_0.6/
└── ...
```

### Validation

```python
import pandas as pd

# Load results
df = pd.read_parquet("data/phase3_5_humaneval/temperature_0.0/dataset_*.parquet")

print(f"Total problems: {len(df)}")
print(f"Correct: {df['is_correct'].sum()}")
print(f"Incorrect: {(~df['is_correct']).sum()}")
print(f"Pass rate: {df['is_correct'].mean():.2%}")
print(f"First task: {df['task_id'].iloc[0]}")
print(f"Last task: {df['task_id'].iloc[-1]}")
```

**Checklist**:
- [ ] Start screen session
- [ ] Run Phase 3.5 on full HumanEval
- [ ] Monitor progress (reattach periodically)
- [ ] Wait for completion (~2-6 hours)
- [ ] Validate: 164 problems processed
- [ ] Validate: Activations saved at temp=0
- [ ] Validate: Pass rate reasonable (Gemma-2B ~20-40% on HumanEval)
- [ ] Document results in notes

---

## Step 8: Update Phase 3.8 (AUROC/F1 Evaluation)

**File**: `phase3_8_evaluation/auroc_f1_evaluator.py`

**Task**: Make Phase 3.8 work with HumanEval activations

### Current Flow

1. Load Phase 3.5 temp=0 output from `data/phase3_5/temperature_0.0/`
2. Load features from Phase 2.10 `data/phase2_10/`
3. Load activations (already SAE-decomposed in Phase 3.5)
4. Compute AUROC and F1 scores
5. Save to `data/phase3_8/`

### Required Changes

1. **Input discovery**: Auto-discover `data/phase3_5_humaneval/temperature_0.0/` when `--dataset humaneval`
2. **Feature loading**: Still load from `data/phase2_10/` (MBPP+Gemma features)
3. **Output directory**: Save to `data/phase3_8_humaneval/`

### Implementation Plan

Update input discovery:
```python
from common.config import config, get_phase_output_dir

# Auto-discover Phase 3.5 temp=0 output
phase3_5_dir = get_phase_output_dir("3.5", config.dataset_name)
temp_0_dir = phase3_5_dir / "temperature_0.0"
dataset_file = discover_latest_file(temp_0_dir, "dataset_*.parquet")

# Load features from MBPP+Gemma (no suffix)
phase2_10_dir = get_phase_output_dir("2.10")  # Always MBPP+Gemma for HumanEval+Gemma
features_file = discover_latest_file(phase2_10_dir, "top_20_features.json")

# Output directory
output_dir = get_phase_output_dir("3.8", config.dataset_name)
```

### Testing

```bash
python3 run.py phase 3.8 --dataset humaneval
```

**Expected**:
- Loads `data/phase3_5_humaneval/temperature_0.0/dataset_*.parquet`
- Loads activations from `data/phase3_5_humaneval/temperature_0.0/activations/`
- Loads features from `data/phase2_10/top_20_features.json`
- Computes AUROC/F1 scores
- Saves to `data/phase3_8_humaneval/results_*.json`

**Checklist**:
- [ ] Update input discovery in Phase 3.8
- [ ] Update output directory
- [ ] Test: Run Phase 3.8 on HumanEval
- [ ] Verify: Loads HumanEval data from Phase 3.5
- [ ] Verify: Loads MBPP features (cross-dataset transfer)
- [ ] Verify: Computes metrics
- [ ] Verify: Saves to `data/phase3_8_humaneval/`
- [ ] Compare: AUROC/F1 for HumanEval vs MBPP (expect similar or slightly different)
- [ ] Commit: `git add phase3_8_evaluation/ && git commit -m "Add HumanEval support to Phase 3.8"`

---

## Step 9: Update Remaining Phases

**Phases to update**: 4.8, 5.3, 6.3, 8.3

For each phase:
1. Update input discovery to use `get_phase_output_dir()`
2. Update output directory
3. Test on small subset
4. Run full pipeline
5. Validate results

### Phase 4.8: Steering Analysis

**Changes**:
- Input: `data/phase3_5_humaneval/temperature_0.0/`
- Features: `data/phase2_10/` (MBPP+Gemma)
- Coefficients: `data/phase4_6/` (MBPP+Gemma)
- Output: `data/phase4_8_humaneval/`

**Test**:
```bash
python3 run.py phase 4.8 --dataset humaneval
```

### Phase 5.3: Weight Orthogonalization

**Changes**:
- Input: `data/phase3_5_humaneval/temperature_0.0/`
- Features: `data/phase2_10/`
- Output: `data/phase5_3_humaneval/`

**Test**:
```bash
python3 run.py phase 5.3 --dataset humaneval
```

### Phase 6.3: Attention Analysis

**Changes**:
- Input: `data/phase3_5_humaneval/temperature_0.0/`
- Features: `data/phase2_10/`
- Output: `data/phase6_3_humaneval/`

**Test**:
```bash
python3 run.py phase 6.3 --dataset humaneval
```

### Phase 8.3: Selective Steering

**Changes**:
- Input: `data/phase3_5_humaneval/temperature_0.0/`
- Features: `data/phase2_10/`
- Thresholds: `data/phase8_2/`
- Output: `data/phase8_3_humaneval/`

**Test**:
```bash
python3 run.py phase 8.3 --dataset humaneval
```

**Checklist**:
- [ ] Update Phase 4.8
- [ ] Test Phase 4.8 on HumanEval
- [ ] Update Phase 5.3
- [ ] Test Phase 5.3 on HumanEval
- [ ] Update Phase 6.3
- [ ] Test Phase 6.3 on HumanEval
- [ ] Update Phase 8.3
- [ ] Test Phase 8.3 on HumanEval
- [ ] Commit all changes

---

## Step 10: Validate Full Pipeline

### Checklist

**Data validation**:
- [ ] `data/phase3_5_humaneval/temperature_0.0/` exists with 164 problems + activations
- [ ] `data/phase3_8_humaneval/` has AUROC/F1 results
- [ ] `data/phase4_8_humaneval/` has steering results
- [ ] `data/phase5_3_humaneval/` has orthogonalization results
- [ ] `data/phase6_3_humaneval/` has attention results
- [ ] `data/phase8_3_humaneval/` has selective steering results

**Result validation**:
- [ ] HumanEval pass rate reasonable (20-40% for Gemma-2B)
- [ ] AUROC/F1 scores reasonable (compare to MBPP)
- [ ] Steering effects present (correction > 0%, corruption > 0%)
- [ ] Attention patterns similar to MBPP

**Documentation**:
- [ ] Update experiment log with HumanEval results
- [ ] Note differences from MBPP (if any)
- [ ] Document any issues or surprises

---

## Summary Checklist

### Steps Completed

- [ ] Step 1: Understand HumanEval format
- [ ] Step 2: Create HumanEval prompt builder
- [ ] Step 3: Create HumanEval dataset loader
- [ ] Step 4: Update config for HumanEval
- [ ] Step 5: Update CLI arguments
- [ ] Step 6: Update Phase 1
- [ ] Step 7: Run full Phase 1
- [ ] Step 8: Update Phase 3.8
- [ ] Step 9: Update remaining phases (4.8, 5.3, 6.3, 8.3)
- [ ] Step 10: Validate full pipeline

### Files Modified

- [ ] `common/prompt_utils.py` - HumanEval prompt builder
- [ ] `common/dataset_loader.py` - NEW FILE
- [ ] `common/config.py` - Dataset config and helpers
- [ ] `run.py` - CLI arguments
- [ ] `phase1_simplified/runner.py` - HumanEval support
- [ ] `phase3_8_evaluation/` - HumanEval support
- [ ] `phase4_8_steering_analysis/` - HumanEval support
- [ ] `phase5_3_*/` - HumanEval support
- [ ] `phase6_3_*/` - HumanEval support
- [ ] `phase8_3_*/` - HumanEval support

### Expected Outputs

```
data/
├── phase3_5_humaneval/
│   └── temperature_0.0/         # ✓ 164 problems + activations
├── phase3_8_humaneval/          # ✓ AUROC/F1 results
├── phase4_8_humaneval/          # ✓ Steering results
├── phase5_3_humaneval/          # ✓ Orthogonalization results
├── phase6_3_humaneval/          # ✓ Attention results
└── phase8_3_humaneval/          # ✓ Selective steering results
```

---

## Next Steps

After completing HumanEval + Gemma:
1. Update paper with HumanEval results
2. Compare MBPP vs HumanEval findings
3. Move to LLAMA + MBPP implementation
4. Then LLAMA + HumanEval

**Success criteria**: All phases run successfully on HumanEval, results are reasonable and comparable to MBPP.
