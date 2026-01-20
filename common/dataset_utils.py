"""
Dataset utilities for common data operations.

This module provides utilities for:
- Splitting datasets by correctness (pass/fail)
- Discovering task IDs from activation files
- Code extraction and evaluation
- Activation loading and SAE encoding
"""

import contextlib
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from common.config import Config

import numpy as np
import pandas as pd
import torch
from einops import rearrange

from .logging import get_logger
from .tensor_utils import load_activation

logger = get_logger("common.dataset_utils")


# ============================================================================
# Error Type Constants
# ============================================================================

# Canonical error type categories (order for display)
ERROR_TYPES = ["passed", "syntax", "name", "type", "logic", "runtime", "timeout"]


# ============================================================================
# Error Type Distribution Helper
# ============================================================================

def compute_error_type_distribution(
    records: Union[pd.DataFrame, list[dict]],
    error_type_column: str
) -> dict:
    """
    Compute error type counts and percentages for a collection of records.

    This is the single source of truth for error type distribution computation.
    All generation phases use this function to ensure consistent output schema.

    Args:
        records: DataFrame or list of dicts containing error type data
        error_type_column: Name of the column containing error type values

    Returns:
        Dictionary with structure:
        {
            "total": int,
            "counts": {"passed": int, "syntax": int, ...},
            "percentages": {"passed": float, "syntax": float, ...}
        }

    Example:
        >>> from common.dataset_utils import compute_error_type_distribution
        >>> dist = compute_error_type_distribution(results_df, "baseline_error_type")
        >>> print(f"Pass rate: {dist['percentages']['passed']:.1f}%")
    """
    # Convert to DataFrame if needed
    if isinstance(records, list):
        if not records:
            return {
                "total": 0,
                "counts": {et: 0 for et in ERROR_TYPES},
                "percentages": {et: 0.0 for et in ERROR_TYPES}
            }
        df = pd.DataFrame(records)
    else:
        df = records

    if error_type_column not in df.columns:
        logger.warning(f"Column '{error_type_column}' not found in data")
        return {
            "total": 0,
            "counts": {et: 0 for et in ERROR_TYPES},
            "percentages": {et: 0.0 for et in ERROR_TYPES}
        }

    total = len(df)
    counts = df[error_type_column].value_counts()

    result = {
        "total": total,
        "counts": {},
        "percentages": {}
    }

    for error_type in ERROR_TYPES:
        count = int(counts.get(error_type, 0))
        result["counts"][error_type] = count
        result["percentages"][error_type] = round(count / total * 100, 2) if total > 0 else 0.0

    return result


# ============================================================================
# Evaluation Result Data Structure
# ============================================================================

@dataclass
class EvaluationResult:
    """
    Result of code evaluation with error type classification.

    Error types:
        - "passed": All tests pass
        - "syntax": SyntaxError, IndentationError (compilation errors)
        - "name": NameError, AttributeError (reference errors)
        - "type": TypeError (type mismatches)
        - "logic": AssertionError (test fails - condition/operation errors)
        - "runtime": IndexError, ValueError, KeyError, etc. (runtime exceptions)
        - "timeout": Execution timeout (infinite loop)
    """
    passed: bool
    error_type: str  # "passed", "syntax", "name", "type", "logic", "runtime", "timeout"
    error_message: Optional[str]
    exception_class: Optional[str]  # e.g., "IndexError"


# ============================================================================
# Dataset Splitting
# ============================================================================

def split_by_correctness(
    df: pd.DataFrame,
    correctness_col: str = 'baseline_passed',
    verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into correct and incorrect subsets.

    Args:
        df: DataFrame with correctness column
        correctness_col: Name of the boolean column indicating correctness
        verbose: If True, log the split statistics

    Returns:
        Tuple of (correct_df, incorrect_df)

    Example:
        >>> correct, incorrect = split_by_correctness(baseline_data)
        >>> print(f"Correct: {len(correct)}, Incorrect: {len(incorrect)}")
    """
    correct = df[df[correctness_col] == True].copy()
    incorrect = df[df[correctness_col] == False].copy()

    if verbose:
        total = len(df)
        logger.info(f"Split complete: {len(correct)} correct ({len(correct)/total*100:.1f}%), "
                   f"{len(incorrect)} incorrect ({len(incorrect)/total*100:.1f}%)")

    return correct, incorrect


def discover_task_ids(
    directory: Path,
    pattern: str = "*_layer_*.safetensors"
) -> list[str]:
    """
    Extract unique task IDs from activation files.

    Activation files are expected to follow the naming convention:
    {task_id}_layer_{layer_num}.safetensors

    Args:
        directory: Directory containing activation files
        pattern: Glob pattern to match activation files

    Returns:
        Sorted list of unique task IDs

    Example:
        >>> task_ids = discover_task_ids(Path("data/phase1_0/activations/correct"))
        >>> print(f"Found {len(task_ids)} tasks")
    """
    task_ids = set()

    for file in directory.glob(pattern):
        parts = file.stem.split('_layer_')
        if len(parts) == 2:
            task_ids.add(parts[0])

    return sorted(list(task_ids))


def discover_layer_indices(
    directory: Path,
    pattern: str = "*_layer_*.safetensors"
) -> list[int]:
    """
    Extract unique layer indices from activation files.

    Args:
        directory: Directory containing activation files
        pattern: Glob pattern to match activation files

    Returns:
        Sorted list of unique layer indices
    """
    layer_indices = set()

    for file in directory.glob(pattern):
        parts = file.stem.split('_layer_')
        if len(parts) == 2:
            try:
                layer_indices.add(int(parts[1]))
            except ValueError:
                continue

    return sorted(list(layer_indices))


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset_split(split_name: str, phase0_1_dir: Path, config: 'Config') -> pd.DataFrame:
    """
    Load dataset split for any supported dataset.

    This is the preferred function for loading dataset splits as it
    supports both MBPP and HumanEval transparently based on config.

    Args:
        split_name: Split identifier ("selection", "tuning", or "analysis")
        phase0_1_dir: Phase 0.1 output directory
        config: Config object with dataset_name attribute

    Returns:
        DataFrame with columns: task_id, text, code, test_list, cyclomatic_complexity

    Raises:
        FileNotFoundError: If split file doesn't exist

    Example:
        >>> from common.config import Config
        >>> config = Config()  # Uses config.dataset_name
        >>> df = load_dataset_split("selection", Path("data/phase0_1"), config)
    """
    split_file = phase0_1_dir / f"{split_name}_{config.dataset_name}.parquet"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Run: python3 run.py phase 0.1 (with dataset_name='{config.dataset_name}' in config)"
        )

    df = pd.read_parquet(split_file)
    logger.info(f"Loaded {len(df)} problems from {split_name} split ({config.dataset_name})")
    return df


# ============================================================================
# Code Extraction and Evaluation
# ============================================================================

def _extract_raw_code(generated_text: str, prompt: str) -> str:
    """
    Try multiple extraction methods, return first success.

    Args:
        generated_text: Generated text (may or may not include prompt)
        prompt: Original prompt to remove if present

    Returns:
        Extracted raw code (may include extra content after function)
    """
    # Method 1: Exact prompt match (works for Gemma)
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()

    # Method 2: Solution marker (our code_initiator)
    solution_marker = "# Solution:"
    marker_idx = generated_text.find(solution_marker)
    if marker_idx != -1:
        return generated_text[marker_idx + len(solution_marker):].strip()

    # Method 3: After last assert (handles whitespace differences)
    last_assert_idx = generated_text.rfind("assert ")
    if last_assert_idx != -1:
        newline_after = generated_text.find('\n', last_assert_idx)
        if newline_after != -1:
            return generated_text[newline_after:].strip()

    # Fallback: entire text
    return generated_text.strip()


def _trim_to_function(code: str) -> str:
    """
    Extract just the function definition from code.

    Args:
        code: Code that may contain extra content after the function

    Returns:
        Just the function definition
    """
    def_index = code.find('def ')
    if def_index == -1:
        return code.strip()

    code = code[def_index:]

    # Find end of function: newline followed by non-whitespace
    for i in range(4, len(code) - 1):  # Skip past "def "
        if code[i] == '\n' and code[i + 1] not in ' \t\n':
            return code[:i].rstrip()

    return code.strip()


def extract_code(generated_text: str, prompt: str) -> str:
    """
    Extract generated code from model output.

    Args:
        generated_text: Generated text (may or may not include prompt)
        prompt: Original prompt to remove if present

    Returns:
        Extracted code
    """
    raw_code = _extract_raw_code(generated_text, prompt)
    return _trim_to_function(raw_code)


@contextlib.contextmanager
def timeout(seconds):
    """
    Context manager for timeout protection.
    Note: Only works on Unix/Mac systems (uses SIGALRM).
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {seconds} seconds")

    # Set up the timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Restore previous handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _classify_exception(exc: Exception) -> tuple[str, str]:
    """
    Classify an exception into an error type category.

    Returns:
        Tuple of (error_type, exception_class_name)
    """
    exc_class = type(exc).__name__

    # Syntax errors (compilation)
    if isinstance(exc, (SyntaxError, IndentationError)):
        return "syntax", exc_class

    # Name/reference errors
    if isinstance(exc, (NameError, AttributeError, UnboundLocalError)):
        return "name", exc_class

    # Type errors
    if isinstance(exc, TypeError):
        return "type", exc_class

    # Logic errors (test assertion failures)
    if isinstance(exc, AssertionError):
        return "logic", exc_class

    # Runtime errors (IndexError, ValueError, KeyError, ZeroDivisionError, etc.)
    return "runtime", exc_class


def _prepare_namespace() -> dict:
    """
    Create namespace with pre-imported modules for code execution.

    Returns:
        Namespace dict with common imports loaded
    """
    namespace = {}

    # Pre-import dataset-specific imports
    # For HumanEval: load from Phase 0.3, For MBPP: load from Phase 0.4
    try:
        from common.config import Config
        config = Config()

        import_file = None
        if config.dataset_name == "humaneval":
            import_file = Path("data/phase0_3_humaneval/required_imports.json")
        elif config.dataset_name == "mbpp":
            import_file = Path("data/phase0_4_mbpp_imports/required_imports.json")

        if import_file and import_file.exists():
            import json
            with open(import_file) as f:
                imports_data = json.load(f)
                import_code = '\n'.join(imports_data['imports'])
                # Execute imports in namespace
                exec(import_code, namespace)
        # If file doesn't exist yet, silently continue
    except Exception:
        # If config loading fails, continue without pre-imports
        pass

    return namespace


def evaluate_code_with_error_type(
    code: str,
    test_list: list[str],
    timeout_seconds: int = 5
) -> EvaluationResult:
    """
    Evaluate generated code against test cases with detailed error type classification.

    Algorithm:
    1. compile() check -> syntax error
    2. exec(code) -> name/type/runtime/timeout
    3. exec(test) -> logic (AssertionError) or other
    4. All pass -> passed

    Args:
        code: Generated code to test
        test_list: List of test assertion strings
        timeout_seconds: Timeout per execution step (default: 5)

    Returns:
        EvaluationResult with passed status, error_type, error_message, and exception_class

    Example:
        >>> result = evaluate_code_with_error_type("def foo(", [])
        >>> assert result.error_type == "syntax"
        >>> result = evaluate_code_with_error_type("def foo(): return 1", ["assert foo() == 2"])
        >>> assert result.error_type == "logic"
    """
    # Step 1: Compile check (catches syntax errors without execution)
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        error_type, exc_class = _classify_exception(e)
        return EvaluationResult(
            passed=False,
            error_type=error_type,
            error_message=str(e),
            exception_class=exc_class
        )

    # Prepare namespace with common imports
    namespace = _prepare_namespace()

    # Step 2: Execute the code definition
    try:
        with timeout(timeout_seconds):
            exec(code, namespace)
    except TimeoutError as e:
        return EvaluationResult(
            passed=False,
            error_type="timeout",
            error_message=str(e),
            exception_class="TimeoutError"
        )
    except Exception as e:
        error_type, exc_class = _classify_exception(e)
        return EvaluationResult(
            passed=False,
            error_type=error_type,
            error_message=str(e),
            exception_class=exc_class
        )

    # Step 3: Run each test
    for test in test_list:
        try:
            with timeout(timeout_seconds):
                exec(test, namespace)
        except TimeoutError as e:
            return EvaluationResult(
                passed=False,
                error_type="timeout",
                error_message=str(e),
                exception_class="TimeoutError"
            )
        except Exception as e:
            error_type, exc_class = _classify_exception(e)
            return EvaluationResult(
                passed=False,
                error_type=error_type,
                error_message=str(e),
                exception_class=exc_class
            )

    # Step 4: All tests passed
    return EvaluationResult(
        passed=True,
        error_type="passed",
        error_message=None,
        exception_class=None
    )


def evaluate_code(code: str, test_list: list[str]) -> bool:
    """
    Evaluate generated code against test cases with timeout protection.

    This is a backward-compatible wrapper around evaluate_code_with_error_type().

    Args:
        code: Generated code to test
        test_list: List of test assertion strings

    Returns:
        True if all tests pass, False otherwise
    """
    return evaluate_code_with_error_type(code, test_list).passed


# ============================================================================
# Activation Loading and SAE Encoding
# ============================================================================

def load_and_encode_activation(
    task_id: str,
    layer: int,
    latent_idx: int,
    sae,
    device: torch.device,
    activation_dir: Path
) -> Optional[float]:
    """
    Load activation from safetensors and encode through SAE to get latent activation.

    This is the common pattern used across evaluation phases:
    1. Load raw activation from safetensors file
    2. Convert to correct dtype
    3. Encode through SAE
    4. Extract specific latent activation

    Args:
        task_id: Task identifier (used in filename)
        layer: Layer number
        latent_idx: SAE latent index to extract
        sae: SAE model with encode() method
        device: Target device for tensors
        activation_dir: Directory containing activation files

    Returns:
        Latent activation value as float, or None if file doesn't exist

    Example:
        >>> value = load_and_encode_activation(
        ...     task_id="42", layer=16, latent_idx=1234,
        ...     sae=my_sae, device=torch.device("cuda"),
        ...     activation_dir=Path("data/phase1_0/activations/task_activations")
        ... )
        >>> if value is not None:
        ...     print(f"Latent activation: {value:.4f}")
    """
    filepath = activation_dir / f"{task_id}_layer_{layer}.safetensors"

    if not filepath.exists():
        return None

    # Load from safetensors (preserves bfloat16)
    raw_activation = load_activation(filepath, device)

    # Match SAE dtype
    raw_activation = raw_activation.to(sae.W_enc.dtype)

    # Ensure [1, d_model] shape for SAE encoding
    if raw_activation.ndim == 1:
        raw_activation = rearrange(raw_activation, 'd -> 1 d')

    # Encode and extract latent activation
    with torch.no_grad():
        latent_activations = sae.encode(raw_activation)

    return latent_activations[0, latent_idx].item()


def load_raw_activation(
    task_id: str,
    layer: int,
    activation_dir: Path,
    device: Optional[torch.device] = None
) -> Optional[torch.Tensor]:
    """
    Load raw activation tensor from .safetensors file (preserves bfloat16).

    Args:
        task_id: Task identifier (used in filename)
        layer: Layer number
        activation_dir: Directory containing activation files
        device: Target device (if None, stays on CPU)

    Returns:
        Activation tensor, or None if file doesn't exist
    """
    filepath = activation_dir / f"{task_id}_layer_{layer}.safetensors"

    if not filepath.exists():
        return None

    return load_activation(filepath, device if device is not None else "cpu")
