"""
Common steering metrics and utilities for model steering experiments.

Provides shared functionality for calculating correction/corruption rates,
code similarity metrics, and creating steering hooks for SAE-based model interventions.
"""

from typing import Callable, Optional, Union
import pandas as pd
import torch
import tokenize
import io
from difflib import SequenceMatcher
from common.logging import get_logger
from common.direction_utils import assert_normalized

logger = get_logger("common.steering_metrics")


def _detect_modified_column(data: Union[list[dict], pd.DataFrame]) -> Optional[str]:
    """
    Detect whether data uses 'steered_correct' or 'orthogonalized_correct'.

    Args:
        data: Either a list of dicts or DataFrame with steering results

    Returns:
        Column/key name if found, None otherwise
    """
    if isinstance(data, pd.DataFrame):
        if 'steered_correct' in data.columns:
            return 'steered_correct'
        if 'orthogonalized_correct' in data.columns:
            return 'orthogonalized_correct'
        return None

    # List case - check first element
    if 'steered_correct' in data[0]:
        return 'steered_correct'
    if 'orthogonalized_correct' in data[0]:
        return 'orthogonalized_correct'
    return None


def calculate_correction_rate(results: Union[list[dict], pd.DataFrame]) -> float:
    """
    Calculate percentage of incorrect→correct transitions.

    Used for evaluating "correct" steering applied to initially incorrect problems.
    Measures how effectively the steering/orthogonalization intervention fixes incorrect solutions.

    Args:
        results: Either a list of dicts with 'baseline_passed' and
                'steered_correct'/'orthogonalized_correct' keys, or a DataFrame with those columns

    Returns:
        Correction rate as percentage (0-100)
    """
    # Early return: empty data
    if isinstance(results, pd.DataFrame) and results.empty:
        return 0.0
    if isinstance(results, list) and not results:
        return 0.0

    # Detect column/key name
    modified_col = _detect_modified_column(results)
    if modified_col is None:
        raise ValueError("Results missing 'steered_correct' or 'orthogonalized_correct' column")

    # Compute based on type - flat structure
    if isinstance(results, pd.DataFrame):
        corrected = len(results[(results['baseline_passed'] == False) & results[modified_col]])
        total_incorrect = len(results[results['baseline_passed'] == False])
    elif isinstance(results, list):
        corrected = sum(1 for r in results if not r['baseline_passed'] and r[modified_col])
        total_incorrect = sum(1 for r in results if not r['baseline_passed'])
    else:
        raise TypeError(f"Expected list or DataFrame, got {type(results)}")

    # Early return: no incorrect samples
    if total_incorrect == 0:
        logger.warning("No initially incorrect problems found for correction rate calculation")
        return 0.0

    correction_rate = (corrected / total_incorrect) * 100
    logger.debug(f"Correction rate: {corrected}/{total_incorrect} = {correction_rate:.1f}%")

    return correction_rate


def calculate_corruption_rate(results: Union[list[dict], pd.DataFrame]) -> float:
    """
    Calculate percentage of correct→incorrect transitions.

    Used for evaluating "incorrect" steering applied to initially correct problems.
    Measures how effectively the steering/orthogonalization intervention introduces bugs.

    Args:
        results: Either a list of dicts with 'baseline_passed' and
                'steered_correct'/'orthogonalized_correct' keys, or a DataFrame with those columns

    Returns:
        Corruption rate as percentage (0-100)
    """
    # Early return: empty data
    if isinstance(results, pd.DataFrame) and results.empty:
        return 0.0
    if isinstance(results, list) and not results:
        return 0.0

    # Detect column/key name
    modified_col = _detect_modified_column(results)
    if modified_col is None:
        raise ValueError("Results missing 'steered_correct' or 'orthogonalized_correct' column")

    # Compute based on type - flat structure
    if isinstance(results, pd.DataFrame):
        corrupted = len(results[results['baseline_passed'] & (results[modified_col] == False)])
        total_correct = len(results[results['baseline_passed']])
    elif isinstance(results, list):
        corrupted = sum(1 for r in results if r['baseline_passed'] and not r[modified_col])
        total_correct = sum(1 for r in results if r['baseline_passed'])
    else:
        raise TypeError(f"Expected list or DataFrame, got {type(results)}")

    # Early return: no correct samples
    if total_correct == 0:
        logger.warning("No initially correct problems found for corruption rate calculation")
        return 0.0

    corruption_rate = (corrupted / total_correct) * 100
    logger.debug(f"Corruption rate: {corrupted}/{total_correct} = {corruption_rate:.1f}%")

    return corruption_rate


def calculate_preservation_rate(results: Union[list[dict], pd.DataFrame]) -> float:
    """
    Calculate percentage of correct problems that remain correct after steering.

    Inverse of corruption rate. Used for evaluating if "correct" steering
    preserves already-correct solutions.

    Args:
        results: Either a list of dicts or DataFrame with steering results

    Returns:
        Preservation rate as percentage (0-100), or NaN if no correct problems exist
    """
    # Check for zero correct problems (preservation is undefined in this case)
    if isinstance(results, pd.DataFrame):
        if results.empty:
            return float('nan')
        modified_col = _detect_modified_column(results)
        if modified_col is None:
            raise ValueError("Results missing 'steered_correct' or 'orthogonalized_correct' column")
        total_correct = len(results[results['baseline_passed']])
    elif isinstance(results, list):
        if not results:
            return float('nan')
        total_correct = sum(1 for r in results if r.get('baseline_passed', False))
    else:
        raise TypeError(f"Expected list or DataFrame, got {type(results)}")

    if total_correct == 0:
        logger.warning("No initially correct problems found for preservation rate calculation")
        return float('nan')

    corruption = calculate_corruption_rate(results)
    return 100 - corruption


def calculate_code_similarity(code1: str, code2: str) -> float:
    """
    Calculate token-based similarity between two code strings.
    
    Uses Python's tokenizer to properly parse code into tokens, ignoring
    formatting differences like whitespace and comments. Falls back to
    simple splitting for syntactically invalid code.
    
    Args:
        code1: First code string
        code2: Second code string
        
    Returns:
        Similarity score from 0.0 (completely different) to 1.0 (identical)
    """
    def get_tokens(code: str) -> list[str]:
        """Extract meaningful tokens from Python code."""
        tokens = []
        try:
            # Use Python's tokenizer for accurate parsing
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                # Skip non-semantic tokens
                if tok.type not in (tokenize.COMMENT,     # Skip comments
                                   tokenize.NEWLINE,      # Skip newlines
                                   tokenize.NL,           # Skip non-terminating newlines
                                   tokenize.INDENT,       # Skip indentation
                                   tokenize.DEDENT,       # Skip dedentation
                                   tokenize.ENCODING,     # Skip encoding declarations
                                   tokenize.ENDMARKER,    # Skip end markers
                                   tokenize.ERRORTOKEN):  # Skip error tokens
                    tokens.append(tok.string)
            return tokens
        except (tokenize.TokenError, IndentationError, SyntaxError):
            # Fallback for broken/incomplete code
            # Simple split on whitespace and common delimiters
            logger.debug("Tokenization failed, using fallback splitting")
            return code.replace('\n', ' ').replace('\t', ' ').split()
    
    # Get tokens for both code strings
    tokens1 = get_tokens(code1)
    tokens2 = get_tokens(code2)
    
    # Handle empty code cases
    if not tokens1 and not tokens2:
        return 1.0  # Both empty = identical
    if not tokens1 or not tokens2:
        return 0.0  # One empty = completely different
    
    # Calculate similarity using SequenceMatcher
    similarity = SequenceMatcher(None, tokens1, tokens2).ratio()
    
    logger.debug(f"Token similarity: {similarity:.3f} "
                f"({len(tokens1)} tokens vs {len(tokens2)} tokens)")
    
    return similarity


def create_last_position_steering_hook(latent_direction: torch.Tensor,
                                       coefficient: float) -> Callable:
    """
    Create a steering hook that modifies ONLY the last position.

    During autoregressive generation:
    - Prefill: steers only the last prompt token (position -1)
    - Generation: steers each new token (seq_len=1, so position 0 = last)

    This is more targeted than steering all positions - only affects
    where next-token prediction happens.

    IMPORTANT: The direction MUST be pre-normalized to unit L2 norm by the caller.
    This ensures the coefficient directly controls perturbation magnitude.
    Use normalize_direction() from common.direction_utils before calling this function.

    Args:
        latent_direction: Unit-normalized decoder weight vector for a latent [d_model].
                         MUST have L2 norm == 1.0 (validated at creation time).
        coefficient: Scalar multiplier for steering strength

    Returns:
        Hook function for forward_pre_hook registration

    Raises:
        ValueError: If latent_direction is not unit-normalized
    """
    # Validate that caller has properly normalized the direction
    # This catches bugs where raw W_dec is passed without normalization
    assert_normalized(latent_direction, name="latent_direction")

    def hook_fn(module, input):
        # input[0] is residual stream: [batch_size, seq_len, d_model]
        residual = input[0]

        # Only modify the LAST position
        steering = latent_direction * coefficient
        residual = residual.clone()  # Don't modify original tensor
        residual[:, -1, :] = residual[:, -1, :] + steering.to(residual.device, residual.dtype)

        return (residual,) + input[1:]

    return hook_fn