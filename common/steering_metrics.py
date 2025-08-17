"""
Common steering metrics and utilities for model steering experiments.

Provides shared functionality for calculating correction/corruption rates,
code similarity metrics, and creating steering hooks for SAE-based model interventions.
"""

from typing import List, Dict, Callable, Union
import pandas as pd
import torch
import tokenize
import io
from difflib import SequenceMatcher
from common.logging import get_logger

logger = get_logger("common.steering_metrics")


def calculate_correction_rate(results: Union[List[Dict], pd.DataFrame]) -> float:
    """
    Calculate percentage of incorrect→correct transitions.
    
    Used for evaluating "correct" steering applied to initially incorrect problems.
    Measures how effectively the steering/orthogonalization intervention fixes incorrect solutions.
    
    Args:
        results: Either a list of dicts with 'baseline_passed'/'test_passed' and 
                'steered_passed'/'orthogonalized_passed' keys, or a DataFrame with those columns
                
    Returns:
        Correction rate as percentage (0-100)
    """
    if isinstance(results, pd.DataFrame):
        if results.empty:
            return 0.0
        
        # DataFrame path - check for either column name
        if 'steered_passed' in results.columns:
            modified_col = 'steered_passed'
        elif 'orthogonalized_passed' in results.columns:
            modified_col = 'orthogonalized_passed'
        else:
            logger.warning("DataFrame missing 'steered_passed' or 'orthogonalized_passed' column")
            return 0.0
        
        # Count incorrect→correct transitions
        corrected = len(results[(results['test_passed'] == False) & results[modified_col]])
        total_incorrect = len(results[results['test_passed'] == False])
            
    elif isinstance(results, list):
        if not results:
            return 0.0
        
        # List of dicts path - check which key is present
        if results and len(results) > 0:
            # Check first item to determine key name
            if 'steered_passed' in results[0]:
                modified_key = 'steered_passed'
            elif 'orthogonalized_passed' in results[0]:
                modified_key = 'orthogonalized_passed'
            else:
                logger.warning("Results missing 'steered_passed' or 'orthogonalized_passed' key")
                return 0.0
            
            corrected = sum(1 for r in results if not r.get('baseline_passed', r.get('test_passed', False)) and r[modified_key])
            total_incorrect = sum(1 for r in results if not r.get('baseline_passed', r.get('test_passed', False)))
        else:
            return 0.0
        
    else:
        raise TypeError(f"Expected list or DataFrame, got {type(results)}")
    
    if total_incorrect == 0:
        logger.warning("No initially incorrect problems found for correction rate calculation")
        return 0.0
        
    correction_rate = (corrected / total_incorrect) * 100
    logger.debug(f"Correction rate: {corrected}/{total_incorrect} = {correction_rate:.1f}%")
    
    return correction_rate


def calculate_corruption_rate(results: Union[List[Dict], pd.DataFrame]) -> float:
    """
    Calculate percentage of correct→incorrect transitions.
    
    Used for evaluating "incorrect" steering applied to initially correct problems.
    Measures how effectively the steering/orthogonalization intervention introduces bugs.
    
    Args:
        results: Either a list of dicts with 'baseline_passed'/'test_passed' and 
                'steered_passed'/'orthogonalized_passed' keys, or a DataFrame with those columns
                
    Returns:
        Corruption rate as percentage (0-100)
    """
    if isinstance(results, pd.DataFrame):
        if results.empty:
            return 0.0
        
        # DataFrame path - check for either column name
        if 'steered_passed' in results.columns:
            modified_col = 'steered_passed'
        elif 'orthogonalized_passed' in results.columns:
            modified_col = 'orthogonalized_passed'
        else:
            logger.warning("DataFrame missing 'steered_passed' or 'orthogonalized_passed' column")
            return 0.0
        
        # Count correct→incorrect transitions
        corrupted = len(results[results['test_passed'] & (results[modified_col] == False)])
        total_correct = len(results[results['test_passed']])
            
    elif isinstance(results, list):
        if not results:
            return 0.0
        
        # List of dicts path - check which key is present
        if results and len(results) > 0:
            # Check first item to determine key name
            if 'steered_passed' in results[0]:
                modified_key = 'steered_passed'
            elif 'orthogonalized_passed' in results[0]:
                modified_key = 'orthogonalized_passed'
            else:
                logger.warning("Results missing 'steered_passed' or 'orthogonalized_passed' key")
                return 0.0
            
            corrupted = sum(1 for r in results if r.get('baseline_passed', r.get('test_passed', True)) and not r[modified_key])
            total_correct = sum(1 for r in results if r.get('baseline_passed', r.get('test_passed', True)))
        else:
            return 0.0
        
    else:
        raise TypeError(f"Expected list or DataFrame, got {type(results)}")
    
    if total_correct == 0:
        logger.warning("No initially correct problems found for corruption rate calculation")
        return 0.0
        
    corruption_rate = (corrupted / total_correct) * 100
    logger.debug(f"Corruption rate: {corrupted}/{total_correct} = {corruption_rate:.1f}%")
    
    return corruption_rate


def calculate_preservation_rate(results: Union[List[Dict], pd.DataFrame]) -> float:
    """
    Calculate percentage of correct problems that remain correct after steering.
    
    Inverse of corruption rate. Used for evaluating if "correct" steering 
    preserves already-correct solutions.
    
    Args:
        results: Either a list of dicts or DataFrame with steering results
                
    Returns:
        Preservation rate as percentage (0-100)
    """
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
    def get_tokens(code: str) -> List[str]:
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


def create_steering_hook(sae_decoder_direction: torch.Tensor, 
                        coefficient: float) -> Callable:
    """
    Create a hook that adds SAE decoder direction to residual stream.
    
    This hook modifies the model's internal representations by adding
    a scaled SAE feature direction to steer the model's behavior.
    
    Args:
        sae_decoder_direction: Decoder vector from SAE [d_model]
        coefficient: Scalar multiplier for steering strength
    
    Returns:
        Hook function for forward_pre_hook registration
    """
    def hook_fn(module, input):
        # input[0] is residual stream: [batch_size, seq_len, d_model]
        residual = input[0]
        
        # Add steering vector scaled by coefficient to all positions
        steering = sae_decoder_direction.unsqueeze(0).unsqueeze(0) * coefficient
        residual = residual + steering.to(residual.device)
        
        return (residual,) + input[1:]
    
    return hook_fn