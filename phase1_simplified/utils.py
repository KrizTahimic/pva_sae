"""Utility functions for Phase 1 dataset building."""

from typing import Dict, Tuple, Optional, List
from common.logging import get_logger

logger = get_logger("phase1_simplified.utils", phase="1.0")


def extract_code(generated_text: str, prompt: str) -> str:
    """
    Extract generated code from model output.
    
    Args:
        generated_text: Full text from model including prompt
        prompt: Original prompt to remove
        
    Returns:
        Extracted code
    """
    # Remove the prompt from the beginning
    code = generated_text[len(prompt):].strip()
    
    # Basic cleanup - remove common artifacts
    if code.startswith("```python"):
        code = code[9:]  # Remove ```python
    if code.startswith("```"):
        code = code[3:]  # Remove ```
        
    if code.endswith("```"):
        code = code[:-3]  # Remove trailing ```
        
    return code.strip()


def evaluate_code(code: str, test_list: List[str]) -> bool:
    """
    Evaluate generated code against test cases.
    
    Args:
        code: Generated code to test
        test_list: List of test assertion strings
        
    Returns:
        True if all tests pass, False otherwise
    """
    # Create namespace for execution
    namespace = {}
    
    # Execute the code
    try:
        exec(code, namespace)
    except Exception:
        return False
    
    # Run each test
    for test in test_list:
        try:
            exec(test, namespace)
        except Exception:
            return False
    
    # All tests passed
    return True


def create_activation_filename(task_id: int, layer: int) -> str:
    """Create consistent filename for activation storage."""
    return f"{task_id}_layer_{layer}.npz"