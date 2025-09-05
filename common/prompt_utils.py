"""
Simplified prompt utilities for consistent prompt generation across all phases.

This module provides basic prompt building for code generation.
Following YAGNI principle: only temperature variation needed for robustness.
"""

from typing import Optional


class PromptBuilder:
    """Simplified prompt builder focused on standard prompt generation."""
    TEMPLATE = """{problem_description}

{test_cases}

{code_initiator}"""
    @classmethod
    def build_prompt(
        cls,
        problem_description: str,
        test_cases: str,
        # code_initiator: str = "# Write your function definition here:"
        # code_initiator: str = "# Your code here:"
        code_initiator: str = "# Solution:"
    ) -> str:
        """
        Build prompt template.
        
        Args:
            problem_description: The problem statement
            test_cases: Test cases as a string (one per line)
            code_initiator: The code starter prompt
            
        Returns:
            Formatted prompt template
        """
        return cls.TEMPLATE.format(
            problem_description=problem_description,
            test_cases=test_cases,
            code_initiator=code_initiator
        )