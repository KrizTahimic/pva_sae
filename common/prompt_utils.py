"""
Simplified prompt utilities for consistent prompt generation across all phases.

This module provides basic prompt building for code generation.
Following YAGNI principle: only temperature variation needed for robustness.
"""

from typing import Optional


class PromptBuilder:
    """Simplified prompt builder focused on standard prompt generation."""
    
    # Standard prompt template
#     TEMPLATE = """Write a Python function to solve the following problem.

# Problem:
# {problem_description}

# Tests:
# {test_cases}

# Write only the function definition. Do not include test code or explanations.

# {code_initiator}

# """
    TEMPLATE = """{problem_description}

{test_cases}

{code_initiator}"""
#     TEMPLATE = """{problem_description}

# {test_cases}"""

    
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