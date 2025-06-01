"""
Shared prompt utilities for consistent prompt generation across phases.
"""

def build_prompt_template(problem_description: str, test_cases: str, code_initiator: str = "# Your code here:") -> str:
    """
    Build standardized prompt template used across all phases.
    
    Args:
        problem_description: The problem statement
        test_cases: Test cases as a string (one per line)
        code_initiator: The code starter prompt
        
    Returns:
        Formatted prompt template
    """
    template = f"""You are an expert Python programmer. Write a Python function to solve the following problem.

Problem:
{problem_description}

Your function must pass all of these test cases:
{test_cases}

Write only the function definition. Do not include test code or explanations.

{code_initiator}"""
    
    return template