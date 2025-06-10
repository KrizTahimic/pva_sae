"""
Enhanced prompt utilities for consistent prompt generation across all phases.

This module provides prompt building, variation, and management utilities
for code generation, robustness testing, and experimental variations.
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class PromptVariation:
    """Container for prompt variation data."""
    variation_type: str  # 'instruction', 'format', 'style'
    original_prompt: str
    varied_prompt: str
    variation_description: str


class PromptBuilder:
    """Enhanced prompt builder with variation support."""
    
    # Standard prompt template
    STANDARD_TEMPLATE = """You are an expert Python programmer. Write a Python function to solve the following problem.

Problem:
{problem_description}

Your function must pass all of these test cases:
{test_cases}

Write only the function definition. Do not include test code or explanations.

{code_initiator}"""
    
    # Alternative instruction variations for robustness testing
    INSTRUCTION_VARIATIONS = [
        "You are a skilled Python developer. Create a function that solves this problem.",
        "As a Python expert, implement a solution for the following task.",
        "Write a Python function that addresses the problem below.",
        "Develop a Python function to handle this programming challenge.",
        "Implement a Python solution for the given problem."
    ]
    
    # Code initiator variations
    CODE_INITIATORS = [
        "# Your code here:",
        "# Solution:",
        "# Implementation:",
        "def solution():",
        "# Write your function below:"
    ]
    
    # Format variations (different ways to present test cases)
    TEST_CASE_FORMATS = {
        'standard': lambda cases: cases,
        'numbered': lambda cases: '\n'.join(f"{i+1}. {case}" for i, case in enumerate(cases.split('\n'))),
        'bullet': lambda cases: '\n'.join(f"• {case}" for case in cases.split('\n')),
        'verbose': lambda cases: '\n'.join(f"Test case: {case}" for case in cases.split('\n'))
    }
    
    @classmethod
    def build_standard_prompt(
        cls,
        problem_description: str,
        test_cases: str,
        code_initiator: str = "# Your code here:"
    ) -> str:
        """
        Build standard prompt template (backward compatible).
        
        Args:
            problem_description: The problem statement
            test_cases: Test cases as a string (one per line)
            code_initiator: The code starter prompt
            
        Returns:
            Formatted prompt template
        """
        return cls.STANDARD_TEMPLATE.format(
            problem_description=problem_description,
            test_cases=test_cases,
            code_initiator=code_initiator
        )
    
    @classmethod
    def build_with_variation(
        cls,
        problem_description: str,
        test_cases: str,
        variation_type: str = "none",
        seed: Optional[int] = None
    ) -> PromptVariation:
        """
        Build prompt with specified variation type.
        
        Args:
            problem_description: The problem statement
            test_cases: Test cases as string
            variation_type: Type of variation ('none', 'instruction', 'format', 'style', 'random')
            seed: Random seed for reproducibility
            
        Returns:
            PromptVariation object with original and varied prompts
        """
        if seed is not None:
            random.seed(seed)
        
        original = cls.build_standard_prompt(problem_description, test_cases)
        
        if variation_type == "none":
            return PromptVariation(
                variation_type="none",
                original_prompt=original,
                varied_prompt=original,
                variation_description="No variation applied"
            )
        
        elif variation_type == "instruction":
            instruction = random.choice(cls.INSTRUCTION_VARIATIONS)
            varied = f"{instruction}\n\nProblem:\n{problem_description}\n\n"
            varied += f"Your function must pass all of these test cases:\n{test_cases}\n\n"
            varied += "Write only the function definition. Do not include test code or explanations.\n\n"
            varied += "# Your code here:"
            
            return PromptVariation(
                variation_type="instruction",
                original_prompt=original,
                varied_prompt=varied,
                variation_description=f"Instruction variation: '{instruction[:50]}...'"
            )
        
        elif variation_type == "format":
            format_name = random.choice(list(cls.TEST_CASE_FORMATS.keys()))
            formatter = cls.TEST_CASE_FORMATS[format_name]
            formatted_cases = formatter(test_cases)
            
            varied = cls.STANDARD_TEMPLATE.format(
                problem_description=problem_description,
                test_cases=formatted_cases,
                code_initiator="# Your code here:"
            )
            
            return PromptVariation(
                variation_type="format",
                original_prompt=original,
                varied_prompt=varied,
                variation_description=f"Test case format: {format_name}"
            )
        
        elif variation_type == "style":
            code_initiator = random.choice(cls.CODE_INITIATORS)
            varied = cls.STANDARD_TEMPLATE.format(
                problem_description=problem_description,
                test_cases=test_cases,
                code_initiator=code_initiator
            )
            
            return PromptVariation(
                variation_type="style",
                original_prompt=original,
                varied_prompt=varied,
                variation_description=f"Code initiator: '{code_initiator}'"
            )
        
        elif variation_type == "random":
            # Apply random combination of variations
            variations_applied = []
            
            # Random instruction
            if random.random() > 0.5:
                instruction = random.choice(cls.INSTRUCTION_VARIATIONS)
                variations_applied.append(f"instruction")
            else:
                instruction = "You are an expert Python programmer. Write a Python function to solve the following problem."
            
            # Random format
            format_name = random.choice(list(cls.TEST_CASE_FORMATS.keys()))
            formatter = cls.TEST_CASE_FORMATS[format_name]
            formatted_cases = formatter(test_cases)
            if format_name != 'standard':
                variations_applied.append(f"format:{format_name}")
            
            # Random code initiator
            code_initiator = random.choice(cls.CODE_INITIATORS)
            if code_initiator != "# Your code here:":
                variations_applied.append(f"initiator")
            
            # Build varied prompt
            varied = f"{instruction}\n\nProblem:\n{problem_description}\n\n"
            varied += f"Your function must pass all of these test cases:\n{formatted_cases}\n\n"
            varied += "Write only the function definition. Do not include test code or explanations.\n\n"
            varied += code_initiator
            
            return PromptVariation(
                variation_type="random",
                original_prompt=original,
                varied_prompt=varied,
                variation_description=f"Random variations: {', '.join(variations_applied) if variations_applied else 'none'}"
            )
        
        else:
            raise ValueError(f"Unknown variation type: {variation_type}")
    
    @classmethod
    def generate_robustness_prompts(
        cls,
        problem_description: str,
        test_cases: str,
        num_variations: int = 5,
        variation_types: Optional[List[str]] = None
    ) -> List[PromptVariation]:
        """
        Generate multiple prompt variations for robustness testing.
        
        Args:
            problem_description: The problem statement
            test_cases: Test cases
            num_variations: Number of variations to generate
            variation_types: List of variation types to use (None = all types)
            
        Returns:
            List of PromptVariation objects
        """
        if variation_types is None:
            variation_types = ["instruction", "format", "style", "random"]
        
        variations = []
        
        # Always include the original
        variations.append(cls.build_with_variation(
            problem_description, test_cases, "none"
        ))
        
        # Generate variations
        for i in range(num_variations - 1):
            variation_type = variation_types[i % len(variation_types)]
            variation = cls.build_with_variation(
                problem_description, test_cases, variation_type, seed=i
            )
            variations.append(variation)
        
        return variations


# Backward compatibility function
def build_prompt_template(
    problem_description: str, 
    test_cases: str, 
    code_initiator: str = "# Your code here:"
) -> str:
    """
    Build standardized prompt template (backward compatible).
    
    Args:
        problem_description: The problem statement
        test_cases: Test cases as a string (one per line)
        code_initiator: The code starter prompt
        
    Returns:
        Formatted prompt template
    """
    return PromptBuilder.build_standard_prompt(
        problem_description, test_cases, code_initiator
    )


class PromptManager:
    """Manages prompt templates and variations for experiments."""
    
    def __init__(self):
        """Initialize prompt manager."""
        self.templates: Dict[str, str] = {}
        self.variations: Dict[str, List[PromptVariation]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_template(self, name: str, template: str) -> None:
        """Register a custom prompt template."""
        self.templates[name] = template
        self.logger.info(f"Registered template: {name}")
    
    def get_template(self, name: str) -> Optional[str]:
        """Get registered template by name."""
        return self.templates.get(name)
    
    def save_variations(self, experiment_id: str, variations: List[PromptVariation]) -> None:
        """Save prompt variations for an experiment."""
        self.variations[experiment_id] = variations
        self.logger.info(f"Saved {len(variations)} variations for experiment: {experiment_id}")
    
    def get_variations(self, experiment_id: str) -> Optional[List[PromptVariation]]:
        """Get saved variations for an experiment."""
        return self.variations.get(experiment_id)