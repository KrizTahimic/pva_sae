"""
Dataset management classes for Phase 1 of the PVA-SAE project.

This module contains classes for managing the MBPP dataset, including
loading, prompt template generation, and data validation.
"""

import pandas as pd
import logging
from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, asdict

# Import common utilities and phase config
from common.utils import get_phase_dir, discover_latest_phase_output
from common.prompt_utils import PromptBuilder

# Configuration constants
DEFAULT_CODE_INITIATOR = "# Your code here:"
MAX_PROMPT_LENGTH = 10000  # Safety limit for prompt size
MIN_TEST_CASES = 1  # Minimum required test cases per problem


def _validate_mbpp_record(record: Dict[str, Any], idx: int) -> None:
    """Validate MBPP record has required fields and valid content.
    
    Args:
        record: MBPP dataset record to validate
        idx: Record index for error context
        
    Raises:
        ValueError: If record is invalid with specific error context
    """
    if 'text' not in record:
        raise ValueError(
            f"MBPP record at index {idx} missing 'text' field. "
            f"Available fields: {list(record.keys())}"
        )
    
    if not isinstance(record['text'], str) or not record['text'].strip():
        raise ValueError(
            f"MBPP record at index {idx} has invalid 'text' field. "
            f"Expected non-empty string, got: {type(record.get('text', None))}"
        )
    
    test_list = record.get('test_list', [])
    # Accept both list and numpy array (from parquet files)
    import numpy as np
    if not (isinstance(test_list, (list, np.ndarray))) or len(test_list) < MIN_TEST_CASES:
        raise ValueError(
            f"MBPP record at index {idx} has invalid 'test_list' field. "
            f"Expected list/array with >= {MIN_TEST_CASES} items, got: {type(test_list).__name__} with {len(test_list) if hasattr(test_list, '__len__') else 'no length'}"
        )


def _extract_prompt_components(record: Dict[str, Any]) -> Tuple[str, str]:
    """Extract and clean prompt components from MBPP record.
    
    Args:
        record: Valid MBPP dataset record
        
    Returns:
        Tuple of (problem_description, test_cases_string)
    """
    problem_description = record['text'].strip()
    
    # Filter out empty or None test cases with list comprehension
    valid_test_cases = [
        test.strip() for test in record['test_list'] 
        if isinstance(test, str) and test.strip()
    ]
    
    if not valid_test_cases:
        raise ValueError("No valid test cases found after filtering empty strings")
    
    test_cases = '\n'.join(valid_test_cases)
    return problem_description, test_cases


@dataclass
class CodeTestResult:
    """Encapsulates code test execution results"""
    passed: int
    total: int
    errors: List[str]
    
    @property
    def success_rate(self) -> float:
        """Calculate success percentage"""
        return (self.passed / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def failed(self) -> int:
        """Number of failed tests"""
        return self.total - self.passed
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class CodeGenerationResult:
    """Encapsulates code generation and testing results"""
    task_id: str
    prompt: str
    generated_code: str
    test_result: CodeTestResult
    is_correct: bool
    generation_time: float
    complexity_score: int = 1
    
    @property
    def success_rate(self) -> float:
        """Get test success rate"""
        return self.test_result.success_rate
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'prompt': self.prompt,
            'generated_code': self.generated_code,
            'is_correct': self.is_correct,
            'passed_tests': self.test_result.passed,
            'total_tests': self.test_result.total,
            'success_rate': self.success_rate,
            'test_errors': self.test_result.errors,
            'generation_time': self.generation_time,
            'complexity_score': self.complexity_score,
        }
    
    def to_dataframe_row(self) -> dict:
        """Convert to flattened dictionary optimized for DataFrame storage"""
        return {
            'task_id': self.task_id,
            'generated_code': self.generated_code,
            'test_passed': self.is_correct,
            'complexity_score': self.complexity_score,
        }


# Note: Removed PromptTemplateBuilder class - functionality moved to common.prompt_utils.PromptBuilder
# to eliminate code duplication and centralize prompt generation logic


class DatasetManager:
    """Manages MBPP dataset operations including loading, access, and prompt generation"""
    
    def __init__(self):
        """Initialize DatasetManager."""
        self.dataset = None
        self.test_data = None
        self._is_loaded = False
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self, enriched_path: Optional[str] = None):
        """Load MBPP dataset from local enriched file.
        
        Args:
            enriched_path: Optional path to enriched dataset. If None, auto-discovers from Phase 0.
            
        Raises:
            RuntimeError: If enriched dataset is not found or cannot be loaded.
        """
        if self._is_loaded:
            self.logger.info("MBPP dataset already loaded")
            return
        
        try:
            # Auto-discover enriched dataset if path not provided
            if enriched_path is None:
                enriched_path = discover_latest_phase_output("0")
                if not enriched_path:
                    raise RuntimeError(
                        "No enriched dataset found. Please run Phase 0 first: python3 run.py phase 0"
                    )
                self.logger.info(f"Auto-discovered enriched dataset: {enriched_path}")
            
            # Load enriched dataset
            self.logger.info(f"Loading enriched MBPP dataset from: {enriched_path}")
            df = pd.read_parquet(enriched_path)
            
            # Validate required columns
            required_columns = ['task_id', 'text', 'code', 'test_list', 'cyclomatic_complexity']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise RuntimeError(
                    f"Enriched dataset missing required columns: {missing_columns}. "
                    "Please re-run Phase 0 with the latest code."
                )
            
            # Convert DataFrame to list of dicts for compatibility
            self.test_data = df.to_dict('records')
            self._is_loaded = True
            self.logger.info(f"Enriched MBPP dataset loaded: {len(self.test_data)} examples")
            
        except Exception as e:
            self.logger.error(f"Failed to load enriched dataset: {str(e)}")
            raise RuntimeError(f"Failed to load enriched dataset: {str(e)}") from e
    
    def get_record(self, idx: int) -> dict:
        """Retrieve record by index with validation"""
        self._ensure_loaded()
        
        try:
            return self.test_data[idx]
        except IndexError as e:
            self.logger.error(f"Index {idx} out of range. Dataset has {len(self.test_data)} records.")
            raise ValueError(
                f"Index {idx} out of range. Dataset has {len(self.test_data)} records."
            ) from e
    
    def get_size(self) -> int:
        """Get dataset size"""
        return len(self.test_data) if self._is_loaded else 0
    
    def is_loaded(self) -> bool:
        """Check if dataset is loaded"""
        return self._is_loaded
    
    def _ensure_loaded(self):
        """Ensure dataset is loaded before access"""
        if not self._is_loaded:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")
    
    def get_prompt_template(self, idx: int) -> str:
        """Get standardized prompt template for record by index.
        
        Args:
            idx: Index of the record in the dataset
            
        Returns:
            Standardized prompt template string
            
        Raises:
            ValueError: If record index is invalid or record is malformed
            RuntimeError: If dataset is not loaded
        """
        self.logger.debug(f"Generating prompt template for record index {idx}")
        
        try:
            record = self.get_record(idx)
            task_id = record.get('task_id', f'index_{idx}')
            
            self.logger.debug(f"Processing MBPP record {task_id} at index {idx}")
            
            # Validate record structure
            _validate_mbpp_record(record, idx)
            
            # Extract and clean components
            problem_description, test_cases = _extract_prompt_components(record)
            
            self.logger.debug(f"Extracted prompt components for {task_id}: "
                             f"description_length={len(problem_description)}, "
                             f"test_cases_count={test_cases.count('assert')}")
            
            # Build prompt using common utilities
            prompt = PromptBuilder.build_prompt(
                problem_description=problem_description,
                test_cases=test_cases,
                code_initiator=DEFAULT_CODE_INITIATOR
            )
            
            # Validate prompt length
            if len(prompt) > MAX_PROMPT_LENGTH:
                self.logger.warning(f"Generated prompt for {task_id} exceeds MAX_PROMPT_LENGTH "
                                  f"({len(prompt)} > {MAX_PROMPT_LENGTH})")
            
            self.logger.info(f"Successfully generated prompt template for {task_id} "
                            f"(total_length={len(prompt)})")
            return prompt
            
        except ValueError as e:
            self.logger.error(f"Validation failed for record {idx}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error generating prompt for record {idx}: {str(e)}")
            raise RuntimeError(f"Failed to generate prompt template for record {idx}") from e

