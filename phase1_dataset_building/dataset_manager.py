"""
Dataset management classes for Phase 1 of the PVA-SAE project.

This module contains classes for managing the MBPP dataset, including
loading, prompt template generation, and data validation.
"""

from datasets import load_dataset
import logging
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, asdict

# Import common utilities
from common import DEFAULT_DATASET_DIR
from phase2_sae_analysis.prompt_utils import build_prompt_template


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


class PromptTemplateBuilder:
    """Constructs standardized prompt templates from MBPP records"""
    
    # Template constants
    CODE_INITIATOR = "# Your code here:"
    
    def __init__(self):
        """Initialize the prompt template builder"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("PromptTemplateBuilder initialized")
    
    def build_prompt(self, record: dict) -> str:
        """
        Build standardized prompt from MBPP record
        
        Args:
            record: MBPP dataset record containing 'text' and 'test_list'
            
        Returns:
            str: Formatted prompt template ready for model input
        """
        try:
            self._validate_record(record)
            
            # Extract components
            problem_description = self._extract_problem_description(record['text'])
            test_cases = self._format_test_cases(record['test_list'])
            
            # Construct standardized template
            prompt = self._construct_template(problem_description, test_cases)
            
            task_id = record.get('task_id', 'unknown')
            self.logger.debug(f"Built prompt template for task_id: {task_id}")
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Failed to build prompt template: {str(e)}")
            raise ValueError(f"Failed to build prompt template: {str(e)}") from e
    
    def build_batch_prompts(self, records: List[dict]) -> List[str]:
        """
        Build prompts for multiple records
        
        Args:
            records: List of MBPP dataset records
            
        Returns:
            list[str]: List of formatted prompt templates
        """
        prompts = []
        failed_records = []
        
        for i, record in enumerate(records):
            try:
                prompt = self.build_prompt(record)
                prompts.append(prompt)
            except Exception as e:
                task_id = record.get('task_id', f'index_{i}')
                failed_records.append(task_id)
                self.logger.error(f"Failed to build prompt for task_id {task_id}: {str(e)}")
        
        if failed_records:
            self.logger.warning(f"Failed to build prompts for {len(failed_records)} records: {failed_records}")
        
        self.logger.info(f"Successfully built {len(prompts)} prompts from {len(records)} records")
        return prompts
    
    def preview_template(self, record: dict, max_length: int = 200) -> str:
        """
        Generate a preview of the template for debugging
        
        Args:
            record: MBPP dataset record
            max_length: Maximum length for preview
            
        Returns:
            str: Truncated template preview
        """
        prompt = self.build_prompt(record)
        
        if len(prompt) <= max_length:
            return prompt
        
        preview = prompt[:max_length] + "..."
        return preview
    
    def get_template_stats(self, record: dict) -> Dict[str, Any]:
        """
        Get statistics about the generated template
        
        Args:
            record: MBPP dataset record
            
        Returns:
            dict containing template statistics
        """
        prompt = self.build_prompt(record)
        
        stats = {
            'total_length': len(prompt),
            'total_lines': prompt.count('\n') + 1,
            'problem_description_length': len(record['text'].strip()),
            'num_test_cases': len(record['test_list']),
            'test_cases_length': sum(len(test.strip()) for test in record['test_list']),
            'task_id': record.get('task_id', 'unknown')
        }
        
        return stats
    
    def _validate_record(self, record: dict):
        """
        Validate MBPP record has required fields
        
        Args:
            record: MBPP dataset record to validate
            
        Raises:
            ValueError: If record is missing required fields or has invalid format
        """
        required_fields = ['text', 'test_list']
        
        # Check required fields exist
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate field types and content
        if not isinstance(record['text'], str):
            raise ValueError("'text' field must be a string")
        
        if not isinstance(record['test_list'], list):
            raise ValueError("'test_list' field must be a list")
        
        if len(record['test_list']) == 0:
            raise ValueError("'test_list' cannot be empty")
        
        # Validate test cases are strings
        for i, test in enumerate(record['test_list']):
            if not isinstance(test, str):
                raise ValueError(f"Test case {i} must be a string")
            if not test.strip():
                raise ValueError(f"Test case {i} cannot be empty")
        
        # Log successful validation
        task_id = record.get('task_id', 'unknown')
        self.logger.debug(f"Record validation passed for task_id: {task_id}")
    
    def _extract_problem_description(self, text: str) -> str:
        """
        Extract and clean problem description
        
        Args:
            text: Raw problem description from MBPP record
            
        Returns:
            str: Cleaned problem description
        """
        # Clean and normalize the problem description
        description = text.strip()
        
        # Remove any extra whitespace while preserving line breaks
        lines = [line.strip() for line in description.split('\n')]
        cleaned_description = '\n'.join(line for line in lines if line)
        
        return cleaned_description
    
    def _format_test_cases(self, test_list: List[str]) -> str:
        """
        Format test cases for template inclusion
        
        Args:
            test_list: List of test case strings from MBPP record
            
        Returns:
            str: Formatted test cases ready for template
        """
        formatted_tests = []
        
        for test in test_list:
            # Clean up each test case
            cleaned_test = test.strip()
            if cleaned_test:
                formatted_tests.append(cleaned_test)
        
        # Join test cases with newlines
        return '\n'.join(formatted_tests)
    
    def _construct_template(self, problem_description: str, test_cases: str) -> str:
        """
        Construct the final standardized template
        
        Args:
            problem_description: Cleaned problem description
            test_cases: Formatted test cases
            
        Returns:
            str: Complete standardized prompt template
        """
        # Use shared prompt template
        return build_prompt_template(problem_description, test_cases, self.CODE_INITIATOR)


class DatasetManager:
    """Manages MBPP dataset operations"""
    
    def __init__(self):
        self.dataset = None
        self.test_data = None
        self._is_loaded = False
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self):
        """Load MBPP dataset from Hugging Face"""
        if self._is_loaded:
            self.logger.info("MBPP dataset already loaded")
            return
        
        try:
            self.logger.info("Loading MBPP dataset...")
            self.dataset = load_dataset("Muennighoff/mbpp", "full")
            self.test_data = self.dataset['test']
            self._is_loaded = True
            
            self.logger.info(f"MBPP dataset loaded: {len(self.test_data)} examples")
            
        except Exception as e:
            self.logger.error(f"Failed to load MBPP dataset: {str(e)}")
            raise RuntimeError(f"Failed to load MBPP dataset: {str(e)}") from e
    
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


class PromptAwareDatasetManager(DatasetManager):
    """Dataset manager with integrated prompt template functionality"""
    
    def __init__(self):
        super().__init__()
        self.template_builder = PromptTemplateBuilder()
    
    def get_prompt_template(self, idx: int) -> str:
        """
        Get standardized prompt template for record by index
        
        Args:
            idx: Index of the record in the dataset
            
        Returns:
            str: Standardized prompt template
        """
        record = self.get_record(idx)
        return self.template_builder.build_prompt(record)
    
    def get_batch_prompts(self, start_idx: int, end_idx: int) -> List[str]:
        """
        Get prompt templates for a range of records
        
        Args:
            start_idx: Starting index (inclusive)
            end_idx: Ending index (inclusive)
            
        Returns:
            list[str]: List of prompt templates
        """
        self._ensure_loaded()
        
        # Validate range
        dataset_size = self.get_size()
        if start_idx < 0 or start_idx >= dataset_size:
            raise ValueError(f"start_idx {start_idx} out of range [0, {dataset_size-1}]")
        if end_idx < start_idx or end_idx >= dataset_size:
            raise ValueError(f"end_idx {end_idx} out of range [{start_idx}, {dataset_size-1}]")
        
        # Extract records and build prompts
        records = [self.test_data[i] for i in range(start_idx, end_idx + 1)]
        return self.template_builder.build_batch_prompts(records)
    
    def preview_prompt_template(self, idx: int, max_length: int = 200) -> str:
        """
        Preview prompt template for debugging
        
        Args:
            idx: Index of the record
            max_length: Maximum length for preview
            
        Returns:
            str: Truncated prompt preview
        """
        record = self.get_record(idx)
        return self.template_builder.preview_template(record, max_length)
    
    def get_prompt_stats(self, idx: int) -> Dict[str, Any]:
        """
        Get statistics for prompt template
        
        Args:
            idx: Index of the record
            
        Returns:
            dict: Template statistics
        """
        record = self.get_record(idx)
        return self.template_builder.get_template_stats(record)