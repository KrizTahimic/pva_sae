"""
Difficulty analysis module for MBPP dataset preprocessing.

This module analyzes the complexity and difficulty of MBPP programming problems
without any LLM involvement. It creates a lightweight mapping of difficulty
scores that can be used by subsequent phases.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from common.utils import get_cyclomatic_complexity, get_timestamp, ensure_directory_exists
from common.logging import LoggingManager


@dataclass
class DifficultyMetrics:
    """Encapsulates difficulty analysis metrics for a single MBPP problem"""
    task_id: str
    cyclomatic_complexity: int
    reference_code: str
    num_test_cases: int
    avg_test_case_length: float
    problem_description_length: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class MBPPDifficultyAnalyzer:
    """Analyzes difficulty metrics for the entire MBPP dataset"""
    
    def __init__(self, output_dir: str = "data/datasets"):
        """
        Initialize difficulty analyzer
        
        Args:
            output_dir: Directory to save difficulty mapping
        """
        self.output_dir = Path(output_dir)
        logging_manager = LoggingManager(log_dir="data/logs")
        self.logger = logging_manager.setup_logging(__name__)
        self.difficulty_mapping: Dict[str, DifficultyMetrics] = {}
        
        ensure_directory_exists(str(self.output_dir))
        self.logger.info("MBPPDifficultyAnalyzer initialized")
    
    def analyze_dataset(self, dataset_manager) -> Dict[str, DifficultyMetrics]:
        """
        Analyze difficulty for entire MBPP dataset
        
        Args:
            dataset_manager: Loaded DatasetManager instance
            
        Returns:
            dict: Mapping of task_id to DifficultyMetrics
        """
        if not dataset_manager.is_loaded():
            raise ValueError("Dataset manager must be loaded before analysis")
        
        dataset_size = dataset_manager.get_size()
        self.logger.info(f"Starting difficulty analysis for {dataset_size} MBPP problems")
        
        difficulty_mapping = {}
        failed_analyses = []
        
        for idx in range(dataset_size):
            try:
                record = dataset_manager.get_record(idx)
                metrics = self._analyze_single_problem(record)
                difficulty_mapping[metrics.task_id] = metrics
                
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"Analyzed {idx + 1}/{dataset_size} problems")
                    
            except Exception as e:
                task_id = record.get('task_id', f'index_{idx}') if 'record' in locals() else f'index_{idx}'
                failed_analyses.append(task_id)
                self.logger.error(f"Failed to analyze problem {task_id}: {str(e)}")
        
        self.difficulty_mapping = difficulty_mapping
        
        if failed_analyses:
            self.logger.warning(f"Failed to analyze {len(failed_analyses)} problems: {failed_analyses[:10]}...")
        
        self.logger.info(f"Difficulty analysis completed: {len(difficulty_mapping)} problems analyzed")
        return difficulty_mapping
    
    def _analyze_single_problem(self, record: Dict[str, Any]) -> DifficultyMetrics:
        """
        Analyze difficulty metrics for a single MBPP problem
        
        Args:
            record: MBPP dataset record
            
        Returns:
            DifficultyMetrics: Computed difficulty metrics
        """
        task_id = str(record.get('task_id', 'unknown'))
        reference_code = record.get('code', '')
        test_list = record.get('test_list', [])
        problem_text = record.get('text', '')
        
        # Calculate cyclomatic complexity
        cyclomatic_complexity = get_cyclomatic_complexity(reference_code)
        
        # Calculate additional metrics
        num_test_cases = len(test_list)
        avg_test_case_length = sum(len(test.strip()) for test in test_list) / num_test_cases if num_test_cases > 0 else 0.0
        problem_description_length = len(problem_text.strip())
        
        return DifficultyMetrics(
            task_id=task_id,
            cyclomatic_complexity=cyclomatic_complexity,
            reference_code=reference_code,
            num_test_cases=num_test_cases,
            avg_test_case_length=avg_test_case_length,
            problem_description_length=problem_description_length
        )
    
    
    def save_difficulty_mapping(self, filepath: Optional[str] = None) -> str:
        """
        Save difficulty mapping to file
        
        Args:
            filepath: Optional custom filepath. If None, generates timestamped filename
            
        Returns:
            str: Path to saved file
        """
        if not self.difficulty_mapping:
            raise ValueError("No difficulty mapping to save. Run analyze_dataset() first.")
        
        if filepath is None:
            timestamp = get_timestamp()
            filepath = self.output_dir / f"mbpp_difficulty_mapping_{timestamp}.parquet"
        else:
            filepath = Path(filepath)
        
        # Convert to DataFrame for efficient storage
        df_data = []
        for task_id, metrics in self.difficulty_mapping.items():
            row = metrics.to_dict()
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save as parquet for efficiency
        df.to_parquet(filepath, index=False)
        
        # Also save summary statistics
        self._save_difficulty_summary(filepath)
        
        self.logger.info(f"Difficulty mapping saved to {filepath}")
        self.logger.info(f"Saved {len(self.difficulty_mapping)} difficulty entries")
        
        return str(filepath)
    
    def _save_difficulty_summary(self, mapping_filepath: Path) -> None:
        """
        Save summary statistics about difficulty distribution
        
        Args:
            mapping_filepath: Path to the main mapping file
        """
        summary_filepath = mapping_filepath.with_suffix('.summary.json')
        
        # Calculate complexity statistics
        complexity_scores = []
        
        for metrics in self.difficulty_mapping.values():
            complexity_scores.append(metrics.cyclomatic_complexity)
        
        total_problems = len(self.difficulty_mapping)
        
        # Calculate complexity distribution
        import numpy as np
        complexity_array = np.array(complexity_scores) if complexity_scores else np.array([])
        
        summary = {
            'total_problems_analyzed': total_problems,
            'complexity_statistics': {
                'min': int(complexity_array.min()) if len(complexity_array) > 0 else 0,
                'max': int(complexity_array.max()) if len(complexity_array) > 0 else 0,
                'mean': round(float(complexity_array.mean()), 2) if len(complexity_array) > 0 else 0,
                'median': float(np.median(complexity_array)) if len(complexity_array) > 0 else 0,
                'std': round(float(complexity_array.std()), 2) if len(complexity_array) > 0 else 0,
                'percentiles': {
                    '25th': float(np.percentile(complexity_array, 25)) if len(complexity_array) > 0 else 0,
                    '75th': float(np.percentile(complexity_array, 75)) if len(complexity_array) > 0 else 0,
                    '90th': float(np.percentile(complexity_array, 90)) if len(complexity_array) > 0 else 0
                }
            },
            'methodology_note': 'Uses interleaved sampling based on complexity scores, not categorical buckets',
            'timestamp': get_timestamp()
        }
        
        # Save summary as JSON
        import json
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Difficulty summary saved to {summary_filepath}")
    
    @classmethod
    def load_difficulty_mapping(cls, filepath: str) -> Dict[str, DifficultyMetrics]:
        """
        Load difficulty mapping from file
        
        Args:
            filepath: Path to difficulty mapping file
            
        Returns:
            dict: Mapping of task_id to DifficultyMetrics
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Difficulty mapping file not found: {filepath}")
        
        # Load DataFrame
        df = pd.read_parquet(filepath)
        
        # Convert back to DifficultyMetrics objects
        difficulty_mapping = {}
        for _, row in df.iterrows():
            metrics = DifficultyMetrics(
                task_id=row['task_id'],
                cyclomatic_complexity=row['cyclomatic_complexity'],
                reference_code=row['reference_code'],
                num_test_cases=row['num_test_cases'],
                avg_test_case_length=row['avg_test_case_length'],
                problem_description_length=row['problem_description_length']
            )
            difficulty_mapping[metrics.task_id] = metrics
        
        return difficulty_mapping
    
    def get_complexity_distribution(self) -> Dict[str, Any]:
        """
        Get complexity distribution statistics for analyzed problems
        
        Returns:
            dict: Complexity distribution statistics
        """
        if not self.difficulty_mapping:
            return {}
        
        complexity_scores = []
        
        for metrics in self.difficulty_mapping.values():
            complexity_scores.append(metrics.cyclomatic_complexity)
        
        total = len(self.difficulty_mapping)
        
        if not complexity_scores:
            return {'total_analyzed': 0}
        
        import numpy as np
        complexity_array = np.array(complexity_scores)
        
        return {
            'total_analyzed': total,
            'complexity_range': (int(complexity_array.min()), int(complexity_array.max())),
            'mean': round(float(complexity_array.mean()), 2),
            'median': float(np.median(complexity_array)),
            'std': round(float(complexity_array.std()), 2),
            'percentiles': {
                '25th': float(np.percentile(complexity_array, 25)),
                '75th': float(np.percentile(complexity_array, 75)),
                '90th': float(np.percentile(complexity_array, 90))
            }
        }