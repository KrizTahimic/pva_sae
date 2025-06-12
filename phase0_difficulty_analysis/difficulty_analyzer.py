"""
Difficulty analysis module for MBPP dataset preprocessing.

This module analyzes the complexity and difficulty of MBPP programming problems
without any LLM involvement. It creates a lightweight mapping of difficulty
scores that can be used by subsequent phases.
"""

from pandas import DataFrame, read_parquet
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from common.utils import get_timestamp, ensure_directory_exists
from common.logging import LoggingManager
from common.utils import get_phase_dir


def get_cyclomatic_complexity(code: str) -> int:
    """
    Calculate cyclomatic complexity for Python code
    
    Args:
        code: Python source code string
        
    Returns:
        int: Cyclomatic complexity score
    """
    try:
        from radon.complexity import cc_visit
        results = cc_visit(code)
        complexity = max(r.complexity for r in results) if results else 1
    except Exception:
        complexity = 1
    
    return complexity


@dataclass
class DifficultyMetrics:
    """Encapsulates difficulty analysis metrics for a single MBPP problem"""
    task_id: int
    cyclomatic_complexity: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class MBPPDifficultyAnalyzer:
    """Analyzes difficulty metrics for the entire MBPP dataset"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize difficulty analyzer
        
        Args:
            output_dir: Directory to save difficulty mapping
        """
        self.output_dir = Path(output_dir or get_phase_dir(0))
        logging_manager = LoggingManager(log_dir="data/logs")
        self.logger = logging_manager.setup_logging(__name__)
        self.difficulty_mapping: Dict[int, DifficultyMetrics] = {}
        
        ensure_directory_exists(str(self.output_dir))
        self.logger.info("MBPPDifficultyAnalyzer initialized")
    
    def analyze_dataset(self, dataset_manager) -> Dict[int, DifficultyMetrics]:
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
        task_id = record.get('task_id', -1)  # Use -1 as sentinel for missing task_id
        reference_code = record.get('code', '')
        test_list = record.get('test_list', [])
        problem_text = record.get('text', '')
        
        # Calculate cyclomatic complexity
        cyclomatic_complexity = get_cyclomatic_complexity(reference_code)
        
        return DifficultyMetrics(
            task_id=task_id,
            cyclomatic_complexity=cyclomatic_complexity
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
        df_data = [metrics.to_dict() for metrics in self.difficulty_mapping.values()]
        
        df = DataFrame(df_data)
        
        # Save as parquet for efficiency
        df.to_parquet(filepath, index=False)
        
        self.logger.info(f"Difficulty mapping saved to {filepath}")
        self.logger.info(f"Saved {len(self.difficulty_mapping)} difficulty entries")
        
        return str(filepath)
    
    @classmethod
    def load_difficulty_mapping(cls, filepath: str) -> Dict[int, DifficultyMetrics]:
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
        df = read_parquet(filepath)
        
        # Convert back to DifficultyMetrics objects
        difficulty_mapping = {}
        for _, row in df.iterrows():
            metrics = DifficultyMetrics(
                task_id=row['task_id'],
                cyclomatic_complexity=row['cyclomatic_complexity']
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
        
        complexity_scores = [metrics.cyclomatic_complexity for metrics in self.difficulty_mapping.values()]
        
        total = len(self.difficulty_mapping)
        
        if not complexity_scores:
            return {'total_analyzed': 0}
        
        from numpy import array, min as np_min, max as np_max, mean, median, std, percentile
        complexity_array = array(complexity_scores)
        
        return {
            'total_analyzed': total,
            'complexity_range': (int(np_min(complexity_array)), int(np_max(complexity_array))),
            'mean': round(float(mean(complexity_array)), 2),
            'median': float(median(complexity_array)),
            'std': round(float(std(complexity_array)), 2),
            'percentiles': {
                '25th': float(percentile(complexity_array, 25)),
                '75th': float(percentile(complexity_array, 75)),
                '90th': float(percentile(complexity_array, 90))
            }
        }