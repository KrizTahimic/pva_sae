"""
Difficulty analysis module for MBPP dataset preprocessing.

This module analyzes the complexity and difficulty of MBPP programming problems
without any LLM involvement. It creates a lightweight mapping of difficulty
scores that can be used by subsequent phases.
"""

import pandas as pd
from pandas import DataFrame, read_parquet
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from common.utils import get_timestamp, ensure_directory_exists, get_phase_dir
from common.logging import get_logger


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
        self.logger = get_logger("difficulty_analyzer", phase="0.0")
        
        ensure_directory_exists(str(self.output_dir))
        self.logger.info("MBPPDifficultyAnalyzer initialized")
    
    def analyze_dataset(self, dataset_manager) -> pd.DataFrame:
        """
        Analyze difficulty for entire MBPP dataset and create enriched DataFrame
        
        Args:
            dataset_manager: Loaded DatasetManager instance
            
        Returns:
            pd.DataFrame: Full MBPP dataset with cyclomatic_complexity column added
        """
        if not dataset_manager.is_loaded():
            raise ValueError("Dataset manager must be loaded before analysis")
        
        dataset_size = dataset_manager.get_size()
        self.logger.info(f"Starting difficulty analysis for {dataset_size} MBPP problems")
        
        # Collect all records with complexity analysis
        enriched_records = []
        failed_analyses = []
        
        for idx in range(dataset_size):
            try:
                record = dataset_manager.get_record(idx)
                
                # Calculate cyclomatic complexity
                reference_code = record.get('code', '')
                cyclomatic_complexity = get_cyclomatic_complexity(reference_code)
                
                # Create enriched record with all original fields plus complexity
                enriched_record = {
                    'task_id': record.get('task_id', -1),
                    'text': record.get('text', ''),
                    'code': record.get('code', ''),
                    'test_list': record.get('test_list', []),
                    'cyclomatic_complexity': cyclomatic_complexity
                }
                enriched_records.append(enriched_record)
                
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"Analyzed {idx + 1}/{dataset_size} problems")
                    
            except Exception as e:
                task_id = record.get('task_id', f'index_{idx}') if 'record' in locals() else f'index_{idx}'
                failed_analyses.append(task_id)
                self.logger.error(f"Failed to analyze problem {task_id}: {str(e)}")
        
        # Create DataFrame from enriched records
        enriched_df = pd.DataFrame(enriched_records)
        
        if failed_analyses:
            self.logger.warning(f"Failed to analyze {len(failed_analyses)} problems: {failed_analyses[:10]}...")
        
        self.logger.info(f"Difficulty analysis completed: {len(enriched_df)} problems analyzed")
        return enriched_df
    
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
    
    
    
    def save_enriched_dataset(self, df: pd.DataFrame, filepath: Optional[str] = None) -> str:
        """
        Save enriched MBPP dataset with complexity to file
        
        Args:
            df: Enriched DataFrame to save
            filepath: Optional custom filepath. If None, generates timestamped filename
        
        Returns:
            str: Path to saved file
        """
        # Generate filename if not provided
        if filepath is None:
            timestamp = get_timestamp()
            filename = f"mbpp_with_complexity_{timestamp}.parquet"
            filepath = self.output_dir / filename
        else:
            filepath = Path(filepath)
        
        # Save to file
        df.to_parquet(filepath, index=False)
        self.logger.info(f"Saved enriched dataset to {filepath}")
        self.logger.info(f"Dataset shape: {df.shape}")
        
        return str(filepath)
    
    
    def get_complexity_distribution(self, enriched_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Get complexity distribution statistics for analyzed problems
        
        Args:
            enriched_df: DataFrame with cyclomatic_complexity column
            
        Returns:
            dict: Complexity distribution statistics
        """
        if enriched_df is None or enriched_df.empty:
            return {'total_analyzed': 0}
        
        complexity_scores = enriched_df['cyclomatic_complexity'].tolist()
        total = len(enriched_df)
        
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