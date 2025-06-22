"""
MBPP preprocessor for Phase 0 difficulty analysis.

This module provides the main orchestrator for preprocessing the MBPP dataset
with difficulty analysis before any LLM interaction.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from common.logging import get_logger
from .difficulty_analyzer import MBPPDifficultyAnalyzer
from common.utils import get_phase_dir
from datasets import load_dataset


class MBPPPreprocessor:
    """Main orchestrator for Phase 0 MBPP difficulty preprocessing"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize MBPP preprocessor
        
        Args:
            output_dir: Directory to save preprocessed data
        """
        self.output_dir = Path(output_dir or get_phase_dir(0))
        self.logger = get_logger("mbpp_preprocessor", phase="0.0")
        
        # Initialize difficulty analyzer
        self.difficulty_analyzer = MBPPDifficultyAnalyzer(str(output_dir))
        
        # We'll load MBPP dataset directly from HuggingFace
        self.dataset = None
        self.test_data = None
        
        self.logger.info("MBPPPreprocessor initialized")
    
    def preprocess_dataset(self, save_mapping: bool = True) -> pd.DataFrame:
        """
        Run complete Phase 0 preprocessing on MBPP dataset
        
        Args:
            save_mapping: Whether to save difficulty mapping to file
            
        Returns:
            dict: Difficulty mapping for all MBPP problems
        """
        self.logger.info("Starting Phase 0: MBPP difficulty preprocessing")
        
        # Step 1: Load MBPP dataset directly from HuggingFace
        self.logger.info("Step 1: Loading MBPP dataset from HuggingFace")
        self._load_mbpp_from_huggingface()
        dataset_size = len(self.test_data)
        self.logger.info(f"Loaded MBPP dataset with {dataset_size} problems")
        
        # Step 2: Analyze difficulty for all problems
        self.logger.info("Step 2: Analyzing difficulty metrics")
        enriched_df = self.difficulty_analyzer.analyze_dataset(self)
        
        # Step 3: Save enriched dataset if requested
        enriched_filepath = None
        if save_mapping:
            self.logger.info("Step 3: Saving enriched dataset")
            enriched_filepath = self.difficulty_analyzer.save_enriched_dataset(enriched_df)
            self.logger.info(f"Enriched dataset saved to: {enriched_filepath}")
        
        # Step 4: Report summary statistics
        complexity_stats = self.difficulty_analyzer.get_complexity_distribution(enriched_df)
        self._report_preprocessing_summary(complexity_stats, enriched_filepath)
        
        self.logger.info("Phase 0 preprocessing completed successfully")
        return enriched_df
    
    def _report_preprocessing_summary(self, 
                                    distribution: Dict[str, Any],
                                    enriched_filepath: Optional[str] = None) -> None:
        """
        Report summary of preprocessing results
        
        Args:
            distribution: Complexity distribution statistics
            enriched_filepath: Path to saved enriched dataset file
        """
        
        self.logger.info("=== Phase 0 Preprocessing Summary ===")
        self.logger.info(f"Total problems analyzed: {distribution['total_analyzed']}")
        self.logger.info(f"Complexity range: {distribution['complexity_range']}")
        self.logger.info(f"Complexity statistics:")
        self.logger.info(f"  Mean: {distribution['mean']}")
        self.logger.info(f"  Median: {distribution['median']}")
        self.logger.info(f"  Standard deviation: {distribution['std']}")
        self.logger.info(f"  25th percentile: {distribution['percentiles']['25th']}")
        self.logger.info(f"  75th percentile: {distribution['percentiles']['75th']}")
        self.logger.info(f"  90th percentile: {distribution['percentiles']['90th']}")
        
        if enriched_filepath:
            self.logger.info(f"Enriched dataset saved to: {enriched_filepath}")
        
        self.logger.info("=== End Summary ===")
    
    def _load_mbpp_from_huggingface(self):
        """Load MBPP dataset directly from HuggingFace."""
        try:
            self.logger.info("Loading MBPP dataset from HuggingFace...")
            self.dataset = load_dataset("Muennighoff/mbpp", "full")
            self.test_data = self.dataset['test']
            self.logger.info(f"Loaded {len(self.test_data)} MBPP problems")
        except Exception as e:
            self.logger.error(f"Failed to load MBPP from HuggingFace: {str(e)}")
            raise RuntimeError(f"Failed to load MBPP dataset: {str(e)}") from e
    
    def get_size(self) -> int:
        """Get dataset size."""
        return len(self.test_data) if self.test_data else 0
    
    def get_record(self, idx: int) -> dict:
        """Get record by index."""
        if not self.test_data:
            raise RuntimeError("Dataset not loaded")
        return self.test_data[idx]
    
    def is_loaded(self) -> bool:
        """Check if dataset is loaded."""
        return self.test_data is not None
    
    def get_latest_enriched_dataset_path(self) -> Optional[str]:
        """
        Get path to the most recently created enriched dataset file
        
        Returns:
            str: Path to latest enriched dataset file, or None if none found
        """
        # Create glob pattern to match enriched dataset files
        pattern = "mbpp_with_complexity_*.parquet"
        
        # Search output_dir for all matching files
        enriched_files = list(self.output_dir.glob(pattern))
        
        if not enriched_files:
            return None
        
        # Find file with most recent modification time
        latest_file = max(enriched_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)
    
    
    
