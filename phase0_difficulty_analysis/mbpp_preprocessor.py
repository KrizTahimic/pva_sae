"""
MBPP preprocessor for Phase 0 difficulty analysis.

This module provides the main orchestrator for preprocessing the MBPP dataset
with difficulty analysis before any LLM interaction.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from common.logging import LoggingManager
from phase1_dataset_building.dataset_manager import DatasetManager
from .difficulty_analyzer import MBPPDifficultyAnalyzer, DifficultyMetrics


class MBPPPreprocessor:
    """Main orchestrator for Phase 0 MBPP difficulty preprocessing"""
    
    def __init__(self, output_dir: str = "data/datasets"):
        """
        Initialize MBPP preprocessor
        
        Args:
            output_dir: Directory to save preprocessed data
        """
        self.output_dir = Path(output_dir)
        logging_manager = LoggingManager(log_dir="data/logs")
        self.logger = logging_manager.setup_logging(__name__)
        
        # Initialize components
        self.dataset_manager = DatasetManager()
        self.difficulty_analyzer = MBPPDifficultyAnalyzer(str(output_dir))
        
        self.logger.info("MBPPPreprocessor initialized")
    
    def preprocess_dataset(self, save_mapping: bool = True) -> Dict[str, DifficultyMetrics]:
        """
        Run complete Phase 0 preprocessing on MBPP dataset
        
        Args:
            save_mapping: Whether to save difficulty mapping to file
            
        Returns:
            dict: Difficulty mapping for all MBPP problems
        """
        self.logger.info("Starting Phase 0: MBPP difficulty preprocessing")
        
        # Step 1: Load MBPP dataset
        self.logger.info("Step 1: Loading MBPP dataset")
        self.dataset_manager.load_dataset()
        dataset_size = self.dataset_manager.get_size()
        self.logger.info(f"Loaded MBPP dataset with {dataset_size} problems")
        
        # Step 2: Analyze difficulty for all problems
        self.logger.info("Step 2: Analyzing difficulty metrics")
        difficulty_mapping = self.difficulty_analyzer.analyze_dataset(self.dataset_manager)
        
        # Step 3: Save difficulty mapping if requested
        mapping_filepath = None
        if save_mapping:
            self.logger.info("Step 3: Saving difficulty mapping")
            mapping_filepath = self.difficulty_analyzer.save_difficulty_mapping()
            self.logger.info(f"Difficulty mapping saved to: {mapping_filepath}")
        
        # Step 4: Report summary statistics
        self._report_preprocessing_summary(difficulty_mapping, mapping_filepath)
        
        self.logger.info("Phase 0 preprocessing completed successfully")
        return difficulty_mapping
    
    def _report_preprocessing_summary(self, 
                                    difficulty_mapping: Dict[str, DifficultyMetrics],
                                    mapping_filepath: Optional[str] = None) -> None:
        """
        Report summary of preprocessing results
        
        Args:
            difficulty_mapping: Generated difficulty mapping
            mapping_filepath: Path to saved mapping file
        """
        distribution = self.difficulty_analyzer.get_complexity_distribution()
        
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
        
        if mapping_filepath:
            self.logger.info(f"Mapping saved to: {mapping_filepath}")
        
        self.logger.info("=== End Summary ===")
    
    def get_latest_difficulty_mapping_path(self) -> Optional[str]:
        """
        Get path to the most recently created difficulty mapping file
        
        Returns:
            str: Path to latest mapping file, or None if none found
        """
        # Step 1: Create glob pattern to match all difficulty mapping files
        # Matches: mbpp_difficulty_mapping_20250608_160739.parquet, etc.
        pattern = "mbpp_difficulty_mapping_*.parquet"
        
        # Step 2: Search output_dir (data/datasets) for all matching files
        # Returns list of Path objects
        mapping_files = list(self.output_dir.glob(pattern))
        
        if not mapping_files:
            return None
        
        # Step 3: Find file with most recent modification time
        # p.stat().st_mtime gets file's last modification timestamp (Unix epoch)
        # max() finds the Path with highest timestamp = most recently modified
        latest_file = max(mapping_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)
    
    def load_existing_mapping(self, filepath: Optional[str] = None) -> Dict[str, DifficultyMetrics]:
        """
        Load existing difficulty mapping from file
        
        Args:
            filepath: Optional path to mapping file. If None, loads latest mapping
            
        Returns:
            dict: Loaded difficulty mapping
        """
        if filepath is None:
            filepath = self.get_latest_difficulty_mapping_path()
            if filepath is None:
                raise FileNotFoundError("No existing difficulty mapping found")
        
        self.logger.info(f"Loading difficulty mapping from: {filepath}")
        mapping = MBPPDifficultyAnalyzer.load_difficulty_mapping(filepath)
        self.logger.info(f"Loaded difficulty mapping for {len(mapping)} problems")
        
        return mapping
    
    def validate_mapping_completeness(self, 
                                    difficulty_mapping: Dict[str, DifficultyMetrics]) -> bool:
        """
        Validate that difficulty mapping covers all MBPP problems
        
        Args:
            difficulty_mapping: Difficulty mapping to validate
            
        Returns:
            bool: True if mapping is complete
        """
        if not self.dataset_manager.is_loaded():
            self.dataset_manager.load_dataset()
        
        expected_size = self.dataset_manager.get_size()
        actual_size = len(difficulty_mapping)
        
        is_complete = expected_size == actual_size
        
        if is_complete:
            self.logger.info(f"Difficulty mapping validation passed: {actual_size}/{expected_size} problems")
        else:
            self.logger.warning(f"Difficulty mapping incomplete: {actual_size}/{expected_size} problems")
        
        return is_complete