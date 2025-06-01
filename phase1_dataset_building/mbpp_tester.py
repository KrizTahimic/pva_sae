"""
Main orchestrator classes for Phase 1 of the PVA-SAE project.

This module contains the main classes that orchestrate the dataset building
process, coordinating between model management, dataset handling, and result generation.
"""

import logging
import os
from typing import Optional, Any, Dict, List
from tqdm import tqdm

from common import (
    LoggingManager,
    ModelManager,
    ModelConfiguration,
    DatasetConfiguration,
    RobustnessConfig,
    DEFAULT_MODEL_NAME,
    DEFAULT_LOG_DIR,
    DEFAULT_DATASET_DIR,
    MAX_NEW_TOKENS,
    auto_cleanup,
    ensure_directory_exists
)
from phase1_dataset_building.dataset_manager import PromptAwareDatasetManager, CodeTestResult
from phase1_dataset_building.test_executor import TestExecutor
from phase1_dataset_building.dataset_builder import DatasetBuilder, RobustDatasetBuilder


class MBPPTester:
    """Main orchestrator for MBPP testing workflow"""
    
    def __init__(self, 
                 model_name: str = DEFAULT_MODEL_NAME,
                 debug: bool = False, 
                 log_dir: str = DEFAULT_LOG_DIR):
        """
        Initialize MBPP tester
        
        Args:
            model_name: Name of the model to use
            debug: Enable debug logging
            log_dir: Directory for log files
        """
        self.model_name = model_name
        self.debug = debug
        self.log_dir = log_dir
        
        # Initialize components
        self.logging_manager = LoggingManager(
            log_dir=log_dir,
            log_level="DEBUG" if debug else "INFO"
        )
        self.logger = None
        
        # Model and dataset managers
        self.model_manager = None
        self.dataset_manager = PromptAwareDatasetManager()
        
        # Results tracking
        self.total_tests = 0
        self.passed_tests = 0
        self.record_results: List[CodeTestResult] = []
    
    @property
    def log_file(self) -> Optional[str]:
        """Get current log file path"""
        return self.logging_manager.log_file
    
    def setup_logging(self) -> str:
        """Configure logging system"""
        self.logger = self.logging_manager.setup_logging("mbpp_tester")
        return self.logging_manager.log_file
    
    def setup_model(self, model_config: Optional[ModelConfiguration] = None, num_gpus: int = 1):
        """
        Setup model manager
        
        Args:
            model_config: Optional model configuration
            num_gpus: Number of GPUs to use (1 for single GPU, >1 for DataParallel)
        """
        if not model_config:
            model_config = ModelConfiguration(model_name=self.model_name)
        
        self.model_manager = ModelManager(model_config)
        self.model_manager.load_model(num_gpus=num_gpus)
    
    def ensure_dataset_ready(self):
        """Ensure dataset is loaded and ready"""
        if not self.dataset_manager.is_loaded():
            self.dataset_manager.load_dataset()
    
    def test_single_record(self, idx: int) -> CodeTestResult:
        """
        Test single MBPP record by index
        
        Args:
            idx: Index of record to test
            
        Returns:
            CodeTestResult: Test execution results
        """
        try:
            record = self.dataset_manager.get_record(idx)
            
            self.logger.info(f"Processing record {idx} (Task ID: {record['task_id']})")
            
            result = TestExecutor.run_record_tests(record)
            self._update_overall_stats(result)
            
            self.logger.info("-" * 40)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to test record {idx}: {str(e)}")
            raise RuntimeError(f"Failed to test record {idx}: {str(e)}") from e
    
    def test_range(self, start_idx: int = 0, end_idx: int = 3) -> Dict[str, Any]:
        """
        Test range of MBPP records
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (inclusive)
            
        Returns:
            dict: Summary of test results
        """
        try:
            # Ensure prerequisites
            self._ensure_prerequisites()
            
            # Validate range
            validated_end = self._validate_test_range(start_idx, end_idx)
            
            # Execute tests
            self._run_test_sequence(start_idx, validated_end)
            
            # Generate final summary
            return self._create_summary()
            
        except Exception as e:
            self.logger.error(f"Test range execution failed: {str(e)}")
            raise RuntimeError(f"Test range execution failed: {str(e)}") from e
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current test results summary"""
        return {
            'passed': self.passed_tests,
            'total': self.total_tests,
            'success_rate': self._calculate_success_rate(),
            'records_tested': len(self.record_results),
            'log_file': self.log_file
        }
    
    def get_detailed_results(self) -> List[CodeTestResult]:
        """Get detailed results for each tested record"""
        return self.record_results.copy()
    
    def _ensure_prerequisites(self):
        """Ensure all prerequisites are met"""
        if not self.logger:
            self.setup_logging()
        self.ensure_dataset_ready()
    
    def _validate_test_range(self, start_idx: int, end_idx: int) -> int:
        """Validate and adjust test range"""
        dataset_size = self.dataset_manager.get_size()
        
        if start_idx < 0:
            raise ValueError(f"start_idx must be >= 0, got {start_idx}")
        if start_idx >= dataset_size:
            raise ValueError(f"start_idx {start_idx} >= dataset size {dataset_size}")
        
        validated_end = min(end_idx, dataset_size - 1)
        self.logger.info(f"Testing records {start_idx} to {validated_end}")
        return validated_end
    
    def _run_test_sequence(self, start_idx: int, end_idx: int):
        """Execute the test sequence with progress tracking"""
        self._reset_statistics()
        
        for idx in tqdm(range(start_idx, end_idx + 1),
                        desc="Testing MBPP records",
                        unit="record"):
            self.test_single_record(idx)
    
    def _reset_statistics(self):
        """Reset test statistics for new run"""
        self.total_tests = 0
        self.passed_tests = 0
        self.record_results = []
    
    def _update_overall_stats(self, result: CodeTestResult):
        """Update overall statistics with new result"""
        self.total_tests += result.total
        self.passed_tests += result.passed
        self.record_results.append(result)
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        return (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0.0
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create and log final summary"""
        summary = self.get_summary()
        
        self.logger.info(
            f"FINAL SUMMARY: {summary['passed']}/{summary['total']} tests passed "
            f"({summary['success_rate']:.1f}%)"
        )
        
        if self.log_file:
            print(f"ℹ️  Results logged to: {self.log_file}")
        
        return summary


class DatasetBuildingOrchestrator(MBPPTester):
    """Dataset building orchestrator with MBPP code generation capabilities"""
    
    def __init__(self,
                 model_name: str = DEFAULT_MODEL_NAME,
                 debug: bool = False,
                 log_dir: str = DEFAULT_LOG_DIR,
                 dataset_dir: str = DEFAULT_DATASET_DIR,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        """
        Initialize enhanced MBPP tester
        
        Args:
            model_name: Name of the model to use
            debug: Enable debug logging
            log_dir: Directory for log files
            dataset_dir: Directory for dataset files
            max_new_tokens: Maximum tokens to generate
        """
        super().__init__(model_name, debug, log_dir)
        
        self.dataset_dir = dataset_dir
        self.max_new_tokens = max_new_tokens
        self.dataset_builder = None
        
        # Ensure directories exist
        ensure_directory_exists(log_dir)
        ensure_directory_exists(dataset_dir)
    
    def build_dataset_simple(self, 
                             start_idx: int = 0, 
                             end_idx: int = 2, 
                             stream: bool = False) -> str:
        """
        Build dataset with simple configuration
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (inclusive)
            stream: Whether to stream generation output
            
        Returns:
            str: Path to saved dataset
        """
        try:
            # Setup if not already done
            if not self.logger:
                self.setup_logging()
            if not self.model_manager:
                self.setup_model()
            self.ensure_dataset_ready()
            
            # Create dataset builder
            dataset_config = DatasetConfiguration(
                dataset_dir=self.dataset_dir,
                start_idx=start_idx,
                end_idx=end_idx
            )
            
            self.dataset_builder = DatasetBuilder(
                model_manager=self.model_manager,
                dataset_manager=self.dataset_manager,
                config=dataset_config,
                max_new_tokens=self.max_new_tokens,
                stream_output=stream
            )
            
            # Build dataset
            self.logger.info("Starting dataset building process...")
            results = self.dataset_builder.build_dataset(start_idx, end_idx)
            
            # Save results
            dataset_path = self.dataset_builder.save_dataset(format="parquet")
            
            # Display summary
            stats = self.dataset_builder.get_statistics()
            print(f"\n{'='*60}")
            print("DATASET BUILDING COMPLETE")
            print(f"{'='*60}")
            print(f"Total Processed: {stats['total_processed']}")
            print(f"Correct Solutions: {stats['correct_solutions']} ({stats['correct_rate']:.1f}%)")
            print(f"Incorrect Solutions: {stats['incorrect_solutions']}")
            print(f"Dataset saved to: {dataset_path}")
            print(f"{'='*60}")
            
            return dataset_path
            
        except Exception as e:
            self.logger.error(f"Dataset building failed: {str(e)}")
            raise RuntimeError(f"Dataset building failed: {str(e)}") from e
    
    def build_dataset_simple_with_cleanup(self, 
                                          start_idx: int = 0, 
                                          end_idx: int = 2, 
                                          stream: bool = False) -> str:
        """
        Build dataset with simple configuration and automatic cleanup
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (inclusive)
            stream: Whether to stream generation output
            
        Returns:
            str: Path to saved dataset
        """
        # Perform cleanup first
        auto_cleanup(self.dataset_dir, self.log_dir)
        
        # Build dataset
        return self.build_dataset_simple(start_idx, end_idx, stream)
    
    def analyze_dataset(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a built dataset
        
        Args:
            dataset_path: Path to dataset file (uses latest if None)
            
        Returns:
            dict: Analysis results
        """
        if self.dataset_builder and not dataset_path:
            # Use current dataset builder
            return self.dataset_builder.analyze_results()
        
        # Load and analyze from file
        if not dataset_path:
            # Find latest dataset
            import glob
            pattern = os.path.join(self.dataset_dir, "mbpp_dataset_*.parquet")
            files = glob.glob(pattern)
            if not files:
                raise ValueError("No dataset files found")
            dataset_path = max(files, key=os.path.getmtime)
        
        # Load dataset
        import pandas as pd
        df = pd.read_parquet(dataset_path)
        
        # Analyze
        analysis = {
            'overview': {
                'total_records': len(df),
                'correct_solutions': df['is_correct'].sum(),
                'success_rate': df['is_correct'].mean() * 100,
                'avg_generation_time': df['generation_time'].mean(),
                'avg_code_length': df['code_length'].mean()
            },
            'test_performance': {
                'avg_tests_passed': df['passed_tests'].mean(),
                'perfect_scores': (df['passed_tests'] == df['total_tests']).sum(),
                'zero_scores': (df['passed_tests'] == 0).sum()
            },
            'dataset_file': dataset_path
        }
        
        return analysis


class ProductionDatasetBuilder(DatasetBuildingOrchestrator):
    """Production-ready dataset builder with robustness features"""
    
    def __init__(self,
                 model_name: str = DEFAULT_MODEL_NAME,
                 debug: bool = False,
                 log_dir: str = DEFAULT_LOG_DIR,
                 dataset_dir: str = DEFAULT_DATASET_DIR,
                 max_new_tokens: int = MAX_NEW_TOKENS,
                 robustness_config: Optional[RobustnessConfig] = None):
        """
        Initialize production MBPP tester
        
        Args:
            model_name: Name of the model to use
            debug: Enable debug logging
            log_dir: Directory for log files
            dataset_dir: Directory for dataset files
            max_new_tokens: Maximum tokens to generate
            robustness_config: Robustness configuration
        """
        super().__init__(model_name, debug, log_dir, dataset_dir, max_new_tokens)
        
        self.robustness_config = robustness_config or RobustnessConfig()
        self.robust_builder = None
    
    def build_dataset_production(self,
                                 start_idx: int = 0,
                                 end_idx: int = 973,
                                 stream: bool = False,
                                 resume_from_checkpoint: Optional[str] = None) -> str:
        """
        Build dataset with production robustness
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (inclusive)
            stream: Whether to stream generation output
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            str: Path to saved dataset
        """
        try:
            # Setup if not already done
            if not self.logger:
                self.setup_logging()
            if not self.model_manager:
                self.setup_model()
            self.ensure_dataset_ready()
            
            # Log experiment configuration
            self.logging_manager.log_experiment_info({
                'model_name': self.model_name,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'max_new_tokens': self.max_new_tokens,
                'robustness_config': self.robustness_config.to_dict(),
                'resume_from_checkpoint': resume_from_checkpoint
            })
            
            # Create robust dataset builder
            dataset_config = DatasetConfiguration(
                dataset_dir=self.dataset_dir,
                start_idx=start_idx,
                end_idx=end_idx
            )
            
            self.robust_builder = RobustDatasetBuilder(
                model_manager=self.model_manager,
                dataset_manager=self.dataset_manager,
                config=dataset_config,
                robustness_config=self.robustness_config,
                max_new_tokens=self.max_new_tokens,
                stream_output=stream
            )
            
            # Start phase
            self.logging_manager.log_phase_start(
                "Dataset Building",
                total_items=end_idx - start_idx + 1
            )
            
            # Build dataset with resume capability
            start_time = time.time()
            results = self.robust_builder.build_dataset_with_resume(
                start_idx, end_idx, resume_from_checkpoint
            )
            duration = time.time() - start_time
            
            # Log phase end
            stats = self.robust_builder.get_statistics()
            self.logging_manager.log_phase_end(
                "Dataset Building",
                duration=duration,
                success_count=stats['correct_solutions'],
                error_count=stats['incorrect_solutions']
            )
            
            # Get saved dataset path
            dataset_files = self.robust_builder.save_dataset(format="both")
            dataset_path = dataset_files[1]  # Parquet file
            
            # Display final summary
            self._display_production_summary(stats, dataset_path, duration)
            
            return dataset_path
            
        except Exception as e:
            self.logger.error(f"Production dataset building failed: {str(e)}")
            # Log error with context
            self.logging_manager.log_error_with_context(e, {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'processed': self.robust_builder.total_processed if self.robust_builder else 0
            })
            raise RuntimeError(f"Production dataset building failed: {str(e)}") from e
    
    def build_dataset_with_config_file(self, config_file: str) -> str:
        """
        Build dataset using configuration file
        
        Args:
            config_file: Path to configuration JSON file
            
        Returns:
            str: Path to saved dataset
        """
        # Load configurations
        import json
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Extract configurations
        model_config = ModelConfiguration(**config_data.get('model', {}))
        dataset_config = DatasetConfiguration(**config_data.get('dataset', {}))
        robustness_config = RobustnessConfig(**config_data.get('robustness', {}))
        
        # Update tester configuration
        self.model_name = model_config.model_name
        self.dataset_dir = dataset_config.dataset_dir
        self.max_new_tokens = model_config.max_new_tokens
        self.robustness_config = robustness_config
        
        # Setup model with config
        self.setup_model(model_config)
        
        # Build dataset
        return self.build_dataset_production(
            start_idx=dataset_config.start_idx,
            end_idx=dataset_config.end_idx or 973,
            stream=config_data.get('stream_output', False)
        )
    
    def _display_production_summary(self, stats: Dict[str, Any], 
                                    dataset_path: str, duration: float):
        """Display comprehensive production run summary"""
        print(f"\n{'='*80}")
        print("PRODUCTION DATASET BUILDING COMPLETE")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Total Processed: {stats['total_processed']}")
        print(f"Correct Solutions: {stats['correct_solutions']} ({stats['correct_rate']:.1f}%)")
        print(f"Incorrect Solutions: {stats['incorrect_solutions']}")
        print(f"Total Duration: {format_duration(duration)}")
        print(f"Average Speed: {stats['total_processed']/duration:.2f} records/second")
        print(f"Dataset saved to: {dataset_path}")
        print(f"Log file: {self.log_file}")
        print(f"{'='*80}")


# Import time for production tester
import time
from common import format_duration