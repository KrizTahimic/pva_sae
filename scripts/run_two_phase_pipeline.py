#!/usr/bin/env python3
"""
Script to run the complete two-phase pipeline: 
Phase 0 (Difficulty Analysis) → Phase 1 (LLM Dataset Building)

This script combines difficulty preprocessing with LLM-based dataset generation,
using pre-computed difficulty scores to enrich the final dataset.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.logging import LoggingManager
from common.config import DatasetConfiguration, RobustnessConfig, ModelConfiguration
from common.models import ModelManager
from phase0_difficulty_analysis.mbpp_preprocessor import MBPPPreprocessor
from phase0_difficulty_analysis.difficulty_analyzer import MBPPDifficultyAnalyzer
from phase1_dataset_building.dataset_manager import PromptAwareDatasetManager
from phase1_dataset_building.dataset_builder import RobustDatasetBuilder


def main():
    """Main entry point for two-phase pipeline"""
    parser = argparse.ArgumentParser(
        description="Run two-phase pipeline: Difficulty Analysis + LLM Dataset Building"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b",
        help="Model name for LLM generation (default: google/gemma-2-2b)"
    )
    
    # Phase configuration
    parser.add_argument(
        "--skip-phase0",
        action="store_true",
        help="Skip Phase 0 and use existing difficulty mapping"
    )
    parser.add_argument(
        "--difficulty-mapping",
        type=str,
        help="Path to existing difficulty mapping file (auto-detect if not specified)"
    )
    
    # Range configuration
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index for MBPP records (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=10,
        help="Ending index for MBPP records (default: 10)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets",
        help="Directory to save outputs (default: data/datasets)"
    )
    parser.add_argument(
        "--save-format",
        choices=["json", "parquet", "both"],
        default="both",
        help="Dataset save format (default: both)"
    )
    
    # Robustness configuration
    parser.add_argument(
        "--enable-checkpoints",
        action="store_true",
        help="Enable checkpoint recovery for Phase 1"
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=50,
        help="Checkpoint frequency for Phase 1 (default: 50)"
    )
    
    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running pipeline"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logging_manager = LoggingManager(log_level=log_level, log_dir="data/logs")
    logger = logging_manager.setup_logging(__name__)
    
    try:
        # Show configuration
        logger.info("=== Two-Phase Pipeline Configuration ===")
        logger.info(f"Model: {args.model}")
        logger.info(f"Record range: {args.start}-{args.end}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Save format: {args.save_format}")
        logger.info(f"Skip Phase 0: {args.skip_phase0}")
        logger.info(f"Enable checkpoints: {args.enable_checkpoints}")
        
        if args.dry_run:
            logger.info("Dry run completed - no processing performed")
            return
        
        # Phase 0: Difficulty Analysis
        difficulty_mapping = None
        if not args.skip_phase0:
            logger.info("=== Starting Phase 0: Difficulty Analysis ===")
            preprocessor = MBPPPreprocessor(output_dir=args.output_dir)
            difficulty_mapping = preprocessor.preprocess_dataset(save_mapping=True)
            
            logger.info(f"✅ Phase 0 completed: {len(difficulty_mapping)} problems analyzed")
            
            # Get mapping file path for Phase 1
            mapping_filepath = preprocessor.get_latest_difficulty_mapping_path()
            logger.info(f"Difficulty mapping saved to: {mapping_filepath}")
        
        else:
            # Load existing difficulty mapping
            logger.info("=== Loading existing difficulty mapping ===")
            if args.difficulty_mapping:
                mapping_filepath = args.difficulty_mapping
            else:
                preprocessor = MBPPPreprocessor(output_dir=args.output_dir)
                mapping_filepath = preprocessor.get_latest_difficulty_mapping_path()
                if mapping_filepath is None:
                    raise FileNotFoundError("No existing difficulty mapping found. Run without --skip-phase0 first.")
            
            logger.info(f"Loading difficulty mapping from: {mapping_filepath}")
            difficulty_mapping = MBPPDifficultyAnalyzer.load_difficulty_mapping(mapping_filepath)
            logger.info(f"✅ Loaded difficulty mapping for {len(difficulty_mapping)} problems")
        
        # Phase 1: LLM Dataset Building
        logger.info("=== Starting Phase 1: LLM Dataset Building ===")
        
        # Initialize components
        model_config = ModelConfiguration(model_name=args.model)
        model_manager = ModelManager(model_config)
        model_manager.load_model()
        
        dataset_manager = PromptAwareDatasetManager()
        dataset_manager.load_dataset()
        
        # Setup configurations
        dataset_config = DatasetConfiguration(dataset_dir=args.output_dir)
        
        if args.enable_checkpoints:
            robustness_config = RobustnessConfig(
                enable_checkpointing=True,
                checkpoint_frequency=args.checkpoint_frequency
            )
            builder = RobustDatasetBuilder(
                model_manager=model_manager,
                dataset_manager=dataset_manager,
                config=dataset_config,
                robustness_config=robustness_config,
                difficulty_mapping=difficulty_mapping
            )
            
            # Build with robustness features
            results = builder.build_dataset_with_resume(
                start_idx=args.start,
                end_idx=args.end
            )
        else:
            from phase1_dataset_building.dataset_builder import DatasetBuilder
            builder = DatasetBuilder(
                model_manager=model_manager,
                dataset_manager=dataset_manager,
                config=dataset_config,
                difficulty_mapping=difficulty_mapping
            )
            
            # Build dataset
            results = builder.build_dataset(
                start_idx=args.start,
                end_idx=args.end
            )
        
        # Save dataset with difficulty information
        saved_paths = builder.save_dataset(format=args.save_format)
        
        # Report final results
        stats = builder.get_statistics()
        logger.info("=== Two-Phase Pipeline Summary ===")
        logger.info(f"Total problems processed: {stats['total_processed']}")
        logger.info(f"Correct solutions: {stats['correct_solutions']}")
        logger.info(f"Incorrect solutions: {stats['incorrect_solutions']}")
        logger.info(f"Success rate: {stats['correct_rate']:.1f}%")
        
        if isinstance(saved_paths, tuple):
            logger.info(f"Dataset saved to: {saved_paths[0]} and {saved_paths[1]}")
        else:
            logger.info(f"Dataset saved to: {saved_paths}")
        
        logger.info("✅ Two-phase pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"Two-phase pipeline failed: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()