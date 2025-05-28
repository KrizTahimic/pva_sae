#!/usr/bin/env python3
"""
Script to run Phase 0: MBPP difficulty analysis preprocessing.

This script analyzes the complexity and difficulty of all MBPP problems
without any LLM involvement, creating a lightweight mapping file for
use by subsequent phases.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.logging import LoggingManager
from phase0_difficulty_analysis.mbpp_preprocessor import MBPPPreprocessor


def main():
    """Main entry point for difficulty analysis"""
    parser = argparse.ArgumentParser(
        description="Run Phase 0: MBPP difficulty analysis preprocessing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets",
        help="Directory to save difficulty mapping (default: data/datasets)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving difficulty mapping to file"
    )
    parser.add_argument(
        "--load-existing",
        type=str,
        help="Load and validate existing difficulty mapping file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logging_manager = LoggingManager(log_level=log_level, log_dir="data/logs")
    logger = logging_manager.setup_logging(__name__)
    
    try:
        # Initialize preprocessor
        preprocessor = MBPPPreprocessor(output_dir=args.output_dir)
        
        if args.load_existing:
            # Load and validate existing mapping
            logger.info(f"Loading existing difficulty mapping: {args.load_existing}")
            difficulty_mapping = preprocessor.load_existing_mapping(args.load_existing)
            
            # Validate completeness
            is_complete = preprocessor.validate_mapping_completeness(difficulty_mapping)
            
            if is_complete:
                logger.info("✅ Existing difficulty mapping is complete and valid")
                
                # Show distribution
                distribution = preprocessor.difficulty_analyzer.get_complexity_distribution()
                logger.info("Complexity distribution:")
                if distribution:
                    logger.info(f"  Mean: {distribution['mean']}")
                    logger.info(f"  Median: {distribution['median']}")
                    logger.info(f"  Range: {distribution['complexity_range']}")
                    logger.info(f"  Standard deviation: {distribution['std']}")
            else:
                logger.error("❌ Existing difficulty mapping is incomplete")
                sys.exit(1)
        
        else:
            # Run full preprocessing
            logger.info("Starting Phase 0: MBPP difficulty analysis")
            difficulty_mapping = preprocessor.preprocess_dataset(save_mapping=not args.no_save)
            
            if difficulty_mapping:
                logger.info("✅ Phase 0 completed successfully")
                logger.info(f"Analyzed {len(difficulty_mapping)} MBPP problems")
                
                if not args.no_save:
                    latest_mapping = preprocessor.get_latest_difficulty_mapping_path()
                    logger.info(f"Difficulty mapping available at: {latest_mapping}")
            else:
                logger.error("❌ Phase 0 failed - no difficulty mapping generated")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Phase 0 failed with error: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()