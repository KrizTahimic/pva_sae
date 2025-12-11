"""
Phase 0.2 Runner: HumanEval to MBPP Conversion.

Converts the entire HumanEval dataset to MBPP format for seamless
integration with existing pipeline phases.
"""

from pathlib import Path
from .converter import convert_humaneval_to_mbpp, inspect_sample_conversions
from common.config import Config
from common.logging import get_logger
from common.utils import get_phase_output_dir

logger = get_logger("phase0_2.runner")


def run_phase_0_2(config: Config):
    """
    Execute Phase 0.2: Convert HumanEval to MBPP format.

    Args:
        config: Configuration object with phase0_2_output_dir
    """
    logger.info("=" * 80)
    logger.info("PHASE 0.2: HUMANEVAL TO MBPP CONVERSION")
    logger.info("=" * 80)

    # Get output directory from registry
    output_dir = get_phase_output_dir("0.2", config)

    # Run conversion
    df = convert_humaneval_to_mbpp(output_dir=output_dir)

    # Inspect samples
    inspect_sample_conversions(df, num_samples=5)

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 0.2 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput: {output_dir}/humaneval.parquet")
    logger.info(f"Total problems converted: {len(df)}")
    logger.info("\nNext steps:")
    logger.info("  1. Verify schema matches validation_mbpp.parquet")
    logger.info("  2. Manually inspect converted problems")
    logger.info("  3. Test Phase 3.5 with converted dataset")
