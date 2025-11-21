"""
Phase 0.2 Runner: HumanEval to MBPP Conversion.

Converts the entire HumanEval dataset to MBPP format for seamless
integration with existing pipeline phases.
"""

from pathlib import Path
from .converter import convert_humaneval_to_mbpp, inspect_sample_conversions
from common.config import Config


def run_phase_0_2(config: Config):
    """
    Execute Phase 0.2: Convert HumanEval to MBPP format.

    Args:
        config: Configuration object with phase0_2_output_dir
    """
    print("=" * 80)
    print("PHASE 0.2: HUMANEVAL TO MBPP CONVERSION")
    print("=" * 80)

    # Get output directory from config
    output_dir = config.phase0_2_output_dir

    # Run conversion
    df = convert_humaneval_to_mbpp(output_dir=output_dir)

    # Inspect samples
    inspect_sample_conversions(df, num_samples=5)

    print("\n" + "=" * 80)
    print("PHASE 0.2 COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {output_dir}/humaneval.parquet")
    print(f"Total problems converted: {len(df)}")
    print("\nNext steps:")
    print("  1. Verify schema matches validation_mbpp.parquet")
    print("  2. Manually inspect converted problems")
    print("  3. Test Phase 3.5 with converted dataset")
