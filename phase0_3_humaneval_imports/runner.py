"""Runner for Phase 0.3: HumanEval import scanning."""

from pathlib import Path
from common.config import Config
from common.logging import get_logger
from common.utils import get_phase_output_dir, write_phase_output
from .scanner import scan_humaneval_imports, save_imports

logger = get_logger("phase0_3.runner", phase="0.3")


def run_phase_0_3(config: Config):
    """
    Run Phase 0.3: Scan HumanEval prompts for required imports.

    Args:
        config: Configuration object

    Outputs:
        - data/phase0_3_humaneval/required_imports.json
    """
    logger.info("=" * 80)
    logger.info("PHASE 0.3: HUMANEVAL IMPORT SCANNING")
    logger.info("=" * 80)

    # Output directory from registry
    output_dir = Path(get_phase_output_dir("0.3", config))
    logger.info(f"Output directory: {output_dir}")

    # Scan imports
    logger.info("\nScanning HumanEval prompts for import statements...")
    imports_data = scan_humaneval_imports()

    # Save results
    logger.info("\nSaving results...")
    output_file = save_imports(imports_data, output_dir)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 0.3 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Scanned: {imports_data['n_problems']} problems")
    logger.info(f"Found: {imports_data['n_unique_imports']} unique import statements")
    logger.info(f"Output: {output_file}")
    logger.info("")
    logger.info("Import statements:")
    for imp in imports_data['imports']:
        logger.info(f"  {imp}")

    # Write phase output manifest
    write_phase_output(
        phase="0.3",
        outputs={
            "primary": "required_imports.json"
        },
        config=config,
        output_dir=str(output_dir)
    )

    return imports_data
