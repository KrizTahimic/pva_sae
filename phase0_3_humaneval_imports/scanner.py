"""Scanner to extract required imports from HumanEval prompts."""

import json
from pathlib import Path
from datetime import datetime
from typing import Set, List, Dict
from datasets import load_dataset
from common.logging import get_logger

logger = get_logger("phase0_3.scanner")


def scan_humaneval_imports() -> Dict:
    """
    Scan all HumanEval prompts and extract required import statements.

    Returns:
        Dictionary containing:
        - dataset: Dataset name
        - imports: List of import statements
        - scanned_at: Timestamp
        - n_problems: Number of problems scanned
    """
    logger.info("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")

    logger.info(f"Scanning {len(dataset)} problems for import statements...")

    # Collect all unique import statements
    import_statements = set()

    for idx, problem in enumerate(dataset):
        prompt = problem['prompt']

        # Extract import lines (must contain "import" keyword)
        for line in prompt.split('\n'):
            stripped = line.strip()
            # Match lines that are actual imports (contain "import" keyword)
            if stripped.startswith('from ') and ' import ' in stripped:
                import_statements.add(stripped)
                logger.debug(f"Task {idx}: Found import: {stripped}")
            # Also match standalone "import X" statements
            elif stripped.startswith('import '):
                import_statements.add(stripped)
                logger.debug(f"Task {idx}: Found import: {stripped}")

    # Sort for consistency
    imports_list = sorted(list(import_statements))

    logger.info(f"Found {len(imports_list)} unique import statements:")
    for imp in imports_list:
        logger.info(f"  {imp}")

    # Create output dictionary
    result = {
        'dataset': 'humaneval',
        'imports': imports_list,
        'scanned_at': datetime.now().isoformat(),
        'n_problems': len(dataset),
        'n_unique_imports': len(imports_list)
    }

    return result


def save_imports(imports_data: Dict, output_dir: Path) -> Path:
    """
    Save scanned imports to JSON file.

    Args:
        imports_data: Dictionary with import information
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "required_imports.json"

    with open(output_file, 'w') as f:
        json.dump(imports_data, f, indent=2)

    logger.info(f"Saved import list to {output_file}")
    return output_file
