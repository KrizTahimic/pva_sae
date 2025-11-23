#!/usr/bin/env python3
"""
Extract all required imports from raw MBPP dataset solutions.

This script loads the original MBPP solutions (which contain import statements)
and extracts all unique imports. This is the CORRECT way to determine what
libraries MBPP problems actually use.

Usage:
    python3 extract_mbpp_imports.py
"""

import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Set, List, Dict
from collections import Counter

def extract_imports_from_code(code: str) -> Set[str]:
    """
    Extract import statements from Python code.

    Handles:
    - import module
    - import module as alias
    - from module import something
    - from module import something as alias

    Returns set of import lines as written in the code.
    """
    imports = set()

    for line in code.split('\n'):
        stripped = line.strip()

        # Match "from X import Y" statements
        if stripped.startswith('from ') and ' import ' in stripped:
            imports.add(stripped)

        # Match "import X" statements
        elif stripped.startswith('import '):
            imports.add(stripped)

    return imports


def extract_base_library(import_statement: str) -> str:
    """
    Extract base library name from import statement.

    Examples:
        "import math" -> "math"
        "import heapq as hq" -> "heapq"
        "from collections import Counter" -> "collections"
        "from itertools import chain, combinations" -> "itertools"
    """
    # Handle "from X import Y"
    if import_statement.startswith('from '):
        # Extract module name after "from" and before "import"
        match = re.match(r'from\s+([^\s]+)\s+import', import_statement)
        if match:
            return match.group(1).split('.')[0]  # Take first part for nested imports

    # Handle "import X" or "import X as Y"
    elif import_statement.startswith('import '):
        # Extract module name after "import" and before "as" or end
        match = re.match(r'import\s+([^\s,]+)', import_statement)
        if match:
            return match.group(1).split('.')[0]  # Take first part for nested imports

    return None


def main():
    print("="*80)
    print("EXTRACT MBPP REQUIRED IMPORTS FROM ORIGINAL SOLUTIONS")
    print("="*80)
    print("\nThis script analyzes the ORIGINAL MBPP solutions to find actual imports.")
    print("Much more accurate than trying to detect library usage in generated code!\n")

    # Load MBPP dataset from Phase 0.1 parquet files
    print("Loading MBPP dataset from Phase 0.1 parquet files...")

    # Load all three splits
    phase0_1_dir = Path("data/phase0_1")
    sae_df = pd.read_parquet(phase0_1_dir / "sae_mbpp.parquet")
    hyperparams_df = pd.read_parquet(phase0_1_dir / "hyperparams_mbpp.parquet")
    validation_df = pd.read_parquet(phase0_1_dir / "validation_mbpp.parquet")

    # Combine all splits
    all_data = pd.concat([sae_df, hyperparams_df, validation_df], ignore_index=True)

    print(f"✅ Loaded {len(all_data)} total problems")
    print(f"   - SAE set: {len(sae_df)} problems")
    print(f"   - Hyperparams set: {len(hyperparams_df)} problems")
    print(f"   - Validation set: {len(validation_df)} problems\n")

    # Extract all import statements
    print("Extracting import statements from solutions...")
    all_imports = set()
    import_counts = Counter()

    problems_with_imports = 0

    for idx, row in all_data.iterrows():
        code = row['code']
        task_id = row['task_id']

        # Extract imports from this solution
        imports = extract_imports_from_code(code)

        if imports:
            problems_with_imports += 1
            all_imports.update(imports)

            # Count base library usage
            for imp in imports:
                base_lib = extract_base_library(imp)
                if base_lib:
                    import_counts[base_lib] += 1

            print(f"  Task {task_id}: {len(imports)} import(s)")

    # Sort imports for consistency
    sorted_imports = sorted(list(all_imports))
    sorted_libraries = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)

    # Display results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal problems: {len(all_data)}")
    print(f"Problems with imports: {problems_with_imports} ({problems_with_imports/len(all_data)*100:.1f}%)")
    print(f"Problems without imports: {len(all_data) - problems_with_imports} ({(len(all_data)-problems_with_imports)/len(all_data)*100:.1f}%)")
    print(f"\nTotal unique import statements: {len(sorted_imports)}")
    print(f"Total unique base libraries: {len(import_counts)}")

    # Show library usage frequency
    print(f"\n{'='*80}")
    print("📦 BASE LIBRARIES (sorted by usage)")
    print(f"{'='*80}\n")

    for i, (lib, count) in enumerate(sorted_libraries, 1):
        pct = count / len(all_data) * 100
        print(f"  {i:2d}. {lib:20s} used in {count:3d} problems ({pct:4.1f}%)")

    # Show all unique import statements
    print(f"\n{'='*80}")
    print("📋 ALL UNIQUE IMPORT STATEMENTS")
    print(f"{'='*80}\n")

    for imp in sorted_imports:
        print(f"  {imp}")

    # Generate simple import list for copy-paste
    unique_libraries = sorted(set(import_counts.keys()))

    print(f"\n{'='*80}")
    print("📋 SIMPLE IMPORT LIST (for copy-paste)")
    print(f"{'='*80}\n")

    for lib in unique_libraries:
        print(f"  import {lib}")

    # Save results to file
    output_dir = Path("investigation_results")
    output_dir.mkdir(exist_ok=True)

    # Save detailed results
    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'dataset': 'mbpp',
        'split': 'all (sae + hyperparams + validation)',
        'total_problems': len(all_data),
        'problems_with_imports': problems_with_imports,
        'problems_without_imports': len(all_data) - problems_with_imports,
        'n_unique_import_statements': len(sorted_imports),
        'n_unique_libraries': len(import_counts),
        'library_usage_counts': dict(sorted_libraries),
        'all_import_statements': sorted_imports,
        'simple_library_list': unique_libraries
    }

    json_file = output_dir / "mbpp_required_imports.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved detailed results to: {json_file}")

    # Save simple text file with just import statements
    txt_file = output_dir / "mbpp_imports.txt"
    with open(txt_file, 'w') as f:
        f.write("# MBPP Required Imports (from original solutions)\n")
        f.write(f"# Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total libraries: {len(unique_libraries)}\n")
        f.write(f"# Problems analyzed: {len(all_data)}\n\n")

        f.write("# Sorted by usage frequency:\n")
        for lib, count in sorted_libraries:
            pct = count / len(all_data) * 100
            f.write(f"# {lib:20s} - {count:3d} problems ({pct:4.1f}%)\n")

        f.write("\n# Simple import statements:\n")
        for lib in unique_libraries:
            f.write(f"import {lib}\n")

        f.write("\n# All original import statements (with aliases and specific imports):\n")
        for imp in sorted_imports:
            f.write(f"{imp}\n")

    print(f"✅ Saved import list to: {txt_file}")

    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print("\n💡 Key Finding:")
    print(f"   Only {problems_with_imports}/{len(all_data)} ({problems_with_imports/len(all_data)*100:.1f}%) of MBPP problems require imports!")
    print(f"   {len(all_data) - problems_with_imports}/{len(all_data)} ({(len(all_data)-problems_with_imports)/len(all_data)*100:.1f}%) use pure Python with no imports.\n")


if __name__ == "__main__":
    main()
