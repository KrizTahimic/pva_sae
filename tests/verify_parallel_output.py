#!/usr/bin/env python3
"""
Quick verification of parallel merge outputs.

Checks existing output files for common parallel merge issues:
- Duplicate task_ids
- Missing task_ids
- Metric calculation mismatches

Usage:
    # Check a specific merged file
    python tests/verify_parallel_output.py data/phase4_8/steering_effect_analysis.json

    # Check all merged outputs for a phase
    python tests/verify_parallel_output.py --phase 4.8

    # Check all phases
    python tests/verify_parallel_output.py --all

    # Verbose mode shows individual task_ids
    python tests/verify_parallel_output.py data/phase4_8/*.json -v
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def check_parquet(filepath: Path, verbose: bool = False) -> dict:
    """Check parquet file for merge issues."""
    df = pd.read_parquet(filepath)

    issues = []

    # Check for duplicate task_ids
    if 'task_id' in df.columns:
        duplicates = df[df.duplicated(subset=['task_id'], keep=False)]
        if len(duplicates) > 0:
            dup_ids = duplicates['task_id'].unique().tolist()
            issues.append(f"DUPLICATE task_ids: {len(dup_ids)} duplicated")
            if verbose:
                issues.append(f"  IDs: {dup_ids[:10]}...")

        # Check for None task_ids
        none_count = df['task_id'].isna().sum()
        if none_count > 0:
            issues.append(f"NULL task_ids: {none_count}")

    return {
        'filepath': str(filepath),
        'n_rows': len(df),
        'n_unique_tasks': df['task_id'].nunique() if 'task_id' in df.columns else 'N/A',
        'issues': issues,
        'ok': len(issues) == 0
    }


def check_json(filepath: Path, verbose: bool = False) -> dict:
    """Check JSON file for merge issues."""
    with open(filepath) as f:
        data = json.load(f)

    issues = []

    # Collect all results from various locations
    all_results = []

    # Direct results list
    if 'results' in data and isinstance(data['results'], list):
        all_results.extend(data['results'])

    # Detailed results structure (Phase 4.8, 7.6)
    if 'detailed_results' in data:
        dr = data['detailed_results']
        for key in ['correction', 'corruption', 'preservation']:
            if key in dr and isinstance(dr[key], list):
                all_results.extend(dr[key])

    # Search history structure (Phase 4.5)
    for steering_key in ['correct_steering', 'incorrect_steering']:
        if steering_key in data:
            history = data[steering_key].get('search_history', [])
            for entry in history:
                if 'results' in entry:
                    all_results.extend(entry['results'])

    # Check for duplicates
    task_ids = [r.get('task_id') for r in all_results]
    unique_ids = set(tid for tid in task_ids if tid is not None)

    if len(task_ids) != len(unique_ids):
        n_dups = len(task_ids) - len(unique_ids)
        issues.append(f"DUPLICATE task_ids: {n_dups} duplicated across results")

    # Check for None task_ids
    none_count = sum(1 for tid in task_ids if tid is None)
    if none_count > 0:
        issues.append(f"NULL task_ids: {none_count}")

    # Verify metric calculations if rates are present
    if 'correction_rate' in data and all_results:
        # Recalculate
        corrections = sum(1 for r in all_results
                         if not r.get('baseline_passed', True) and r.get('steered_correct', False))
        incorrect_baseline = sum(1 for r in all_results if not r.get('baseline_passed', True))
        expected_rate = (corrections / incorrect_baseline * 100) if incorrect_baseline > 0 else 0

        reported_rate = data['correction_rate']
        if abs(reported_rate - expected_rate) > 0.1:
            issues.append(f"METRIC MISMATCH: correction_rate reported={reported_rate:.2f}% "
                         f"calculated={expected_rate:.2f}%")

    return {
        'filepath': str(filepath),
        'n_results': len(all_results),
        'n_unique_tasks': len(unique_ids),
        'issues': issues,
        'ok': len(issues) == 0
    }


def check_file(filepath: Path, verbose: bool = False) -> dict:
    """Check a file based on its extension."""
    if filepath.suffix == '.parquet':
        return check_parquet(filepath, verbose)
    elif filepath.suffix == '.json':
        return check_json(filepath, verbose)
    else:
        return {'filepath': str(filepath), 'issues': ['Unknown file type'], 'ok': False}


def find_phase_outputs(phase_id: str, data_dir: Path = None) -> list[Path]:
    """Find all output files for a phase."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    # Normalize phase_id (4.8 -> 4_8)
    phase_dir_pattern = f"phase{phase_id.replace('.', '_')}*"

    files = []
    for phase_dir in data_dir.glob(phase_dir_pattern):
        if phase_dir.is_dir():
            files.extend(phase_dir.glob("*.parquet"))
            files.extend(phase_dir.glob("*.json"))

    return files


def main():
    parser = argparse.ArgumentParser(description="Verify parallel merge outputs")
    parser.add_argument('files', nargs='*', help='Files to check')
    parser.add_argument('--phase', type=str, help='Check all outputs for a specific phase')
    parser.add_argument('--all', action='store_true', help='Check all phase outputs')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--data-dir', type=Path, default=None, help='Data directory')

    args = parser.parse_args()

    files_to_check = []

    if args.files:
        files_to_check = [Path(f) for f in args.files]
    elif args.phase:
        files_to_check = find_phase_outputs(args.phase, args.data_dir)
    elif args.all:
        data_dir = args.data_dir or Path(__file__).parent.parent / "data"
        for phase_dir in data_dir.glob("phase*"):
            if phase_dir.is_dir():
                files_to_check.extend(phase_dir.glob("*.parquet"))
                # Only check main JSON files, not all
                for pattern in ["*_summary.json", "*_analysis.json", "phase_output.json"]:
                    files_to_check.extend(phase_dir.glob(pattern))
    else:
        parser.print_help()
        return 0

    if not files_to_check:
        print("No files found to check")
        return 1

    print(f"Checking {len(files_to_check)} files...\n")

    all_ok = True
    for filepath in sorted(files_to_check):
        if not filepath.exists():
            print(f"SKIP: {filepath} (not found)")
            continue

        result = check_file(filepath, args.verbose)

        status = "OK" if result['ok'] else "FAIL"
        print(f"[{status}] {filepath.name}")

        if result.get('n_rows'):
            print(f"      Rows: {result['n_rows']}, Unique tasks: {result['n_unique_tasks']}")
        elif result.get('n_results'):
            print(f"      Results: {result['n_results']}, Unique tasks: {result['n_unique_tasks']}")

        if result['issues']:
            all_ok = False
            for issue in result['issues']:
                print(f"      {issue}")

        print()

    if all_ok:
        print("All checks passed!")
        return 0
    else:
        print("Some checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
