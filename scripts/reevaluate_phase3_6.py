#!/usr/bin/env python3
"""
Re-evaluate Phase 3.6 results to fix spurious timeout classifications.

Phase 3.6 had high timeout rates due to CPU contention during parallel execution:
- Gemma-2B: 23/97 (23.7%) timeouts
- Gemma-9B: 9/97 (9.3%) timeouts
- LLAMA: 10/97 (10.3%) timeouts

This script re-evaluates all generated_code and produces corrected parquet files.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc

    # Re-evaluate Gemma-2B (default directory)
    python3 scripts/reevaluate_phase3_6.py --model gemma2b

    # Re-evaluate Gemma-9B
    python3 scripts/reevaluate_phase3_6.py --model gemma9b

    # Re-evaluate LLAMA
    python3 scripts/reevaluate_phase3_6.py --model llama

    # Re-evaluate all models
    python3 scripts/reevaluate_phase3_6.py --model all

Output:
    - data/phase3_6_corrected/ (for gemma2b)
    - data/phase3_6_{model}_corrected/ (for gemma9b, llama)
    - Includes reevaluation_summary.json with change stats
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from tqdm import tqdm

from common.dataset_utils import evaluate_code_with_error_type


# Model to directory mapping (input -> output)
MODEL_DIRS = {
    'gemma2b': ('data/phase3_6', 'data/phase3_6_corrected'),
    'gemma9b': ('data/phase3_6_gemma9b', 'data/phase3_6_gemma9b_corrected'),
    'llama': ('data/phase3_6_llama', 'data/phase3_6_llama_corrected'),
}


def load_test_cases() -> dict[str, list[str]]:
    """Load test cases from MBPP splits, keyed by task_id."""
    test_cases = {}

    phase0_1_dir = Path("data/phase0_1")
    splits = ["selection_mbpp.parquet", "tuning_mbpp.parquet", "analysis_mbpp.parquet"]

    for split_file in splits:
        split_path = phase0_1_dir / split_file
        if split_path.exists():
            df = pd.read_parquet(split_path)
            for _, row in df.iterrows():
                task_id = str(row["task_id"])
                test_list = row["test_list"]
                if isinstance(test_list, str):
                    test_list = json.loads(test_list)
                elif hasattr(test_list, 'tolist'):
                    test_list = test_list.tolist()
                test_cases[task_id] = list(test_list)

    print(f"Loaded test cases for {len(test_cases)} tasks")
    return test_cases


def reevaluate_file(
    parquet_path: Path,
    test_cases: dict[str, list[str]],
    output_dir: Path
) -> dict:
    """
    Re-evaluate a Phase 3.6 parquet file.

    Returns:
        dict with change statistics
    """
    print(f"\n{'='*60}")
    print(f"Re-evaluating {parquet_path.name}")
    print(f"{'='*60}")

    df = pd.read_parquet(parquet_path)
    n_total = len(df)

    # Track changes
    changes = []
    old_error_types = []
    new_error_types = []

    # Re-evaluate each row
    for idx in tqdm(range(len(df)), desc="Re-evaluating"):
        row = df.iloc[idx]
        task_id = str(row['task_id'])

        # Get old error type (Phase 3.6 uses baseline_error_type)
        old_error = row.get('baseline_error_type') or row.get('error_type', 'unknown')
        if hasattr(old_error, 'item'):
            old_error = old_error.item()
        old_error_types.append(old_error)

        # Get code and tests
        code = row.get('generated_code', '')
        if hasattr(code, 'item'):
            code = str(code)

        tests = test_cases.get(task_id)

        # Skip if no code or tests
        if not code or not str(code).strip() or not tests:
            new_error_types.append(old_error)
            continue

        # Re-evaluate
        result = evaluate_code_with_error_type(str(code), tests)
        new_error = result.error_type
        new_error_types.append(new_error)

        # Update dataframe
        if 'baseline_error_type' in df.columns:
            df.at[idx, 'baseline_error_type'] = new_error
        if 'error_type' in df.columns:
            df.at[idx, 'error_type'] = new_error
        if 'baseline_passed' in df.columns:
            df.at[idx, 'baseline_passed'] = result.passed

        # Track changes
        if old_error != new_error:
            changes.append({
                'task_id': task_id,
                'old': old_error,
                'new': new_error,
                'old_passed': old_error == 'passed',
                'new_passed': result.passed
            })

    # Save corrected parquet with same filename
    output_path = output_dir / parquet_path.name
    df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")

    # Compute statistics
    old_dist = Counter(old_error_types)
    new_dist = Counter(new_error_types)

    old_timeout = old_dist.get('timeout', 0)
    new_timeout = new_dist.get('timeout', 0)
    timeout_resolved = old_timeout - new_timeout

    print(f"\nChanges: {len(changes)}/{n_total}")
    print(f"Timeouts: {old_timeout} → {new_timeout} ({timeout_resolved} resolved)")

    return {
        'file': parquet_path.name,
        'total': n_total,
        'changes': len(changes),
        'old_timeout': old_timeout,
        'new_timeout': new_timeout,
        'timeout_resolved': timeout_resolved,
        'old_distribution': dict(old_dist),
        'new_distribution': dict(new_dist),
        'changes_detail': changes[:20]  # Store first 20 for reference
    }


def reevaluate_model(model: str, test_cases: dict[str, list[str]]) -> dict:
    """
    Re-evaluate all Phase 3.6 files for a specific model.

    Returns:
        Summary statistics dict
    """
    input_dir_str, output_dir_str = MODEL_DIRS[model]
    phase_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    print("\n" + "=" * 70)
    print(f"Processing Phase 3.6 for {model.upper()}")
    print(f"Input:  {phase_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    if not phase_dir.exists():
        print(f"ERROR: Input directory does not exist: {phase_dir}")
        return None

    output_dir.mkdir(exist_ok=True)

    # Find all parquet files (Phase 3.6 uses dataset_merged_*.parquet or dataset_hyperparams_*.parquet)
    parquet_files = sorted(phase_dir.glob("dataset_*.parquet"))
    print(f"\nFound {len(parquet_files)} parquet files to process")

    if not parquet_files:
        print("ERROR: No parquet files found")
        return None

    # Process each file
    all_stats = []
    total_changes = 0
    total_timeout_resolved = 0
    total_records = 0

    for parquet_path in parquet_files:
        stats = reevaluate_file(parquet_path, test_cases, output_dir)
        all_stats.append(stats)
        total_changes += stats['changes']
        total_timeout_resolved += stats['timeout_resolved']
        total_records += stats['total']

    # Copy manifest/metadata if exists
    for manifest_name in ["phase_output.json", "metadata.json"]:
        manifest_file = phase_dir / manifest_name
        if manifest_file.exists():
            shutil.copy(manifest_file, output_dir / manifest_name)
            print(f"Copied: {manifest_name}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY for {model.upper()}")
    print("=" * 70)

    print(f"\n{'File':<50} {'Total':>8} {'Changes':>10} {'Old TO':>8} {'New TO':>8} {'Resolved':>10}")
    print("-" * 100)

    for stats in all_stats:
        filename = stats['file'][:48] if len(stats['file']) > 48 else stats['file']
        print(f"{filename:<50} {stats['total']:>8} {stats['changes']:>10} "
              f"{stats['old_timeout']:>8} {stats['new_timeout']:>8} {stats['timeout_resolved']:>10}")

    print("-" * 100)
    print(f"{'TOTAL':<50} {total_records:>8} {total_changes:>10} "
          f"{sum(s['old_timeout'] for s in all_stats):>8} "
          f"{sum(s['new_timeout'] for s in all_stats):>8} "
          f"{total_timeout_resolved:>10}")

    # Compute overall error distribution change
    print("\n\nError Type Distribution:")
    print(f"{'Type':<12} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 45)

    old_combined = Counter()
    new_combined = Counter()
    for stats in all_stats:
        old_combined.update(stats['old_distribution'])
        new_combined.update(stats['new_distribution'])

    for et in ['passed', 'syntax', 'name', 'type', 'logic', 'runtime', 'timeout']:
        old_count = old_combined.get(et, 0)
        new_count = new_combined.get(et, 0)
        change = new_count - old_count
        change_str = f"+{change}" if change > 0 else str(change)
        print(f"{et:<12} {old_count:>10} {new_count:>10} {change_str:>10}")

    # Save summary
    summary = {
        'phase': output_dir_str.replace('data/', ''),
        'model': model,
        'description': f'Re-evaluated Phase 3.6 {model.upper()} results (fixing spurious timeouts)',
        'timestamp': datetime.now().isoformat(),
        'total_records': total_records,
        'total_changes': total_changes,
        'total_timeout_resolved': total_timeout_resolved,
        'per_file': all_stats,
        'old_distribution': dict(old_combined),
        'new_distribution': dict(new_combined)
    }

    summary_path = output_dir / "reevaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nResults saved to: {output_dir}")
    print(f"Summary: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate Phase 3.6 results for all models"
    )
    parser.add_argument(
        '--model',
        choices=['gemma2b', 'gemma9b', 'llama', 'all'],
        required=True,
        help="Model to re-evaluate (gemma2b, gemma9b, llama, or all)"
    )
    args = parser.parse_args()

    # Load test cases
    test_cases = load_test_cases()

    # Determine which models to process
    if args.model == 'all':
        models = ['gemma2b', 'gemma9b', 'llama']
    else:
        models = [args.model]

    # Process each model
    summaries = {}
    for model in models:
        summary = reevaluate_model(model, test_cases)
        if summary:
            summaries[model] = summary

    # Final summary
    if len(summaries) > 1:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY (ALL MODELS)")
        print("=" * 70)

        for model, summary in summaries.items():
            print(f"\n{model.upper()}: {summary['total_changes']} changes, "
                  f"{summary['total_timeout_resolved']} timeouts resolved")

    print("\nDone!")


if __name__ == "__main__":
    main()
