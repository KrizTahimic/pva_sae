#!/usr/bin/env python3
"""
Re-evaluate Phase 3.5 results for Gemma-9B and LLAMA models.

Phase 3.5 had high timeout rates due to CPU contention during parallel execution:
- Gemma-9B: 422/3104 (13.6%) timeouts
- LLAMA: 388/3104 (12.5%) timeouts

This script re-evaluates all generated_code and produces corrected parquet files.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc

    # Re-evaluate Gemma-9B
    python3 scripts/reevaluate_phase3_5_multimodel.py --model gemma9b

    # Re-evaluate LLAMA
    python3 scripts/reevaluate_phase3_5_multimodel.py --model llama

    # Re-evaluate all models
    python3 scripts/reevaluate_phase3_5_multimodel.py --model all

Output:
    - data/phase3_5_{model}_corrected/dataset_temp_*.parquet (corrected files)
    - data/phase3_5_{model}_corrected/reevaluation_summary.json (change stats)
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


# Model to directory suffix mapping
MODEL_SUFFIXES = {
    'gemma9b': 'gemma9b',
    'llama': 'llama',
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


def reevaluate_temperature_file(
    parquet_path: Path,
    test_cases: dict[str, list[str]],
    output_dir: Path
) -> dict:
    """
    Re-evaluate a single temperature parquet file.

    Returns:
        dict with change statistics
    """
    temp_name = parquet_path.stem  # e.g., "dataset_temp_0_8"
    temp_value = temp_name.replace('dataset_temp_', '').replace('_', '.')

    print(f"\n{'='*60}")
    print(f"Re-evaluating {temp_name} (temperature={temp_value})")
    print(f"{'='*60}")

    df = pd.read_parquet(parquet_path)
    n_total = len(df)

    # Track changes
    changes = []
    old_error_types = []
    new_error_types = []

    # Re-evaluate each row
    for idx in tqdm(range(len(df)), desc=f"Temp {temp_value}"):
        row = df.iloc[idx]
        task_id = str(row['task_id'])

        # Get old error type
        old_error = row.get('error_type') or row.get('baseline_error_type', 'unknown')
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
        if 'error_type' in df.columns:
            df.at[idx, 'error_type'] = new_error
        if 'baseline_error_type' in df.columns:
            df.at[idx, 'baseline_error_type'] = new_error
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

    # Save corrected parquet
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
        'temperature': temp_value,
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
    Re-evaluate all Phase 3.5 files for a specific model.

    Returns:
        Summary statistics dict
    """
    suffix = MODEL_SUFFIXES[model]
    phase_dir = Path(f"data/phase3_5_{suffix}")
    output_dir = Path(f"data/phase3_5_{suffix}_corrected")

    print("\n" + "=" * 70)
    print(f"Processing Phase 3.5 for {model.upper()}")
    print(f"Input:  {phase_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    if not phase_dir.exists():
        print(f"ERROR: Input directory does not exist: {phase_dir}")
        return None

    output_dir.mkdir(exist_ok=True)

    # Find all temperature parquet files
    parquet_files = sorted(phase_dir.glob("dataset_temp_*.parquet"))
    print(f"\nFound {len(parquet_files)} temperature files to process")

    if not parquet_files:
        print("ERROR: No temperature parquet files found")
        return None

    # Process each file
    all_stats = []
    total_changes = 0
    total_timeout_resolved = 0
    total_records = 0

    for parquet_path in parquet_files:
        stats = reevaluate_temperature_file(parquet_path, test_cases, output_dir)
        all_stats.append(stats)
        total_changes += stats['changes']
        total_timeout_resolved += stats['timeout_resolved']
        total_records += stats['total']

    # Copy metadata if exists
    metadata_file = phase_dir / "metadata.json"
    if metadata_file.exists():
        shutil.copy(metadata_file, output_dir / "metadata.json")

    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY for {model.upper()}")
    print("=" * 70)

    print(f"\n{'Temperature':<12} {'Total':>8} {'Changes':>10} {'Old TO':>10} {'New TO':>10} {'Resolved':>10}")
    print("-" * 70)

    for stats in all_stats:
        print(f"{stats['temperature']:<12} {stats['total']:>8} {stats['changes']:>10} "
              f"{stats['old_timeout']:>10} {stats['new_timeout']:>10} {stats['timeout_resolved']:>10}")

    print("-" * 70)
    print(f"{'TOTAL':<12} {total_records:>8} {total_changes:>10} "
          f"{sum(s['old_timeout'] for s in all_stats):>10} "
          f"{sum(s['new_timeout'] for s in all_stats):>10} "
          f"{total_timeout_resolved:>10}")

    # Compute overall error distribution change
    print("\n\nError Type Distribution (all temperatures combined):")
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
        'phase': f'phase3_5_{suffix}_corrected',
        'model': model,
        'description': f'Re-evaluated Phase 3.5 {model.upper()} results (fixing spurious timeouts)',
        'timestamp': datetime.now().isoformat(),
        'total_records': total_records,
        'total_changes': total_changes,
        'total_timeout_resolved': total_timeout_resolved,
        'per_temperature': all_stats,
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
        description="Re-evaluate Phase 3.5 results for Gemma-9B and LLAMA models"
    )
    parser.add_argument(
        '--model',
        choices=['gemma9b', 'llama', 'all'],
        required=True,
        help="Model to re-evaluate (gemma9b, llama, or all)"
    )
    args = parser.parse_args()

    # Load test cases
    test_cases = load_test_cases()

    # Determine which models to process
    if args.model == 'all':
        models = ['gemma9b', 'llama']
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
