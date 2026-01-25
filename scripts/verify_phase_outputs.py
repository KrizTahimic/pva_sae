#!/usr/bin/env python3
"""
Verify phase outputs for completeness and correctness.

Checks:
1. Expected vs actual record counts
2. Duplicate task IDs
3. Missing task IDs (gaps between phases)
4. Cross-phase consistency
"""

import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import Config
from common.phase_discovery import get_phase_output_dir


def load_split_counts(config: Config) -> dict[str, int]:
    """Load expected task counts from Phase 0.1 splits."""
    phase0_1_dir = Path(get_phase_output_dir("0.1", config))

    counts = {}
    for split in ["selection", "tuning", "analysis"]:
        split_file = phase0_1_dir / f"{split}_{config.dataset_name}.parquet"
        if split_file.exists():
            df = pd.read_parquet(split_file)
            counts[split] = len(df)
            counts[f"{split}_task_ids"] = set(df['task_id'].astype(str))

    return counts


def check_phase_output(phase: str, config: Config, expected_task_ids: set = None) -> dict:
    """Check a single phase's output for issues."""
    output_dir = Path(get_phase_output_dir(phase, config))

    result = {
        "phase": phase,
        "dir": str(output_dir),
        "exists": output_dir.exists(),
        "issues": [],
        "warnings": [],
        "record_count": 0,
        "unique_tasks": 0,
        "duplicates": 0,
        "missing": 0,
    }

    if not output_dir.exists():
        result["issues"].append(f"Output directory does not exist: {output_dir}")
        return result

    # Find parquet files
    parquet_files = list(output_dir.glob("dataset_*.parquet")) + list(output_dir.glob("results_*.parquet"))

    # Exclude GPU-specific intermediate files
    parquet_files = [f for f in parquet_files if "_gpu" not in f.stem]

    if not parquet_files:
        result["issues"].append("No dataset/results parquet files found")
        return result

    # Load and combine all parquet files
    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
            result["files"] = result.get("files", []) + [f.name]
        except Exception as e:
            result["issues"].append(f"Failed to load {f.name}: {e}")

    if not dfs:
        return result

    combined = pd.concat(dfs, ignore_index=True)
    result["record_count"] = len(combined)

    if 'task_id' not in combined.columns:
        result["warnings"].append("No 'task_id' column - cannot check for duplicates/gaps")
        return result

    # Check for duplicates
    task_ids = combined['task_id'].astype(str)
    unique_tasks = task_ids.nunique()
    result["unique_tasks"] = unique_tasks

    duplicates = len(combined) - unique_tasks
    if duplicates > 0:
        result["duplicates"] = duplicates
        dup_ids = task_ids[task_ids.duplicated()].unique().tolist()[:5]
        result["issues"].append(f"Found {duplicates} duplicate records. Sample IDs: {dup_ids}")

    # Check for missing tasks (if expected set provided)
    if expected_task_ids:
        actual_task_ids = set(task_ids)
        missing = expected_task_ids - actual_task_ids
        extra = actual_task_ids - expected_task_ids

        result["missing"] = len(missing)
        if missing:
            sample = list(missing)[:5]
            result["issues"].append(f"Missing {len(missing)} expected tasks. Sample: {sample}")

        if extra:
            result["warnings"].append(f"Found {len(extra)} unexpected task IDs")

    return result


def check_steering_phase(phase: str, config: Config, expected_task_ids: set = None) -> dict:
    """Check steering phases that have correction/corruption/preservation experiments."""
    output_dir = Path(get_phase_output_dir(phase, config))

    result = {
        "phase": phase,
        "dir": str(output_dir),
        "exists": output_dir.exists(),
        "issues": [],
        "warnings": [],
        "experiments": {}
    }

    if not output_dir.exists():
        result["issues"].append(f"Output directory does not exist: {output_dir}")
        return result

    # Check for experiment-specific files
    for exp_type in ["correction", "corruption", "preservation"]:
        exp_files = list(output_dir.glob(f"*{exp_type}*.parquet"))
        if exp_files:
            try:
                df = pd.read_parquet(exp_files[0])
                result["experiments"][exp_type] = {
                    "file": exp_files[0].name,
                    "records": len(df),
                    "unique_tasks": df['task_id'].nunique() if 'task_id' in df.columns else "N/A"
                }
            except Exception as e:
                result["issues"].append(f"Failed to load {exp_type}: {e}")

    return result


def main():
    config = Config()
    print(f"Verifying phase outputs for: {config.model_name} / {config.dataset_name}")
    print("=" * 70)

    # Load expected counts from splits
    split_counts = load_split_counts(config)
    print(f"\nExpected task counts from Phase 0.1:")
    for split in ["selection", "tuning", "analysis"]:
        if split in split_counts:
            print(f"  {split}: {split_counts[split]} tasks")
    print()

    # Define phases to check with their expected split
    phases_to_check = [
        ("1", "selection", "Phase 1: Dataset Building"),
        ("3.5", "analysis", "Phase 3.5: Temperature Robustness"),
        ("3.6", "tuning", "Phase 3.6: Hyperparameter Baseline"),
        ("4.5", "selection", "Phase 4.5: Coefficient Grid Search"),
        ("4.6", "selection", "Phase 4.6: Golden Section Refinement"),
        ("4.8", "selection", "Phase 4.8: Steering Effect Analysis"),
        ("7.3", "analysis", "Phase 7.3: Instruct Baseline"),
    ]

    all_issues = []
    all_warnings = []

    for phase, split, desc in phases_to_check:
        expected_ids = split_counts.get(f"{split}_task_ids", set())
        result = check_phase_output(phase, config, expected_ids)

        print(f"\n{desc}")
        print(f"  Directory: {result['dir']}")

        if not result["exists"]:
            print(f"  ❌ NOT FOUND")
            all_issues.extend(result["issues"])
            continue

        print(f"  Records: {result['record_count']}, Unique tasks: {result['unique_tasks']}")
        print(f"  Expected: {split_counts.get(split, 'unknown')} ({split} split)")

        if result["duplicates"]:
            print(f"  ⚠️  Duplicates: {result['duplicates']}")

        if result["missing"]:
            print(f"  ❌ Missing: {result['missing']} tasks")

        if result["issues"]:
            for issue in result["issues"]:
                print(f"  ❌ {issue}")
            all_issues.extend(result["issues"])

        if result["warnings"]:
            for warn in result["warnings"]:
                print(f"  ⚠️  {warn}")
            all_warnings.extend(result["warnings"])

        if not result["issues"] and not result["duplicates"] and not result["missing"]:
            if result["unique_tasks"] == split_counts.get(split, -1):
                print(f"  ✅ COMPLETE")
            elif result["unique_tasks"] > 0:
                print(f"  ⚠️  Count mismatch (may be from --end subset)")

    # Check steering phases
    steering_phases = [
        ("4.8", "Phase 4.8: Steering Effects"),
        ("4.12", "Phase 4.12: Zero-Disc Steering"),
        ("5.3", "Phase 5.3: Layer Sweep"),
        ("5.6", "Phase 5.6: Multi-Latent"),
    ]

    print("\n" + "=" * 70)
    print("STEERING PHASES (experiment-based)")

    for phase, desc in steering_phases:
        result = check_steering_phase(phase, config)
        print(f"\n{desc}")

        if not result["exists"]:
            print(f"  ❌ NOT FOUND")
            continue

        if result["experiments"]:
            for exp_type, exp_data in result["experiments"].items():
                print(f"  {exp_type}: {exp_data['records']} records, {exp_data['unique_tasks']} unique tasks")
        else:
            print(f"  ⚠️  No experiment files found")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_issues:
        print(f"\n❌ {len(all_issues)} ISSUES FOUND:")
        for issue in all_issues[:10]:
            print(f"   - {issue}")
        if len(all_issues) > 10:
            print(f"   ... and {len(all_issues) - 10} more")
        print("\n⚠️  RECOMMENDATION: Rerun affected phases")
    else:
        print("\n✅ No critical issues found")

    if all_warnings:
        print(f"\n⚠️  {len(all_warnings)} warnings (may not be problems)")


if __name__ == "__main__":
    main()
