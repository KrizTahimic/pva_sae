#!/usr/bin/env python3
"""
Re-evaluate Phase 4.8 results to fix spurious timeout classifications.

The Feb 2 run had system-state issues causing legitimate code to timeout.
This script re-evaluates all steered_code using the same test harness
and produces corrected metrics.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc
    python3 scripts/reevaluate_phase4_8.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from common.dataset_utils import evaluate_code_with_error_type


def load_test_cases() -> dict[int, list[str]]:
    """Load test cases from MBPP splits, keyed by task_id."""
    test_cases = {}

    phase0_1_dir = Path("data/phase0_1")
    splits = ["selection_mbpp.parquet", "tuning_mbpp.parquet", "analysis_mbpp.parquet"]

    for split_file in splits:
        split_path = phase0_1_dir / split_file
        if split_path.exists():
            df = pd.read_parquet(split_path)
            for _, row in df.iterrows():
                task_id = int(row["task_id"])
                test_list = row["test_list"]
                # Handle JSON string or list
                if isinstance(test_list, str):
                    test_list = json.loads(test_list)
                test_cases[task_id] = test_list

    print(f"Loaded test cases for {len(test_cases)} tasks")
    return test_cases


def reevaluate_results(
    results: list[dict],
    test_cases: dict[int, list[str]],
    result_type: str
) -> tuple[list[dict], dict]:
    """
    Re-evaluate all results and track changes.

    Returns:
        Tuple of (updated_results, change_stats)
    """
    updated = []
    changes = []

    for r in results:
        task_id = r["task_id"]
        old_error_type = r.get("steered_error_type", "unknown")
        steered_code = r.get("steered_code", "")

        # Get test cases for this task
        tests = test_cases.get(task_id)
        if tests is None:
            print(f"  Warning: No test cases for task {task_id}")
            updated.append(r)
            continue

        # Skip if no code to evaluate
        if not steered_code or not steered_code.strip():
            updated.append(r)
            continue

        # Re-evaluate
        eval_result = evaluate_code_with_error_type(steered_code, tests)
        new_error_type = eval_result.error_type
        new_passed = eval_result.passed

        # Track changes
        if old_error_type != new_error_type:
            changes.append({
                "task_id": task_id,
                "old": old_error_type,
                "new": new_error_type,
                "old_passed": r.get("steered_correct", False),
                "new_passed": new_passed
            })

        # Update result
        updated_r = r.copy()
        updated_r["steered_correct"] = new_passed
        updated_r["steered_error_type"] = new_error_type
        updated_r["flipped"] = r["baseline_passed"] != new_passed
        updated.append(updated_r)

    # Compute change stats
    old_types = Counter(r.get("steered_error_type", "unknown") for r in results if r.get("steered_code", "").strip())
    new_types = Counter(r["steered_error_type"] for r in updated if r.get("steered_code", "").strip())

    stats = {
        "total": len(results),
        "changes": len(changes),
        "old_distribution": dict(old_types),
        "new_distribution": dict(new_types),
        "changes_detail": changes
    }

    return updated, stats


def main():
    print("=" * 70)
    print("Phase 4.8 Re-evaluation")
    print("=" * 70)

    # Load test cases
    test_cases = load_test_cases()

    # Phase 4.8 directory
    phase_dir = Path("data/phase4_8")
    output_dir = Path("data/phase4_8_corrected")
    output_dir.mkdir(exist_ok=True)

    # Load original summary
    with open(phase_dir / "phase_4_8_summary.json") as f:
        original_summary = json.load(f)

    print(f"\nOriginal Phase 4.8 Summary:")
    print(f"  Correction rate: {original_summary['results']['correction_rate']:.2f}%")
    print(f"  Corruption rate: {original_summary['results']['corruption_rate']:.2f}%")
    print(f"  Preservation rate: {original_summary['results']['preservation_rate']:.2f}%")
    print(f"  Total timeouts: {original_summary['steered_error_type_distribution']['counts']['timeout']}")

    # Re-evaluate each result set
    result_files = {
        "preservation": "all_preservation_results.json",
        "correction": "all_correction_results.json",
        "corruption": "all_corruption_results.json"
    }

    all_stats = {}
    all_updated = {}

    for result_type, filename in result_files.items():
        filepath = phase_dir / filename
        if not filepath.exists():
            print(f"\nSkipping {result_type}: file not found")
            continue

        print(f"\n{'=' * 50}")
        print(f"Re-evaluating {result_type} results...")
        print("=" * 50)

        with open(filepath) as f:
            results = json.load(f)

        print(f"  Loaded {len(results)} {result_type} results")

        updated, stats = reevaluate_results(results, test_cases, result_type)
        all_stats[result_type] = stats
        all_updated[result_type] = updated

        # Print change summary
        if stats["changes"] > 0:
            print(f"\n  Changes detected: {stats['changes']}")
            print(f"  Old distribution: {stats['old_distribution']}")
            print(f"  New distribution: {stats['new_distribution']}")

            # Show timeout -> other changes
            timeout_fixes = [c for c in stats["changes_detail"] if c["old"] == "timeout"]
            if timeout_fixes:
                print(f"\n  Timeout → Other transitions ({len(timeout_fixes)}):")
                by_new_type = Counter(c["new"] for c in timeout_fixes)
                for new_type, count in by_new_type.items():
                    print(f"    timeout → {new_type}: {count}")
        else:
            print("  No changes detected")

        # Save updated results
        with open(output_dir / filename, "w") as f:
            json.dump(updated, f, indent=2)

    # Compute corrected metrics
    print("\n" + "=" * 70)
    print("Corrected Metrics")
    print("=" * 70)

    # Preservation: baseline_passed=True, count how many stayed correct
    if "preservation" in all_updated:
        pres = all_updated["preservation"]
        n_total = len(pres)
        n_preserved = sum(1 for r in pres if r["steered_correct"])
        preservation_rate = n_preserved / n_total * 100 if n_total > 0 else 0

        print(f"\nPreservation (baseline correct → steered correct):")
        print(f"  Original: {original_summary['results']['preservation_rate']:.2f}% ({int(original_summary['results']['preservation_rate'] * n_total / 100)}/{n_total})")
        print(f"  Corrected: {preservation_rate:.2f}% ({n_preserved}/{n_total})")

    # Correction: baseline_passed=False, count how many became correct
    if "correction" in all_updated:
        corr = all_updated["correction"]
        n_total = len(corr)
        n_corrected = sum(1 for r in corr if r["steered_correct"])
        correction_rate = n_corrected / n_total * 100 if n_total > 0 else 0

        print(f"\nCorrection (baseline incorrect → steered correct):")
        print(f"  Original: {original_summary['results']['correction_rate']:.2f}%")
        print(f"  Corrected: {correction_rate:.2f}% ({n_corrected}/{n_total})")

    # Corruption: computed from preservation results (not corrected results!)
    if "preservation" in all_updated:
        pres = all_updated["preservation"]
        n_pres_total = len(pres)
        n_corrupted = sum(1 for r in pres if not r["steered_correct"])
        corruption_rate = n_corrupted / n_pres_total * 100 if n_pres_total > 0 else 0

        print(f"\nCorruption (baseline correct → steered incorrect):")
        print(f"  Original: {original_summary['results']['corruption_rate']:.2f}%")
        print(f"  Corrected: {corruption_rate:.2f}% ({n_corrupted}/{n_pres_total})")

    # Combined error type distribution
    all_results = []
    for result_type in ["preservation", "correction"]:
        if result_type in all_updated:
            all_results.extend(all_updated[result_type])

    if all_results:
        new_dist = Counter(r["steered_error_type"] for r in all_results if r.get("steered_code", "").strip())
        old_dist = original_summary["steered_error_type_distribution"]["counts"]

        print(f"\nError Type Distribution (all results):")
        print(f"  {'Type':<10} {'Original':>10} {'Corrected':>10} {'Change':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for et in ["passed", "syntax", "name", "type", "logic", "runtime", "timeout"]:
            old_count = old_dist.get(et, 0)
            new_count = new_dist.get(et, 0)
            change = new_count - old_count
            change_str = f"+{change}" if change > 0 else str(change)
            print(f"  {et:<10} {old_count:>10} {new_count:>10} {change_str:>10}")

    # Save corrected summary
    corrected_summary = {
        "phase": "4.8_corrected",
        "description": "Re-evaluated Phase 4.8 results (fixing spurious timeouts)",
        "timestamp": datetime.now().isoformat(),
        "original_run": original_summary["timestamp"],
        "change_stats": all_stats,
        "results": {
            "preservation_rate": preservation_rate if "preservation" in all_updated else None,
            "correction_rate": correction_rate if "correction" in all_updated else None,
            "corruption_rate": corruption_rate if "preservation" in all_updated else None,
        }
    }

    with open(output_dir / "phase_4_8_corrected_summary.json", "w") as f:
        json.dump(corrected_summary, f, indent=2)

    print(f"\n\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
