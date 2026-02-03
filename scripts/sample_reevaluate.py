#!/usr/bin/env python3
"""
Sample and re-evaluate timeout cases from any phase to verify if spurious.

This script helps diagnose whether timeout issues are due to:
1. CPU contention (spurious) - timeouts resolve on re-evaluation
2. Legitimate complexity - code actually runs long

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc
    python3 scripts/sample_reevaluate.py --phase 3.5 --sample 20
    python3 scripts/sample_reevaluate.py --phase 4.8 --sample 50

The script will:
1. Load results from the specified phase
2. Filter to timeout cases only
3. Sample N cases randomly
4. Re-evaluate each with fresh CPU (no contention)
5. Report how many timeouts were spurious vs legitimate
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from common.dataset_utils import evaluate_code_with_error_type
from common.phase_discovery import discover_latest_phase_output


def load_test_cases(dataset_name: str = "mbpp") -> dict[str, list[str]]:
    """Load test cases from dataset splits, keyed by task_id."""
    test_cases = {}

    phase0_1_dir = Path("data/phase0_1")
    splits = [
        f"selection_{dataset_name}.parquet",
        f"tuning_{dataset_name}.parquet",
        f"analysis_{dataset_name}.parquet"
    ]

    for split_file in splits:
        split_path = phase0_1_dir / split_file
        if split_path.exists():
            df = pd.read_parquet(split_path)
            for _, row in df.iterrows():
                task_id = str(row["task_id"])
                test_list = row["test_list"]
                if isinstance(test_list, str):
                    test_list = json.loads(test_list)
                test_cases[task_id] = test_list

    print(f"Loaded test cases for {len(test_cases)} tasks from {dataset_name}")
    return test_cases


def find_timeout_cases(phase_id: str) -> list[dict]:
    """Find all timeout cases from a phase's results."""
    timeout_cases = []

    # Try to find the phase output
    try:
        phase_output = discover_latest_phase_output(phase_id)
        phase_dir = Path(phase_output).parent
    except Exception as e:
        print(f"Could not find Phase {phase_id} output: {e}")
        return []

    print(f"Loading results from: {phase_dir}")

    # Phase 3.5: Temperature robustness results
    if phase_id == "3.5":
        results_file = phase_dir / "temperature_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            for temp_result in data.get("temperature_results", []):
                for r in temp_result.get("results", []):
                    if r.get("error_type") == "timeout" or r.get("baseline_error_type") == "timeout":
                        r["temperature"] = temp_result.get("temperature")
                        r["source_phase"] = "3.5"
                        timeout_cases.append(r)

    # Phase 4.8: Steering effect analysis
    elif phase_id == "4.8":
        for result_type in ["correction", "preservation", "corruption"]:
            filename = f"all_{result_type}_results.json"
            filepath = phase_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    results = json.load(f)
                for r in results:
                    if r.get("steered_error_type") == "timeout":
                        r["result_type"] = result_type
                        r["source_phase"] = "4.8"
                        timeout_cases.append(r)

    # Phase 4.5: Coefficient grid search
    elif phase_id == "4.5":
        analysis_file = phase_dir / "coefficient_analysis.json"
        if analysis_file.exists():
            with open(analysis_file) as f:
                data = json.load(f)
            for steering_key in ["correct_steering", "incorrect_steering"]:
                if steering_key in data:
                    for hist in data[steering_key].get("search_history", []):
                        for r in hist.get("results", []):
                            if r.get("steered_error_type") == "timeout":
                                r["coefficient"] = hist.get("coefficient")
                                r["steering_type"] = steering_key
                                r["source_phase"] = "4.5"
                                timeout_cases.append(r)

    # Phase 4.12: Zero-disc steering
    elif phase_id == "4.12":
        results_file = phase_dir / "zero_disc_steering_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            for result_type in ["correction_results", "preservation_results", "corruption_results"]:
                for task_id, r in data.get(result_type, {}).items():
                    if r.get("steered_error_type") == "timeout":
                        r["task_id"] = task_id
                        r["result_type"] = result_type
                        r["source_phase"] = "4.12"
                        timeout_cases.append(r)

    # Phase 1: Baseline generation
    elif phase_id == "1":
        parquet_files = list(phase_dir.glob("dataset_*.parquet"))
        if parquet_files:
            df = pd.read_parquet(parquet_files[0])
            for _, row in df.iterrows():
                if row.get("baseline_error_type") == "timeout":
                    timeout_cases.append({
                        "task_id": str(row["task_id"]),
                        "generated_code": row.get("generated_code", ""),
                        "baseline_error_type": row.get("baseline_error_type"),
                        "source_phase": "1"
                    })

    else:
        print(f"Phase {phase_id} not yet supported by this script")
        return []

    return timeout_cases


def get_code_for_case(case: dict) -> str:
    """Extract the code to evaluate from a case."""
    # Try various field names used by different phases
    for field in ["steered_code", "generated_code", "code"]:
        if field in case and case[field]:
            return case[field]
    return ""


def sample_and_reevaluate(
    timeout_cases: list[dict],
    test_cases: dict[str, list[str]],
    sample_size: int
) -> dict:
    """Sample timeout cases and re-evaluate them."""

    if len(timeout_cases) < sample_size:
        sample = timeout_cases
        print(f"Using all {len(sample)} timeout cases (fewer than requested {sample_size})")
    else:
        sample = random.sample(timeout_cases, sample_size)
        print(f"Sampled {sample_size} timeout cases from {len(timeout_cases)} total")

    results = []
    still_timeout = 0
    resolved = 0

    print(f"\nRe-evaluating {len(sample)} cases...")
    print("-" * 60)

    for i, case in enumerate(sample, 1):
        task_id = str(case.get("task_id", "unknown"))
        code = get_code_for_case(case)

        if not code or not code.strip():
            print(f"  [{i}/{len(sample)}] Task {task_id}: No code to evaluate")
            continue

        tests = test_cases.get(task_id)
        if tests is None:
            print(f"  [{i}/{len(sample)}] Task {task_id}: No test cases found")
            continue

        # Re-evaluate (main process = signal-based timeout, no contention)
        eval_result = evaluate_code_with_error_type(code, tests)
        new_error_type = eval_result.error_type

        if new_error_type == "timeout":
            still_timeout += 1
            status = "STILL TIMEOUT"
        else:
            resolved += 1
            status = f"RESOLVED → {new_error_type}"

        print(f"  [{i}/{len(sample)}] Task {task_id}: {status}")

        results.append({
            "task_id": task_id,
            "original_error": "timeout",
            "new_error": new_error_type,
            "resolved": new_error_type != "timeout",
            "source_phase": case.get("source_phase", "unknown"),
            "extra_info": {k: v for k, v in case.items()
                         if k in ["temperature", "coefficient", "result_type", "steering_type"]}
        })

    print("-" * 60)

    # Compute summary statistics
    total_evaluated = len(results)
    spurious_rate = (resolved / total_evaluated * 100) if total_evaluated > 0 else 0

    # Group resolved by new error type
    new_types = Counter(r["new_error"] for r in results if r["resolved"])

    return {
        "total_timeout_cases": len(timeout_cases),
        "sample_size": len(sample),
        "evaluated": total_evaluated,
        "still_timeout": still_timeout,
        "resolved": resolved,
        "spurious_rate": spurious_rate,
        "resolved_to": dict(new_types),
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sample and re-evaluate timeout cases from a phase"
    )
    parser.add_argument(
        "--phase", "-p",
        required=True,
        help="Phase ID to check (e.g., 3.5, 4.8, 4.5, 4.12, 1)"
    )
    parser.add_argument(
        "--sample", "-n",
        type=int,
        default=20,
        help="Number of timeout cases to sample (default: 20)"
    )
    parser.add_argument(
        "--dataset",
        default="mbpp",
        choices=["mbpp", "humaneval"],
        help="Dataset name for test cases (default: mbpp)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 70)
    print(f"Timeout Verification for Phase {args.phase}")
    print("=" * 70)

    # Load test cases
    test_cases = load_test_cases(args.dataset)

    # Find timeout cases
    print(f"\nSearching for timeout cases in Phase {args.phase}...")
    timeout_cases = find_timeout_cases(args.phase)

    if not timeout_cases:
        print(f"No timeout cases found in Phase {args.phase}")
        return

    print(f"Found {len(timeout_cases)} timeout cases")

    # Sample and re-evaluate
    summary = sample_and_reevaluate(timeout_cases, test_cases, args.sample)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total timeout cases in phase: {summary['total_timeout_cases']}")
    print(f"Sample size: {summary['sample_size']}")
    print(f"Successfully evaluated: {summary['evaluated']}")
    print()
    print(f"Still timeout (legitimate): {summary['still_timeout']} ({100 - summary['spurious_rate']:.1f}%)")
    print(f"Resolved (spurious): {summary['resolved']} ({summary['spurious_rate']:.1f}%)")

    if summary['resolved_to']:
        print(f"\nResolved timeouts became:")
        for error_type, count in sorted(summary['resolved_to'].items()):
            print(f"  {error_type}: {count}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if summary['spurious_rate'] > 50:
        print("⚠️  HIGH SPURIOUS RATE - These timeouts are likely due to CPU contention")
        print("   Recommendation: Run full re-evaluation for this phase")
        print(f"   Expected improvement: ~{summary['spurious_rate']:.0f}% of timeouts should resolve")
    elif summary['spurious_rate'] > 20:
        print("⚡ MODERATE SPURIOUS RATE - Some timeouts are due to CPU contention")
        print("   Recommendation: Consider re-evaluation if timeout rate is significant")
    else:
        print("✓  LOW SPURIOUS RATE - Most timeouts appear legitimate (complex code)")
        print("   Recommendation: No re-evaluation needed for this phase")


if __name__ == "__main__":
    main()
