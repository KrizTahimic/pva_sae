"""
HumanEval to MBPP format converter.

Converts HumanEval dataset to match MBPP schema for seamless integration
with existing pipeline phases.
"""

import re
from typing import List
from datasets import load_dataset
import pandas as pd
from pathlib import Path


def parse_humaneval_test(test_code: str, entry_point: str) -> List[str]:
    """
    Parse HumanEval test function and extract assertions.

    HumanEval tests use the format:
        def check(candidate):
            assert candidate(...) == expected

    We extract the assertions and replace 'candidate' with the actual
    function name for consistency with MBPP format.

    Args:
        test_code: The test code string containing check(candidate) function
        entry_point: The actual function name to replace 'candidate' with

    Returns:
        List of assertion strings with candidate replaced by function name
    """
    assertions = []

    # Split by lines and process each
    for line in test_code.split('\n'):
        line = line.strip()

        # Look for assert statements
        if line.startswith('assert '):
            # Replace 'candidate' with actual function name
            assertion = line.replace('candidate', entry_point)
            assertions.append(assertion)

    return assertions


def convert_humaneval_to_mbpp(output_dir: str = "data/phase0_2_humaneval") -> pd.DataFrame:
    """
    Convert HumanEval dataset to MBPP format.

    Output schema matches Phase 0.1 validation_mbpp.parquet:
    - task_id: int64 (sequential 0-163)
    - text: object (problem description/prompt)
    - code: object (canonical solution)
    - test_list: object (list of assertion strings)
    - cyclomatic_complexity: int64 (set to 0 as not applicable)

    Args:
        output_dir: Directory to save the converted parquet file

    Returns:
        DataFrame with converted data
    """
    print("=" * 80)
    print("LOADING HUMANEVAL DATASET")
    print("=" * 80)

    # Load HumanEval dataset
    dataset = load_dataset("openai_humaneval", split="test")
    print(f"\nLoaded {len(dataset)} problems from HumanEval")

    print("\n" + "=" * 80)
    print("CONVERTING TO MBPP FORMAT")
    print("=" * 80)

    records = []
    conversion_errors = []

    for idx, problem in enumerate(dataset):
        try:
            # Parse test assertions
            test_list = parse_humaneval_test(
                problem['test'],
                problem['entry_point']
            )

            # Check if we got any assertions
            if not test_list:
                conversion_errors.append({
                    'task_id': idx,
                    'original_task_id': problem['task_id'],
                    'error': 'No assertions found in test code'
                })

            # Create record matching MBPP schema
            record = {
                'task_id': idx,  # Sequential 0-163
                'text': problem['prompt'],
                'code': problem['canonical_solution'],
                'test_list': test_list,
                'cyclomatic_complexity': 0  # Not applicable for HumanEval
            }
            records.append(record)

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} problems...")

        except Exception as e:
            conversion_errors.append({
                'task_id': idx,
                'original_task_id': problem.get('task_id', 'unknown'),
                'error': str(e)
            })
            print(f"\n⚠️  Error processing problem {idx}: {e}")

    print(f"\nProcessed all {len(dataset)} problems")

    # Report conversion errors if any
    if conversion_errors:
        print(f"\n⚠️  {len(conversion_errors)} conversion warnings/errors:")
        for err in conversion_errors[:5]:  # Show first 5
            print(f"  - Task {err['task_id']} ({err['original_task_id']}): {err['error']}")
        if len(conversion_errors) > 5:
            print(f"  ... and {len(conversion_errors) - 5} more")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Ensure correct data types to match MBPP schema
    df['task_id'] = df['task_id'].astype('int64')
    df['cyclomatic_complexity'] = df['cyclomatic_complexity'].astype('int64')

    print("\n" + "=" * 80)
    print("SCHEMA VALIDATION")
    print("=" * 80)
    print("\nDataFrame shape:", df.shape)
    print("\nColumn types:")
    print(df.dtypes)
    print("\nSample test_list lengths:")
    print(df['test_list'].apply(len).describe())

    # Save to parquet
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "humaneval.parquet"

    df.to_parquet(output_file, index=False)
    print(f"\n✓ Saved converted dataset to: {output_file}")
    print(f"  Total records: {len(df)}")

    return df


def inspect_sample_conversions(df: pd.DataFrame, num_samples: int = 3):
    """
    Inspect sample conversions for manual verification.

    Args:
        df: Converted DataFrame
        num_samples: Number of samples to inspect
    """
    print("\n" + "=" * 80)
    print(f"SAMPLE CONVERSIONS (First {num_samples})")
    print("=" * 80)

    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        print(f"\n{'=' * 80}")
        print(f"TASK {i}: {row['task_id']}")
        print('=' * 80)

        print("\n--- PROMPT (first 200 chars) ---")
        print(row['text'][:200] + ("..." if len(row['text']) > 200 else ""))

        print("\n--- CODE (first 300 chars) ---")
        print(row['code'][:300] + ("..." if len(row['code']) > 300 else ""))

        print("\n--- TEST LIST ---")
        print(f"Number of assertions: {len(row['test_list'])}")
        for j, test in enumerate(row['test_list'][:5]):  # Show first 5 tests
            print(f"  {j+1}. {test}")
        if len(row['test_list']) > 5:
            print(f"  ... and {len(row['test_list']) - 5} more assertions")

        print(f"\n--- CYCLOMATIC COMPLEXITY ---")
        print(row['cyclomatic_complexity'])
