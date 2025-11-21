"""
Explore HumanEval dataset structure to inform Phase 0.2 design.
"""

from datasets import load_dataset
import pandas as pd

print("="*80)
print("LOADING HUMANEVAL DATASET")
print("="*80)

# Load dataset
dataset = load_dataset("openai_humaneval", split="test")

print(f"\nTotal problems: {len(dataset)}")
print(f"\nDataset features: {list(dataset.features.keys())}")

# Convert to DataFrame for easier viewing
df = pd.DataFrame(dataset)

print("\n" + "="*80)
print("DATASET COLUMNS")
print("="*80)
print(df.columns.tolist())

print("\n" + "="*80)
print("FIRST 3 PROBLEMS (Overview)")
print("="*80)
print(df[['task_id', 'entry_point']].head(3))

print("\n" + "="*80)
print("EXAMPLE PROBLEM (HumanEval/0) - DETAILED")
print("="*80)

problem = dataset[0]
for key in ['task_id', 'prompt', 'entry_point', 'canonical_solution', 'test']:
    print(f"\n--- {key.upper()} ---")
    value = problem[key]
    if len(str(value)) > 500:
        print(str(value)[:500] + "\n... (truncated)")
    else:
        print(value)

print("\n" + "="*80)
print("TEST FORMAT EXAMPLES (First 3 problems)")
print("="*80)

for i in range(min(3, len(dataset))):
    print(f"\n### {dataset[i]['task_id']}: {dataset[i]['entry_point']}")
    test_str = dataset[i]['test']
    # Show first 300 chars of test
    print(test_str[:300] + ("..." if len(test_str) > 300 else ""))
    print()

print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Total problems: {len(df)}")
print(f"Unique task_ids: {df['task_id'].nunique()}")
print(f"\nSample task_ids:")
print(df['task_id'].head(10).tolist())
