# %%
import pandas as pd
import os
from pathlib import Path
import glob

# Auto-discovery of latest Phase 3.5 data
datasets_dir = "../data/phase3_5/"
pattern = os.path.join(datasets_dir, "dataset_temp_*.parquet")
matching_files = glob.glob(pattern)

if matching_files:
    matching_files.sort()
    print(f"🔍 Found {len(matching_files)} temperature files")
    for file in matching_files:
        print(f"  📁 {Path(file).name}")
else:
    raise FileNotFoundError(f"No temperature dataset files found in {datasets_dir}")

# %%
# Load and display first 5 records for each temperature dataset
for file_path in matching_files:
    file_name = Path(file_path).name
    print(f"\n{'='*50}")
    print(f"Dataset: {file_name}")
    print(f"{'='*50}")
    
    df = pd.read_parquet(file_path)
    print(f"Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 records:")
    display(df.head(5))
# %%
