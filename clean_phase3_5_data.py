#!/usr/bin/env python3
"""Clean up old Phase 3.5 data files to force regeneration with correct column names."""

import os
from pathlib import Path
import shutil

def clean_phase3_5():
    """Remove old Phase 3.5 data to force regeneration."""
    phase3_5_dir = Path("data/phase3_5")
    
    if not phase3_5_dir.exists():
        print("No Phase 3.5 directory found.")
        return
    
    # Find all parquet files
    parquet_files = list(phase3_5_dir.glob("dataset_temp_*.parquet"))
    
    if not parquet_files:
        print("No Phase 3.5 dataset files found.")
        return
    
    print(f"Found {len(parquet_files)} Phase 3.5 dataset files:")
    for f in parquet_files:
        print(f"  - {f.name}")
    
    response = input("\nDo you want to delete these files to force regeneration with correct column names? (y/N): ")
    
    if response.lower() == 'y':
        for f in parquet_files:
            os.remove(f)
            print(f"✅ Deleted {f.name}")
        print("\n✨ Old Phase 3.5 data cleaned. Run Phase 3.5 again to generate with correct column names.")
    else:
        print("❌ Cleanup cancelled.")

if __name__ == "__main__":
    clean_phase3_5()