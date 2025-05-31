#!/usr/bin/env python3
"""
Merge multiple dataset files from parallel GPU processing.

Usage:
    python3 merge_datasets.py --output merged_dataset.parquet
    python3 merge_datasets.py --pattern "mbpp_dataset_*.parquet" --output final_dataset.parquet
"""

import argparse
import pandas as pd
import glob
import os
from pathlib import Path
import json
from datetime import datetime

def merge_parquet_files(pattern: str, output_file: str, dataset_dir: str = "data/datasets"):
    """
    Merge multiple parquet files into one
    
    Args:
        pattern: Glob pattern to match files
        output_file: Output filename
        dataset_dir: Directory containing datasets
    """
    # Find matching files
    search_pattern = os.path.join(dataset_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"No files found matching pattern: {search_pattern}")
        return None
    
    print(f"Found {len(files)} files to merge:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # Load and concatenate
    dfs = []
    total_records = 0
    
    for file in files:
        df = pd.read_parquet(file)
        total_records += len(df)
        dfs.append(df)
        print(f"Loaded {len(df)} records from {os.path.basename(file)}")
    
    # Merge
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by task_id to maintain order
    if 'task_id' in merged_df.columns:
        # Extract numeric part from task_id for proper sorting
        merged_df['task_num'] = merged_df['task_id'].str.extract('(\d+)').astype(int)
        merged_df = merged_df.sort_values('task_num').drop('task_num', axis=1)
    
    # Save merged dataset
    output_path = os.path.join(dataset_dir, output_file)
    merged_df.to_parquet(output_path, index=False)
    
    # Calculate statistics
    stats = {
        'total_records': len(merged_df),
        'correct_solutions': merged_df['is_correct'].sum() if 'is_correct' in merged_df else 0,
        'files_merged': len(files),
        'merge_timestamp': datetime.now().isoformat()
    }
    
    if 'is_correct' in merged_df:
        stats['success_rate'] = (stats['correct_solutions'] / stats['total_records'] * 100)
    
    # Save metadata
    metadata_file = output_path.replace('.parquet', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'merged_files': [os.path.basename(f) for f in files],
            'statistics': stats,
            'output_file': output_file
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Total records: {stats['total_records']}")
    if 'success_rate' in stats:
        print(f"Correct solutions: {stats['correct_solutions']} ({stats['success_rate']:.1f}%)")
    print(f"Output file: {output_path}")
    print(f"Metadata: {metadata_file}")
    print(f"{'='*60}")
    
    return output_path


def cleanup_partial_datasets(dataset_dir: str = "data/datasets", 
                           pattern: str = "mbpp_dataset_*.parquet",
                           keep_merged: bool = True):
    """
    Clean up partial dataset files after merging
    
    Args:
        dataset_dir: Directory containing datasets
        pattern: Pattern of files to clean
        keep_merged: Keep the merged file
    """
    search_pattern = os.path.join(dataset_dir, pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        print("No files to clean up")
        return
    
    print(f"\nCleaning up {len(files)} partial dataset files...")
    
    for file in files:
        # Skip merged files
        if keep_merged and 'merged' in file:
            continue
            
        os.remove(file)
        
        # Also remove metadata
        metadata = file.replace('.parquet', '_metadata.json')
        if os.path.exists(metadata):
            os.remove(metadata)
    
    print("Cleanup complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge dataset files from parallel GPU processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='mbpp_dataset_*.parquet',
        help='Glob pattern to match dataset files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='merged_dataset.parquet',
        help='Output filename for merged dataset'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='data/datasets',
        help='Directory containing dataset files'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove partial datasets after merging'
    )
    
    args = parser.parse_args()
    
    # Merge files
    output_path = merge_parquet_files(
        pattern=args.pattern,
        output_file=args.output,
        dataset_dir=args.dataset_dir
    )
    
    # Cleanup if requested
    if args.cleanup and output_path:
        cleanup_partial_datasets(
            dataset_dir=args.dataset_dir,
            pattern=args.pattern,
            keep_merged=True
        )


if __name__ == "__main__":
    main()