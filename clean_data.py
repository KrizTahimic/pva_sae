#!/usr/bin/env python3
"""
Clean all generated data files from the project.

Usage:
    python3 clean_data.py           # Interactive mode - asks for confirmation
    python3 clean_data.py --force   # Force mode - deletes without asking
    python3 clean_data.py --dry-run # Dry run - shows what would be deleted
"""

import argparse
import os
import shutil
from pathlib import Path
import glob


def get_files_to_delete(data_dir: str = "data") -> dict:
    """
    Get all files and directories that would be deleted
    
    Args:
        data_dir: Root data directory
        
    Returns:
        dict: Categorized files and directories to delete
    """
    files_to_delete = {
        "datasets": [],
        "logs": [],
        "checkpoints": [],
        "test_checkpoints": [],
        "other": []
    }
    
    # Dataset files
    dataset_patterns = [
        "data/datasets/*.json",
        "data/datasets/*.parquet",
        "data/datasets/difficulty_mapping_*.json",
        "data/datasets/checkpoint_*.json",
        "data/datasets/autosave_*.parquet"
    ]
    
    for pattern in dataset_patterns:
        files_to_delete["datasets"].extend(glob.glob(pattern))
    
    # Log files
    log_patterns = [
        "data/logs/*.log",
        "data/logs/mbpp_test_*.log"
    ]
    
    for pattern in log_patterns:
        files_to_delete["logs"].extend(glob.glob(pattern))
    
    # Checkpoint directory
    checkpoint_dir = "data/datasets/checkpoints"
    if os.path.exists(checkpoint_dir):
        files_to_delete["checkpoints"].append(checkpoint_dir)
    
    # Test checkpoint files
    test_checkpoint_patterns = [
        "data/test_checkpoints/*.json",
        "data/test_checkpoints/*.parquet"
    ]
    
    for pattern in test_checkpoint_patterns:
        files_to_delete["test_checkpoints"].extend(glob.glob(pattern))
    
    return files_to_delete


def print_files_summary(files_to_delete: dict) -> int:
    """
    Print summary of files to be deleted
    
    Args:
        files_to_delete: Categorized files to delete
        
    Returns:
        int: Total number of items to delete
    """
    total_count = 0
    
    print("\n" + "="*60)
    print("FILES TO BE DELETED")
    print("="*60)
    
    # Datasets
    dataset_count = len(files_to_delete["datasets"])
    if dataset_count > 0:
        print(f"\nDatasets ({dataset_count} files):")
        for f in sorted(files_to_delete["datasets"])[:5]:
            print(f"  - {os.path.basename(f)}")
        if dataset_count > 5:
            print(f"  ... and {dataset_count - 5} more")
        total_count += dataset_count
    
    # Logs
    log_count = len(files_to_delete["logs"])
    if log_count > 0:
        print(f"\nLogs ({log_count} files):")
        for f in sorted(files_to_delete["logs"])[:5]:
            print(f"  - {os.path.basename(f)}")
        if log_count > 5:
            print(f"  ... and {log_count - 5} more")
        total_count += log_count
    
    # Checkpoints
    if files_to_delete["checkpoints"]:
        print(f"\nCheckpoint directories:")
        for d in files_to_delete["checkpoints"]:
            print(f"  - {d}/")
        total_count += len(files_to_delete["checkpoints"])
    
    # Test checkpoints
    test_checkpoint_count = len(files_to_delete["test_checkpoints"])
    if test_checkpoint_count > 0:
        print(f"\nTest checkpoint files ({test_checkpoint_count} files):")
        for f in sorted(files_to_delete["test_checkpoints"])[:5]:
            print(f"  - {os.path.basename(f)}")
        if test_checkpoint_count > 5:
            print(f"  ... and {test_checkpoint_count - 5} more")
        total_count += test_checkpoint_count
    
    if total_count == 0:
        print("\nNo files found to delete. Data directory is already clean!")
    
    print("="*60)
    return total_count


def delete_files(files_to_delete: dict, dry_run: bool = False) -> int:
    """
    Delete the specified files and directories
    
    Args:
        files_to_delete: Categorized files to delete
        dry_run: If True, don't actually delete anything
        
    Returns:
        int: Number of items deleted
    """
    deleted_count = 0
    
    # Delete dataset files
    for file_path in files_to_delete["datasets"]:
        try:
            if not dry_run:
                os.remove(file_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Delete log files
    for file_path in files_to_delete["logs"]:
        try:
            if not dry_run:
                os.remove(file_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Delete checkpoint directories
    for dir_path in files_to_delete["checkpoints"]:
        try:
            if not dry_run:
                shutil.rmtree(dir_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'} directory: {dir_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting directory {dir_path}: {e}")
    
    # Delete test checkpoint files
    for file_path in files_to_delete["test_checkpoints"]:
        try:
            if not dry_run:
                os.remove(file_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    return deleted_count


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Clean all generated data files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Delete files without confirmation prompt'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory to clean'
    )
    
    args = parser.parse_args()
    
    # Get files to delete
    files_to_delete = get_files_to_delete(args.data_dir)
    
    # Print summary
    total_count = print_files_summary(files_to_delete)
    
    if total_count == 0:
        return 0
    
    # Handle dry run
    if args.dry_run:
        print(f"\n[DRY RUN] Would delete {total_count} items total")
        return 0
    
    # Confirm deletion
    if not args.force:
        print(f"\n⚠️  This will permanently delete {total_count} files/directories!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return 0
    
    # Delete files
    print("\nDeleting files...")
    deleted_count = delete_files(files_to_delete)
    
    print(f"\n✅ Cleanup complete! Deleted {deleted_count} items.")
    return 0


if __name__ == "__main__":
    exit(main())