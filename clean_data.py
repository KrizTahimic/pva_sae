#!/usr/bin/env python3
"""
Clean all generated data files from the project.

Usage:
    python3 clean_data.py           # Interactive mode - asks for confirmation
    python3 clean_data.py --force   # Force mode - deletes without asking
    python3 clean_data.py --dry-run # Dry run - shows what would be deleted
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import remove
from os.path import exists, basename
from shutil import rmtree
from pathlib import Path
from glob import glob


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
        "phase0": [],
        "phase0_1": [],
        "phase1": [],
        "phase2_2": [],
        "phase2_5": [],
        "phase3": [],
        "phase3_5": [],
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
    
    # Phase 0 files
    phase0_patterns = [
        "data/phase0/*.json",
        "data/phase0/*.parquet",
        "data/phase0/mbpp_difficulty_mapping_*.parquet"
    ]
    
    files_to_delete["datasets"] = [f for pattern in dataset_patterns for f in glob(pattern)]
    
    files_to_delete["phase0"] = [f for pattern in phase0_patterns for f in glob(pattern)]
    
    # Phase 0.1 files (Problem splitting)
    phase0_1_patterns = [
        "data/phase0_1/*.json",
        "data/phase0_1/*.parquet"
    ]
    
    files_to_delete["phase0_1"] = [f for pattern in phase0_1_patterns for f in glob(pattern)]
    
    # Phase 1 files (including new structure)
    phase1_patterns = [
        "data/phase1/*.json",
        "data/phase1/*.parquet",
        "data/phase1/dataset_*.parquet",
        "data/phase1/checkpoints/*.json",
        "data/phase1_0/*.json",
        "data/phase1_0/*.parquet",
        "data/phase1_0/dataset_*.parquet",
        "data/phase1_0/checkpoints/*.json",
        "data/phase1_0/activations/**/*.npz"
    ]
    
    files_to_delete["phase1"] = [f for pattern in phase1_patterns for f in glob(pattern)]
    
    # Phase 2.2 files (Pile activation caching)
    phase2_2_patterns = [
        "data/phase2_2/*.json",
        "data/phase2_2/pile_activations/*.npz"
    ]
    
    files_to_delete["phase2_2"] = [f for pattern in phase2_2_patterns for f in glob(pattern)]
    
    # Phase 2.5 files (SAE analysis with pile filtering)
    phase2_5_patterns = [
        "data/phase2_5/*.json",
        "data/phase2_5/checkpoints/*.json",
        "data/phase2_5/activation_cache*"
    ]
    
    files_to_delete["phase2_5"] = [f for pattern in phase2_5_patterns for f in glob(pattern)]
    
    # Phase 3 files
    phase3_patterns = [
        "data/phase3/*.json",
        "data/phase3/*.parquet"
    ]
    
    files_to_delete["phase3"] = [f for pattern in phase3_patterns for f in glob(pattern)]
    
    # Phase 3.5 files (Temperature robustness)
    phase3_5_patterns = [
        "data/phase3_5/*.json",
        "data/phase3_5/*.parquet",
        "data/phase3_5/dataset_*.parquet",
        "data/phase3_5/activations/**/*.npz"
    ]
    
    files_to_delete["phase3_5"] = [f for pattern in phase3_5_patterns for f in glob(pattern)]
    
    # Log files
    log_patterns = [
        "data/logs/*.log",
        "data/logs/mbpp_test_*.log"
    ]
    
    files_to_delete["logs"] = [f for pattern in log_patterns for f in glob(pattern)]
    
    # Checkpoint directory
    checkpoint_dir = "data/datasets/checkpoints"
    if exists(checkpoint_dir):
        files_to_delete["checkpoints"].append(checkpoint_dir)
    
    # Test checkpoint files
    test_checkpoint_patterns = [
        "data/phase2_5/test_checkpoints/*.json",
        "data/phase2_5/test_checkpoints/*.parquet"
    ]
    
    files_to_delete["test_checkpoints"] = [f for pattern in test_checkpoint_patterns for f in glob(pattern)]
    
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
            print(f"  - {basename(f)}")
        if dataset_count > 5:
            print(f"  ... and {dataset_count - 5} more")
        total_count += dataset_count
    
    # Phase 0 files
    phase0_count = len(files_to_delete["phase0"])
    if phase0_count > 0:
        print(f"\nPhase 0 files ({phase0_count} files):")
        for f in sorted(files_to_delete["phase0"])[:5]:
            print(f"  - {basename(f)}")
        if phase0_count > 5:
            print(f"  ... and {phase0_count - 5} more")
        total_count += phase0_count
    
    # Phase 1 files
    phase1_count = len(files_to_delete["phase1"])
    if phase1_count > 0:
        print(f"\nPhase 1 files ({phase1_count} files):")
        for f in sorted(files_to_delete["phase1"])[:5]:
            print(f"  - {basename(f)}")
        if phase1_count > 5:
            print(f"  ... and {phase1_count - 5} more")
        total_count += phase1_count
    
    # Phase 2 files
    phase2_count = len(files_to_delete["phase2"])
    if phase2_count > 0:
        print(f"\nPhase 2 files ({phase2_count} files):")
        for f in sorted(files_to_delete["phase2"])[:5]:
            print(f"  - {basename(f)}")
        if phase2_count > 5:
            print(f"  ... and {phase2_count - 5} more")
        total_count += phase2_count
    
    # Phase 3 files
    phase3_count = len(files_to_delete["phase3"])
    if phase3_count > 0:
        print(f"\nPhase 3 files ({phase3_count} files):")
        for f in sorted(files_to_delete["phase3"])[:5]:
            print(f"  - {basename(f)}")
        if phase3_count > 5:
            print(f"  ... and {phase3_count - 5} more")
        total_count += phase3_count
    
    # Logs
    log_count = len(files_to_delete["logs"])
    if log_count > 0:
        print(f"\nLogs ({log_count} files):")
        for f in sorted(files_to_delete["logs"])[:5]:
            print(f"  - {basename(f)}")
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
            print(f"  - {basename(f)}")
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
                remove(file_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Delete phase 0 files
    for file_path in files_to_delete["phase0"]:
        try:
            if not dry_run:
                remove(file_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Delete phase 1 files
    for file_path in files_to_delete["phase1"]:
        try:
            if not dry_run:
                remove(file_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Delete phase 2 files
    for file_path in files_to_delete["phase2"]:
        try:
            if not dry_run:
                remove(file_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Delete phase 3 files
    for file_path in files_to_delete["phase3"]:
        try:
            if not dry_run:
                remove(file_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Delete log files
    for file_path in files_to_delete["logs"]:
        try:
            if not dry_run:
                remove(file_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Delete checkpoint directories
    for dir_path in files_to_delete["checkpoints"]:
        try:
            if not dry_run:
                rmtree(dir_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'} directory: {dir_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting directory {dir_path}: {e}")
    
    # Delete test checkpoint files
    for file_path in files_to_delete["test_checkpoints"]:
        try:
            if not dry_run:
                remove(file_path)
            print(f"{'[DRY RUN] Would delete' if dry_run else 'Deleted'}: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    return deleted_count


def main():
    """Main entry point"""
    parser = ArgumentParser(
        description="Clean all generated data files",
        formatter_class=ArgumentDefaultsHelpFormatter
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