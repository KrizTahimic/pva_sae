#!/usr/bin/env python3
"""
Upload data/ directory to HuggingFace Hub.

Usage:
    # Preview what will upload
    python3 scripts/upload_to_hf.py --dry-run

    # Full upload
    python3 scripts/upload_to_hf.py

    # Custom repo
    python3 scripts/upload_to_hf.py --repo-id your-username/your-dataset
"""

import argparse
from pathlib import Path


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def format_size(size_mb: float) -> str:
    """Format size for display."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"
    return f"{size_mb:.2f} MB"


def collect_files(data_dir: Path) -> list[Path]:
    """Collect all files to upload, excluding temporary files."""
    files = list(data_dir.rglob("*"))
    files = [f for f in files if f.is_file()]

    # Exclude patterns
    exclude_patterns = [
        "__pycache__",
        ".pyc",
        "checkpoint_",
        "autosave",
        ".DS_Store",
        "*.log",
    ]

    filtered = []
    for f in files:
        path_str = str(f)
        if not any(pattern in path_str for pattern in exclude_patterns):
            filtered.append(f)

    return filtered


def print_summary(files: list[Path], data_dir: Path) -> None:
    """Print summary of files to upload."""
    # Calculate sizes
    total_size = sum(get_file_size_mb(f) for f in files)

    # Count by extension
    by_ext: dict[str, tuple[int, float]] = {}
    for f in files:
        ext = f.suffix.lower() or "(no ext)"
        count, size = by_ext.get(ext, (0, 0.0))
        by_ext[ext] = (count + 1, size + get_file_size_mb(f))

    # Count by phase
    by_phase: dict[str, tuple[int, float]] = {}
    for f in files:
        rel_path = f.relative_to(data_dir)
        phase = rel_path.parts[0] if rel_path.parts else "root"
        count, size = by_phase.get(phase, (0, 0.0))
        by_phase[phase] = (count + 1, size + get_file_size_mb(f))

    print(f"\n{'=' * 60}")
    print(f"UPLOAD SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total files: {len(files):,}")
    print(f"Total size:  {format_size(total_size)}")

    print(f"\nBy file type:")
    for ext, (count, size) in sorted(by_ext.items(), key=lambda x: -x[1][1]):
        print(f"  {ext:12} {count:>8,} files  {format_size(size):>10}")

    print(f"\nBy phase directory:")
    for phase, (count, size) in sorted(by_phase.items(), key=lambda x: -x[1][1])[:10]:
        print(f"  {phase:25} {count:>8,} files  {format_size(size):>10}")
    if len(by_phase) > 10:
        print(f"  ... and {len(by_phase) - 10} more directories")

    print(f"{'=' * 60}\n")


def upload_to_hf(repo_id: str, data_dir: Path, dry_run: bool = False) -> None:
    """Upload data directory to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()

    # Collect files
    print(f"Scanning {data_dir}...")
    files = collect_files(data_dir)

    if not files:
        print("No files found to upload!")
        return

    print_summary(files, data_dir)

    if dry_run:
        print("DRY RUN - No files will be uploaded")
        print("\nSample files that would be uploaded:")
        for f in files[:20]:
            size = format_size(get_file_size_mb(f))
            print(f"  {f.relative_to(data_dir)} ({size})")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20:,} more files")
        return

    # Check if repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Repository {repo_id} exists, uploading...")
    except Exception:
        print(f"Repository {repo_id} not found. Creating...")
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Created repository: https://huggingface.co/datasets/{repo_id}")

    # Upload using upload_large_folder for >25k files
    print(f"\nUploading to {repo_id}...")
    print("Using upload_large_folder for large number of files...")
    print("This may take 30-60 minutes...")

    api.upload_large_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=[
            "__pycache__",
            "*.pyc",
            "checkpoint_*",
            "autosave*",
            ".DS_Store",
            "*.log",
        ],
    )

    print(f"\nUpload complete!")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload data/ directory to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/upload_to_hf.py --dry-run     # Preview upload
  python3 scripts/upload_to_hf.py               # Full upload
  python3 scripts/upload_to_hf.py --repo-id myuser/mydata  # Custom repo
        """,
    )
    parser.add_argument(
        "--repo-id",
        default="kriztahimic/sae-code-correctness-data",
        help="HuggingFace dataset repository ID (default: kriztahimic/sae-code-correctness-data)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Data directory to upload (default: data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    # Resolve data dir relative to script location
    if not args.data_dir.is_absolute():
        script_dir = Path(__file__).parent.parent
        args.data_dir = script_dir / args.data_dir

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    upload_to_hf(args.repo_id, args.data_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    exit(main())
