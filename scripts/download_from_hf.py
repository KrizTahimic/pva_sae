#!/usr/bin/env python3
"""
Download data from HuggingFace to set up a new machine (e.g. laptop).

Two-phase download:
  1. Normal phases — snapshot_download (ignoring archives/) → downloads all 156
     individual phase dirs directly into data/
  2. Large phases  — for each archives/*.tar on HF: download to /tmp/,
     extract to data/, delete tar

Resumable: tracks completed phases in data/.download_progress.json.

Usage:
    python3 scripts/download_from_hf.py --dry-run        # Preview plan
    python3 scripts/download_from_hf.py                   # Full download (resumable)
    python3 scripts/download_from_hf.py --skip-archives  # Normal phases only
    python3 scripts/download_from_hf.py --data-dir /path/to/data
    python3 scripts/download_from_hf.py --repo-id user/repo
"""

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

DEFAULT_REPO_ID = "kriztahimic/sae-code-correctness-data"
PROGRESS_FILE = ".download_progress.json"


def load_progress(data_dir: Path) -> dict:
    """Load download progress from JSON file."""
    progress_file = data_dir / PROGRESS_FILE
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"completed_archives": []}


def save_progress(data_dir: Path, progress: dict) -> None:
    """Save download progress to JSON file."""
    progress_file = data_dir / PROGRESS_FILE
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def list_archive_files(repo_id: str) -> list[str]:
    """List all archive tar files available on HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi()
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return [f for f in all_files if f.startswith("archives/") and f.endswith(".tar")]


def download_normal_phases(repo_id: str, data_dir: Path) -> None:
    """Download all normal (non-archive) phases via snapshot_download."""
    from huggingface_hub import snapshot_download

    print("\n--- Downloading normal phases ---")
    print(f"  Destination: {data_dir}")
    print("  (This may take a while for large repos)")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(data_dir),
        ignore_patterns=["archives/*", ".download_progress.json"],
    )
    print("  Normal phases download complete.")


def download_and_extract_archive(
    repo_id: str,
    data_dir: Path,
    archive_path: str,
    progress: dict,
) -> None:
    """Download a single archive from HF, extract to data_dir, then delete the tar."""
    from huggingface_hub import hf_hub_download

    phase_name = Path(archive_path).stem  # e.g. "phase2_2" from "archives/phase2_2.tar"
    completed = progress.setdefault("completed_archives", [])

    if phase_name in completed:
        print(f"  [{phase_name}] Already extracted, skipping")
        return

    print(f"  [{phase_name}] Downloading {archive_path} ...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = hf_hub_download(
            repo_id=repo_id,
            filename=archive_path,
            repo_type="dataset",
            local_dir=tmp_dir,
        )
        tar_path = Path(tar_path)
        print(f"  [{phase_name}] Extracting to {data_dir} ...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=data_dir)

    completed.append(phase_name)
    save_progress(data_dir, progress)
    print(f"  [{phase_name}] Done")


def dry_run_report(repo_id: str, data_dir: Path, skip_archives: bool) -> None:
    """Print download plan without downloading."""
    print(f"\n{'=' * 70}")
    print("DOWNLOAD PLAN (DRY RUN)")
    print(f"{'=' * 70}")
    print(f"  Repo:        {repo_id}")
    print(f"  Destination: {data_dir}")

    print("\n--- Step 1: Normal phases ---")
    print("  snapshot_download(ignore_patterns=['archives/*'])")
    print("  → Downloads all individual phase directories directly into data/")

    if skip_archives:
        print("\n--- Step 2: Large phase archives ---")
        print("  SKIPPED (--skip-archives)")
    else:
        print("\n--- Step 2: Large phase archives ---")
        print("  Checking HF for archives/*.tar files...")
        try:
            archives = list_archive_files(repo_id)
            if archives:
                progress = load_progress(data_dir)
                completed = set(progress.get("completed_archives", []))
                for a in archives:
                    phase_name = Path(a).stem
                    status = "DONE" if phase_name in completed else "pending"
                    print(f"  {a:50} [{status}]")
            else:
                print("  No archives found on HF yet.")
        except Exception as e:
            print(f"  Could not list archives: {e}")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Download data from HuggingFace to set up a new machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/download_from_hf.py --dry-run          # Preview plan
  python3 scripts/download_from_hf.py                     # Full download (resumable)
  python3 scripts/download_from_hf.py --skip-archives    # Normal phases only
  python3 scripts/download_from_hf.py --repo-id user/ds  # Custom repo
        """,
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace dataset repository ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        type=Path,
        help="Local destination directory (default: data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview plan without downloading",
    )
    parser.add_argument(
        "--skip-archives",
        action="store_true",
        help="Skip large phase archives (download normal phases only)",
    )
    args = parser.parse_args()

    # Resolve data dir relative to repo root
    if not args.data_dir.is_absolute():
        script_dir = Path(__file__).parent.parent
        args.data_dir = script_dir / args.data_dir

    if args.dry_run:
        dry_run_report(args.repo_id, args.data_dir, args.skip_archives)
        return 0

    # Ensure data dir exists
    args.data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download normal phases
    download_normal_phases(args.repo_id, args.data_dir)

    # Step 2: Download and extract archives
    if not args.skip_archives:
        print("\n--- Downloading large phase archives ---")
        archives = list_archive_files(args.repo_id)
        if not archives:
            print("  No archives found on HF.")
        else:
            progress = load_progress(args.data_dir)
            for archive_path in sorted(archives):
                download_and_extract_archive(
                    args.repo_id, args.data_dir, archive_path, progress
                )

    print(f"\nDownload complete! Data is in: {args.data_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
