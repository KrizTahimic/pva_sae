#!/usr/bin/env python3
"""
Download data from HuggingFace to set up a new machine (e.g. laptop).

Two-step download:
  Step 1 — Normal phases (CDN): enumerates all files via a partial git clone
    (tree objects only, no file content downloaded — zero HF API calls, no rate
    limits, no 429s). Then downloads each file via direct CDN URL.
    Resume: skips already-present files.
  Step 2 — Archive phases (tar): downloads each archives/*.tar via hf_hub_download,
    extracts to data/, deletes the tar. Resume: tracks completed phases in
    data/.download_progress.json.

Usage:
    python3 scripts/download_from_hf.py --dry-run        # Preview plan
    python3 scripts/download_from_hf.py                   # Full download (resumable)
    python3 scripts/download_from_hf.py --skip-archives   # Normal phases only
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
    progress_file = data_dir / PROGRESS_FILE
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"completed_archives": []}


def save_progress(data_dir: Path, progress: dict) -> None:
    progress_file = data_dir / PROGRESS_FILE
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def list_all_repo_files_via_git(repo_id: str) -> list[str]:
    """
    Enumerate every file in the HF dataset repo using a partial git clone.

    Uses --filter=blob:none --no-checkout so only tree/commit objects are
    fetched (a few MB), not actual file content. Then `git ls-tree -r` lists
    every path. Zero HF API calls — immune to rate limiting.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        url = f"https://user:{token}@huggingface.co/datasets/{repo_id}"
    else:
        url = f"https://huggingface.co/datasets/{repo_id}"

    with tempfile.TemporaryDirectory() as tmp:
        print("  Cloning git metadata (no file content, ~seconds)...")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", url, tmp],
            check=True,
            capture_output=True,  # suppress git output
        )
        result = subprocess.run(
            ["git", "-C", tmp, "ls-tree", "-r", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )

    all_files = [f for f in result.stdout.strip().split("\n") if f]
    return all_files


def list_archive_files_via_git(repo_id: str) -> list[str]:
    all_files = list_all_repo_files_via_git(repo_id)
    return [f for f in all_files if f.startswith("archives/") and f.endswith(".tar")]


def list_normal_files_via_git(repo_id: str) -> list[str]:
    all_files = list_all_repo_files_via_git(repo_id)
    return [f for f in all_files if not f.startswith("archives/") and not f.startswith(".")]


def download_file_with_timeout(url: str, local_path: Path, headers: dict,
                               chunk_size: int = 65536, stall_timeout: int = 30) -> None:
    """
    Download a file via streaming with per-chunk stall detection.

    Uses a background thread to enforce a per-chunk read timeout, since
    requests' timeout= only covers the initial connection/headers, not
    streaming body chunks (stalled streams hang forever otherwise).
    """
    import socket
    import requests

    # socket-level timeout catches stalls at the TCP layer
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(stall_timeout)
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=(10, stall_timeout))
        resp.raise_for_status()
        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        try:
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
            tmp_path.rename(local_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    finally:
        socket.setdefaulttimeout(old_timeout)


def download_normal_phases_via_cdn(repo_id: str, data_dir: Path) -> None:
    """Download all non-archive files directly from HF CDN."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    normal_files = list_normal_files_via_git(repo_id)
    print(f"  Found {len(normal_files)} normal-phase files.", flush=True)

    skipped = 0
    downloaded = 0
    errors = 0
    for i, remote_path in enumerate(normal_files):
        local_path = data_dir / remote_path
        if local_path.exists():
            skipped += 1
            continue
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{remote_path}"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                download_file_with_timeout(url, local_path, headers)
                downloaded += 1
                if downloaded % 100 == 1:
                    print(f"  [{i + 1}/{len(normal_files)}] {remote_path}", flush=True)
                break
            except Exception as e:
                if attempt == 2:
                    errors += 1
                    print(f"  ERROR {remote_path}: {e}", flush=True)
                else:
                    print(f"  Retry {attempt + 1} for {remote_path}: {e}", flush=True)

    print(f"  Normal phases complete: {downloaded} downloaded, {skipped} already present, {errors} errors.", flush=True)


def download_and_extract_archive(
    repo_id: str,
    data_dir: Path,
    archive_path: str,
    progress: dict,
) -> None:
    """Download a single archive from HF, extract to data_dir, then delete the tar."""
    from huggingface_hub import hf_hub_download

    phase_name = Path(archive_path).stem
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
    print(f"\n{'=' * 70}")
    print("DOWNLOAD PLAN (DRY RUN)")
    print(f"{'=' * 70}")
    print(f"  Repo:        {repo_id}")
    print(f"  Destination: {data_dir}")

    print("\n  Fetching file list via git metadata clone...")
    try:
        all_files = list_all_repo_files_via_git(repo_id)
        normal_files = [f for f in all_files if not f.startswith("archives/") and not f.startswith(".")]
        archive_files = [f for f in all_files if f.startswith("archives/") and f.endswith(".tar")]

        print(f"\n--- Step 1: Normal phases (CDN download) ---")
        present = sum(1 for f in normal_files if (data_dir / f).exists())
        print(f"  {len(normal_files)} files total, {present} already present, "
              f"{len(normal_files) - present} to download.")

        if skip_archives:
            print(f"\n--- Step 2: Archives (SKIPPED via --skip-archives) ---")
        else:
            print(f"\n--- Step 2: Archive phases ---")
            progress = load_progress(data_dir)
            completed = set(progress.get("completed_archives", []))
            if archive_files:
                for a in sorted(archive_files):
                    phase_name = Path(a).stem
                    status = "DONE" if phase_name in completed else "pending"
                    print(f"  {a:50} [{status}]")
            else:
                print("  No archives found in repo.")
    except Exception as e:
        print(f"  Failed: {e}")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Download data from HuggingFace to set up a new machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/download_from_hf.py --dry-run          # Preview plan
  python3 scripts/download_from_hf.py                     # Full download (resumable)
  python3 scripts/download_from_hf.py --skip-archives     # Normal phases only
  python3 scripts/download_from_hf.py --repo-id user/ds  # Custom repo
        """,
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-archives", action="store_true")
    args = parser.parse_args()

    if not args.data_dir.is_absolute():
        args.data_dir = Path(__file__).parent.parent / args.data_dir

    if args.dry_run:
        dry_run_report(args.repo_id, args.data_dir, args.skip_archives)
        return 0

    args.data_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Step 1: Downloading normal phases via CDN ---")
    download_normal_phases_via_cdn(args.repo_id, args.data_dir)

    if not args.skip_archives:
        print("\n--- Step 2: Downloading phase archives ---")
        archive_files = list_archive_files_via_git(args.repo_id)
        if not archive_files:
            print("  No archives found.")
        else:
            progress = load_progress(args.data_dir)
            for archive_path in sorted(archive_files):
                download_and_extract_archive(
                    args.repo_id, args.data_dir, archive_path, progress
                )
    else:
        print("\n--- Step 2: Skipped (--skip-archives) ---")

    print(f"\nDownload complete! Data is in: {args.data_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
