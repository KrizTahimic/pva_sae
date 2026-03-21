#!/usr/bin/env python3
"""
Upload ALL phase directories to HuggingFace as tar archives.

Instead of per-file LFS uploads (which exhausts HF's API rate limits with
98k+ files), each phase is tarred into a single file and uploaded in one
API call. ~30 tar uploads ≈ 1-3 hours; enables O(N_phases) download.

Resumable: tracks completed archives in data/.upload_progress.json under
"archived_phases" key.

Usage:
    python3 scripts/archive_upload.py --dry-run          # Preview plan
    python3 scripts/archive_upload.py                     # Full upload (resumable)
    python3 scripts/archive_upload.py --phase phase2_2   # Single phase only
    python3 scripts/archive_upload.py --repo-id user/repo # Custom repo
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import requests as _requests

# Monkey-patch requests.Session.send to add a default read timeout.
_orig_send = _requests.Session.send


def _send_with_timeout(self, request, **kwargs):
    kwargs.setdefault("timeout", (10, 600))  # (connect_timeout, read_timeout) seconds
    return _orig_send(self, request, **kwargs)


_requests.Session.send = _send_with_timeout

DEFAULT_REPO_ID = "kriztahimic/sae-code-correctness-data"

# Phases already archived (8 large phases). Listed to document history; no
# longer used to restrict which phases get archived — all phases are archived.
LARGE_PHASES = [
    "phase1_0_gemma9b",
    "phase1_0_llama",
    "phase3_5",
    "phase3_5_gemma9b",
    "phase3_5_llama",
    "phase2_2",
    "phase2_2_gemma9b",
    "phase2_2_llama",
]


def discover_all_phases(data_dir: Path) -> list[str]:
    """Return sorted list of all phase directories in data_dir."""
    if not data_dir.exists():
        return []
    phases = sorted(
        d.name
        for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("phase")
    )
    return phases


def format_size(size_bytes: int) -> str:
    """Format byte size for display."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MB"
    return f"{size_bytes / 1024:.2f} KB"


def get_dir_size(directory: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for root, _dirs, files in os.walk(directory):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def load_progress(data_dir: Path) -> dict:
    """Load upload progress from JSON file."""
    progress_file = data_dir / ".upload_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"completed_phases": [], "large_phase_chunks": {}, "commit_count": 0}


def save_progress(data_dir: Path, progress: dict) -> None:
    """Save upload progress to JSON file."""
    progress_file = data_dir / ".upload_progress.json"
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def upload_with_retry(api, commit_fn, max_retries: int = 5) -> None:
    """Execute an upload function with retry on 429 and 502 errors."""
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(max_retries):
        try:
            commit_fn()
            return
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else None

            if status == 429:
                match = re.search(
                    r"retry this action in about (\d+) (minute|hour)", str(e)
                )
                if match:
                    val, unit = int(match.group(1)), match.group(2)
                    wait = val * 3600 if unit == "hour" else val * 60
                else:
                    wait = 65 * 60
                print(f"  429 rate limit. Waiting {wait // 60} min before retry...")
                time.sleep(wait + 30)
                continue

            if status in (502, 504):
                wait = min(60 * (2**attempt), 600)
                print(
                    f"  {status} server error (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue

            raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(60 * (2**attempt), 600)
                print(
                    f"  Error: {e.__class__.__name__}: {e} "
                    f"(attempt {attempt + 1}/{max_retries}). Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Failed after {max_retries} retries")


def create_tar(phase_dir: Path, tar_path: Path) -> None:
    """Create uncompressed tar archive of a phase directory."""
    print(f"  Creating tar: {tar_path} ...")
    # -C data_dir {phase_name} so tar paths are relative to data/
    result = subprocess.run(
        ["tar", "-cf", str(tar_path), "-C", str(phase_dir.parent), phase_dir.name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tar failed: {result.stderr}")
    size = tar_path.stat().st_size
    print(f"  Tar created: {format_size(size)}")


def upload_archive(
    api, repo_id: str, data_dir: Path, phase_name: str, progress: dict
) -> None:
    """Tar and upload a single phase directory as an archive."""
    archived = progress.setdefault("archived_phases", [])
    if phase_name in archived:
        print(f"  [{phase_name}] Already archived+uploaded, skipping")
        return

    phase_dir = data_dir / phase_name
    if not phase_dir.exists():
        print(f"  [{phase_name}] Directory not found, skipping")
        return

    tar_path = Path(f"/tmp/{phase_name}.tar")

    try:
        # Step 1: Create tar
        create_tar(phase_dir, tar_path)

        # Step 2: Upload
        repo_path = f"archives/{phase_name}.tar"
        print(f"  [{phase_name}] Uploading to {repo_path} ...")

        def do_upload():
            api.upload_file(
                path_or_fileobj=str(tar_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Archive upload: {phase_name}",
            )

        upload_with_retry(api, do_upload)

        # Step 3: Track progress
        archived.append(phase_name)
        progress["commit_count"] = progress.get("commit_count", 0) + 1
        save_progress(data_dir, progress)
        print(f"  [{phase_name}] Done")

    finally:
        # Always clean up temp tar
        if tar_path.exists():
            tar_path.unlink()
            print(f"  Deleted temp tar: {tar_path}")


def dry_run_report(data_dir: Path, phases: list[str]) -> None:
    """Print archive upload plan without uploading."""
    progress = load_progress(data_dir)
    archived = set(progress.get("archived_phases", []))

    print(f"\n{'=' * 70}")
    print("ARCHIVE UPLOAD PLAN (DRY RUN)")
    print(f"{'=' * 70}")
    print(f"\nTarget repo: {DEFAULT_REPO_ID}")
    print(f"Archive path: archives/{{phase}}.tar\n")

    total_size = 0
    for phase in phases:
        phase_dir = data_dir / phase
        if not phase_dir.exists():
            print(f"  {phase:40} [NOT FOUND]")
            continue
        sz = get_dir_size(phase_dir)
        total_size += sz
        status = "DONE" if phase in archived else "pending"
        print(f"  {phase:40} {format_size(sz):>10}  [{status}]")

    print(f"\n{'=' * 70}")
    print(f"Total size:    {format_size(total_size):>12}")
    print(f"Archives done: {len(archived)}/{len(phases)}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload large phase directories as tar archives to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/archive_upload.py --dry-run              # Preview plan
  python3 scripts/archive_upload.py                         # Full upload (resumable)
  python3 scripts/archive_upload.py --phase phase2_2        # Single phase
  python3 scripts/archive_upload.py --repo-id user/dataset  # Custom repo
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
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview plan without uploading",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Upload a single phase directory only",
    )
    args = parser.parse_args()

    # Resolve data dir relative to repo root
    if not args.data_dir.is_absolute():
        script_dir = Path(__file__).parent.parent
        args.data_dir = script_dir / args.data_dir

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    phases = [args.phase] if args.phase else discover_all_phases(args.data_dir)

    if args.dry_run:
        dry_run_report(args.data_dir, phases)
        return 0

    from huggingface_hub import HfApi

    api = HfApi()

    # Ensure repo exists
    try:
        api.repo_info(repo_id=args.repo_id, repo_type="dataset")
        print(f"Repository {args.repo_id} exists.")
    except Exception:
        print(f"Creating repository {args.repo_id}...")
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    progress = load_progress(args.data_dir)
    archived = set(progress.get("archived_phases", []))
    remaining = [p for p in phases if p not in archived]

    print(f"\nArchive upload plan:")
    print(f"  Phases to archive: {len(remaining)} remaining (of {len(phases)})")
    print(f"  Previous commits: {progress.get('commit_count', 0)}")
    print()

    for phase in remaining:
        upload_archive(api, args.repo_id, args.data_dir, phase, progress)

    print(f"\nArchive upload complete!")
    print(f"Total commits: {progress.get('commit_count', 0)}")
    print(f"View at: https://huggingface.co/datasets/{args.repo_id}/tree/main/archives")
    return 0


if __name__ == "__main__":
    exit(main())
