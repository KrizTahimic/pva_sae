#!/usr/bin/env python3
"""
Upload data/ directory to HuggingFace Hub, phase-by-phase.

Handles the 2M+ file data directory by uploading one phase directory per commit,
with chunking for any phase with >8k files (HF commit API times out at ~15-20k files).

Resumable: tracks progress in data/.upload_progress.json.

Usage:
    python3 scripts/upload_to_hf.py --dry-run          # Preview plan
    python3 scripts/upload_to_hf.py                     # Full upload (resumable)
    python3 scripts/upload_to_hf.py --phase phase2_2    # Single phase only
    python3 scripts/upload_to_hf.py --repo-id user/repo # Custom repo
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests as _requests

# Monkey-patch requests.Session.send to add a default read timeout.
# HF's upload_folder -> create_commit() uses requests with no timeout, causing
# infinite hangs on slow commits. 120s read timeout converts hangs to retryable
# ReadTimeout exceptions.
_orig_send = _requests.Session.send


def _send_with_timeout(self, request, **kwargs):
    kwargs.setdefault("timeout", (10, 120))  # (connect_timeout, read_timeout) seconds
    return _orig_send(self, request, **kwargs)


_requests.Session.send = _send_with_timeout

# Threshold for "large" phase dirs that need chunking
# HF commit API times out at ~15-20k files; 8k gives safe margin
LARGE_PHASE_THRESHOLD = 8_000
# How many files per chunk for large dirs.
# 1k files × ~500B LFS pointer ≈ 500KB commit payload → completes well within
# the 120s read timeout. Trades more commits for per-commit reliability.
FILES_PER_CHUNK = 1_000
# Preemptive pause after this many commits
COMMITS_BEFORE_PAUSE = 100
# Pause duration in seconds (60 minutes)
PAUSE_DURATION = 60 * 60

IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "checkpoint_*",
    "autosave*",
    ".DS_Store",
    "*.log",
    ".upload_progress.json",
]


def count_files(directory: Path) -> int:
    """Count files in a directory (non-recursive for speed, then recurse)."""
    count = 0
    for _root, _dirs, files in os.walk(directory):
        count += len(files)
    return count


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


def classify_phases(data_dir: Path) -> tuple[list[str], list[str]]:
    """Classify phase directories into normal and large.

    Returns:
        (normal_phases, large_phases) - sorted lists of directory names
    """
    normal = []
    large = []

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name == "logs":
            continue

        file_count = count_files(entry)
        if file_count >= LARGE_PHASE_THRESHOLD:
            large.append(entry.name)
        else:
            normal.append(entry.name)

    return normal, large


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


def get_file_chunks(phase_dir: Path) -> list[list[str]]:
    """For a large phase dir, enumerate all files and batch into chunks of FILES_PER_CHUNK.

    Works for any directory structure by using exact relative paths as allow_patterns.

    Returns:
        List of file path lists (allow_patterns), one per chunk.
    """
    all_files = []
    for root, _dirs, files in os.walk(phase_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), phase_dir)
            all_files.append(rel)
    all_files.sort()

    if not all_files:
        return [[]]

    chunks = []
    for i in range(0, len(all_files), FILES_PER_CHUNK):
        chunks.append(all_files[i : i + FILES_PER_CHUNK])
    return chunks


def upload_with_retry(api, commit_fn, max_retries: int = 5) -> None:
    """Execute an upload function with retry on 429 and 502 errors.

    Args:
        api: HfApi instance (unused but kept for consistency)
        commit_fn: Callable that performs the upload
        max_retries: Maximum retry attempts for 502 errors
    """
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(max_retries):
        try:
            commit_fn()
            return
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else None

            if status == 429:
                # Rate limit - parse wait time
                match = re.search(
                    r"retry this action in about (\d+) (minute|hour)", str(e)
                )
                if match:
                    val, unit = int(match.group(1)), match.group(2)
                    wait = val * 3600 if unit == "hour" else val * 60
                else:
                    wait = 65 * 60  # Default 65 minutes
                print(
                    f"  429 rate limit. Waiting {wait // 60} min before retry..."
                )
                time.sleep(wait + 30)
                continue

            if status in (502, 504):
                # Server error - exponential backoff
                wait = min(60 * (2**attempt), 600)  # Max 10 min
                print(
                    f"  {status} server error (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue

            raise
        except Exception as e:
            # Connection errors etc - retry with backoff
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


def upload_phase(
    api, repo_id: str, data_dir: Path, phase_name: str, progress: dict
) -> int:
    """Upload a normal-sized phase directory. Returns number of commits made (0 or 1)."""
    if phase_name in progress["completed_phases"]:
        print(f"  [{phase_name}] Already uploaded, skipping")
        return 0

    phase_dir = data_dir / phase_name

    def do_upload():
        api.upload_folder(
            folder_path=str(phase_dir),
            path_in_repo=phase_name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {phase_name}",
            ignore_patterns=IGNORE_PATTERNS,
        )

    file_count = count_files(phase_dir)
    print(f"  [{phase_name}] Uploading {file_count:,} files...")
    upload_with_retry(api, do_upload)

    progress["completed_phases"].append(phase_name)
    progress["commit_count"] = progress.get("commit_count", 0) + 1
    save_progress(data_dir, progress)
    print(f"  [{phase_name}] Done")
    return 1


def upload_large_phase(
    api, repo_id: str, data_dir: Path, phase_name: str, progress: dict
) -> int:
    """Upload a large phase directory in chunks. Returns number of commits made."""
    phase_dir = data_dir / phase_name
    chunks = get_file_chunks(phase_dir)
    total_chunks = len(chunks)

    # Initialize chunk tracking
    if phase_name not in progress.get("large_phase_chunks", {}):
        progress.setdefault("large_phase_chunks", {})[phase_name] = {
            "completed": [],
            "total": total_chunks,
        }
        save_progress(data_dir, progress)

    chunk_progress = progress["large_phase_chunks"][phase_name]
    completed_chunks = chunk_progress["completed"]
    commits_made = 0

    print(f"  [{phase_name}] Large phase: {total_chunks} chunks")

    for chunk_idx, patterns in enumerate(chunks):
        if chunk_idx in completed_chunks:
            print(
                f"  [{phase_name}] Chunk {chunk_idx + 1}/{total_chunks} already done"
            )
            continue

        def do_upload(p=patterns):
            api.upload_folder(
                folder_path=str(phase_dir),
                path_in_repo=phase_name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {phase_name} (chunk {chunk_idx + 1}/{total_chunks})",
                allow_patterns=p,
                ignore_patterns=IGNORE_PATTERNS,
            )

        print(
            f"  [{phase_name}] Uploading chunk {chunk_idx + 1}/{total_chunks} "
            f"({len(patterns):,} files)..."
        )
        upload_with_retry(api, do_upload)

        chunk_progress["completed"].append(chunk_idx)
        progress["commit_count"] = progress.get("commit_count", 0) + 1
        save_progress(data_dir, progress)
        commits_made += 1

    # Mark phase as fully completed
    progress["completed_phases"].append(phase_name)
    save_progress(data_dir, progress)
    print(f"  [{phase_name}] All chunks done")
    return commits_made


def maybe_pause(progress: dict, commits_this_session: int) -> None:
    """Preemptive pause after COMMITS_BEFORE_PAUSE commits to avoid rate limits."""
    total = progress.get("commit_count", 0)
    if commits_this_session > 0 and commits_this_session % COMMITS_BEFORE_PAUSE == 0:
        print(
            f"\n  Preemptive pause after {commits_this_session} commits this session "
            f"({total} total). Waiting {PAUSE_DURATION // 60} min..."
        )
        time.sleep(PAUSE_DURATION)
        print("  Resuming uploads.\n")


def dry_run_report(data_dir: Path) -> None:
    """Print detailed upload plan without uploading."""
    normal, large = classify_phases(data_dir)

    print(f"\n{'=' * 70}")
    print("UPLOAD PLAN (DRY RUN)")
    print(f"{'=' * 70}")

    total_files = 0
    total_size = 0
    total_commits = 0

    print(f"\n--- Normal phases ({len(normal)} dirs, 1 commit each) ---")
    for phase in normal:
        phase_dir = data_dir / phase
        fc = count_files(phase_dir)
        sz = get_dir_size(phase_dir)
        total_files += fc
        total_size += sz
        total_commits += 1
        print(f"  {phase:40} {fc:>8,} files  {format_size(sz):>10}")

    print(f"\n--- Large phases ({len(large)} dirs, chunked) ---")
    for phase in large:
        phase_dir = data_dir / phase
        fc = count_files(phase_dir)
        sz = get_dir_size(phase_dir)
        chunks = get_file_chunks(phase_dir)
        total_files += fc
        total_size += sz
        total_commits += len(chunks)
        print(
            f"  {phase:40} {fc:>8,} files  {format_size(sz):>10}  "
            f"({len(chunks)} chunks)"
        )

    print(f"\n{'=' * 70}")
    print(f"Total files:   {total_files:>12,}")
    print(f"Total size:    {format_size(total_size):>12}")
    print(f"Total commits: {total_commits:>12,}")
    print(f"{'=' * 70}")

    # Check progress
    progress = load_progress(data_dir)
    completed = progress.get("completed_phases", [])
    if completed:
        remaining_normal = [p for p in normal if p not in completed]
        remaining_large = [p for p in large if p not in completed]
        print(f"\nProgress: {len(completed)} phases already uploaded")
        print(f"Remaining: {len(remaining_normal)} normal + {len(remaining_large)} large")

        # Check partial large phases
        for phase in large:
            if phase in progress.get("large_phase_chunks", {}):
                info = progress["large_phase_chunks"][phase]
                done = len(info["completed"])
                total = info["total"]
                if done < total:
                    print(f"  {phase}: {done}/{total} chunks done")


def upload_to_hf(
    repo_id: str,
    data_dir: Path,
    dry_run: bool = False,
    single_phase: str | None = None,
) -> None:
    """Upload data directory to HuggingFace Hub, phase by phase."""
    if dry_run:
        dry_run_report(data_dir)
        return

    from huggingface_hub import HfApi

    api = HfApi()

    # Ensure repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Repository {repo_id} exists.")
    except Exception:
        print(f"Creating repository {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Created: https://huggingface.co/datasets/{repo_id}")

    # Classify phases
    normal, large = classify_phases(data_dir)
    progress = load_progress(data_dir)

    # Filter to single phase if requested
    if single_phase:
        if single_phase in normal:
            normal = [single_phase]
            large = []
        elif single_phase in large:
            normal = []
            large = [single_phase]
        else:
            print(f"Error: Phase '{single_phase}' not found in {data_dir}")
            return

    completed = set(progress.get("completed_phases", []))
    remaining_normal = [p for p in normal if p not in completed]
    remaining_large = [p for p in large if p not in completed]

    print(f"\nUpload plan:")
    print(f"  Normal phases: {len(remaining_normal)} remaining (of {len(normal)})")
    print(f"  Large phases:  {len(remaining_large)} remaining (of {len(large)})")
    print(f"  Previous commits: {progress.get('commit_count', 0)}")
    print()

    commits_this_session = 0

    # Phase 1: Normal phases
    for phase in remaining_normal:
        commits = upload_phase(api, repo_id, data_dir, phase, progress)
        commits_this_session += commits
        maybe_pause(progress, commits_this_session)

    # Phase 2: Large phases (chunked)
    for phase in remaining_large:
        commits = upload_large_phase(api, repo_id, data_dir, phase, progress)
        commits_this_session += commits
        maybe_pause(progress, commits_this_session)

    print(f"\nUpload complete! {commits_this_session} commits this session.")
    print(f"Total commits: {progress.get('commit_count', 0)}")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload data/ directory to HuggingFace Hub (phase-by-phase)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/upload_to_hf.py --dry-run              # Preview plan
  python3 scripts/upload_to_hf.py                         # Full upload (resumable)
  python3 scripts/upload_to_hf.py --phase phase2_2        # Single phase
  python3 scripts/upload_to_hf.py --repo-id user/dataset  # Custom repo
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
        help="Preview upload plan without uploading",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Upload a single phase directory only",
    )
    args = parser.parse_args()

    # Resolve data dir relative to script location
    if not args.data_dir.is_absolute():
        script_dir = Path(__file__).parent.parent
        args.data_dir = script_dir / args.data_dir

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    upload_to_hf(args.repo_id, args.data_dir, args.dry_run, args.phase)
    return 0


if __name__ == "__main__":
    exit(main())
