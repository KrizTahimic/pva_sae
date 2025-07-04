#!/usr/bin/env python3
"""
Chunk runner for sequential processing of dataset chunks on a single GPU.

This script is launched by multi_gpu_launcher.py to process chunks sequentially
while multiple GPUs run in parallel.
"""

import sys
import subprocess
from pathlib import Path
import os
import json
from typing import List, Tuple


def calculate_chunks(start: int, end: int, chunk_size: int) -> List[Tuple[int, int]]:
    """Calculate chunk boundaries."""
    chunks = []
    for chunk_start in range(start, end + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end)
        chunks.append((chunk_start, chunk_end))
    return chunks


def get_phase_dir(phase: str) -> str:
    """Get the output directory for a phase."""
    phase_dirs = {
        "1": "data/phase1_0",
        "3.5": "data/phase3_5"
    }
    return phase_dirs.get(phase, f"data/phase{phase}")


def process_chunk(phase: str, chunk_start: int, chunk_end: int, 
                  model: str, chunk_dir: Path, gpu_id: int) -> bool:
    """
    Process a single chunk.
    
    Returns:
        True if successful, False otherwise
    """
    # Build command
    cmd = [
        "python3", "run.py", "phase", phase,
        "--start", str(chunk_start),
        "--end", str(chunk_end),
        "--model", model
    ]
    
    # Set environment
    env = os.environ.copy()
    # GPU already set by parent process
    
    # Set output directory for the phase runner
    if phase == "1":
        env["PHASE1_OUTPUT_DIR"] = str(chunk_dir)
    elif phase == "3.5":
        env["PHASE3_5_OUTPUT_DIR"] = str(chunk_dir)
    
    # Run the chunk
    log_file = chunk_dir / "run.log"
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False  # Don't raise on non-zero exit
            )
        
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: Failed to run chunk: {e}")
        return False


def main():
    """Main entry point for chunk processing."""
    if len(sys.argv) != 7:
        print(f"Usage: {sys.argv[0]} <gpu_id> <phase> <start> <end> <chunk_size> <model>")
        sys.exit(1)
    
    # Parse arguments
    gpu_id = int(sys.argv[1])
    phase = sys.argv[2]
    start_idx = int(sys.argv[3])
    end_idx = int(sys.argv[4])
    chunk_size = int(sys.argv[5])
    model = sys.argv[6]
    
    print(f"\nGPU {gpu_id}: Processing chunks for range {start_idx}-{end_idx}")
    print(f"Phase: {phase}, Chunk size: {chunk_size}")
    
    # Calculate chunks
    chunks = calculate_chunks(start_idx, end_idx, chunk_size)
    print(f"Total chunks: {len(chunks)}")
    
    # Base directory for chunks
    chunk_base = Path(get_phase_dir(phase)) / "chunks" / f"gpu{gpu_id}"
    chunk_base.mkdir(parents=True, exist_ok=True)
    
    # Process each chunk sequentially
    failed_chunks = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_dir = chunk_base / f"chunk_{chunk_start:05d}-{chunk_end:05d}"
        
        # Check if already completed
        if chunk_dir.exists() and any(chunk_dir.glob("dataset_*.parquet")):
            print(f"  Chunk {i}/{len(chunks)-1}: {chunk_start}-{chunk_end} [SKIPPED - Already completed]")
            continue
        
        # Create chunk directory
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Process the chunk
        print(f"  Chunk {i}/{len(chunks)-1}: {chunk_start}-{chunk_end} [PROCESSING]", end='', flush=True)
        
        success = process_chunk(phase, chunk_start, chunk_end, model, chunk_dir, gpu_id)
        
        if success:
            print(" [DONE]")
        else:
            print(" [FAILED]")
            failed_chunks.append((chunk_start, chunk_end))
            # Continue processing other chunks instead of failing immediately
    
    # Summary
    print(f"\nGPU {gpu_id}: Chunk processing complete")
    print(f"  Successful: {len(chunks) - len(failed_chunks)}/{len(chunks)}")
    
    if failed_chunks:
        print(f"  Failed chunks: {failed_chunks}")
        sys.exit(1)  # Exit with error if any chunks failed
    
    print(f"GPU {gpu_id}: All chunks completed successfully!")
    
    # Write completion marker
    completion_file = chunk_base / "completed.json"
    with open(completion_file, 'w') as f:
        json.dump({
            "gpu_id": gpu_id,
            "phase": phase,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "chunks": len(chunks),
            "chunk_size": chunk_size
        }, f, indent=2)


if __name__ == "__main__":
    main()