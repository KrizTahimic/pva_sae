#!/usr/bin/env python3
"""
Multi-GPU launcher for parallel dataset processing.

This script automatically distributes work across available GPUs by launching
separate processes for each GPU with non-overlapping dataset chunks.

Usage:
    python3 multi_gpu_launcher.py --start 0 --end 973 --model google/gemma-2-2b
    python3 multi_gpu_launcher.py --gpus 0,1,2 --start 0 --end 973 --model google/gemma-2-2b
"""

import argparse
from subprocess import Popen, STDOUT, TimeoutExpired, run as subprocess_run
from os import environ, setsid, killpg, getpgid
import time
from signal import signal as signal_register, SIGINT, SIGTERM, SIGKILL
import sys
from typing import List, Tuple, Optional, Dict
import math
import torch
from pathlib import Path
import pandas as pd
import shutil
import glob

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common.logging import LoggingManager
from common.utils import managed_subprocess, get_phase_dir, get_timestamp
from common.config import Config


class MultiGPULauncher:
    """Manages parallel execution across multiple GPUs"""
    
    def __init__(self, gpus: List[int] = None):
        """
        Initialize launcher
        
        Args:
            gpus: List of GPU indices to use (None for auto-detect)
        """
        self.processes = []
        self.logger = None
        
        # Detect available GPUs
        if gpus is None:
            self.gpus = list(range(torch.cuda.device_count()))
        else:
            self.gpus = gpus
            
        if not self.gpus:
            raise RuntimeError("No GPUs available")
            
        # Setup signal handlers for cleanup
        signal_register(SIGINT, self._signal_handler)
        signal_register(SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals"""
        print("\nReceived interrupt signal. Cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def split_workload(self, start_idx: int, end_idx: int) -> List[Tuple[int, int]]:
        """
        Split dataset range across GPUs
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (inclusive)
            
        Returns:
            List of (start, end) tuples for each GPU
        """
        total_items = end_idx - start_idx + 1
        items_per_gpu = math.ceil(total_items / len(self.gpus))
        
        splits = []
        for i in range(len(self.gpus)):
            gpu_start = start_idx + (i * items_per_gpu)
            gpu_end = min(gpu_start + items_per_gpu - 1, end_idx)
            
            if gpu_start <= end_idx:
                splits.append((gpu_start, gpu_end))
            
        return splits
    
    def split_into_chunks(self, start: int, end: int, chunk_size: int) -> List[Tuple[int, int]]:
        """
        Split range into fixed-size chunks for checkpointing.
        
        Args:
            start: Starting index
            end: Ending index (inclusive)
            chunk_size: Size of each chunk
            
        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        chunks = []
        for chunk_start in range(start, end + 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size - 1, end)
            chunks.append((chunk_start, chunk_end))
        return chunks
    
    def launch_phase1_processes(self, 
                               start_idx: int, 
                               end_idx: int,
                               model: str,
                               dataset_dir: str = "data/datasets",
                               extra_args: List[str] = None,
                               no_checkpoint: bool = False):
        """
        Launch Phase 1 parallel processes on different GPUs
        
        Args:
            start_idx: Starting index
            end_idx: Ending index
            model: Model name to use
            dataset_dir: Directory for datasets
            extra_args: Additional arguments to pass
            no_checkpoint: If True, disable checkpointing
        """
        # Setup logging
        logging_manager = LoggingManager(
            phase="1.0",
            log_dir="data/logs"
        )
        self.logger = logging_manager.setup_logging("multi_gpu_launcher")
        
        # Check SAE split exists
        sae_split_path = Path(get_phase_dir('0.1')) / "sae_mbpp.parquet"
        if not sae_split_path.exists():
            raise FileNotFoundError(
                f"SAE split not found at {sae_split_path}. "
                "Please run Phase 0.1 first: python3 run.py phase 0.1"
            )
        
        # Load SAE split to get actual size
        sae_df = pd.read_parquet(sae_split_path)
        actual_size = len(sae_df)
        
        # Adjust end_idx if it exceeds actual SAE split size
        if end_idx >= actual_size:
            self.logger.info(f"Adjusting end_idx from {end_idx} to {actual_size - 1} (SAE split size: {actual_size})")
            end_idx = actual_size - 1
        
        # Split workload
        splits = self.split_workload(start_idx, end_idx)
        
        # Get checkpoint size from config
        config = Config()
        checkpoint_size = config.checkpoint_frequency
        
        print(f"\n{'='*60}")
        print(f"MULTI-GPU PARALLEL PROCESSING - PHASE 1")
        print(f"{'='*60}")
        print(f"Total GPUs: {len(self.gpus)}")
        print(f"SAE split size: {actual_size} problems")
        print(f"Processing range: {start_idx}-{end_idx} ({end_idx - start_idx + 1} items)")
        print(f"Model: {model}")
        print(f"Checkpointing: {'DISABLED' if no_checkpoint else f'ENABLED (chunk size: {checkpoint_size})'}")
        print(f"\nWorkload distribution:")
        
        for i, (gpu_id, (split_start, split_end)) in enumerate(zip(self.gpus, splits)):
            items = split_end - split_start + 1
            print(f"  GPU {gpu_id}: {split_start}-{split_end} ({items} items)")
        
        print(f"\n{'='*60}")
        print("Starting processes...\n")
        
        # Launch processes
        for gpu_id, (split_start, split_end) in zip(self.gpus, splits):
            if no_checkpoint:
                # Original behavior - single run
                cmd = [
                    "python3", "run.py", "phase", "1",
                    "--start", str(split_start),
                    "--end", str(split_end),
                    "--model", model,
                    "--dataset-dir", dataset_dir
                ]
            else:
                # New behavior - use chunk runner
                cmd = [
                    "python3", "chunk_runner.py",
                    str(gpu_id), "1", str(split_start), str(split_end),
                    str(checkpoint_size), model
                ]
            
            # Add extra arguments (only for direct run)
            if extra_args and no_checkpoint:
                cmd.extend(extra_args)
            
            # Set environment for GPU
            env = environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Create log file for this GPU
            timestamp = get_timestamp()
            log_file = f"data/logs/gpu_{gpu_id}_phase1_{split_start}-{split_end}_{timestamp}.log"
            
            print(f"Launching on GPU {gpu_id}: {split_start}-{split_end}")
            self.logger.info(f"Launching process on GPU {gpu_id}: {' '.join(cmd)}")
            
            # Launch process
            log_handle = open(log_file, 'w')
            process = Popen(
                cmd,
                env=env,
                stdout=log_handle,
                stderr=STDOUT,
                preexec_fn=setsid  # Create new process group
            )
            self.processes.append((gpu_id, process, log_file, log_handle))
        
        print(f"\nAll processes launched. Monitoring progress...")
        print(f"Check individual logs in data/logs/")
        print(f"\nPress Ctrl+C to stop all processes\n")
    
    def launch_phase2_2_processes(self,
                                  start_idx: int,
                                  end_idx: int,
                                  model: str,
                                  extra_args: List[str] = None,
                                  no_checkpoint: bool = False):
        """
        Launch Phase 2.2 pile caching processes on different GPUs
        
        Args:
            start_idx: Starting index for pile dataset
            end_idx: Ending index for pile dataset (inclusive)
            model: Model name to use
            extra_args: Additional arguments to pass
            no_checkpoint: If True, disable checkpointing
        """
        # Setup logging
        logging_manager = LoggingManager(
            phase="2.2",
            log_dir="data/logs"
        )
        self.logger = logging_manager.setup_logging("multi_gpu_launcher")
        
        # Split workload
        splits = self.split_workload(start_idx, end_idx)
        
        print(f"\n{'='*60}")
        print(f"MULTI-GPU PARALLEL PROCESSING - PHASE 2.2")
        print(f"{'='*60}")
        print(f"Total GPUs: {len(self.gpus)}")
        print(f"Processing range: {start_idx}-{end_idx} ({end_idx - start_idx + 1} pile samples)")
        print(f"Model: {model}")
        print(f"\nWorkload distribution:")
        
        for i, (gpu_id, (split_start, split_end)) in enumerate(zip(self.gpus, splits)):
            items = split_end - split_start + 1
            print(f"  GPU {gpu_id}: {split_start}-{split_end} ({items} samples)")
        
        print(f"\n{'='*60}")
        print("Starting processes...\n")
        
        # Launch processes
        for gpu_id, (split_start, split_end) in zip(self.gpus, splits):
            cmd = [
                "python3", "run.py", "phase", "2.2",
                "--start", str(split_start),
                "--end", str(split_end),
                "--model", model
            ]
            
            # Add extra arguments
            if extra_args:
                cmd.extend(extra_args)
            
            # Set environment for GPU
            env = environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Log file for this GPU
            log_file = f"data/logs/gpu_{gpu_id}_output.log"
            log_handle = open(log_file, "w")
            
            print(f"Launching on GPU {gpu_id}: python3 run.py phase 2.2 --start {split_start} --end {split_end}")
            
            # Launch process
            process = Popen(
                cmd,
                env=env,
                stdout=log_handle,
                stderr=STDOUT,
                preexec_fn=setsid  # Create new process group
            )
            self.processes.append((gpu_id, process, log_file, log_handle))
        
        print(f"\nAll processes launched. Monitoring progress...")
        print(f"Check individual logs in data/logs/")
        print(f"\nPress Ctrl+C to stop all processes\n")
    
    def launch_phase3_5_processes(self,
                                  start_idx: int,
                                  end_idx: int,
                                  model: str,
                                  extra_args: List[str] = None,
                                  no_checkpoint: bool = False):
        """
        Launch Phase 3.5 temperature robustness processes on different GPUs
        
        Args:
            start_idx: Starting index for validation dataset
            end_idx: Ending index for validation dataset (inclusive)
            model: Model name to use
            extra_args: Additional arguments to pass
            no_checkpoint: If True, disable checkpointing
        """
        # Setup logging
        logging_manager = LoggingManager(
            phase="3.5",
            log_dir="data/logs"
        )
        self.logger = logging_manager.setup_logging("multi_gpu_launcher")
        
        # Check validation split exists
        validation_file = Path("data/phase0_1/validation_mbpp.parquet")
        if not validation_file.exists():
            raise FileNotFoundError(
                f"Validation data not found at {validation_file}. "
                "Please run Phase 0.1 first."
            )
        
        # Load validation data to check size
        validation_df = pd.read_parquet(validation_file)
        actual_size = len(validation_df)
        
        # Adjust end_idx if it exceeds validation size
        if end_idx >= actual_size:
            self.logger.info(f"Adjusting end_idx from {end_idx} to {actual_size - 1} (validation size: {actual_size})")
            end_idx = actual_size - 1
        
        # Split workload
        splits = self.split_workload(start_idx, end_idx)
        
        # Get checkpoint size from config
        config = Config()
        checkpoint_size = config.checkpoint_frequency
        
        print(f"\n{'='*60}")
        print(f"MULTI-GPU PARALLEL PROCESSING - PHASE 3.5")
        print(f"{'='*60}")
        print(f"Total GPUs: {len(self.gpus)}")
        print(f"Validation dataset size: {actual_size} problems")
        print(f"Processing range: {start_idx}-{end_idx} ({end_idx - start_idx + 1} items)")
        print(f"Model: {model}")
        print(f"Checkpointing: {'DISABLED' if no_checkpoint else f'ENABLED (chunk size: {checkpoint_size})'}")
        print(f"\nWorkload distribution:")
        
        for i, (gpu_id, (split_start, split_end)) in enumerate(zip(self.gpus, splits)):
            items = split_end - split_start + 1
            print(f"  GPU {gpu_id}: {split_start}-{split_end} ({items} items)")
        
        print(f"\n{'='*60}")
        print("Starting processes...\n")
        
        # Launch processes
        for gpu_id, (split_start, split_end) in zip(self.gpus, splits):
            if no_checkpoint:
                # Original behavior - single run
                cmd = [
                    "python3", "run.py", "phase", "3.5",
                    "--start", str(split_start),
                    "--end", str(split_end),
                    "--model", model
                ]
            else:
                # New behavior - use chunk runner
                cmd = [
                    "python3", "chunk_runner.py",
                    str(gpu_id), "3.5", str(split_start), str(split_end),
                    str(checkpoint_size), model
                ]
            
            # Add extra arguments (only for direct run)
            if extra_args and no_checkpoint:
                cmd.extend(extra_args)
            
            # Set environment for GPU
            env = environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Create log file for this GPU
            timestamp = get_timestamp()
            log_file = f"data/logs/gpu_{gpu_id}_phase3.5_{split_start}-{split_end}_{timestamp}.log"
            
            print(f"Launching on GPU {gpu_id}: {split_start}-{split_end}")
            self.logger.info(f"Launching process on GPU {gpu_id}: {' '.join(cmd)}")
            
            # Launch process
            log_handle = open(log_file, 'w')
            process = Popen(
                cmd,
                env=env,
                stdout=log_handle,
                stderr=STDOUT,
                preexec_fn=setsid  # Create new process group
            )
            self.processes.append((gpu_id, process, log_file, log_handle))
        
        print(f"\nAll processes launched. Monitoring progress...")
        print(f"Check individual logs in data/logs/")
        print(f"\nPress Ctrl+C to stop all processes\n")
    
    def monitor_processes(self):
        """Monitor running processes and wait for completion"""
        start_time = time.time()
        
        try:
            while True:
                # Check process status
                running = []
                completed = []
                failed = []
                
                for gpu_id, process, log_file, log_handle in self.processes:
                    poll = process.poll()
                    if poll is None:
                        running.append(gpu_id)
                    elif poll == 0:
                        completed.append(gpu_id)
                        # Close log file for completed process
                        if log_handle and not log_handle.closed:
                            log_handle.close()
                    else:
                        failed.append((gpu_id, poll))
                        # Close log file for failed process
                        if log_handle and not log_handle.closed:
                            log_handle.close()
                
                # Display status
                elapsed = time.time() - start_time
                elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
                
                status = f"[{elapsed_str}] "
                status += f"Running: {len(running)} "
                status += f"Completed: {len(completed)} "
                status += f"Failed: {len(failed)}"
                
                print(f"\r{status}", end='', flush=True)
                
                # Check if all done
                if not running:
                    print("\n\nAll processes completed!")
                    break
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            self.cleanup()
            return
        
        # Final summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total time: {elapsed_str}")
        print(f"Completed: {len(completed)} GPUs")
        print(f"Failed: {len(failed)} GPUs")
        
        if failed:
            print("\nFailed GPUs:")
            for gpu_id, code in failed:
                print(f"  GPU {gpu_id}: exit code {code}")
        
        print(f"\nCheck logs in data/logs/gpu_*_output.log for details")
    
    def cleanup(self):
        """Clean up all running processes and resources"""
        for gpu_id, process, log_file, log_handle in self.processes:
            # Close log file if still open
            if log_handle and not log_handle.closed:
                log_handle.close()
            
            # Terminate process if still running
            if process.poll() is None:
                print(f"Terminating process on GPU {gpu_id}")
                try:
                    # Try graceful termination first
                    killpg(getpgid(process.pid), SIGTERM)
                    # Wait briefly for termination
                    try:
                        process.wait(timeout=5)
                    except TimeoutExpired:
                        # Force kill if necessary
                        killpg(getpgid(process.pid), SIGKILL)
                        process.wait()
                except Exception as e:
                    print(f"Error terminating process: {e}")
    
    def merge_checkpoint_chunks(self, phase: str, gpu_id: int) -> Optional[Path]:
        """
        Merge all chunks for a GPU into final dataset.
        
        Args:
            phase: Phase number (e.g., "1" or "3.5")
            gpu_id: GPU ID
            
        Returns:
            Path to merged dataset file, or None if no chunks found
        """
        chunk_base = Path(get_phase_dir(phase)) / "chunks" / f"gpu{gpu_id}"
        
        if not chunk_base.exists():
            self.logger.warning(f"No chunk directory found at {chunk_base}")
            return None
        
        # Find all chunk directories
        chunk_dirs = sorted([d for d in chunk_base.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
        
        if not chunk_dirs:
            self.logger.warning(f"No chunk directories found in {chunk_base}")
            return None
        
        print(f"\nMerging {len(chunk_dirs)} chunks for GPU {gpu_id}...")
        
        # Collect all parquet files from chunk directories
        dfs = []
        for chunk_dir in chunk_dirs:
            parquet_files = list(chunk_dir.glob("dataset_*.parquet"))
            if parquet_files:
                # Take the first (should be only) parquet file
                df = pd.read_parquet(parquet_files[0])
                dfs.append(df)
                print(f"  Loaded {chunk_dir.name}: {len(df)} records")
            else:
                self.logger.warning(f"  No dataset file found in {chunk_dir}")
        
        if not dfs:
            self.logger.error(f"No valid chunk data found for GPU {gpu_id}")
            return None
        
        # Concatenate all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Save merged file with timestamp
        timestamp = get_timestamp()
        output_file = Path(get_phase_dir(phase)) / f"dataset_gpu{gpu_id}_{timestamp}.parquet"
        merged_df.to_parquet(output_file, index=False)
        
        print(f"  Saved merged dataset: {output_file.name} ({len(merged_df)} records)")
        
        return output_file
    
    def monitor_and_merge(self, phase: str, checkpointing_enabled: bool = True):
        """
        Monitor running processes and merge chunks after completion.
        
        Args:
            phase: Phase number for merging
            checkpointing_enabled: Whether checkpointing was used
        """
        # First monitor until all processes complete
        self.monitor_processes()
        
        # If checkpointing was enabled, merge the chunks
        if checkpointing_enabled:
            print(f"\n{'='*60}")
            print("MERGING CHECKPOINT CHUNKS")
            print(f"{'='*60}")
            
            merged_files = []
            for gpu_id, _, _, _ in self.processes:
                merged_file = self.merge_checkpoint_chunks(phase, gpu_id)
                if merged_file:
                    merged_files.append(merged_file)
                else:
                    print(f"WARNING: Failed to merge chunks for GPU {gpu_id}")
            
            if merged_files:
                print(f"\nSuccessfully created {len(merged_files)} merged datasets:")
                for f in merged_files:
                    print(f"  - {f}")
            
            print(f"{'='*60}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Launch parallel dataset processing across multiple GPUs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--phase',
        type=float,
        choices=[1, 2.2, 3.5],
        default=1,
        help='Phase to run (1=Dataset Building, 2.2=Pile Caching, 3.5=Temperature Robustness)'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        help='Comma-separated list of GPU indices (e.g., "0,1,2"). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--start',
        type=int,
        help='Starting index for dataset (required for Phase 1)'
    )
    parser.add_argument(
        '--end',
        type=int,
        help='Ending index for dataset (required for Phase 1)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemma-2-2b',
        help='Model to use for generation'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        help='Directory for dataset outputs (default depends on phase)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Run cleanup before processing'
    )
    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Disable checkpointing (run entire range at once without chunking)'
    )
    
    args = parser.parse_args()
    
    # Parse GPU list
    gpus = None
    if args.gpus:
        gpus = [int(g.strip()) for g in args.gpus.split(',')]
    
    # Create launcher
    launcher = MultiGPULauncher(gpus=gpus)
    
    # Build extra arguments
    extra_args = []
    if args.cleanup:
        extra_args.append('--cleanup')
    
    # Phase-specific handling
    if args.phase == 1:
        # Phase 1 requires start/end indices
        if args.start is None or args.end is None:
            parser.error("Phase 1 requires --start and --end arguments")
        
        # Set default dataset directory if not specified
        if args.dataset_dir is None:
            args.dataset_dir = get_phase_dir("1")
        
        # Launch processes
        launcher.launch_phase1_processes(
            start_idx=args.start,
            end_idx=args.end,
            model=args.model,
            dataset_dir=args.dataset_dir,
            extra_args=extra_args,
            no_checkpoint=args.no_checkpoint
        )
        
        # Monitor and potentially merge
        launcher.monitor_and_merge(phase="1", checkpointing_enabled=not args.no_checkpoint)
    
    elif args.phase == 2.2:
        # Phase 2.2 pile caching
        if args.start is None:
            args.start = 0
        if args.end is None:
            args.end = 9999  # Default to 10,000 samples (0-9999)
        
        # Handle --run-count if provided
        if '--run-count' in extra_args:
            idx = extra_args.index('--run-count')
            if idx + 1 < len(extra_args):
                run_count = int(extra_args[idx + 1])
                args.end = min(args.end, run_count - 1)
        
        # Launch Phase 2.2 processes
        launcher.launch_phase2_2_processes(
            start_idx=args.start,
            end_idx=args.end,
            model=args.model,
            extra_args=extra_args,
            no_checkpoint=args.no_checkpoint
        )
        
        # Monitor (no merging needed for Phase 2.2)
        launcher.monitor_and_merge(phase="2.2", checkpointing_enabled=False)
    
    elif args.phase == 3.5:
        # Phase 3.5 will auto-detect dataset size if end is None
        if args.start is None:
            args.start = 0
        
        # Launch Phase 3.5 processes
        launcher.launch_phase3_5_processes(
            start_idx=args.start,
            end_idx=args.end,
            model=args.model,
            extra_args=extra_args,
            no_checkpoint=args.no_checkpoint
        )
        
        # Monitor and potentially merge
        launcher.monitor_and_merge(phase="3.5", checkpointing_enabled=not args.no_checkpoint)


if __name__ == "__main__":
    main()