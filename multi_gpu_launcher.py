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
from subprocess import Popen, STDOUT, TimeoutExpired
from os import environ, setsid, killpg, getpgid
import time
from signal import signal as signal_register, SIGINT, SIGTERM, SIGKILL
import sys
from typing import List, Tuple
import math
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common.logging import LoggingManager
from common.utils import managed_subprocess
from common.utils import get_phase_dir


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
    
    def launch_phase1_processes(self, 
                               start_idx: int, 
                               end_idx: int,
                               model: str,
                               dataset_dir: str = "data/datasets",
                               extra_args: List[str] = None):
        """
        Launch Phase 1 parallel processes on different GPUs
        
        Args:
            start_idx: Starting index
            end_idx: Ending index
            model: Model name to use
            dataset_dir: Directory for datasets
            extra_args: Additional arguments to pass
        """
        # Setup logging
        logging_manager = LoggingManager(log_dir="data/logs")
        self.logger = logging_manager.setup_logging("multi_gpu_launcher")
        
        # Split workload
        splits = self.split_workload(start_idx, end_idx)
        
        print(f"\n{'='*60}")
        print(f"MULTI-GPU PARALLEL PROCESSING")
        print(f"{'='*60}")
        print(f"Total GPUs: {len(self.gpus)}")
        print(f"Total range: {start_idx}-{end_idx} ({end_idx - start_idx + 1} items)")
        print(f"Model: {model}")
        print(f"\nWorkload distribution:")
        
        for i, (gpu_id, (split_start, split_end)) in enumerate(zip(self.gpus, splits)):
            items = split_end - split_start + 1
            print(f"  GPU {gpu_id}: {split_start}-{split_end} ({items} items)")
        
        print(f"\n{'='*60}")
        print("Starting processes...\n")
        
        # Launch processes
        for gpu_id, (split_start, split_end) in zip(self.gpus, splits):
            cmd = [
                "python3", "run.py", "phase", "1",
                "--start", str(split_start),
                "--end", str(split_end),
                "--model", model,
                "--dataset-dir", dataset_dir
            ]
            
            # Add extra arguments
            if extra_args:
                cmd.extend(extra_args)
            
            # Set environment for GPU
            env = environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Create log file for this GPU
            from common.utils import get_readable_timestamp
            timestamp = get_readable_timestamp()
            log_file = f"data/logs/gpu_{gpu_id}_{model.split('/')[-1]}_{split_start}-{split_end}_{timestamp}.log"
            
            print(f"Launching on GPU {gpu_id}: {split_start}-{split_end}")
            self.logger.info(f"Launching process on GPU {gpu_id}: {' '.join(cmd)}")
            
            # Launch process with context manager
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
        print(f"Check individual logs in data/logs/gpu_*_output.log")
        print(f"\nPress Ctrl+C to stop all processes\n")
    
    def launch_phase12_processes(self, 
                                model: str,
                                extra_args: List[str] = None):
        """
        Launch Phase 1.2 temperature variation processes on different GPUs
        
        For Phase 1.2, we load validation task IDs from Phase 0.1 and split
        them across GPUs for parallel temperature generation.
        
        Args:
            model: Model name to use
            extra_args: Additional arguments to pass
        """
        # Setup logging
        logging_manager = LoggingManager(log_dir="data/logs")
        self.logger = logging_manager.setup_logging("multi_gpu_launcher")
        
        # Load validation indices to determine workload
        import json
        validation_task_ids_file = Path("data/phase0_1/validation_task_ids.json")
        
        if not validation_task_ids_file.exists():
            raise FileNotFoundError(
                f"Validation task IDs not found at {validation_task_ids_file}. "
                "Please run Phase 0.1 first."
            )
        
        with open(validation_task_ids_file, 'r') as f:
            validation_task_ids = json.load(f)
        
        total_tasks = len(validation_task_ids)
        
        print(f"\n{'='*60}")
        print(f"MULTI-GPU PHASE 1.2 TEMPERATURE GENERATION")
        print(f"{'='*60}")
        print(f"Total GPUs: {len(self.gpus)}")
        print(f"Total validation tasks: {total_tasks}")
        print(f"Model: {model}")
        
        # Split validation task IDs across GPUs
        tasks_per_gpu = math.ceil(total_tasks / len(self.gpus))
        
        print(f"\nWorkload distribution:")
        
        # Launch processes
        for i, gpu_id in enumerate(self.gpus):
            # Calculate indices slice for this GPU
            start_idx = i * tasks_per_gpu
            end_idx = min((i + 1) * tasks_per_gpu, total_tasks)
            
            if start_idx >= total_tasks:
                break
                
            gpu_task_count = end_idx - start_idx
            print(f"  GPU {gpu_id}: {gpu_task_count} tasks (indices {start_idx}-{end_idx-1})")
            
            cmd = [
                "python3", "run.py", "phase", "1.2",
                "--model", model,
                # Pass the index range via environment variable since Phase 1.2
                # doesn't have --start/--end arguments
            ]
            
            # Add extra arguments
            if extra_args:
                cmd.extend(extra_args)
            
            # Set environment for GPU and task range
            env = environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["TASK_START_IDX"] = str(start_idx)
            env["TASK_END_IDX"] = str(end_idx)
            
            # Create log file for this GPU
            from common.utils import get_readable_timestamp
            timestamp = get_readable_timestamp()
            log_file = f"data/logs/gpu_{gpu_id}_phase1.2_{timestamp}.log"
            
            # Launch process
            log_handle = open(log_file, 'w')
            process = Popen(
                cmd,
                stdout=log_handle,
                stderr=STDOUT,
                env=env,
                preexec_fn=setsid  # Create new process group
            )
            self.processes.append((gpu_id, process, log_file, log_handle))
        
        print(f"\n{'='*60}")
        print("Starting Phase 1.2 processes...\n")
        print(f"\nAll processes launched. Monitoring progress...")
        print(f"Check individual logs in data/logs/gpu_*_phase1.2_*.log")
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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Launch parallel dataset processing across multiple GPUs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--phase',
        type=float,
        choices=[1, 1.2],
        default=1,
        help='Phase to run (1=Dataset Building, 1.2=Temperature Generation)'
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
            extra_args=extra_args
        )
    
    elif args.phase == 1.2:
        # Phase 1.2 uses validation task IDs from Phase 0.1
        # No start/end required
        if args.start is not None or args.end is not None:
            print("Warning: Phase 1.2 ignores --start and --end arguments")
        
        # Launch Phase 1.2 processes
        launcher.launch_phase12_processes(
            model=args.model,
            extra_args=extra_args
        )
    
    # Monitor until completion
    launcher.monitor_processes()


if __name__ == "__main__":
    main()