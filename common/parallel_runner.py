"""
Multi-GPU parallel execution for SAE-Code-Correctness phases.

This module provides data parallelism by splitting tasks across multiple GPUs.
Each GPU loads its own model and processes a subset of tasks.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    run.py (orchestrator)                     │
    │  --parallel 4                                               │
    └─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ GPU 0   │     │ GPU 1   │     │ GPU 2   │ ...
        │ Worker  │     │ Worker  │     │ Worker  │
        │ Tasks   │     │ Tasks   │     │ Tasks   │
        │ 0,4,8.. │     │ 1,5,9.. │     │ 2,6,10..│
        └────┬────┘     └────┬────┘     └────┬────┘
              │               │               │
              ▼               ▼               ▼
        results_0.parquet  results_1.parquet  results_2.parquet
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌─────────────────┐
                    │  Merge Results  │
                    │ final.parquet   │
                    └─────────────────┘

Usage:
    python3 run.py phase 1 --parallel 4 --start 0 --end 40
"""

import os
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import shutil

import pandas as pd
import torch

from common.logging import get_logger
from common.config import Config
from common.utils import get_timestamp


logger = get_logger(__name__)


# Phases that support multi-GPU parallelization via task_subset
# These phases iterate over tasks and can split work across GPUs
PARALLELIZABLE_PHASES = {
    "1",     # Code generation + activation extraction
    "3.5",   # Temperature robustness
    "3.6",   # Hyperparameter baseline
    "4.5",   # Coefficient grid search
    "4.6",   # Golden section refinement
    "4.8",   # Steering effect analysis
    "4.12",  # Zero-disc steering
    "5.3",   # Weight orthogonalization
    "5.6",   # Zero-disc orthogonalization
    "7.3",   # Instruct baseline
    "7.6",   # Instruct steering
    "8.2",   # Threshold optimizer
    "8.3",   # Selective steering
}


def _get_gpu_task_indices(total_tasks: int, n_gpus: int, gpu_id: int) -> list[int]:
    """
    Get task indices for a specific GPU using round-robin distribution.

    Args:
        total_tasks: Total number of tasks
        n_gpus: Number of GPUs
        gpu_id: This GPU's ID (0-indexed)

    Returns:
        List of task indices for this GPU
    """
    return [i for i in range(total_tasks) if i % n_gpus == gpu_id]


def _worker_phase(
    gpu_id: int,
    phase_id: str,
    config_dict: dict,
    n_gpus: int,
    output_dir: str
) -> dict:
    """
    Worker function that runs on a single GPU.

    This function:
    1. Sets CUDA_VISIBLE_DEVICES to use only the assigned GPU
    2. Creates a config with task_subset filter
    3. Runs the phase
    4. Saves results to a GPU-specific file

    Args:
        gpu_id: GPU index (0-indexed)
        phase_id: Phase to run (e.g., "1", "4.5")
        config_dict: Config as dictionary (for pickling)
        n_gpus: Total number of GPUs
        output_dir: Directory to save GPU-specific results

    Returns:
        dict with status and result info
    """
    # Set CUDA device BEFORE importing torch or any phase modules
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Import phase modules after setting CUDA device
    import importlib
    from common.phase_registry import get_phase
    from common.config import Config
    from common.logging import get_logger

    worker_logger = get_logger(f"parallel_worker_{gpu_id}")
    worker_logger.info(f"Worker {gpu_id}: Starting on GPU {gpu_id}")

    try:
        # Recreate config from dict
        config = Config(**config_dict)

        # Get phase info
        phase = get_phase(phase_id)

        # Import phase module
        module = importlib.import_module(phase.module)
        runner_cls = getattr(module, phase.runner)

        # Create runner with gpu_id for task filtering
        runner = runner_cls(config, gpu_id=gpu_id, n_gpus=n_gpus)

        # Run phase
        result = runner.run()

        worker_logger.info(f"Worker {gpu_id}: Completed successfully")

        return {
            'gpu_id': gpu_id,
            'status': 'success',
            'result': result
        }

    except Exception as e:
        worker_logger.error(f"Worker {gpu_id}: Failed with error: {e}")
        import traceback
        return {
            'gpu_id': gpu_id,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def run_phase_parallel(phase_id: str, config: Config, n_gpus: int) -> dict:
    """
    Run a phase in parallel across multiple GPUs.

    This function:
    1. Validates the phase supports parallelization
    2. Spawns worker processes (one per GPU)
    3. Each worker processes its subset of tasks
    4. Merges results after all workers complete

    Args:
        phase_id: Phase to run
        config: Configuration object
        n_gpus: Number of GPUs to use

    Returns:
        Merged results from all workers
    """
    if phase_id not in PARALLELIZABLE_PHASES:
        raise ValueError(f"Phase {phase_id} does not support parallel execution")

    logger.info(f"Starting parallel execution for Phase {phase_id} with {n_gpus} GPUs")

    # Convert config to dict for pickling
    config_dict = asdict(config)

    # Get output directory for this phase
    from common.phase_discovery import get_phase_output_dir
    output_dir = get_phase_output_dir(phase_id, config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Use spawn context for CUDA compatibility
    ctx = mp.get_context('spawn')

    # Prepare worker arguments
    worker_args = [
        (gpu_id, phase_id, config_dict, n_gpus, str(output_dir))
        for gpu_id in range(n_gpus)
    ]

    # Run workers in parallel
    results = []
    failed_gpus = []

    with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx) as executor:
        futures = {
            executor.submit(_worker_phase, *args): args[0]
            for args in worker_args
        }

        for future in as_completed(futures):
            gpu_id = futures[future]
            try:
                worker_result = future.result()
                results.append(worker_result)

                if worker_result['status'] == 'success':
                    logger.info(f"GPU {gpu_id}: Completed successfully")
                else:
                    logger.error(f"GPU {gpu_id}: Failed - {worker_result.get('error', 'Unknown error')}")
                    failed_gpus.append(gpu_id)

            except Exception as e:
                logger.error(f"GPU {gpu_id}: Exception - {e}")
                failed_gpus.append(gpu_id)
                results.append({
                    'gpu_id': gpu_id,
                    'status': 'error',
                    'error': str(e)
                })

    # Check for failures
    if failed_gpus:
        logger.warning(f"Some GPUs failed: {failed_gpus}")

    # Merge results
    logger.info("Merging results from all GPUs...")
    merged_result = _merge_parallel_results(phase_id, output_dir, n_gpus, config)

    logger.info(f"Parallel execution complete for Phase {phase_id}")

    return merged_result


def _merge_parallel_results(
    phase_id: str,
    output_dir: str,
    n_gpus: int,
    config: Config
) -> dict:
    """
    Merge results from parallel GPU workers.

    For most phases, this merges parquet files.
    For phases with special result formats, handles them appropriately.

    Args:
        phase_id: Phase ID
        output_dir: Output directory containing per-GPU results
        n_gpus: Number of GPUs used
        config: Config object

    Returns:
        Merged result dict
    """
    output_path = Path(output_dir)

    # Find per-GPU result files
    gpu_result_files = sorted(output_path.glob("results_gpu*.parquet"))

    if not gpu_result_files:
        logger.warning("No per-GPU result files found to merge")
        return {}

    logger.info(f"Found {len(gpu_result_files)} GPU result files to merge")

    # Merge parquet files
    dfs = []
    for result_file in gpu_result_files:
        df = pd.read_parquet(result_file)
        dfs.append(df)
        logger.info(f"  Loaded {len(df)} rows from {result_file.name}")

    merged_df = pd.concat(dfs, ignore_index=True)

    # Sort by task_id if present
    if 'task_id' in merged_df.columns:
        merged_df = merged_df.sort_values('task_id').reset_index(drop=True)

    # Save merged result
    timestamp = get_timestamp()
    merged_file = output_path / f"dataset_merged_{timestamp}.parquet"
    merged_df.to_parquet(merged_file, index=False)
    logger.info(f"Saved merged dataset: {merged_file} ({len(merged_df)} rows)")

    # Clean up per-GPU files
    for result_file in gpu_result_files:
        result_file.unlink()
        logger.info(f"  Cleaned up {result_file.name}")

    return {
        'merged_file': str(merged_file),
        'total_rows': len(merged_df),
        'n_gpus': n_gpus
    }


def create_task_filter(gpu_id: int, n_gpus: int):
    """
    Create a filter function for task assignment.

    This returns a function that takes an index and returns True
    if the task at that index should be processed by this GPU.

    Args:
        gpu_id: This GPU's ID
        n_gpus: Total number of GPUs

    Returns:
        Filter function: (index) -> bool
    """
    def should_process(idx: int) -> bool:
        return idx % n_gpus == gpu_id
    return should_process


def filter_dataframe_for_gpu(df: pd.DataFrame, gpu_id: int, n_gpus: int) -> pd.DataFrame:
    """
    Filter a DataFrame to only include rows for this GPU.

    Uses round-robin assignment: GPU k processes indices where idx % n_gpus == k.

    Args:
        df: Input DataFrame
        gpu_id: This GPU's ID
        n_gpus: Total number of GPUs

    Returns:
        Filtered DataFrame
    """
    if n_gpus == 1:
        return df

    # Get indices for this GPU
    indices = [i for i in range(len(df)) if i % n_gpus == gpu_id]

    return df.iloc[indices].copy()
