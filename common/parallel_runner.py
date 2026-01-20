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
import json
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
from common.phase_discovery import write_phase_output


logger = get_logger(__name__)


# One-shot parallelization: distribute problems, merge at end
# These phases iterate over tasks and can split work across GPUs
DATA_PARALLEL_PHASES = {
    "1",     # Code generation + activation extraction
    "3.6",   # Hyperparameter baseline
    "4.8",   # Steering effect analysis (fixed coefficient)
    "4.12",  # Zero-disc steering
    "5.3",   # Weight orthogonalization
    "5.6",   # Zero-disc orthogonalization
    "7.3",   # Instruct baseline
    "7.6",   # Instruct steering
    "8.3",   # Selective steering (fixed threshold)
}

# Iterative parallelization: distribute problems, merge after each value
# Used for grid search phases where early stopping needs full data
ITERATIVE_PARALLEL_PHASES = {
    "3.5",   # Temperature robustness (iterate over temperatures)
    "4.5",   # Coefficient grid search (iterate over coefficients)
    "4.6",   # Golden section (iterate over refinement points)
    "8.2",   # Threshold optimizer (iterate over percentiles)
}

# Backward compatibility - all phases that support parallelization
PARALLELIZABLE_PHASES = DATA_PARALLEL_PHASES | ITERATIVE_PARALLEL_PHASES


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
        runner_obj = getattr(module, phase.runner)

        # Handle both class-based and function-based runners
        if phase.runner_type == "function":
            # Function-based: call directly with parameters
            result = runner_obj(config, gpu_id=gpu_id, n_gpus=n_gpus)
        else:
            # Class-based: instantiate then call .run()
            runner = runner_obj(config, gpu_id=gpu_id, n_gpus=n_gpus)
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
    2. Routes to appropriate parallelization strategy:
       - Data-parallel: distribute problems, merge at end
       - Iterative-parallel: distribute problems, merge after each value
    3. Merges results after all workers complete

    Args:
        phase_id: Phase to run
        config: Configuration object
        n_gpus: Number of GPUs to use

    Returns:
        Merged results from all workers
    """
    if phase_id not in PARALLELIZABLE_PHASES:
        raise ValueError(f"Phase {phase_id} does not support parallel execution")

    # Route iterative phases to the iterative parallel runner
    if phase_id in ITERATIVE_PARALLEL_PHASES:
        logger.info(f"Using iterative parallelization for Phase {phase_id}")
        from common.iterative_parallel_runner import run_iterative_parallel
        return run_iterative_parallel(phase_id, config, n_gpus)

    logger.info(f"Starting parallel execution for Phase {phase_id} with {n_gpus} GPUs")

    # Convert config to dict for pickling
    config_dict = asdict(config)

    # Get output directory for this phase
    from common.phase_discovery import get_phase_output_dir
    output_dir = get_phase_output_dir(phase_id, config)

    # Add _probe suffix for probe-based steering phases
    if phase_id in ("4.5", "4.6", "4.7", "4.8") and getattr(config, 'direction_source', 'sae') == 'probe_mass_mean':
        output_dir = str(Path(output_dir).parent / (Path(output_dir).name + "_probe"))

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


def _merge_phase4_5_json_results(
    output_path: Path,
    n_gpus: int,
    config: Config,
    phase_id: str
) -> dict:
    """
    Merge Phase 4.5/4.6 JSON results from parallel GPU workers.

    Phase 4.5 produces coefficient_analysis_gpu{N}.json files with structure:
    {
        "correct_steering": {
            "optimal_coefficient": float,
            "best_result": {...},
            "search_history": [{"coefficient": float, "metrics": {...}, "results": [...]}]
        },
        "incorrect_steering": {...}
    }

    This function merges results by:
    1. Combining per-problem results from all GPUs for each coefficient
    2. Recalculating metrics based on merged results
    3. Determining overall optimal coefficients
    """
    # Find per-GPU JSON files (Phase 4.5 uses coefficient_analysis, Phase 4.6 uses refinement_analysis)
    if phase_id == "4.6":
        json_files = sorted(output_path.glob("refinement_analysis_gpu*.json"))
        output_filename = "refinement_analysis.json"
        selected_filename = "refined_coefficients.json"
        selected_pattern = "refined_coefficients_gpu*.json"
    else:
        json_files = sorted(output_path.glob("coefficient_analysis_gpu*.json"))
        output_filename = "coefficient_analysis.json"
        selected_filename = "selected_coefficients.json"
        selected_pattern = "selected_coefficients_gpu*.json"

    if not json_files:
        raise RuntimeError(
            f"No per-GPU coefficient_analysis files found in {output_path}. "
            f"Expected pattern: coefficient_analysis_gpu*.json"
        )

    logger.info(f"Found {len(json_files)} GPU JSON files to merge")

    # Load all per-GPU results
    gpu_results = []
    for json_file in json_files:
        with open(json_file) as f:
            gpu_results.append(json.load(f))
        logger.info(f"  Loaded {json_file.name}")

    # Phase 4.6 has different structure - pick best result from each GPU
    if phase_id == "4.6":
        merged = {}
        for steering_key in ['correct_steering', 'incorrect_steering']:
            steering_type = steering_key.replace('_steering', '')

            best_coefficient = None
            best_score = -1
            best_result = None

            for gpu_data in gpu_results:
                if steering_key not in gpu_data:
                    continue
                gpu_result = gpu_data[steering_key]
                score = gpu_result.get('best_score', 0)
                if score > best_score:
                    best_score = score
                    best_coefficient = gpu_result.get('optimal_coefficient')
                    best_result = gpu_result

            if best_result:
                merged[steering_key] = best_result
                logger.info(f"  {steering_key}: optimal_coefficient={best_coefficient}, "
                           f"best_score={best_score:.1f}%")
            else:
                logger.warning(f"No {steering_key} results found across GPUs")

        # Save and cleanup handled below
        merged_file = output_path / output_filename
        with open(merged_file, 'w') as f:
            json.dump(merged, f, indent=2)
        logger.info(f"Saved merged analysis: {merged_file}")

        selected_files = sorted(output_path.glob(selected_pattern))
        if selected_files:
            with open(selected_files[0]) as f:
                selected = json.load(f)
            for steering_type in ['correct', 'incorrect']:
                steering_key = f'{steering_type}_steering'
                if steering_key in merged and steering_type in selected:
                    selected[steering_type]['coefficient'] = merged[steering_key].get('optimal_coefficient')
            selected_merged_file = output_path / selected_filename
            with open(selected_merged_file, 'w') as f:
                json.dump(selected, f, indent=2)
            logger.info(f"Saved merged coefficients: {selected_merged_file}")

        write_phase_output(phase=phase_id, outputs={"primary": output_filename},
                          config=config, output_dir=str(output_path))
        logger.info("Wrote phase_output.json manifest")

        for f in json_files:
            f.unlink()
            logger.info(f"  Cleaned up {f.name}")
        for f in selected_files:
            f.unlink()
            logger.info(f"  Cleaned up {f.name}")

        return {'merged_file': str(merged_file), 'steering_types': list(merged.keys()), 'n_gpus': n_gpus}

    # Phase 4.5 merging - combine per-problem results across GPUs
    merged = {}
    for steering_key in ['correct_steering', 'incorrect_steering']:
        steering_type = steering_key.replace('_steering', '')

        # Collect search histories from all GPUs
        all_histories = []
        for gpu_data in gpu_results:
            if steering_key in gpu_data and gpu_data[steering_key].get('search_history'):
                all_histories.extend(gpu_data[steering_key]['search_history'])

        if not all_histories:
            logger.warning(f"No {steering_key} results found across GPUs")
            continue

        # Group by coefficient
        coeff_results = {}
        for hist in all_histories:
            coeff = hist['coefficient']
            if coeff not in coeff_results:
                coeff_results[coeff] = {
                    'coefficient': coeff,
                    'steering_type': steering_type,
                    'all_results': [],
                    'metrics': {}
                }
            # Extend with this GPU's problem results
            if 'results' in hist:
                coeff_results[coeff]['all_results'].extend(hist['results'])

        # Recalculate metrics for each coefficient
        merged_history = []
        best_coefficient = None
        best_score = -1
        best_result = None

        for coeff, data in sorted(coeff_results.items()):
            results = data['all_results']
            n_problems = len(results)

            if n_problems == 0:
                continue

            # Calculate metrics based on steering type
            if steering_type == 'correct':
                # Correction rate: incorrect baseline → correct steered
                corrections = sum(1 for r in results
                                 if not r.get('baseline_passed', True) and r.get('steered_correct', False))
                incorrect_baseline = sum(1 for r in results if not r.get('baseline_passed', True))
                correction_rate = (corrections / incorrect_baseline * 100) if incorrect_baseline > 0 else 0

                metrics = {'correction_rate': correction_rate}
                score = correction_rate
            else:
                # Corruption rate: correct baseline → incorrect steered
                corruptions = sum(1 for r in results
                                 if r.get('baseline_passed', False) and not r.get('steered_correct', True))
                correct_baseline = sum(1 for r in results if r.get('baseline_passed', False))
                corruption_rate = (corruptions / correct_baseline * 100) if correct_baseline > 0 else 0

                # Average similarity
                similarities = [r.get('code_similarity', 0) for r in results if 'code_similarity' in r]
                avg_similarity = (sum(similarities) / len(similarities) * 100) if similarities else 0

                # Composite score (same formula as Phase 4.5)
                composite_score = corruption_rate * 0.5 + avg_similarity * 0.5

                metrics = {
                    'corruption_rate': corruption_rate,
                    'avg_similarity': avg_similarity,
                    'composite_score': composite_score
                }
                score = composite_score

            # Calculate divergence metrics
            similarities = [r.get('code_similarity', 0) for r in results if 'code_similarity' in r]
            mean_similarity = sum(similarities) / len(similarities) if similarities else 0

            hist_entry = {
                'coefficient': coeff,
                'steering_type': steering_type,
                'metrics': metrics,
                'divergence': {'mean_code_similarity': mean_similarity},
                'n_problems': n_problems,
                'results': results
            }
            merged_history.append(hist_entry)

            # Track best
            if score > best_score:
                best_score = score
                best_coefficient = coeff
                best_result = hist_entry

        merged[steering_key] = {
            'optimal_coefficient': best_coefficient,
            'best_result': best_result,
            'search_history': merged_history
        }

        logger.info(f"  {steering_key}: optimal_coefficient={best_coefficient}, "
                   f"best_score={best_score:.1f}%, n_coefficients={len(merged_history)}")

    # Save merged analysis file
    merged_file = output_path / output_filename
    with open(merged_file, 'w') as f:
        json.dump(merged, f, indent=2)
    logger.info(f"Saved merged analysis: {merged_file}")

    # Also merge and save selected/refined coefficients
    selected_files = sorted(output_path.glob(selected_pattern))
    if selected_files:
        # Use the first GPU's selected coefficients as base, update with merged optimal
        with open(selected_files[0]) as f:
            selected = json.load(f)

        # Update with merged optimal coefficients
        for steering_type in ['correct', 'incorrect']:
            steering_key = f'{steering_type}_steering'
            if steering_key in merged and steering_type in selected:
                selected[steering_type]['coefficient'] = merged[steering_key]['optimal_coefficient']
                if merged[steering_key]['best_result']:
                    selected[steering_type]['metrics'] = merged[steering_key]['best_result']['metrics']

        selected_merged_file = output_path / selected_filename
        with open(selected_merged_file, 'w') as f:
            json.dump(selected, f, indent=2)
        logger.info(f"Saved merged coefficients: {selected_merged_file}")

    # Write phase_output.json manifest
    write_phase_output(
        phase=phase_id,
        outputs={"primary": output_filename},
        config=config,
        output_dir=str(output_path)
    )
    logger.info("Wrote phase_output.json manifest")

    # Clean up per-GPU files
    for f in json_files:
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")
    for f in selected_files:
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")

    return {
        'merged_file': str(merged_file),
        'steering_types': list(merged.keys()),
        'n_gpus': n_gpus
    }


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

    # Phase 4.5/4.6 use JSON output format, not parquet
    if phase_id in ("4.5", "4.6"):
        return _merge_phase4_5_json_results(output_path, n_gpus, config, phase_id)

    # Find per-GPU result files
    # Phase 3.5 uses a different pattern for temperature experiments
    if phase_id == "3.5":
        gpu_result_files = sorted(output_path.glob("results_gpu*_temp_*.parquet"))
    else:
        gpu_result_files = sorted(output_path.glob("results_gpu*.parquet"))

    if not gpu_result_files:
        raise RuntimeError(
            f"No per-GPU result files found in {output_dir}. "
            f"Expected pattern: results_gpu*.parquet. "
            f"Check worker logs for errors."
        )

    if len(gpu_result_files) < n_gpus:
        logger.warning(
            f"Only found {len(gpu_result_files)}/{n_gpus} GPU result files. "
            f"Some workers may have failed."
        )

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

    # Write phase_output.json manifest for downstream discovery
    write_phase_output(
        phase=phase_id,
        outputs={"primary": merged_file.name},
        config=config,
        output_dir=str(output_path)
    )
    logger.info("Wrote phase_output.json manifest")

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
