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
    "2.2",   # Pile activation caching
    "3.6",   # Hyperparameter baseline
    "4.8",   # Steering effect analysis (fixed coefficient)
    "4.12",  # Zero-disc steering
    "5.3",   # Weight orthogonalization
    "5.6",   # Zero-disc orthogonalization
    "7.3",   # Instruct baseline
    "7.6",   # Instruct steering
    "8.3",   # Selective steering (outputs parquet in parallel mode)
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
    output_dir: str,
    eval_semaphore=None
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
        eval_semaphore: Optional shared semaphore for serializing code evaluations

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
    from common.dataset_utils import set_eval_semaphore

    worker_logger = get_logger(f"parallel_worker_{gpu_id}")
    worker_logger.info(f"Worker {gpu_id}: Starting on GPU {gpu_id}")

    # Set the evaluation semaphore for this worker process
    # All evaluate_code_with_error_type() calls will use this semaphore
    if eval_semaphore is not None:
        set_eval_semaphore(eval_semaphore)
        worker_logger.info(f"Worker {gpu_id}: Evaluation semaphore configured")

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
    direction_source = getattr(config, 'direction_source', 'sae')
    if phase_id in ("4.5", "4.6", "4.7", "4.8", "8.2", "8.3") and direction_source in ('probe_mass_mean', 'probe_logreg'):
        output_dir = str(Path(output_dir).parent / (Path(output_dir).name + "_probe"))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Use spawn context for CUDA compatibility
    ctx = mp.get_context('spawn')

    # Create a shared semaphore to prevent CPU contention during code evaluation.
    # When multiple GPU workers finish generation simultaneously and all try to
    # evaluate code, they compete for CPU resources, causing spurious timeouts.
    # This semaphore limits concurrent evaluations (2 = reasonable parallelism
    # while avoiding the 4+ concurrent evals that cause problems).
    manager = ctx.Manager()
    eval_semaphore = manager.Semaphore(2)
    logger.info("Created shared evaluation semaphore (max 2 concurrent evals)")

    # Prepare worker arguments
    worker_args = [
        (gpu_id, phase_id, config_dict, n_gpus, str(output_dir), eval_semaphore)
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

        logger.info(f"Submitted {len(futures)} worker tasks, waiting for completion...")
        completed_count = 0

        from concurrent.futures import TimeoutError as FuturesTimeoutError

        for future in as_completed(futures):
            gpu_id = futures[future]
            completed_count += 1

            # Check if subprocess crashed (future has exception instead of result)
            try:
                exc = future.exception(timeout=5)
            except (TimeoutError, FuturesTimeoutError):
                exc = TimeoutError("Timeout checking for exception")

            if exc is not None:
                logger.error(f"GPU {gpu_id}: Worker subprocess crashed - {exc}")
                failed_gpus.append(gpu_id)
                results.append({
                    'gpu_id': gpu_id,
                    'status': 'error',
                    'error': f"Subprocess crashed: {exc}"
                })
                continue

            try:
                worker_result = future.result(timeout=30)
                results.append(worker_result)

                if worker_result['status'] == 'success':
                    logger.info(f"GPU {gpu_id}: Completed successfully")
                else:
                    logger.error(f"GPU {gpu_id}: Failed - {worker_result.get('error', 'Unknown error')}")
                    if 'traceback' in worker_result:
                        logger.error(f"GPU {gpu_id} traceback:\n{worker_result['traceback']}")
                    failed_gpus.append(gpu_id)

            except (TimeoutError, FuturesTimeoutError) as e:
                logger.error(f"GPU {gpu_id}: Timeout waiting for result")
                failed_gpus.append(gpu_id)
                results.append({
                    'gpu_id': gpu_id,
                    'status': 'error',
                    'error': 'Timeout waiting for subprocess result'
                })
            except Exception as e:
                logger.error(f"GPU {gpu_id}: Exception - {type(e).__name__}: {e}")
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

    # Merge per-coefficient result files from all GPUs
    # Pattern: {type}_results_coeff_{N}_gpu{M}.json -> {type}_results_coeff_{N}.json
    from common.utils import save_json
    import re

    def dedupe_by_task_id(results_list):
        seen = set()
        deduped = []
        for r in results_list:
            tid = r.get('task_id')
            if tid not in seen:
                seen.add(tid)
                deduped.append(r)
        return deduped

    results_by_coefficient = {}  # {coefficient: {'correction': [], 'corruption': [], 'preservation': []}}
    per_coeff_files_to_cleanup = []

    # Find and merge all per-coefficient files from GPUs
    for result_type in ['correction', 'corruption', 'preservation']:
        pattern = f"{result_type}_results_coeff_*_gpu*.json"
        gpu_files = sorted(output_path.glob(pattern))

        for gpu_file in gpu_files:
            # Extract coefficient from filename (e.g., "correction_results_coeff_10_gpu0.json" -> "10")
            match = re.search(r'coeff_([0-9.]+)_gpu', gpu_file.name)
            if not match:
                continue
            coeff_str = match.group(1)
            coeff = int(coeff_str) if '.' not in coeff_str else float(coeff_str)

            if coeff not in results_by_coefficient:
                results_by_coefficient[coeff] = {'correction': [], 'corruption': [], 'preservation': []}

            # Load and merge results
            with open(gpu_file) as f:
                gpu_results = json.load(f)
                results_by_coefficient[coeff][result_type].extend(gpu_results)

            per_coeff_files_to_cleanup.append(gpu_file)

    # Save merged per-coefficient files and build aggregated lists
    all_correction_results = []
    all_corruption_results = []
    all_preservation_results = []

    for coeff, results in results_by_coefficient.items():
        coeff_str = f"coeff_{int(coeff)}" if coeff == int(coeff) else f"coeff_{coeff}"

        # Deduplicate each result type
        results['correction'] = dedupe_by_task_id(results['correction'])
        results['corruption'] = dedupe_by_task_id(results['corruption'])
        results['preservation'] = dedupe_by_task_id(results['preservation'])

        # Save per-coefficient files
        if results['correction']:
            save_json(results['correction'], output_path / f"correction_results_{coeff_str}.json")
            logger.info(f"Saved {len(results['correction'])} correction results to correction_results_{coeff_str}.json")
            all_correction_results.extend(results['correction'])
        if results['corruption']:
            save_json(results['corruption'], output_path / f"corruption_results_{coeff_str}.json")
            logger.info(f"Saved {len(results['corruption'])} corruption results to corruption_results_{coeff_str}.json")
            all_corruption_results.extend(results['corruption'])
        if results['preservation']:
            save_json(results['preservation'], output_path / f"preservation_results_{coeff_str}.json")
            logger.info(f"Saved {len(results['preservation'])} preservation results to preservation_results_{coeff_str}.json")
            all_preservation_results.extend(results['preservation'])

    # Save aggregated result files for backward compatibility
    if all_correction_results:
        save_json(all_correction_results, output_path / "all_correction_results.json")
        logger.info(f"Saved {len(all_correction_results)} total correction results to all_correction_results.json")
    if all_corruption_results:
        save_json(all_corruption_results, output_path / "all_corruption_results.json")
        logger.info(f"Saved {len(all_corruption_results)} total corruption results to all_corruption_results.json")
    if all_preservation_results:
        save_json(all_preservation_results, output_path / "all_preservation_results.json")
        logger.info(f"Saved {len(all_preservation_results)} total preservation results to all_preservation_results.json")

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

    # Write phase_output.json manifest with all output files
    outputs_dict = {"primary": output_filename}
    if all_correction_results:
        outputs_dict["correction_results"] = "all_correction_results.json"
    if all_corruption_results:
        outputs_dict["corruption_results"] = "all_corruption_results.json"
    if all_preservation_results:
        outputs_dict["preservation_results"] = "all_preservation_results.json"
    # Add per-coefficient result files to manifest
    for coeff in results_by_coefficient.keys():
        coeff_str = f"coeff_{int(coeff)}" if coeff == int(coeff) else f"coeff_{coeff}"
        if results_by_coefficient[coeff]['correction']:
            outputs_dict[f"correction_results_{coeff_str}"] = f"correction_results_{coeff_str}.json"
        if results_by_coefficient[coeff]['corruption']:
            outputs_dict[f"corruption_results_{coeff_str}"] = f"corruption_results_{coeff_str}.json"
        if results_by_coefficient[coeff]['preservation']:
            outputs_dict[f"preservation_results_{coeff_str}"] = f"preservation_results_{coeff_str}.json"

    write_phase_output(
        phase=phase_id,
        outputs=outputs_dict,
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

    # Also clean up per-GPU result files (aggregated + per-coefficient)
    for pattern in ["all_correction_results_gpu*.json", "all_corruption_results_gpu*.json", "all_preservation_results_gpu*.json"]:
        for f in output_path.glob(pattern):
            f.unlink()
            logger.info(f"  Cleaned up {f.name}")
    # Clean up per-coefficient per-GPU files
    for f in per_coeff_files_to_cleanup:
        if f.exists():
            f.unlink()
            logger.info(f"  Cleaned up {f.name}")

    return {
        'merged_file': str(merged_file),
        'steering_types': list(merged.keys()),
        'n_gpus': n_gpus
    }


def _merge_phase5_6_json_results(
    output_path: Path,
    n_gpus: int,
    config: Config
) -> dict:
    """
    Merge Phase 5.6 zero-disc orthogonalization results from parallel workers.

    Phase 5.6 produces zero_disc_orthogonalization_results_gpu{N}.json files with:
    - config: metadata
    - zero_disc_orthogonalization:
      - latent: the zero-disc latent info (same for all GPUs)
      - weight_changes: how weights changed (same for all GPUs)
      - metrics: correction/preservation/corruption rates (recalculated)
      - incorrect_results: list of per-problem results
      - correct_results: {corrected, preserved, corrupted} lists

    This function merges by combining result lists and recalculating metrics.
    """
    from datetime import datetime
    from common.utils import save_json
    import numpy as np

    # Find per-GPU JSON files
    json_files = sorted(output_path.glob("zero_disc_orthogonalization_results_gpu*.json"))
    if not json_files:
        raise RuntimeError(
            f"No zero_disc_orthogonalization_results_gpu*.json files found in {output_path}"
        )

    logger.info(f"Found {len(json_files)} GPU JSON files to merge for Phase 5.6")

    # Load all per-GPU results
    gpu_results = []
    for json_file in json_files:
        with open(json_file) as f:
            gpu_results.append(json.load(f))
        logger.info(f"  Loaded {json_file.name}")

    # Use first GPU's latent and weight_changes (same for all)
    base_result = gpu_results[0]
    latent_info = base_result['zero_disc_orthogonalization']['latent']
    weight_changes = base_result['zero_disc_orthogonalization']['weight_changes']

    # Merge result lists from all GPUs
    all_incorrect_results = []
    all_corrected = []
    all_preserved = []
    all_corrupted = []

    for gpu_data in gpu_results:
        ortho = gpu_data['zero_disc_orthogonalization']

        # Merge incorrect results
        if 'incorrect_results' in ortho:
            all_incorrect_results.extend(ortho['incorrect_results'])

        # Merge correct results
        if 'correct_results' in ortho:
            cr = ortho['correct_results']
            all_corrected.extend(cr.get('corrected', []))
            all_preserved.extend(cr.get('preserved', []))
            all_corrupted.extend(cr.get('corrupted', []))

    # Deduplicate by task_id
    def dedupe_by_task_id(results_list):
        seen = set()
        deduped = []
        for r in results_list:
            tid = r.get('task_id')
            if tid not in seen:
                seen.add(tid)
                deduped.append(r)
        return deduped

    all_incorrect_results = dedupe_by_task_id(all_incorrect_results)
    all_corrected = dedupe_by_task_id(all_corrected)
    all_preserved = dedupe_by_task_id(all_preserved)
    all_corrupted = dedupe_by_task_id(all_corrupted)

    # Recalculate metrics from merged data
    n_incorrect = len(all_incorrect_results)
    n_corrected = sum(1 for r in all_incorrect_results if r.get('orthogonalized_correct', False))
    n_correct = len(all_preserved) + len(all_corrupted)
    n_preserved = len(all_preserved)
    n_corrupted = len(all_corrupted)

    correction_rate = (n_corrected / n_incorrect * 100) if n_incorrect > 0 else 0.0
    preservation_rate = (n_preserved / n_correct * 100) if n_correct > 0 else 0.0
    corruption_rate = (n_corrupted / n_correct * 100) if n_correct > 0 else 0.0

    # Calculate average similarity from preserved results
    similarity_scores = [r.get('similarity', 1.0) for r in all_preserved]
    avg_similarity = float(np.mean(similarity_scores)) if similarity_scores else 0.0

    logger.info(f"  Merged: {n_incorrect} incorrect, {n_correct} correct results")
    logger.info(f"  Correction: {correction_rate:.1f}%, Preservation: {preservation_rate:.1f}%, "
               f"Corruption: {corruption_rate:.1f}%")

    # Build merged result
    merged = {
        'timestamp': datetime.now().isoformat(),
        'parallel_merge': True,
        'n_gpus': n_gpus,
        'config': {
            'model': config.model_name,
            'target_weights': config.orthogonalization_target_weights,
            'n_validation_problems': n_incorrect + n_correct,
            'n_correct_baseline': n_correct,
            'n_incorrect_baseline': n_incorrect
        },
        'zero_disc_orthogonalization': {
            'latent': latent_info,
            'weight_changes': weight_changes,
            'metrics': {
                'correction_rate': correction_rate,
                'preservation_rate': preservation_rate,
                'corruption_rate': corruption_rate,
                'avg_similarity_score': avg_similarity,
                'n_incorrect_baseline': n_incorrect,
                'n_corrected': n_corrected,
                'n_correct_baseline': n_correct,
                'n_preserved': n_preserved,
                'n_corrupted': n_corrupted
            },
            'incorrect_results': all_incorrect_results,
            'correct_results': {
                'corrected': all_corrected,
                'preserved': all_preserved,
                'corrupted': all_corrupted
            }
        }
    }

    # Save merged result
    merged_file = output_path / "zero_disc_orthogonalization_results.json"
    save_json(merged, merged_file)
    logger.info(f"Saved merged results: {merged_file}")

    # Write phase_output.json manifest
    write_phase_output(
        phase="5.6",
        outputs={"primary": "zero_disc_orthogonalization_results.json"},
        config=config,
        output_dir=str(output_path)
    )
    logger.info("Wrote phase_output.json manifest")

    return {
        'merged_file': str(merged_file),
        'n_incorrect': n_incorrect,
        'n_correct': n_correct,
        'correction_rate': correction_rate,
        'preservation_rate': preservation_rate,
        'n_gpus': n_gpus
    }


def _merge_phase8_3_results(
    output_path: Path,
    n_gpus: int,
    config: Config
) -> dict:
    """
    Merge Phase 8.3 selective steering results from parallel workers.

    Phase 8.3 produces results_gpu{N}.parquet files with columns:
    - task_id, baseline_passed, steered, incorrect_pred_activation
    - steered_correct, steered_code, baseline_code, source, experiment_type

    This function:
    1. Loads all per-GPU parquet files
    2. Splits by experiment_type (correction vs preservation)
    3. Recalculates metrics from merged data
    4. Saves JSON outputs and merged parquet
    """
    from datetime import datetime
    from common.utils import save_json

    # Find per-GPU parquet files
    gpu_files = sorted(output_path.glob("results_gpu*.parquet"))
    if not gpu_files:
        raise RuntimeError(f"No results_gpu*.parquet files found in {output_path}")

    logger.info(f"Found {len(gpu_files)} GPU parquet files to merge")

    # Load and merge
    dfs = [pd.read_parquet(f) for f in gpu_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged {len(merged_df)} total results from {len(gpu_files)} GPUs")

    # Deduplicate by task_id + experiment_type (handles cross-run checkpointing)
    if 'task_id' in merged_df.columns:
        before_dedup = len(merged_df)
        merged_df = merged_df.drop_duplicates(
            subset=['task_id', 'experiment_type'], keep='last'
        )
        if before_dedup != len(merged_df):
            logger.info(f"  Deduplicated: {before_dedup} -> {len(merged_df)} rows")

    # Split by experiment type
    correction_df = merged_df[merged_df['experiment_type'] == 'correction']
    preservation_df = merged_df[merged_df['experiment_type'] == 'preservation']

    # Convert to records (matching original JSON format)
    correction_results = correction_df.to_dict('records')
    preservation_results = preservation_df.to_dict('records')

    # === CALCULATE CORRECTION METRICS ===
    n_correction = len(correction_results)
    valid_correction = [r for r in correction_results if r.get('steered_correct') is not None]
    n_valid_correction = len(valid_correction)

    n_steered_correction = sum(1 for r in valid_correction if r.get('steered', False))
    n_not_steered_correction = n_valid_correction - n_steered_correction
    steering_trigger_rate = n_steered_correction / n_valid_correction if n_valid_correction > 0 else 0

    n_corrected = sum(1 for r in valid_correction
                     if not r.get('baseline_passed', True) and r.get('steered_correct', False))
    correction_rate = n_corrected / n_valid_correction if n_valid_correction > 0 else 0
    correction_efficiency = n_corrected / n_steered_correction if n_steered_correction > 0 else 0

    # Activation stats for correction
    correction_activations = [r.get('incorrect_pred_activation') for r in valid_correction
                             if r.get('incorrect_pred_activation') is not None]
    correction_activation_stats = {
        'mean': sum(correction_activations) / len(correction_activations) if correction_activations else None,
        'min': min(correction_activations) if correction_activations else None,
        'max': max(correction_activations) if correction_activations else None
    }

    correction_metrics = {
        'total_problems': n_correction,
        'valid_problems': n_valid_correction,
        'n_steered': n_steered_correction,
        'n_not_steered': n_not_steered_correction,
        'steering_trigger_rate': round(steering_trigger_rate, 4),
        'n_corrected': n_corrected,
        'correction_rate': round(correction_rate, 4),
        'correction_efficiency': round(correction_efficiency, 4),
        'activation_stats': correction_activation_stats
    }

    # === CALCULATE PRESERVATION METRICS ===
    n_preservation = len(preservation_results)
    valid_preservation = [r for r in preservation_results if r.get('steered_correct') is not None]
    n_valid_preservation = len(valid_preservation)

    n_steered_preservation = sum(1 for r in valid_preservation if r.get('steered', False))
    n_not_steered_preservation = n_valid_preservation - n_steered_preservation
    steering_avoidance_rate = n_not_steered_preservation / n_valid_preservation if n_valid_preservation > 0 else 0

    n_preserved = sum(1 for r in valid_preservation
                     if r.get('baseline_passed', False) and r.get('steered_correct', False))
    n_corrupted = sum(1 for r in valid_preservation
                     if r.get('baseline_passed', False) and not r.get('steered_correct', True))
    preservation_rate = n_preserved / n_valid_preservation if n_valid_preservation > 0 else 0
    corruption_rate = n_corrupted / n_valid_preservation if n_valid_preservation > 0 else 0

    # Activation stats for preservation
    preservation_activations = [r.get('incorrect_pred_activation') for r in valid_preservation
                               if r.get('incorrect_pred_activation') is not None]
    preservation_activation_stats = {
        'mean': sum(preservation_activations) / len(preservation_activations) if preservation_activations else None,
        'min': min(preservation_activations) if preservation_activations else None,
        'max': max(preservation_activations) if preservation_activations else None
    }

    preservation_metrics = {
        'total_problems': n_preservation,
        'valid_problems': n_valid_preservation,
        'n_steered': n_steered_preservation,
        'n_not_steered': n_not_steered_preservation,
        'steering_avoidance_rate': round(steering_avoidance_rate, 4),
        'n_preserved': n_preserved,
        'n_corrupted': n_corrupted,
        'preservation_rate': round(preservation_rate, 4),
        'corruption_rate': round(corruption_rate, 4),
        'activation_stats': preservation_activation_stats
    }

    # === COMBINED METRICS ===
    total_problems = n_valid_correction + n_valid_preservation
    total_steered = n_steered_correction + n_steered_preservation
    overall_steering_rate = total_steered / total_problems if total_problems > 0 else 0

    combined_metrics = {
        'total_problems': total_problems,
        'total_steered': total_steered,
        'overall_steering_rate': round(overall_steering_rate, 4)
    }

    # Save JSON files (same format as sequential mode)
    save_json(correction_results, output_path / "all_selective_correction_results.json")
    save_json(preservation_results, output_path / "all_selective_preservation_results.json")
    logger.info(f"Saved {len(correction_results)} correction and {len(preservation_results)} preservation results")

    # Build and save summary
    summary = {
        'phase': '8.3',
        'timestamp': datetime.now().isoformat(),
        'parallel_merge': True,
        'n_gpus': n_gpus,
        'correction_experiment': correction_metrics,
        'preservation_experiment': preservation_metrics,
        'combined_metrics': combined_metrics
    }
    save_json(summary, output_path / "selective_steering_summary.json")
    logger.info(f"Saved selective_steering_summary.json")

    # Save merged parquet
    timestamp = get_timestamp()
    merged_file = output_path / f"dataset_merged_{timestamp}.parquet"
    merged_df.to_parquet(merged_file, index=False)
    logger.info(f"Saved merged dataset: {merged_file}")

    # Write manifest
    write_phase_output(
        phase="8.3",
        outputs={
            "primary": "selective_steering_summary.json",
            "correction_results": "all_selective_correction_results.json",
            "preservation_results": "all_selective_preservation_results.json"
        },
        config=config,
        output_dir=str(output_path)
    )
    logger.info("Wrote phase_output.json manifest")

    # Cleanup per-GPU files
    for f in gpu_files:
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")

    # Print summary
    logger.info("="*60)
    logger.info("PHASE 8.3 PARALLEL MERGE COMPLETE")
    logger.info("="*60)
    logger.info(f"Correction: {n_corrected}/{n_valid_correction} ({correction_rate*100:.2f}%)")
    logger.info(f"Preservation: {n_preserved}/{n_valid_preservation} ({preservation_rate*100:.2f}%)")
    logger.info(f"Corruption: {n_corrupted}/{n_valid_preservation} ({corruption_rate*100:.2f}%)")
    logger.info(f"Overall steering rate: {overall_steering_rate*100:.1f}%")

    return {
        'merged_file': str(merged_file),
        'total_results': len(merged_df),
        'correction_rate': correction_rate,
        'preservation_rate': preservation_rate,
        'corruption_rate': corruption_rate
    }


def _merge_phase4_8_results(
    output_path: Path,
    n_gpus: int,
    config: Config
) -> dict:
    """
    Merge Phase 4.8 steering effect analysis results from parallel workers.

    Phase 4.8 produces steering_effect_analysis_gpu{N}.json files containing:
    - correction_rate, corruption_rate, preservation_rate
    - detailed_results with correction/corruption/preservation lists
    - coefficients, n_problems, exclusion_summary

    This function:
    1. Loads all per-GPU JSON files
    2. Combines detailed_results across GPUs (dedup by task_id)
    3. Recalculates aggregate metrics from merged data
    4. Saves unified JSON outputs and phase manifest
    """
    from datetime import datetime
    from common.utils import save_json
    from common.steering_metrics import calculate_correction_rate, calculate_corruption_rate

    # Find per-GPU JSON files
    gpu_files = sorted(output_path.glob("steering_effect_analysis_gpu*.json"))
    if not gpu_files:
        raise RuntimeError(f"No steering_effect_analysis_gpu*.json files found in {output_path}")

    logger.info(f"Found {len(gpu_files)} GPU JSON files to merge")

    # Load all per-GPU results
    gpu_data = []
    for f in gpu_files:
        with open(f) as fh:
            import json
            gpu_data.append(json.load(fh))
        logger.info(f"  Loaded {f.name}")

    # Merge detailed_results across GPUs
    merged_correction = []
    merged_corruption = []
    merged_preservation = []

    for data in gpu_data:
        detailed = data.get('detailed_results', {})
        merged_correction.extend(detailed.get('correction', []))
        merged_corruption.extend(detailed.get('corruption', []))
        merged_preservation.extend(detailed.get('preservation', []))

    # Deduplicate by task_id within each experiment type
    def dedup_by_task_id(results: list) -> list:
        seen = set()
        deduped = []
        for r in results:
            tid = r.get('task_id')
            if tid not in seen:
                seen.add(tid)
                deduped.append(r)
        return deduped

    merged_correction = dedup_by_task_id(merged_correction)
    merged_corruption = dedup_by_task_id(merged_corruption)
    merged_preservation = dedup_by_task_id(merged_preservation)

    logger.info(f"Merged results: {len(merged_correction)} correction, "
                f"{len(merged_corruption)} corruption, {len(merged_preservation)} preservation")

    # Recalculate rates from merged data
    correction_rate = calculate_correction_rate(merged_correction)
    corruption_rate = calculate_corruption_rate(merged_corruption)

    # Preservation rate: correct→correct
    if merged_preservation:
        preserved = sum(1 for r in merged_preservation
                       if r.get('baseline_passed', False) and r.get('steered_correct', False))
        total_correct = sum(1 for r in merged_preservation
                          if r.get('baseline_passed', False))
        preservation_rate = (preserved / total_correct * 100) if total_correct > 0 else 0.0
    else:
        preservation_rate = 0.0

    # Use first GPU's metadata for coefficients, direction_source, etc.
    ref = gpu_data[0]

    # Sum up n_problems across GPUs
    total_initially_correct = sum(d.get('n_problems', {}).get('initially_correct', 0) for d in gpu_data)
    total_initially_incorrect = sum(d.get('n_problems', {}).get('initially_incorrect', 0) for d in gpu_data)
    total_problems = total_initially_correct + total_initially_incorrect

    # Detect multi-candidate mode (has 'correct' and 'incorrect' candidate lists)
    is_multi_candidate = 'correct' in ref and isinstance(ref.get('correct'), list)

    # Build merged metrics (same format as single-GPU output)
    merged_metrics = {
        'correction_rate': correction_rate,
        'corruption_rate': corruption_rate,
        'preservation_rate': preservation_rate,
        'direction_source': ref.get('direction_source', 'sae'),
        'coefficients': ref.get('coefficients', {}),
        'n_problems': {
            'initially_correct': total_initially_correct,
            'initially_incorrect': total_initially_incorrect,
            'total': total_problems
        },
        'parallel_merge': True,
        'n_gpus': n_gpus,
        'detailed_results': {
            'correction': merged_correction,
            'corruption': merged_corruption,
            'preservation': merged_preservation
        }
    }

    # Handle multi-candidate mode: merge candidate lists with updated counts
    if is_multi_candidate:
        # Merge correct candidates - sum n_total across GPUs, recalculate rates
        merged_correct_candidates = []
        for i, candidate in enumerate(ref.get('correct', [])):
            merged_candidate = candidate.copy()
            merged_candidate['n_total'] = sum(
                d.get('correct', [{}])[i].get('n_total', 0)
                for d in gpu_data if i < len(d.get('correct', []))
            )
            # Recalculate correction_rate from merged detailed_results
            # Note: detailed_results only contains best candidate's results
            merged_correct_candidates.append(merged_candidate)

        # Merge incorrect candidates
        merged_incorrect_candidates = []
        for i, candidate in enumerate(ref.get('incorrect', [])):
            merged_candidate = candidate.copy()
            merged_candidate['n_total'] = sum(
                d.get('incorrect', [{}])[i].get('n_total', 0)
                for d in gpu_data if i < len(d.get('incorrect', []))
            )
            merged_incorrect_candidates.append(merged_candidate)

        merged_metrics['correct'] = merged_correct_candidates
        merged_metrics['incorrect'] = merged_incorrect_candidates
        merged_metrics['best_candidates'] = {
            'correct': merged_correct_candidates[0] if merged_correct_candidates else None,
            'incorrect': merged_incorrect_candidates[0] if merged_incorrect_candidates else None,
        }
        logger.info(f"Multi-candidate mode: merged {len(merged_correct_candidates)} correct, "
                    f"{len(merged_incorrect_candidates)} incorrect candidates")

    # Save merged analysis JSON
    save_json(merged_metrics, output_path / "steering_effect_analysis.json")
    logger.info("Saved merged steering_effect_analysis.json")

    # Collect all steered results for error distribution
    all_steered_results = merged_correction + merged_corruption + merged_preservation

    # Compute error type distribution
    from common.dataset_utils import compute_error_type_distribution
    error_dist = compute_error_type_distribution(
        all_steered_results, 'steered_error_type'
    ) if all_steered_results else None

    # Build and save summary (same format as single-GPU phase_4_8_summary.json)
    summary = {
        'phase': '4.8',
        'description': 'Multi-Candidate Steering Effect Analysis' if is_multi_candidate else 'Steering Effect Analysis',
        'timestamp': datetime.now().isoformat(),
        'parallel_merge': True,
        'n_gpus': n_gpus,
        'mode': 'multi_candidate' if is_multi_candidate else 'probe',
        'config': {
            'model': ref.get('n_problems', {}).get('model', config.model_name if hasattr(config, 'model_name') else 'unknown'),
            'initially_correct_count': total_initially_correct,
            'initially_incorrect_count': total_initially_incorrect,
        },
        'results': {
            'correction_rate': correction_rate,
            'corruption_rate': corruption_rate,
            'preservation_rate': preservation_rate,
            'correct_candidates': len(merged_metrics.get('correct', [])) if is_multi_candidate else 0,
            'incorrect_candidates': len(merged_metrics.get('incorrect', [])) if is_multi_candidate else 0,
        },
        'steered_error_type_distribution': error_dist,
    }

    # Carry over latent/probe info from reference GPU
    if 'latents_used' in ref:
        summary['latents_used'] = ref['latents_used']
    if 'probe_info' in ref:
        summary['probe_info'] = ref['probe_info']

    save_json(summary, output_path / "phase_4_8_summary.json")
    logger.info("Saved merged phase_4_8_summary.json")

    # Save merged all_*_results.json (matching single-GPU format)
    save_json(merged_correction, output_path / "all_correction_results.json")
    save_json(merged_corruption, output_path / "all_corruption_results.json")
    save_json(merged_preservation, output_path / "all_preservation_results.json")
    logger.info("Saved merged all_*_results.json files")

    # Write phase manifest
    write_phase_output(
        phase="4.8",
        outputs={
            "primary": "phase_4_8_summary.json",
            "steering_analysis": "steering_effect_analysis.json",
            "correction_results": "all_correction_results.json",
            "corruption_results": "all_corruption_results.json",
            "preservation_results": "all_preservation_results.json",
        },
        config=config,
        output_dir=str(output_path)
    )
    logger.info("Wrote phase_output.json manifest")

    # Cleanup per-GPU files
    for f in gpu_files:
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")
    for f in sorted(output_path.glob("phase_4_8_summary_gpu*.json")):
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")

    # Print summary
    logger.info("=" * 60)
    logger.info("PHASE 4.8 PARALLEL MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Correction: {correction_rate:.1f}%")
    logger.info(f"Corruption: {corruption_rate:.1f}%")
    logger.info(f"Preservation: {preservation_rate:.1f}%")

    return {
        'correction_rate': correction_rate,
        'corruption_rate': corruption_rate,
        'preservation_rate': preservation_rate,
        'n_gpus': n_gpus
    }


def _merge_phase5_3_json_results(
    output_path: Path,
    n_gpus: int,
    config: Config
) -> dict:
    """
    Merge Phase 5.3 weight orthogonalization results from parallel workers.

    Phase 5.3 produces orthogonalization_results_gpu{N}.json files.
    """
    from datetime import datetime
    from common.utils import save_json

    # Find per-GPU JSON files
    gpu_files = sorted(output_path.glob("orthogonalization_results_gpu*.json"))
    if not gpu_files:
        raise RuntimeError(f"No orthogonalization_results_gpu*.json files found in {output_path}")

    logger.info(f"Found {len(gpu_files)} GPU JSON files to merge for Phase 5.3")

    # Load all per-GPU results
    gpu_results = []
    for f in gpu_files:
        with open(f) as fh:
            gpu_results.append(json.load(fh))
        logger.info(f"  Loaded {f.name}")

    # Use first GPU's structure as base
    merged = gpu_results[0].copy()

    # Merge examples from all GPUs
    for key in ['incorrect_orthogonalization', 'correct_orthogonalization']:
        if key not in merged:
            continue
        all_examples = {k: [] for k in merged[key].get('examples', {}).keys()}
        for gpu_data in gpu_results:
            if key in gpu_data and 'examples' in gpu_data[key]:
                for ex_key, ex_list in gpu_data[key]['examples'].items():
                    if ex_key in all_examples:
                        all_examples[ex_key].extend(ex_list)
        merged[key]['examples'] = all_examples

    # Recalculate metrics from merged examples
    # (simplified - just log the merge for now, metrics can be recomputed if needed)
    merged['parallel_merge'] = True
    merged['n_gpus'] = n_gpus

    # Save merged results
    save_json(merged, output_path / "orthogonalization_results.json")
    logger.info("Saved merged orthogonalization_results.json")

    # Merge and save summary
    summary_files = sorted(output_path.glob("phase_5_3_summary_gpu*.json"))
    if summary_files:
        with open(summary_files[0]) as f:
            summary = json.load(f)
        summary['parallel_merge'] = True
        summary['n_gpus'] = n_gpus
        save_json(summary, output_path / "phase_5_3_summary.json")
        logger.info("Saved merged phase_5_3_summary.json")

    # Write phase manifest
    write_phase_output(
        phase="5.3",
        outputs={
            "primary": "phase_5_3_summary.json",
            "orthogonalization_results": "orthogonalization_results.json",
        },
        config=config,
        output_dir=str(output_path)
    )

    # Cleanup per-GPU files
    for f in gpu_files:
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")
    for f in summary_files:
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")

    logger.info("PHASE 5.3 PARALLEL MERGE COMPLETE")
    return merged


def _merge_phase4_12_json_results(
    output_path: Path,
    n_gpus: int,
    config: Config
) -> dict:
    """
    Merge Phase 4.12 zero-discrimination steering results from parallel workers.

    Phase 4.12 produces zero_disc_steering_results_gpu{N}.json files containing:
    - correction_results, corruption_results, preservation_results (dict keyed by task_id)
    - summary_metrics with correction_rate, corruption_rate, preservation_rate
    """
    from datetime import datetime
    from common.utils import save_json
    from common.steering_metrics import calculate_correction_rate, calculate_corruption_rate
    from common.dataset_utils import compute_error_type_distribution

    # Find per-GPU JSON files
    gpu_files = sorted(output_path.glob("zero_disc_steering_results_gpu*.json"))
    if not gpu_files:
        raise RuntimeError(f"No zero_disc_steering_results_gpu*.json files found in {output_path}")

    logger.info(f"Found {len(gpu_files)} GPU JSON files to merge for Phase 4.12")

    # Load all per-GPU results
    gpu_data = []
    for f in gpu_files:
        with open(f) as fh:
            gpu_data.append(json.load(fh))
        logger.info(f"  Loaded {f.name}")

    # Load existing merged file to preserve previously completed features
    existing_merged_file = output_path / "zero_disc_steering_results.json"
    if existing_merged_file.exists():
        try:
            with open(existing_merged_file) as fh:
                existing = json.load(fh)
            merged_correction = existing.get('correction_results', {})
            merged_corruption = existing.get('corruption_results', {})
            merged_preservation = existing.get('preservation_results', {})
            merged_per_feature = existing.get('per_feature_results', {})
            logger.info(f"Loaded {len(merged_per_feature)} existing features from merged file")
        except Exception as e:
            logger.warning(f"Could not load existing merged file: {e}")
            merged_correction = {}
            merged_corruption = {}
            merged_preservation = {}
            merged_per_feature = {}
    else:
        merged_correction = {}
        merged_corruption = {}
        merged_preservation = {}
        merged_per_feature = {}

    # Merge GPU results on top (Phase 4.12 uses dict keyed by task_id)
    for data in gpu_data:
        # Phase 4.12 stores results as {task_id: result_dict}
        merged_correction.update(data.get('correction_results', {}))
        merged_corruption.update(data.get('corruption_results', {}))
        merged_preservation.update(data.get('preservation_results', {}))
        # Merge per-feature results (multi-feature mode)
        merged_per_feature.update(data.get('per_feature_results', {}))

    # Convert to lists for rate calculation
    correction_list = list(merged_correction.values())
    corruption_list = list(merged_corruption.values())
    preservation_list = list(merged_preservation.values())

    logger.info(f"Merged results: {len(correction_list)} correction, "
                f"{len(corruption_list)} corruption, {len(preservation_list)} preservation")

    # Recalculate rates from merged data
    correction_rate = calculate_correction_rate(correction_list)
    corruption_rate = calculate_corruption_rate(corruption_list)

    # Preservation rate: correct→correct
    if preservation_list:
        preserved = sum(1 for r in preservation_list
                       if r.get('baseline_passed', False) and r.get('steered_correct', False))
        total_correct = sum(1 for r in preservation_list
                          if r.get('baseline_passed', False))
        preservation_rate = (preserved / total_correct * 100) if total_correct > 0 else 0.0
    else:
        preservation_rate = 0.0

    # Use first GPU's metadata
    ref = gpu_data[0]

    # Compute averaged metrics across all features (multi-feature mode)
    averaged_metrics = {}
    if merged_per_feature:
        import numpy as np
        correction_rates = []
        corruption_rates = []
        preservation_rates = []
        for fid, fdata in merged_per_feature.items():
            correction_rates.append(fdata.get('correction_rate', 0))
            corruption_rates.append(fdata.get('corruption_rate', 0))
            preservation_rates.append(fdata.get('preservation_rate', 0))
        averaged_metrics = {
            'correction_rate': float(np.mean(correction_rates)) if correction_rates else 0,
            'corruption_rate': float(np.mean(corruption_rates)) if corruption_rates else 0,
            'preservation_rate': float(np.mean(preservation_rates)) if preservation_rates else 0,
            'std_correction': float(np.std(correction_rates)) if correction_rates else 0,
            'std_corruption': float(np.std(corruption_rates)) if corruption_rates else 0,
            'std_preservation': float(np.std(preservation_rates)) if preservation_rates else 0,
            'n_features': len(merged_per_feature)
        }

    # Build merged results (same structure as single-GPU output)
    merged = {
        'metadata': ref.get('metadata', {}),
        'correction_results': merged_correction,
        'corruption_results': merged_corruption,
        'preservation_results': merged_preservation,
        'per_feature_results': merged_per_feature,
        'averaged_metrics': averaged_metrics,
        'summary_metrics': {
            'correction_rate': correction_rate,
            'corruption_rate': corruption_rate,
            'preservation_rate': preservation_rate,
            'n_corrected': sum(1 for r in correction_list if r.get('steered_correct') and not r.get('baseline_passed')),
            'n_corrupted': sum(1 for r in corruption_list if not r.get('steered_correct') and r.get('baseline_passed')),
            'n_preserved': sum(1 for r in preservation_list if r.get('steered_correct') and r.get('baseline_passed'))
        },
        'steered_error_type_distribution': compute_error_type_distribution(
            correction_list + corruption_list + preservation_list, 'steered_error_type'
        ),
        'parallel_merge': True,
        'n_gpus': n_gpus
    }

    # Add std to summary_metrics if available
    if averaged_metrics:
        merged['summary_metrics']['std_correction'] = averaged_metrics.get('std_correction', 0)
        merged['summary_metrics']['std_corruption'] = averaged_metrics.get('std_corruption', 0)
        merged['summary_metrics']['std_preservation'] = averaged_metrics.get('std_preservation', 0)
        merged['summary_metrics']['n_features'] = averaged_metrics.get('n_features', 0)

    # Update metadata with merged counts
    if 'metadata' in merged:
        merged['metadata']['n_problems_tested'] = {
            'correction': len(correction_list),
            'corruption': len(corruption_list),
            'preservation': len(preservation_list)
        }

    # Save merged results
    save_json(merged, output_path / "zero_disc_steering_results.json")
    logger.info("Saved merged zero_disc_steering_results.json")

    # Write phase manifest
    write_phase_output(
        phase="4.12",
        outputs={
            "primary": "zero_disc_steering_results.json",
        },
        config=config,
        output_dir=str(output_path)
    )
    logger.info("Wrote phase_output.json manifest")

    # Cleanup per-GPU files
    for f in gpu_files:
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")

    # Print summary
    logger.info("=" * 60)
    logger.info("PHASE 4.12 PARALLEL MERGE COMPLETE")
    logger.info("=" * 60)
    if averaged_metrics:
        logger.info(f"Features tested: {averaged_metrics.get('n_features', 0)}")
        logger.info(f"Averaged Correction: {averaged_metrics.get('correction_rate', 0):.1f}% "
                   f"(±{averaged_metrics.get('std_correction', 0):.1f}%)")
        logger.info(f"Averaged Corruption: {averaged_metrics.get('corruption_rate', 0):.1f}% "
                   f"(±{averaged_metrics.get('std_corruption', 0):.1f}%)")
        logger.info(f"Averaged Preservation: {averaged_metrics.get('preservation_rate', 0):.1f}% "
                   f"(±{averaged_metrics.get('std_preservation', 0):.1f}%)")
    logger.info(f"Problem-level Correction: {correction_rate:.1f}%")
    logger.info(f"Problem-level Corruption: {corruption_rate:.1f}%")
    logger.info(f"Problem-level Preservation: {preservation_rate:.1f}%")

    return merged


def _merge_phase7_6_json_results(
    output_path: Path,
    n_gpus: int,
    config: Config
) -> dict:
    """
    Merge Phase 7.6 instruct steering results from parallel workers.

    Similar structure to Phase 4.8 - merges detailed_results and recalculates metrics.
    """
    from datetime import datetime
    from common.utils import save_json
    from common.steering_metrics import calculate_correction_rate, calculate_corruption_rate
    from common.dataset_utils import compute_error_type_distribution

    # Find per-GPU JSON files
    gpu_files = sorted(output_path.glob("steering_effect_analysis_gpu*.json"))
    if not gpu_files:
        raise RuntimeError(f"No steering_effect_analysis_gpu*.json files found in {output_path}")

    logger.info(f"Found {len(gpu_files)} GPU JSON files to merge for Phase 7.6")

    # Load all per-GPU results
    gpu_data = []
    for f in gpu_files:
        with open(f) as fh:
            gpu_data.append(json.load(fh))
        logger.info(f"  Loaded {f.name}")

    # Merge detailed_results across GPUs
    merged_correction = []
    merged_corruption = []
    merged_preservation = []

    for data in gpu_data:
        detailed = data.get('detailed_results', {})
        merged_correction.extend(detailed.get('correction', []))
        merged_corruption.extend(detailed.get('corruption', []))
        merged_preservation.extend(detailed.get('preservation', []))

    # Deduplicate by task_id
    def dedup_by_task_id(results):
        seen = set()
        deduped = []
        for r in results:
            tid = r.get('task_id')
            if tid not in seen:
                seen.add(tid)
                deduped.append(r)
        return deduped

    merged_correction = dedup_by_task_id(merged_correction)
    merged_corruption = dedup_by_task_id(merged_corruption)
    merged_preservation = dedup_by_task_id(merged_preservation)

    logger.info(f"Merged results: {len(merged_correction)} correction, "
                f"{len(merged_corruption)} corruption, {len(merged_preservation)} preservation")

    # Recalculate rates
    correction_rate = calculate_correction_rate(merged_correction)
    corruption_rate = calculate_corruption_rate(merged_corruption)

    if merged_preservation:
        preserved = sum(1 for r in merged_preservation
                       if r.get('baseline_passed', False) and r.get('steered_correct', False))
        total_correct = sum(1 for r in merged_preservation if r.get('baseline_passed', False))
        preservation_rate = (preserved / total_correct * 100) if total_correct > 0 else 0.0
    else:
        preservation_rate = 0.0

    # Use first GPU's metadata
    ref = gpu_data[0]

    # Sum n_problems across GPUs
    total_correct = sum(d.get('n_problems', {}).get('initially_correct', 0) for d in gpu_data)
    total_incorrect = sum(d.get('n_problems', {}).get('initially_incorrect', 0) for d in gpu_data)

    # Build merged metrics
    merged_metrics = {
        'correction_rate': correction_rate,
        'corruption_rate': corruption_rate,
        'preservation_rate': preservation_rate,
        'direction_source': ref.get('direction_source', 'sae'),
        'coefficients': ref.get('coefficients', {}),
        'n_problems': {
            'initially_correct': total_correct,
            'initially_incorrect': total_incorrect,
            'total': total_correct + total_incorrect
        },
        'parallel_merge': True,
        'n_gpus': n_gpus,
        'detailed_results': {
            'correction': merged_correction,
            'corruption': merged_corruption,
            'preservation': merged_preservation
        }
    }

    # Save merged results
    save_json(merged_metrics, output_path / "steering_effect_analysis.json")
    logger.info("Saved merged steering_effect_analysis.json")

    # Build and save summary
    all_steered = merged_correction + merged_corruption + merged_preservation
    error_dist = compute_error_type_distribution(all_steered, 'steered_error_type') if all_steered else None

    summary = {
        'phase': '7.6',
        'description': 'Instruction-Tuned Model Steering Analysis',
        'timestamp': datetime.now().isoformat(),
        'parallel_merge': True,
        'n_gpus': n_gpus,
        'config': {
            'model': ref.get('n_problems', {}).get('model', config.phase7_6_model_name if hasattr(config, 'phase7_6_model_name') else 'unknown'),
            'initially_correct_count': total_correct,
            'initially_incorrect_count': total_incorrect,
        },
        'results': {
            'correction_rate': correction_rate,
            'corruption_rate': corruption_rate,
            'preservation_rate': preservation_rate,
        },
        'steered_error_type_distribution': error_dist,
    }

    save_json(summary, output_path / "phase_7_6_summary.json")
    logger.info("Saved merged phase_7_6_summary.json")

    # Write phase manifest
    write_phase_output(
        phase="7.6",
        outputs={
            "primary": "phase_7_6_summary.json",
            "steering_analysis": "steering_effect_analysis.json",
        },
        config=config,
        output_dir=str(output_path)
    )

    # Cleanup per-GPU files
    for f in gpu_files:
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")
    for f in sorted(output_path.glob("phase_7_6_summary_gpu*.json")):
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")

    logger.info("=" * 60)
    logger.info("PHASE 7.6 PARALLEL MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Correction: {correction_rate:.1f}%")
    logger.info(f"Corruption: {corruption_rate:.1f}%")
    logger.info(f"Preservation: {preservation_rate:.1f}%")

    return merged_metrics


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

    # Phase 2.2: No file merge needed - activations are independent .safetensors files
    # Just write combined manifest for downstream discovery
    if phase_id == "2.2":
        # Count total activations across all GPUs
        activation_dir = output_path / "pile_activations"
        activation_count = len(list(activation_dir.glob("*.safetensors")))

        write_phase_output(
            phase="2.2",
            outputs={"primary": "pile_activations/", "activation_count": str(activation_count)},
            config=config,
            output_dir=str(output_path)
        )
        logger.info(f"Phase 2.2 parallel complete - {activation_count} activations in pile_activations/")
        return {"activation_count": activation_count, "n_gpus": n_gpus}

    # Phase 4.5/4.6 use JSON output format, not parquet
    if phase_id in ("4.5", "4.6"):
        return _merge_phase4_5_json_results(output_path, n_gpus, config, phase_id)

    # Phase 4.8 uses JSON output format (steering effect analysis)
    if phase_id == "4.8":
        return _merge_phase4_8_results(output_path, n_gpus, config)

    # Phase 8.3 needs custom merge (JSON summary recalculated from parquet)
    if phase_id == "8.3":
        return _merge_phase8_3_results(output_path, n_gpus, config)

    # Phase 5.6 uses JSON output format (zero-disc orthogonalization)
    if phase_id == "5.6":
        return _merge_phase5_6_json_results(output_path, n_gpus, config)

    # Phase 5.3 uses JSON output format (weight orthogonalization)
    if phase_id == "5.3":
        return _merge_phase5_3_json_results(output_path, n_gpus, config)

    # Phase 7.6 uses JSON output format (instruct steering)
    if phase_id == "7.6":
        return _merge_phase7_6_json_results(output_path, n_gpus, config)

    # Phase 4.12 uses JSON output format (zero-disc steering)
    if phase_id == "4.12":
        return _merge_phase4_12_json_results(output_path, n_gpus, config)

    # Find per-GPU result files
    # Phase 3.5 uses a different pattern for temperature experiments
    if phase_id == "3.5":
        gpu_result_files = sorted(output_path.glob("results_gpu*_temp_*.parquet"))
    else:
        gpu_result_files = sorted(output_path.glob("results_gpu*.parquet"))

    if not gpu_result_files:
        # Phase 1: Check if all tasks were checkpointed (activations already exist)
        if phase_id == "1":
            activation_dir = output_path / "activations"
            correct_files = list((activation_dir / "correct").glob("*_layer_*.safetensors"))
            incorrect_files = list((activation_dir / "incorrect").glob("*_layer_*.safetensors"))
            if correct_files or incorrect_files:
                logger.info("All tasks already have activations - cross-run checkpointing detected")
                logger.info("No new results to merge. Using existing dataset.")
                # Return indicator that nothing needed to be done
                return {"checkpointed": True, "message": "All tasks already processed"}

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

    # Deduplicate and sort by task_id if present
    # This handles cross-run checkpointing where GPUs may load overlapping records
    if 'task_id' in merged_df.columns:
        before_dedup = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['task_id'], keep='last')
        merged_df = merged_df.sort_values('task_id').reset_index(drop=True)
        if before_dedup != len(merged_df):
            logger.info(f"  Deduplicated: {before_dedup} -> {len(merged_df)} rows")

    # Clean up old merged files before saving new one (prevents duplicate counts)
    old_merged_files = list(output_path.glob("dataset_merged_*.parquet"))
    for old_file in old_merged_files:
        old_file.unlink()
        logger.info(f"  Cleaned up old merged file: {old_file.name}")

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

    # Phase 1: Create summary JSON (required by Phase 9.5)
    if phase_id == "1":
        from datetime import datetime
        from common.dataset_utils import compute_error_type_distribution
        from common.utils import save_json

        error_dist = compute_error_type_distribution(merged_df, "baseline_error_type")

        summary = {
            "phase": "1",
            "description": "Dataset Building",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": config.model_name,
                "dataset": config.dataset_name,
                "split": "selection",
                "temperature": config.model_temperature
            },
            "results": {
                "tasks_attempted": len(merged_df),
                "tasks_included": len(merged_df),
                "tasks_excluded": 0,
                "correct_count": int(merged_df['baseline_passed'].sum()),
                "incorrect_count": int((~merged_df['baseline_passed']).sum()),
                "pass_rate": float(merged_df['baseline_passed'].mean() * 100)
            },
            "baseline_error_type_distribution": error_dist
        }

        summary_file = output_path / "phase_1_summary.json"
        save_json(summary, summary_file)
        logger.info(f"Saved Phase 1 summary: {summary_file}")

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
