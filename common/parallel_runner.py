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
import json
import traceback
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


def _dedup_by_task_id(results_list: list[dict]) -> list[dict]:
    """Remove duplicate results keeping first occurrence per task_id."""
    seen = set()
    deduped = []
    for r in results_list:
        tid = r.get('task_id')
        if tid not in seen:
            seen.add(tid)
            deduped.append(r)
    return deduped


def _load_gpu_json_files(output_path: Path, pattern: str) -> list[dict]:
    """Load and parse GPU JSON result files matching a glob pattern.

    Corrupted files are skipped with a warning rather than failing the
    entire merge, so valid GPU results are preserved.

    Args:
        output_path: Directory containing GPU output files
        pattern: Glob pattern (e.g. "steering_effect_analysis_gpu*.json")

    Returns:
        List of parsed JSON dicts, one per successfully loaded file

    Raises:
        RuntimeError: If no files match the pattern or all files are corrupted
    """
    gpu_files = sorted(output_path.glob(pattern))
    if not gpu_files:
        raise RuntimeError(f"No {pattern} files found in {output_path}")
    logger.info(f"Found {len(gpu_files)} GPU files matching {pattern}")
    results = []
    corrupted = []
    for f in gpu_files:
        try:
            with open(f) as fh:
                results.append(json.load(fh))
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Corrupted GPU result file {f.name}: {e} — skipping")
            corrupted.append(f.name)
    if corrupted:
        logger.warning(
            f"Skipped {len(corrupted)} corrupted file(s): {corrupted}. "
            f"Re-run the phase to regenerate missing GPU data."
        )
    if not results:
        raise RuntimeError(
            f"All GPU files corrupted for pattern {pattern} in {output_path}. "
            f"Re-run the phase."
        )
    return results


def _cleanup_gpu_files(output_path: Path, patterns: list[str]) -> None:
    """Remove per-GPU temporary files matching given glob patterns.

    Args:
        output_path: Directory containing GPU output files
        patterns: List of glob patterns to clean up
    """
    for pattern in patterns:
        for f in sorted(output_path.glob(pattern)):
            f.unlink(missing_ok=True)
            logger.debug(f"Cleaned up {f.name}")


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
    "7.7",   # Instruct zero-disc control
    "8.3",   # Selective steering (outputs parquet in parallel mode)
    "9.5",   # Combined orthogonalization + steering
    "10.5",  # Selective ortho + selective steering
}

# Iterative parallelization: distribute problems, merge after each value
# Used for grid search phases where early stopping needs full data
ITERATIVE_PARALLEL_PHASES = {
    "3.5",   # Temperature robustness (iterate over temperatures)
    "4.5",   # Coefficient grid search (iterate over coefficients)
    "4.6",   # Golden section (iterate over refinement points)
    "8.2",   # Threshold optimizer (iterate over percentiles)
}

# All phases that support parallelization
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
    if phase_id in ("4.5", "4.6", "4.7", "4.8", "5.3", "5.6", "7.6", "8.2", "8.3", "9.5", "10.5") and direction_source in ('probe_mass_mean', 'probe_logreg'):
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
                    'error': f"Subprocess crashed: {exc}",
                    'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if hasattr(exc, '__traceback__') else str(exc)
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

    # Check for failures — refuse to merge incomplete data
    if failed_gpus:
        raise RuntimeError(
            f"GPU workers failed: {failed_gpus}. "
            f"Re-run the same command to resume — successful GPUs' work is checkpointed."
        )

    # viz-only: workers already regenerated visualizations individually; no merge needed
    if getattr(config, 'viz_only', False):
        logger.info("viz-only mode: skipping merge (visualizations regenerated by workers)")
        return {}

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

    # Load all per-GPU results (skip corrupted files)
    gpu_results = []
    corrupted_files = []
    for json_file in json_files:
        try:
            with open(json_file) as f:
                gpu_results.append(json.load(f))
            logger.info(f"  Loaded {json_file.name}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"  Corrupted: {json_file.name}: {e} — skipping")
            corrupted_files.append(json_file.name)
    if corrupted_files:
        logger.warning(
            f"Skipped {len(corrupted_files)} corrupted file(s): {corrupted_files}. "
            f"Re-run the phase to regenerate missing GPU data."
        )
    if not gpu_results:
        raise RuntimeError(
            f"All per-GPU coefficient_analysis files are corrupted in {output_path}. "
            f"Re-run the phase."
        )

    # Phase 4.5/4.6 merging - combine per-problem results across GPUs
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

            # Filter to records with required fields (skip incomplete records)
            valid = [r for r in results if 'baseline_passed' in r and 'steered_correct' in r]
            if len(valid) < len(results):
                logger.warning(f"  Skipped {len(results) - len(valid)} records with missing fields "
                              f"for coefficient {coeff}")

            # Calculate metrics based on steering type
            if steering_type == 'correct':
                # Correction rate: incorrect baseline → correct steered
                corrections = sum(1 for r in valid
                                 if not r['baseline_passed'] and r['steered_correct'])
                incorrect_baseline = sum(1 for r in valid if not r['baseline_passed'])
                correction_rate = (corrections / incorrect_baseline * 100) if incorrect_baseline > 0 else 0

                metrics = {'correction_rate': correction_rate}
                score = correction_rate
            else:
                # Corruption rate: correct baseline → incorrect steered
                corruptions = sum(1 for r in valid
                                 if r['baseline_passed'] and not r['steered_correct'])
                correct_baseline = sum(1 for r in valid if r['baseline_passed'])
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

    # Save merged per-coefficient files
    for coeff, results in results_by_coefficient.items():
        coeff_str = f"coeff_{int(coeff)}" if coeff == int(coeff) else f"coeff_{coeff}"

        # Deduplicate each result type
        results['correction'] = _dedup_by_task_id(results['correction'])
        results['corruption'] = _dedup_by_task_id(results['corruption'])
        results['preservation'] = _dedup_by_task_id(results['preservation'])

        # Save per-coefficient files
        if results['correction']:
            save_json(results['correction'], output_path / f"correction_results_{coeff_str}.json")
            logger.info(f"Saved {len(results['correction'])} correction results to correction_results_{coeff_str}.json")
        if results['corruption']:
            save_json(results['corruption'], output_path / f"corruption_results_{coeff_str}.json")
            logger.info(f"Saved {len(results['corruption'])} corruption results to corruption_results_{coeff_str}.json")
        if results['preservation']:
            save_json(results['preservation'], output_path / f"preservation_results_{coeff_str}.json")
            logger.info(f"Saved {len(results['preservation'])} preservation results to preservation_results_{coeff_str}.json")

    # Also merge and save selected/refined coefficients
    selected_files = sorted(output_path.glob(selected_pattern))
    if selected_files:
        # Use the first GPU's selected coefficients as base, update with merged optimal
        try:
            with open(selected_files[0]) as f:
                selected = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load selected coefficients from {selected_files[0].name}: {e}")
            selected = None

        if selected is not None:
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
    # Per-coefficient result files are the preferred output format
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
        f.unlink(missing_ok=True)
        logger.info(f"  Cleaned up {f.name}")
    for f in selected_files:
        f.unlink(missing_ok=True)
        logger.info(f"  Cleaned up {f.name}")

    # Also clean up per-GPU result files (aggregated + per-coefficient)
    for pattern in ["all_correction_results_gpu*.json", "all_corruption_results_gpu*.json", "all_preservation_results_gpu*.json"]:
        for f in output_path.glob(pattern):
            f.unlink(missing_ok=True)
            logger.info(f"  Cleaned up {f.name}")
    # Clean up per-coefficient per-GPU files
    for f in per_coeff_files_to_cleanup:
        f.unlink(missing_ok=True)
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

    # Load per-GPU JSON files
    gpu_results = _load_gpu_json_files(
        output_path, "zero_disc_orthogonalization_results_gpu*.json"
    )

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
    all_incorrect_results = _dedup_by_task_id(all_incorrect_results)
    all_corrected = _dedup_by_task_id(all_corrected)
    all_preserved = _dedup_by_task_id(all_preserved)
    all_corrupted = _dedup_by_task_id(all_corrupted)

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


def _merge_phase7_3_results(
    output_path: Path,
    n_gpus: int,
    config: Config
) -> dict:
    """
    Merge Phase 7.3 instruct baseline results from parallel workers.

    Phase 7.3 produces results_gpu{N}.parquet files. This function:
    1. Loads all per-GPU parquet files
    2. Deduplicates by task_id
    3. Saves merged result as dataset_instruct_temp_0_0.parquet (required by 7.6, 7.9, 7.12)
    4. Rebuilds metadata.json from merged data
    5. Writes phase_output.json manifest
    6. Cleans up per-GPU files
    """
    from datetime import datetime
    from common.dataset_utils import compute_error_type_distribution
    from common.utils import save_json

    # Find per-GPU parquet files
    gpu_files = sorted(output_path.glob("results_gpu*.parquet"))
    if not gpu_files:
        raise RuntimeError(f"No results_gpu*.parquet files found in {output_path}")

    logger.info(f"Found {len(gpu_files)} GPU parquet files to merge")

    # Load and merge (graceful degradation: skip corrupted files)
    dfs = []
    corrupted_files = []
    for f in gpu_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
            logger.info(f"  Loaded {len(df)} rows from {f.name}")
        except Exception as e:
            logger.error(f"  Corrupted GPU parquet file {f.name}: {e} — skipping")
            corrupted_files.append(f.name)
    if not dfs:
        raise RuntimeError(f"All GPU parquet files corrupted in {output_path}: {corrupted_files}")
    if corrupted_files:
        logger.warning(f"Skipped {len(corrupted_files)} corrupted files, merging {len(dfs)} valid files")

    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged {len(merged_df)} total results from {len(gpu_files)} GPUs")

    # Deduplicate by task_id (handles cross-run checkpointing)
    if 'task_id' in merged_df.columns:
        before_dedup = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['task_id'], keep='last')
        merged_df = merged_df.sort_values('task_id').reset_index(drop=True)
        if before_dedup != len(merged_df):
            logger.info(f"  Deduplicated: {before_dedup} -> {len(merged_df)} rows")

    # Save as expected downstream filename
    merged_file = output_path / "dataset_instruct_temp_0_0.parquet"
    merged_df.to_parquet(merged_file, index=False)
    logger.info(f"Saved merged dataset: {merged_file} ({len(merged_df)} rows)")

    # Rebuild metadata.json from merged data
    correct_count = int(merged_df['baseline_passed'].sum()) if 'baseline_passed' in merged_df.columns else 0
    n_total = len(merged_df)
    metadata = {
        "creation_timestamp": datetime.now().isoformat(),
        "model_name": config.model_name,
        "model_type": "instruction-tuned",
        "temperature": 0.0,
        "n_total_samples": n_total,
        "n_gpus_merged": len(gpu_files),
        "stats": {
            "n_correct": correct_count,
            "n_incorrect": n_total - correct_count,
            "pass_rate": correct_count / n_total if n_total > 0 else 0.0,
        },
        "baseline_error_type_distribution": compute_error_type_distribution(
            merged_df.to_dict('records'), "baseline_error_type"
        ) if 'baseline_error_type' in merged_df.columns else None
    }
    save_json(metadata, output_path / "metadata.json")
    logger.info("Rebuilt metadata.json from merged data")

    # Write phase_output.json manifest for downstream discovery
    write_phase_output(
        phase="7.3",
        outputs={
            "primary": "dataset_instruct_temp_0_0.parquet",
            "metadata": "metadata.json"
        },
        config=config,
        output_dir=str(output_path)
    )
    logger.info("Wrote phase_output.json manifest")

    # Clean up per-GPU files
    for f in gpu_files:
        f.unlink()
        logger.info(f"  Cleaned up {f.name}")

    return {
        'merged_file': str(merged_file),
        'n_total': n_total,
        'n_correct': correct_count,
        'pass_rate': correct_count / n_total if n_total > 0 else 0.0,
        'n_gpus': len(gpu_files)
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

    # Load and merge (graceful degradation: skip corrupted files)
    dfs = []
    corrupted_files = []
    for f in gpu_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.error(f"Corrupted GPU parquet file {f.name}: {e} — skipping")
            corrupted_files.append(f.name)
    if not dfs:
        raise RuntimeError(f"All GPU parquet files corrupted in {output_path}: {corrupted_files}")
    if corrupted_files:
        logger.warning(f"Skipped {len(corrupted_files)} corrupted files, merging {len(dfs)} valid files")
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

    n_steered_correction = sum(1 for r in valid_correction if r.get('was_steered', False))
    n_not_steered_correction = n_valid_correction - n_steered_correction
    steering_trigger_rate = n_steered_correction / n_valid_correction if n_valid_correction > 0 else 0

    n_corrected = sum(1 for r in valid_correction
                     if not r['baseline_passed'] and r['steered_correct'])
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

    n_steered_preservation = sum(1 for r in valid_preservation if r.get('was_steered', False))
    n_not_steered_preservation = n_valid_preservation - n_steered_preservation
    steering_avoidance_rate = n_not_steered_preservation / n_valid_preservation if n_valid_preservation > 0 else 0

    n_preserved = sum(1 for r in valid_preservation
                     if r['baseline_passed'] and r['steered_correct'])
    n_corrupted = sum(1 for r in valid_preservation
                     if r['baseline_passed'] and not r['steered_correct'])
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


def _merge_three_list_results(
    gpu_data: list[dict],
    result_path: str = 'detailed_results',
    list_keys: tuple[str, str, str] = ('correction', 'corruption', 'preservation'),
) -> tuple[list[dict], list[dict], list[dict]]:
    """Extract, extend, and dedup three result lists from per-GPU data.

    Shared pattern for phases 4.8, 7.6, and 5.6 that all merge three
    parallel result lists by task_id.

    Args:
        gpu_data: List of per-GPU JSON dicts
        result_path: Top-level key containing the three lists
        list_keys: Keys for the three result lists within result_path

    Returns:
        Tuple of (list1, list2, list3) — deduplicated by task_id
    """
    lists = {k: [] for k in list_keys}
    for data in gpu_data:
        container = data.get(result_path, {})
        for k in list_keys:
            lists[k].extend(container.get(k, []))

    return tuple(_dedup_by_task_id(lists[k]) for k in list_keys)


def _merge_phase4_8_results(
    output_path: Path,
    n_gpus: int,
    config: Config,
    phase_id: str = "4.8"
) -> dict:
    """
    Merge Phase 4.8/7.6 steering effect analysis results from parallel workers.

    Both phases produce steering_effect_analysis_gpu{N}.json files with the same
    structure: detailed_results with correction/corruption/preservation lists.

    Differences handled via phase_id:
    - Detail files (all_*_results.json): only for 4.8
    - Summary filename: phase_{phase_id}_summary.json (underscores for dots)
    - Multi-candidate mode: only for 4.8
    - Model name in summary: config.model_name (4.8) vs config.phase7_6_model_name (7.6)
    """
    from datetime import datetime
    from common.utils import save_json
    from common.steering_metrics import (
        calculate_correction_rate, calculate_corruption_rate, calculate_preservation_rate
    )
    from common.dataset_utils import compute_error_type_distribution

    # Load per-GPU JSON files
    gpu_data = _load_gpu_json_files(output_path, "steering_effect_analysis_gpu*.json")

    # Merge detailed_results across GPUs
    merged_correction, merged_corruption, merged_preservation = _merge_three_list_results(gpu_data)

    logger.info(f"Merged results: {len(merged_correction)} correction, "
                f"{len(merged_corruption)} corruption, {len(merged_preservation)} preservation")

    # Recalculate rates from merged data
    correction_rate = calculate_correction_rate(merged_correction)
    corruption_rate = calculate_corruption_rate(merged_corruption)
    preservation_rate = calculate_preservation_rate(merged_preservation)

    # Use first GPU's metadata for coefficients, direction_source, etc.
    ref = gpu_data[0]

    # Sum up n_problems across GPUs
    total_initially_correct = sum(d.get('n_problems', {}).get('initially_correct', 0) for d in gpu_data)
    total_initially_incorrect = sum(d.get('n_problems', {}).get('initially_incorrect', 0) for d in gpu_data)
    total_problems = total_initially_correct + total_initially_incorrect

    # Detect multi-candidate mode (Phase 4.8 only)
    is_multi_candidate = phase_id == "4.8" and 'correct' in ref and isinstance(ref.get('correct'), list)

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

    # Handle multi-candidate mode: merge candidate lists by (layer, latent_idx) key
    if is_multi_candidate:
        # Check if per-candidate detailed_results are available (new parallel GPU files)
        sample_candidate = next(
            (c for d in gpu_data for c in d.get('correct', []) + d.get('incorrect', [])),
            {}
        )
        has_candidate_detail = 'detailed_results' in sample_candidate

        if not has_candidate_detail:
            logger.warning(
                "Per-candidate detailed_results missing from GPU files (old format). "
                "Per-candidate rates will reflect GPU 0 only. "
                "Re-run with updated code for accurate merged rates."
            )

        for candidate_key in ('correct', 'incorrect'):
            # Build lookup from each GPU's candidates keyed by (layer, latent_idx)
            candidate_totals = {}
            candidate_details = {}  # key -> list of per-task dicts
            candidate_preservation = {}  # key -> list of per-task dicts
            for d in gpu_data:
                for candidate in d.get(candidate_key, []):
                    key = (candidate.get('layer'), candidate.get('latent_idx'))
                    candidate_totals[key] = candidate_totals.get(key, 0) + candidate.get('n_total', 0)
                    if has_candidate_detail:
                        candidate_details.setdefault(key, []).extend(candidate.get('detailed_results', []))
                        candidate_preservation.setdefault(key, []).extend(candidate.get('preservation_detailed', []))

            # Merge using reference candidate order, matching by key
            merged_candidates = []
            for candidate in ref.get(candidate_key, []):
                merged_candidate = candidate.copy()
                key = (candidate.get('layer'), candidate.get('latent_idx'))
                merged_candidate['n_total'] = candidate_totals.get(key, 0)

                if has_candidate_detail:
                    # Dedup and recalculate rates from merged per-problem data
                    deduped_detail = _dedup_by_task_id(candidate_details.get(key, []))
                    deduped_preservation = _dedup_by_task_id(candidate_preservation.get(key, []))

                    if candidate_key == 'correct':
                        merged_candidate['correction_rate'] = calculate_correction_rate(deduped_detail)
                        merged_candidate['preservation_rate'] = calculate_preservation_rate(deduped_preservation)
                    else:
                        merged_candidate['corruption_rate'] = calculate_corruption_rate(deduped_detail)

                    # Store detail temporarily for best-candidate selection below
                    merged_candidate['_detail'] = deduped_detail
                    merged_candidate['_preservation'] = deduped_preservation

                # Strip per-candidate detail from final output (not needed in saved file)
                merged_candidate.pop('detailed_results', None)
                merged_candidate.pop('preservation_detailed', None)
                merged_candidates.append(merged_candidate)

            merged_metrics[candidate_key] = merged_candidates

        # Rebuild top-level detailed_results from the properly-merged best candidate
        if has_candidate_detail:
            if merged_metrics.get('correct'):
                best_correct = max(merged_metrics['correct'], key=lambda x: x.get('correction_rate', 0))
                best_detail = best_correct.pop('_detail', [])
                best_preservation = best_correct.pop('_preservation', [])
                merged_metrics['detailed_results']['correction'] = best_detail
                merged_metrics['detailed_results']['preservation'] = best_preservation
                # Update top-level rates and local vars for summary/logging
                correction_rate = best_correct['correction_rate']
                preservation_rate = best_correct.get('preservation_rate', preservation_rate)
                merged_metrics['correction_rate'] = correction_rate
                merged_metrics['preservation_rate'] = preservation_rate
                merged_correction = best_detail
                merged_preservation = best_preservation

            if merged_metrics.get('incorrect'):
                best_incorrect = max(merged_metrics['incorrect'], key=lambda x: x.get('corruption_rate', 0))
                best_corruption_detail = best_incorrect.pop('_detail', [])
                merged_metrics['detailed_results']['corruption'] = best_corruption_detail
                corruption_rate = best_incorrect['corruption_rate']
                merged_metrics['corruption_rate'] = corruption_rate
                merged_corruption = best_corruption_detail

            # Clean up temporary _detail/_preservation from non-best candidates
            for candidate_key in ('correct', 'incorrect'):
                for c in merged_metrics.get(candidate_key, []):
                    c.pop('_detail', None)
                    c.pop('_preservation', None)

        merged_metrics['best_candidates'] = {
            'correct': merged_metrics['correct'][0] if merged_metrics.get('correct') else None,
            'incorrect': merged_metrics['incorrect'][0] if merged_metrics.get('incorrect') else None,
        }
        logger.info(f"Multi-candidate mode: merged {len(merged_metrics.get('correct', []))} correct, "
                    f"{len(merged_metrics.get('incorrect', []))} incorrect candidates")

    # Save merged analysis JSON
    save_json(merged_metrics, output_path / "steering_effect_analysis.json")
    logger.info("Saved merged steering_effect_analysis.json")

    # Compute error type distribution
    all_steered_results = merged_correction + merged_corruption + merged_preservation
    error_dist = compute_error_type_distribution(
        all_steered_results, 'steered_error_type'
    ) if all_steered_results else None

    # Determine model name based on phase
    if phase_id == "7.6":
        model_name = getattr(config, 'phase7_6_model_name', 'unknown')
        description = 'Instruction-Tuned Model Steering Analysis'
    else:
        model_name = ref.get('n_problems', {}).get('model', getattr(config, 'model_name', 'unknown'))
        description = 'Multi-Candidate Steering Effect Analysis' if is_multi_candidate else 'Steering Effect Analysis'

    # Build summary filename: phase_4_8_summary.json or phase_7_6_summary.json
    summary_filename = f"phase_{phase_id.replace('.', '_')}_summary.json"

    # Build and save summary
    summary = {
        'phase': phase_id,
        'description': description,
        'timestamp': datetime.now().isoformat(),
        'parallel_merge': True,
        'n_gpus': n_gpus,
        'config': {
            'model': model_name,
            'initially_correct_count': total_initially_correct,
            'initially_incorrect_count': total_initially_incorrect,
        },
        'results': {
            'correction_rate': correction_rate,
            'corruption_rate': corruption_rate,
            'preservation_rate': preservation_rate,
        },
        'steered_error_type_distribution': error_dist,
    }

    # Phase 4.8-specific: add multi-candidate counts and mode
    if phase_id == "4.8":
        summary['mode'] = 'multi_candidate' if is_multi_candidate else 'probe'
        summary['results']['correct_candidates'] = len(merged_metrics.get('correct', [])) if is_multi_candidate else 0
        summary['results']['incorrect_candidates'] = len(merged_metrics.get('incorrect', [])) if is_multi_candidate else 0

    # Phase 7.6: recompute statistical_tests from merged result lists
    # (per-GPU tests are on partial data — must recalculate from full merge)
    if phase_id == "7.6":
        from scipy.stats import binomtest

        stat_tests = {}
        for test_name, results, success_fn in [
            ("correction", merged_correction,
             lambda r: not r.get('baseline_passed') and r.get('steered_correct')),
            ("corruption", merged_corruption,
             lambda r: r.get('baseline_passed') and not r.get('steered_correct')),
            ("preservation", merged_preservation,
             lambda r: r.get('baseline_passed') and r.get('steered_correct')),
        ]:
            successes = sum(1 for r in results if success_fn(r))
            trials = len(results)
            if trials > 0:
                null_p = max(1.0 / trials, 1e-10) if test_name != "preservation" else 0.5
                test_result = binomtest(successes, trials, p=null_p, alternative='greater')
                stat_tests[test_name] = {
                    'successes': successes, 'trials': trials,
                    'rate': successes / trials * 100,
                    'pvalue': float(test_result.pvalue), 'significant': bool(test_result.pvalue < 0.05)
                }
            else:
                stat_tests[test_name] = {
                    'successes': 0, 'trials': 0, 'rate': 0.0,
                    'pvalue': 1.0, 'significant': False
                }
        summary['results']['statistical_tests'] = stat_tests

    # Carry over latent/probe info from reference GPU
    if 'latents_used' in ref:
        summary['latents_used'] = ref['latents_used']
    if 'probe_info' in ref:
        summary['probe_info'] = ref['probe_info']

    save_json(summary, output_path / summary_filename)
    logger.info(f"Saved merged {summary_filename}")

    # Phase 4.8-specific: save per-result-type JSON files
    manifest_outputs = {
        "primary": summary_filename,
        "steering_analysis": "steering_effect_analysis.json",
    }
    if phase_id == "4.8":
        save_json(merged_correction, output_path / "all_correction_results.json")
        save_json(merged_corruption, output_path / "all_corruption_results.json")
        save_json(merged_preservation, output_path / "all_preservation_results.json")
        logger.info("Saved merged all_*_results.json files")
        manifest_outputs.update({
            "correction_results": "all_correction_results.json",
            "corruption_results": "all_corruption_results.json",
            "preservation_results": "all_preservation_results.json",
        })

    # Write phase manifest
    write_phase_output(
        phase=phase_id,
        outputs=manifest_outputs,
        config=config,
        output_dir=str(output_path)
    )
    logger.info("Wrote phase_output.json manifest")

    # Cleanup per-GPU files
    _cleanup_gpu_files(output_path, [
        "steering_effect_analysis_gpu*.json",
        f"{summary_filename.replace('.json', '')}_gpu*.json",
    ])

    # Print summary
    logger.info("=" * 60)
    logger.info(f"PHASE {phase_id} PARALLEL MERGE COMPLETE")
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

    # Load per-GPU JSON files
    gpu_results = _load_gpu_json_files(output_path, "orthogonalization_results_gpu*.json")

    # Use first GPU's structure as base
    merged = gpu_results[0].copy()

    # Merge per-candidate results across GPUs, then pick global best
    import copy
    for direction in ['incorrect', 'correct']:
        key = f'{direction}_orthogonalization'

        # Collect per_candidate data from every GPU and merge by candidate_id
        merged_per_candidate = {}
        for gpu_data in gpu_results:
            per = gpu_data.get('multi_candidate', {}).get(direction, {}).get('per_candidate', {})
            for cand_id, cand_result in per.items():
                if cand_id not in merged_per_candidate:
                    merged_per_candidate[cand_id] = copy.deepcopy(cand_result)
                    for field in ['n_corrected', 'n_incorrect_baseline', 'n_preserved',
                                  'n_correct_baseline', 'n_corrupted', 'accidental_corrections']:
                        merged_per_candidate[cand_id]['metrics'][field] = 0
                    merged_per_candidate[cand_id]['examples'] = {
                        k: [] for k in cand_result.get('examples', {})
                    }
                # Accumulate raw counts
                for field in ['n_corrected', 'n_incorrect_baseline', 'n_preserved',
                              'n_correct_baseline', 'n_corrupted', 'accidental_corrections']:
                    merged_per_candidate[cand_id]['metrics'][field] += (
                        cand_result.get('metrics', {}).get(field, 0)
                    )
                # Merge examples
                for ex_key, ex_list in cand_result.get('examples', {}).items():
                    merged_per_candidate[cand_id]['examples'].setdefault(ex_key, [])
                    merged_per_candidate[cand_id]['examples'][ex_key].extend(ex_list)

        if not merged_per_candidate:
            continue  # No multi-candidate data; fall back to legacy GPU-0 copy

        # Recalculate rates from accumulated counts
        for cand_id, cand_result in merged_per_candidate.items():
            m = cand_result['metrics']
            if direction == 'incorrect':
                n_inc = m.get('n_incorrect_baseline', 0)
                n_cor = m.get('n_correct_baseline', 0)
                m['correction_rate'] = (m['n_corrected'] / n_inc * 100) if n_inc > 0 else 0.0
                m['preservation_rate'] = (m['n_preserved'] / n_cor * 100) if n_cor > 0 else 0.0
            else:
                n_cor = m.get('n_correct_baseline', 0)
                m['corruption_rate'] = (m['n_corrupted'] / n_cor * 100) if n_cor > 0 else 0.0
                all_sims = [
                    ex['similarity']
                    for ex_list in cand_result['examples'].values()
                    for ex in ex_list if 'similarity' in ex
                ]
                m['avg_similarity_score'] = sum(all_sims) / len(all_sims) if all_sims else 0.0

        # Select global best
        rate_key = 'correction_rate' if direction == 'incorrect' else 'corruption_rate'
        best_id = max(merged_per_candidate,
                      key=lambda k: merged_per_candidate[k]['metrics'][rate_key])

        gpu0_best = (gpu_results[0].get('multi_candidate', {})
                     .get(direction, {}).get('best_candidate_id', 'unknown'))
        logger.info(f"  Global best {direction} candidate: {best_id} "
                    f"(GPU-0 local best was: {gpu0_best})")
        if best_id != gpu0_best:
            logger.warning(f"  Global best differs from GPU-0 local best! "
                           f"Bug would have selected {gpu0_best} incorrectly.")

        # Write true global winner into merged dict
        merged[key] = merged_per_candidate[best_id]
        if 'multi_candidate' in merged and direction in merged['multi_candidate']:
            merged['multi_candidate'][direction]['per_candidate'] = merged_per_candidate
            merged['multi_candidate'][direction]['best_candidate_id'] = best_id
            merged['multi_candidate'][direction]['best_candidate'] = merged_per_candidate[best_id]
        if 'best_selection' in merged:
            merged['best_selection'][direction] = best_id

        # Log final merged metrics
        m = merged[key]['metrics']
        if direction == 'incorrect':
            logger.info(f"  {key}: correction_rate={m['correction_rate']:.1f}% "
                        f"({m['n_corrected']}/{m['n_incorrect_baseline']}), "
                        f"preservation_rate={m.get('preservation_rate', 0):.1f}%")
        else:
            logger.info(f"  {key}: corruption_rate={m['corruption_rate']:.1f}% "
                        f"({m['n_corrupted']}/{m['n_correct_baseline']})")

    merged['parallel_merge'] = True
    merged['n_gpus'] = n_gpus

    # Save merged results
    save_json(merged, output_path / "orthogonalization_results.json")
    logger.info("Saved merged orthogonalization_results.json")

    # Merge and save summary
    summary_files = sorted(output_path.glob("phase_5_3_summary_gpu*.json"))
    if summary_files:
        try:
            with open(summary_files[0]) as f:
                summary = json.load(f)
            summary['parallel_merge'] = True
            summary['n_gpus'] = n_gpus
            save_json(summary, output_path / "phase_5_3_summary.json")
            logger.info("Saved merged phase_5_3_summary.json")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load summary from {summary_files[0].name}: {e} — skipping summary merge")

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
    _cleanup_gpu_files(output_path, [
        "orthogonalization_results_gpu*.json",
        "phase_5_3_summary_gpu*.json",
    ])

    logger.info("PHASE 5.3 PARALLEL MERGE COMPLETE")
    return merged


def _merge_phase9_5_results(
    output_path: Path,
    n_gpus: int,
    config: Config,
) -> dict:
    """
    Merge Phase 9.5 combined orthogonalization+steering results from parallel workers.

    Phase 9.5 produces correction_results_gpu{N}.json, preservation_results_gpu{N}.json,
    and corruption_results_gpu{N}.json.
    """
    from common.utils import save_json, load_json
    from common.phase_discovery import write_phase_output
    from datetime import datetime

    # Merge correction results
    correction_gpu_files = sorted(output_path.glob("correction_results_gpu*.json"))
    preservation_gpu_files = sorted(output_path.glob("preservation_results_gpu*.json"))
    corruption_gpu_files = sorted(output_path.glob("corruption_results_gpu*.json"))

    if not correction_gpu_files:
        logger.warning("No correction GPU result files found — parallel merge skipped")
        return {}

    all_correction = []
    for f in correction_gpu_files:
        try:
            data = load_json(f)
            if isinstance(data, dict):
                all_correction.extend(data.values())
            else:
                all_correction.extend(data)
        except Exception as e:
            logger.warning(f"Could not load {f.name}: {e}")

    all_preservation = []
    for f in preservation_gpu_files:
        try:
            data = load_json(f)
            if isinstance(data, dict):
                all_preservation.extend(data.values())
            else:
                all_preservation.extend(data)
        except Exception as e:
            logger.warning(f"Could not load {f.name}: {e}")

    all_corruption = []
    for f in corruption_gpu_files:
        try:
            data = load_json(f)
            if isinstance(data, dict):
                all_corruption.extend(data.values())
            else:
                all_corruption.extend(data)
        except Exception as e:
            logger.warning(f"Could not load {f.name}: {e}")

    # Deduplicate by task_id
    correction_results = list({r['task_id']: r for r in all_correction}.values())
    preservation_results = list({r['task_id']: r for r in all_preservation}.values())
    corruption_results = list({r['task_id']: r for r in all_corruption}.values())

    # Recalculate metrics
    n_incorrect = len([r for r in correction_results if not r['baseline_passed']])
    n_corrected = len([r for r in correction_results if not r['baseline_passed'] and r['combined_correct']])
    correction_rate = (n_corrected / n_incorrect * 100) if n_incorrect > 0 else 0.0

    n_correct_pres = len([r for r in preservation_results if r['baseline_passed']])
    n_preserved = len([r for r in preservation_results if r['baseline_passed'] and r['combined_correct']])
    preservation_rate = (n_preserved / n_correct_pres * 100) if n_correct_pres > 0 else 0.0

    n_correct = len([r for r in corruption_results if r['baseline_passed']])
    n_corrupted = len([r for r in corruption_results if r['baseline_passed'] and not r['combined_correct']])
    corruption_rate = (n_corrupted / n_correct * 100) if n_correct > 0 else 0.0

    sims = [r.get('code_similarity', 1.0) for r in corruption_results if r['baseline_passed']]
    avg_similarity = sum(sims) / len(sims) if sims else 1.0
    composite_score = (corruption_rate + avg_similarity * 100) / 2

    logger.info(f"Phase 9.5 merged: correction={correction_rate:.1f}% ({n_corrected}/{n_incorrect}), "
                f"preservation={preservation_rate:.1f}% ({n_preserved}/{n_correct_pres}), "
                f"corruption={corruption_rate:.1f}% ({n_corrupted}/{n_correct})")

    # Save merged results
    save_json(correction_results, output_path / "correction_results.json")
    save_json(preservation_results, output_path / "preservation_results.json")
    save_json(corruption_results, output_path / "corruption_results.json")

    # Build summary from first GPU's summary (for metadata) + recalculated metrics
    summary = None
    summary_gpu_files = sorted(output_path.glob("phase_9_5_summary_gpu*.json"))
    if summary_gpu_files:
        try:
            summary = load_json(summary_gpu_files[0])
        except Exception as e:
            logger.warning(f"Could not load GPU summary: {e}")

    if summary is None:
        summary = {
            "phase": "9.5",
            "timestamp": datetime.now().isoformat(),
        }

    summary['correction_experiment'] = {
        'correction_rate': correction_rate,
        'n_incorrect_baseline': n_incorrect,
        'n_corrected': n_corrected,
    }
    summary['preservation_experiment'] = {
        'preservation_rate': preservation_rate,
        'n_correct_baseline': n_correct_pres,
        'n_preserved': n_preserved,
    }
    summary['corruption_experiment'] = {
        'corruption_rate': corruption_rate,
        'composite_score': composite_score,
        'avg_code_similarity': avg_similarity,
        'n_correct_baseline': n_correct,
        'n_corrupted': n_corrupted,
    }
    summary['parallel_merge'] = True
    summary['n_gpus'] = n_gpus

    save_json(summary, output_path / "phase_9_5_summary.json")

    # Regenerate visualization from merged summary
    try:
        from phase9_5_combined_analysis.combined_analyzer import CombinedOrthogonalSteeringAnalyzer
        viz_runner = object.__new__(CombinedOrthogonalSteeringAnalyzer)
        viz_runner.output_dir = output_path
        viz_runner._create_visualization(summary)
        logger.info("Saved combined_effects.png from merged summary")
    except Exception as e:
        logger.warning(f"Could not regenerate viz after merge: {e}")

    # Write phase manifest
    write_phase_output(
        phase="9.5",
        outputs={
            "primary": "phase_9_5_summary.json",
            "correction_results": "correction_results.json",
            "preservation_results": "preservation_results.json",
            "corruption_results": "corruption_results.json",
        },
        config=config,
        output_dir=str(output_path),
        config_keys=["model_name", "dataset_name", "direction_source"],
    )

    # Cleanup GPU-specific files
    _cleanup_gpu_files(output_path, [
        "correction_results_gpu*.json",
        "preservation_results_gpu*.json",
        "corruption_results_gpu*.json",
        "phase_9_5_summary_gpu*.json",
        "correction_checkpoint_gpu*.json",
        "preservation_checkpoint_gpu*.json",
        "corruption_checkpoint_gpu*.json",
    ])

    logger.info("PHASE 9.5 PARALLEL MERGE COMPLETE")
    return summary


def _merge_phase10_5_results(
    output_path: Path,
    n_gpus: int,
    config: Config,
) -> dict:
    """
    Merge Phase 10.5 selective ortho+steering results from parallel workers.

    Phase 10.5 produces correction_results_gpu{N}.json and preservation_results_gpu{N}.json.
    """
    from common.utils import save_json, load_json
    from common.phase_discovery import write_phase_output
    from datetime import datetime

    # Merge correction results
    correction_gpu_files = sorted(output_path.glob("correction_results_gpu*.json"))
    preservation_gpu_files = sorted(output_path.glob("preservation_results_gpu*.json"))

    if not correction_gpu_files:
        logger.warning("No correction GPU result files found for Phase 10.5 — merge skipped")
        return {}

    def _load_result_files(files):
        combined = []
        for f in files:
            try:
                data = load_json(f)
                if isinstance(data, dict):
                    combined.extend(data.values())
                else:
                    combined.extend(data)
            except Exception as e:
                logger.warning(f"Could not load {f.name}: {e}")
        return combined

    all_correction = _load_result_files(correction_gpu_files)
    all_preservation = _load_result_files(preservation_gpu_files)

    # Deduplicate by task_id
    correction_results = list({r['task_id']: r for r in all_correction}.values())
    preservation_results = list({r['task_id']: r for r in all_preservation}.values())

    # Recalculate metrics on merged data
    n_incorrect = len([r for r in correction_results if not r['baseline_passed']])
    n_corrected = len([r for r in correction_results if not r['baseline_passed'] and r['steered_correct']])
    correction_rate = (n_corrected / n_incorrect * 100) if n_incorrect > 0 else 0.0
    n_steered_correction = sum(1 for r in correction_results if r.get('was_steered', False))
    steering_trigger_rate = (n_steered_correction / n_incorrect * 100) if n_incorrect > 0 else 0.0

    n_correct = len([r for r in preservation_results if r['baseline_passed']])
    n_preserved = len([r for r in preservation_results if r['baseline_passed'] and r['steered_correct']])
    preservation_rate = (n_preserved / n_correct * 100) if n_correct > 0 else 0.0
    n_steered_preservation = sum(1 for r in preservation_results if r.get('was_steered', False))
    preservation_steer_rate = (n_steered_preservation / n_correct * 100) if n_correct > 0 else 0.0

    logger.info(f"Phase 10.5 merged: correction={correction_rate:.1f}% ({n_corrected}/{n_incorrect}), "
                f"preservation={preservation_rate:.1f}% ({n_preserved}/{n_correct})")

    # Save merged results
    save_json(correction_results, output_path / "correction_results.json")
    save_json(preservation_results, output_path / "preservation_results.json")

    # Build summary from first GPU's summary (for metadata) + recalculated metrics
    summary = None
    summary_gpu_files = sorted(output_path.glob("phase_10_5_summary_gpu*.json"))
    if summary_gpu_files:
        try:
            summary = load_json(summary_gpu_files[0])
        except Exception as e:
            logger.warning(f"Could not load GPU summary: {e}")

    if summary is None:
        summary = {
            "phase": "10.5",
            "timestamp": datetime.now().isoformat(),
        }

    summary['correction_experiment'] = {
        'correction_rate': correction_rate,
        'n_incorrect': n_incorrect,
        'n_corrected': n_corrected,
        'n_steered': n_steered_correction,
        'steering_trigger_rate': steering_trigger_rate,
    }
    summary['preservation_experiment'] = {
        'preservation_rate': preservation_rate,
        'n_correct': n_correct,
        'n_preserved': n_preserved,
        'n_steered': n_steered_preservation,
        'preservation_steer_rate': preservation_steer_rate,
    }
    summary['parallel_merge'] = True
    summary['n_gpus'] = n_gpus

    save_json(summary, output_path / "phase_10_5_summary.json")

    # Regenerate visualization from merged summary
    try:
        from phase10_5_selective_ortho_plus_steering.selective_combined_analyzer import SelectiveCombinedAnalyzer
        viz_runner = object.__new__(SelectiveCombinedAnalyzer)
        viz_runner.output_dir = output_path
        viz_runner.threshold = summary.get('threshold', 0.0)
        viz_runner._create_visualization(summary)
        logger.info("Saved selective_combined_effects.png from merged summary")
    except Exception as e:
        logger.warning(f"Could not regenerate viz after merge: {e}")

    # Write phase manifest
    write_phase_output(
        phase="10.5",
        outputs={
            "primary": "phase_10_5_summary.json",
            "correction_results": "correction_results.json",
            "preservation_results": "preservation_results.json",
        },
        config=config,
        output_dir=str(output_path),
        config_keys=["model_name", "dataset_name", "direction_source"],
    )

    # Cleanup GPU-specific files
    _cleanup_gpu_files(output_path, [
        "correction_results_gpu*.json",
        "preservation_results_gpu*.json",
        "phase_10_5_summary_gpu*.json",
        "correction_checkpoint_gpu*.json",
        "preservation_checkpoint_gpu*.json",
    ])

    logger.info("PHASE 10.5 PARALLEL MERGE COMPLETE")
    return summary


def _merge_phase4_12_json_results(
    output_path: Path,
    n_gpus: int,
    config: Config,
    phase_id: str = "4.12"
) -> dict:
    """
    Merge Phase 4.12 (or 7.7) zero-discrimination steering results from parallel workers.

    Phase 4.12/7.7 produces zero_disc_steering_results_gpu{N}.json files containing:
    - correction_results, corruption_results, preservation_results (dict keyed by task_id)
    - summary_metrics with correction_rate, corruption_rate, preservation_rate
    """
    from datetime import datetime
    from common.utils import save_json
    from common.steering_metrics import calculate_correction_rate, calculate_corruption_rate, calculate_preservation_rate
    from common.dataset_utils import compute_error_type_distribution

    # Load per-GPU JSON files
    gpu_data = _load_gpu_json_files(output_path, "zero_disc_steering_results_gpu*.json")

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
    preservation_rate = calculate_preservation_rate(preservation_list)

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
        phase=phase_id,
        outputs={
            "primary": "zero_disc_steering_results.json",
        },
        config=config,
        output_dir=str(output_path)
    )
    logger.info("Wrote phase_output.json manifest")

    # Cleanup per-GPU files
    _cleanup_gpu_files(output_path, ["zero_disc_steering_results_gpu*.json"])

    # Print summary
    logger.info("=" * 60)
    logger.info(f"PHASE {phase_id} PARALLEL MERGE COMPLETE")
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

    # Phase 4.8 and 7.6 use same JSON format (steering effect analysis)
    if phase_id in ("4.8", "7.6"):
        return _merge_phase4_8_results(output_path, n_gpus, config, phase_id=phase_id)

    # Phase 7.3 needs custom merge (saves as dataset_instruct_temp_0_0.parquet + metadata)
    if phase_id == "7.3":
        return _merge_phase7_3_results(output_path, n_gpus, config)

    # Phase 8.3 needs custom merge (JSON summary recalculated from parquet)
    if phase_id == "8.3":
        return _merge_phase8_3_results(output_path, n_gpus, config)

    # Phase 5.6 uses JSON output format (zero-disc orthogonalization)
    if phase_id == "5.6":
        return _merge_phase5_6_json_results(output_path, n_gpus, config)

    # Phase 5.3 uses JSON output format (weight orthogonalization)
    if phase_id == "5.3":
        return _merge_phase5_3_json_results(output_path, n_gpus, config)

    # Phase 9.5: combined orthogonalization + steering
    if phase_id == "9.5":
        return _merge_phase9_5_results(output_path, n_gpus, config)

    # Phase 10.5: selective ortho + selective steering
    if phase_id == "10.5":
        return _merge_phase10_5_results(output_path, n_gpus, config)

    # Phase 4.12 and 7.7 use JSON output format (zero-disc steering)
    if phase_id in ("4.12", "7.7"):
        return _merge_phase4_12_json_results(output_path, n_gpus, config, phase_id=phase_id)

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
            n_activation_files = len(correct_files) + len(incorrect_files)
            if n_activation_files > 0:
                logger.info(
                    f"Found {n_activation_files} activation files "
                    f"({len(correct_files)} correct, {len(incorrect_files)} incorrect) - "
                    f"cross-run checkpointing detected"
                )
                logger.info("No new results to merge. Using existing dataset.")
                # Return indicator that nothing needed to be done
                return {"checkpointed": True, "message": "All tasks already processed"}

        raise RuntimeError(
            f"No per-GPU result files found in {output_dir}. "
            f"Expected pattern: results_gpu*.parquet. "
            f"Check worker logs for errors."
        )

    if len(gpu_result_files) < n_gpus:
        raise RuntimeError(
            f"Only found {len(gpu_result_files)}/{n_gpus} GPU result files. "
            f"Some workers may have failed. Re-run the same command to resume — "
            f"successful GPUs' work is checkpointed."
        )

    logger.info(f"Found {len(gpu_result_files)} GPU result files to merge")

    # Merge parquet files (graceful degradation: skip corrupted files)
    dfs = []
    corrupted_files = []
    for result_file in gpu_result_files:
        try:
            df = pd.read_parquet(result_file)
            dfs.append(df)
            logger.info(f"  Loaded {len(df)} rows from {result_file.name}")
        except Exception as e:
            logger.error(f"  Corrupted GPU parquet file {result_file.name}: {e} — skipping")
            corrupted_files.append(result_file.name)
    if not dfs:
        raise RuntimeError(f"All GPU parquet files corrupted in {output_path}: {corrupted_files}")
    if corrupted_files:
        logger.warning(f"Skipped {len(corrupted_files)} corrupted files, merging {len(dfs)} valid files")

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
        old_file.unlink(missing_ok=True)
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

    # Phase 1: Create summary JSON (required by Phase 11.5)
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
