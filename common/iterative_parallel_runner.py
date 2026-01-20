"""
Iterative parallelization for grid search phases.

Unlike parallel_runner.py (which runs entire phase independently per GPU),
this module coordinates iteration-by-iteration execution with merged results.

Architecture:
    Orchestrator (controls iteration)
        |
        |  Round 1: "Test value=X"
        |  +----------+----------+
        v  v          v          v
      GPU 0        GPU 1       GPU 2       GPU 3
      (1/4 probs)  (1/4 probs) (1/4 probs) (1/4 probs)
        |             |           |           |
        +-------------+-----------+-----------+
                      |
        Merge -> combined metrics for X (ALL problems)
                      |
        Early stop decision (based on FULL data)
                      |
        Round 2: "Test value=Y"
        ...

Supported phases:
    - 3.5: Temperature robustness (iterate over temperatures)
    - 4.5: Coefficient grid search (iterate over coefficients)
    - 4.6: Golden section refinement (iterate over refinement points)
    - 8.2: Threshold optimizer (iterate over percentiles)
"""

import os
import gc
import queue
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue, get_context
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import torch

from common.config import Config
from common.logging import get_logger

logger = get_logger(__name__)


# Timeout for worker operations (10 minutes per value evaluation)
DEFAULT_WORKER_TIMEOUT = 600


def _no_early_stop(*args) -> bool:
    """Default early stop function that never stops."""
    return False


def _default_merge_fn(gpu_results: list[dict]) -> dict:
    """Default merge function - combines results from all GPUs."""
    all_results = []
    for r in gpu_results:
        if 'results' in r:
            all_results.extend(r['results'])

    return {
        'n_gpus': len(gpu_results),
        'n_problems': len(all_results),
        'results': all_results
    }


@runtime_checkable
class PhaseEvaluator(Protocol):
    """Protocol for phases supporting iterative parallelization."""

    def __init__(self, config: Config, gpu_id: int, n_gpus: int) -> None:
        """Initialize and load model (called once per worker)."""
        ...

    def evaluate_single_value(self, value: Any) -> dict:
        """Evaluate ONE value on this GPU's subset of problems.

        Returns dict with at least:
        - 'value': the value tested
        - 'results': list of per-problem results
        - 'metrics': calculated metrics for this subset
        """
        ...


@dataclass
class IterativeParallelConfig:
    """Configuration for iterative parallel execution."""
    n_gpus: int
    timeout_per_iteration: int = DEFAULT_WORKER_TIMEOUT
    checkpoint_dir: Path | None = None


class IterativeParallelRunner:
    """
    Runs iterative grid search with synchronized rounds across GPUs.

    Usage:
        runner = IterativeParallelRunner(
            phase_evaluator_class=ThresholdEvaluator,
            config=config,
            n_gpus=4,
            values_to_test=[10, 20, 30, ...],
        )
        results = runner.run()
    """

    def __init__(
        self,
        phase_evaluator_class: type,
        config: Config,
        n_gpus: int,
        values_to_test: list[Any],
        early_stop_fn: Callable[[dict, list[dict]], bool] | None = None,
        merge_fn: Callable[[list[dict]], dict] | None = None,
        timeout_per_iteration: int = DEFAULT_WORKER_TIMEOUT,
        checkpoint_dir: Path | None = None,
    ):
        """
        Initialize iterative parallel runner.

        Args:
            phase_evaluator_class: Class implementing PhaseEvaluator protocol
            config: Configuration object
            n_gpus: Number of GPUs to use
            values_to_test: List of values to test (percentiles, coefficients, etc.)
            early_stop_fn: Optional function(current_result, history) -> bool for early stopping
            merge_fn: Optional function(gpu_results) -> merged_result
            timeout_per_iteration: Max time per iteration in seconds
            checkpoint_dir: Optional directory for iteration checkpoints
        """
        self.evaluator_class = phase_evaluator_class
        self.config = config
        self.n_gpus = n_gpus
        self.values_to_test = values_to_test
        self.early_stop_fn = early_stop_fn or _no_early_stop
        self.merge_fn = merge_fn or _default_merge_fn
        self.timeout = timeout_per_iteration
        self.checkpoint_dir = checkpoint_dir

    def run(self) -> dict:
        """
        Run iterative parallel search.

        Returns:
            dict with:
            - 'optimal_value': best value found
            - 'optimal_score': best score achieved
            - 'history': list of merged results per iteration
        """
        logger.info(f"Starting iterative parallel execution with {self.n_gpus} GPUs")
        logger.info(f"Values to test: {self.values_to_test}")

        # Use spawn context for CUDA compatibility
        ctx = get_context('spawn')

        # Create queues for communication
        task_queues: list[Queue] = [ctx.Queue() for _ in range(self.n_gpus)]
        result_queues: list[Queue] = [ctx.Queue() for _ in range(self.n_gpus)]

        # Spawn persistent workers (each loads model once)
        workers: list[Process] = []
        for gpu_id in range(self.n_gpus):
            p = ctx.Process(
                target=self._worker_loop,
                args=(gpu_id, task_queues[gpu_id], result_queues[gpu_id])
            )
            p.start()
            workers.append(p)
            logger.info(f"Started worker for GPU {gpu_id} (PID: {p.pid})")

        # Track optimization progress
        history: list[dict] = []
        optimal_value = None
        optimal_score = float('-inf')

        # Load checkpoint if exists
        completed_values = self._load_checkpoint() if self.checkpoint_dir else set()
        if completed_values:
            logger.info(f"Resuming from checkpoint, {len(completed_values)} values already completed")

        try:
            # Iterate through values
            for value in self.values_to_test:
                if value in completed_values:
                    logger.info(f"Skipping value={value} (already completed)")
                    continue

                logger.info(f"\n{'='*60}")
                logger.info(f"Evaluating value: {value}")
                logger.info(f"{'='*60}")

                # Send value to all workers
                for q in task_queues:
                    q.put(('evaluate', value))

                # Collect results from all workers with timeout
                gpu_results = self._collect_results(result_queues, value)

                if gpu_results is None:
                    # One or more workers failed - abort
                    logger.error(f"Worker failure during value={value}, aborting")
                    break

                # Merge results from all GPUs
                merged = self.merge_fn(gpu_results)
                merged['value'] = value

                # Extract score (support both 'score' and 'net_benefit')
                score = merged.get('score', merged.get('net_benefit', 0.0))
                merged['score'] = score

                # Track best
                if score > optimal_score:
                    optimal_score = score
                    optimal_value = value
                    logger.info(f"New optimal: value={value}, score={score:.4f}")

                history.append(merged)

                # Save checkpoint after each iteration
                if self.checkpoint_dir:
                    self._save_checkpoint(value, merged)

                # Check early stopping (on FULL merged data)
                if self.early_stop_fn(merged, history):
                    logger.info(f"Early stopping triggered at value={value}")
                    break

        finally:
            # Shutdown workers gracefully
            self._shutdown_workers(workers, task_queues)

        logger.info(f"\nIterative parallel execution complete")
        logger.info(f"Optimal value: {optimal_value}, Score: {optimal_score:.4f}")

        return {
            'optimal_value': optimal_value,
            'optimal_score': optimal_score,
            'history': history
        }

    def _worker_loop(self, gpu_id: int, task_queue: Queue, result_queue: Queue):
        """Worker process that stays alive for multiple evaluations."""
        # Set GPU before importing torch-dependent modules
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Import logging after setting GPU
        from common.logging import get_logger
        worker_logger = get_logger(f"iterative_worker_{gpu_id}")

        worker_logger.info(f"Worker {gpu_id}: Initializing on GPU {gpu_id}")

        try:
            # Load model ONCE
            evaluator = self.evaluator_class(
                config=self.config,
                gpu_id=gpu_id,
                n_gpus=self.n_gpus
            )
            worker_logger.info(f"Worker {gpu_id}: Model loaded successfully")

            # Process tasks until shutdown
            while True:
                try:
                    cmd, value = task_queue.get(timeout=self.timeout)

                    if cmd == 'shutdown':
                        worker_logger.info(f"Worker {gpu_id}: Received shutdown signal")
                        break

                    if cmd == 'evaluate':
                        worker_logger.info(f"Worker {gpu_id}: Evaluating value={value}")
                        try:
                            result = evaluator.evaluate_single_value(value)
                            result['gpu_id'] = gpu_id
                            result['status'] = 'success'
                            result_queue.put(result)
                            worker_logger.info(f"Worker {gpu_id}: Completed value={value}")
                        except Exception as e:
                            error_result = {
                                'gpu_id': gpu_id,
                                'value': value,
                                'status': 'error',
                                'error': str(e),
                                'traceback': traceback.format_exc()
                            }
                            result_queue.put(error_result)
                            worker_logger.error(f"Worker {gpu_id}: Failed on value={value}: {e}")

                        # Memory cleanup after each evaluation
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                except queue.Empty:
                    worker_logger.warning(f"Worker {gpu_id}: Timeout waiting for task")
                    continue

        except Exception as e:
            worker_logger.error(f"Worker {gpu_id}: Fatal error: {e}")
            worker_logger.error(traceback.format_exc())
            # Put error result to unblock orchestrator
            result_queue.put({
                'gpu_id': gpu_id,
                'status': 'fatal_error',
                'error': str(e)
            })

    def _collect_results(self, result_queues: list[Queue], value: Any) -> list[dict] | None:
        """Collect results from all workers with timeout and error handling."""
        gpu_results = []

        for i, rq in enumerate(result_queues):
            try:
                result = rq.get(timeout=self.timeout)

                if result.get('status') == 'error':
                    logger.error(f"GPU {i} error on value={value}: {result.get('error')}")
                    if 'traceback' in result:
                        logger.error(f"Traceback: {result['traceback']}")
                    return None

                if result.get('status') == 'fatal_error':
                    logger.error(f"GPU {i} fatal error: {result.get('error')}")
                    return None

                gpu_results.append(result)
                logger.info(f"GPU {i}: Received results for value={value}")

            except queue.Empty:
                logger.error(f"GPU {i} timed out on value={value}")
                return None

        return gpu_results

    def _shutdown_workers(self, workers: list[Process], task_queues: list[Queue]):
        """Shutdown workers gracefully, with fallback to terminate."""
        logger.info("Shutting down workers...")

        # Send shutdown signal
        for q in task_queues:
            try:
                q.put(('shutdown', None), timeout=5)
            except Exception:
                pass

        # Wait for workers to finish
        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Force terminating worker {p.pid}")
                p.terminate()
                p.join(timeout=5)

        logger.info("All workers shut down")

    def _load_checkpoint(self) -> set:
        """Load checkpoint to get completed values."""
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            return set()

        checkpoint_file = self.checkpoint_dir / "iteration_progress.json"
        if not checkpoint_file.exists():
            return set()

        from common.utils import load_json
        try:
            data = load_json(checkpoint_file)
            return set(data.get('completed_values', []))
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return set()

    def _save_checkpoint(self, value: Any, result: dict):
        """Save checkpoint after completing a value."""
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load existing progress
        checkpoint_file = self.checkpoint_dir / "iteration_progress.json"
        from common.utils import load_json, save_json

        if checkpoint_file.exists():
            data = load_json(checkpoint_file)
        else:
            data = {'completed_values': [], 'results': {}}

        # Update with new completion
        if value not in data['completed_values']:
            data['completed_values'].append(value)

        # Store result summary (not full results to save space)
        data['results'][str(value)] = {
            'score': result.get('score', 0),
            'n_problems': result.get('n_problems', 0)
        }

        save_json(data, checkpoint_file)
        logger.debug(f"Checkpoint saved for value={value}")


def run_iterative_parallel(
    phase_id: str,
    config: Config,
    n_gpus: int
) -> dict:
    """
    Entry point for running iterative parallel phases.

    This function creates the appropriate evaluator/orchestrator for the given phase
    and runs the iterative parallel execution.

    Args:
        phase_id: Phase ID (e.g., "3.5", "4.5", "8.2")
        config: Configuration object
        n_gpus: Number of GPUs

    Returns:
        Phase results
    """
    logger.info(f"Starting iterative parallel execution for Phase {phase_id}")

    # Import phase-specific orchestrators
    if phase_id == "8.2":
        from phase8_2_threshold_optimizer.threshold_optimizer import ThresholdOrchestrator
        orchestrator = ThresholdOrchestrator(config, n_gpus=n_gpus)
        return orchestrator.run()

    elif phase_id == "4.5":
        from phase4_5_coefficient_grid_search.steering_coefficient_selector import CoefficientOrchestrator
        orchestrator = CoefficientOrchestrator(config, n_gpus=n_gpus)
        return orchestrator.run()

    elif phase_id == "4.6":
        from phase4_6_golden_section_refinement.golden_section_refiner import RefinementOrchestrator
        orchestrator = RefinementOrchestrator(config, n_gpus=n_gpus)
        return orchestrator.run()

    elif phase_id == "3.5":
        from phase3_5_temperature_robustness.temperature_runner import TemperatureOrchestrator
        orchestrator = TemperatureOrchestrator(config, n_gpus=n_gpus)
        return orchestrator.run()

    else:
        raise ValueError(f"Phase {phase_id} does not support iterative parallelization")
