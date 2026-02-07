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

Per-GPU-Per-Value Checkpointing:
    - Each GPU saves results immediately when it completes (not waiting for merge)
    - Track completed task_ids (strings, not indices)
    - On restart: redistribute ONLY remaining tasks across ALL available GPUs
    - Require all tasks complete before proceeding to next value

Supported phases:
    - 3.5: Temperature robustness (iterate over temperatures)
    - 4.5: Coefficient grid search (iterate over coefficients)
    - 4.6: Golden section refinement (iterate over refinement points)
    - 8.2: Threshold optimizer (iterate over percentiles)
"""

import json
import os
import gc
import queue
import traceback
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Process, Queue, get_context
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import pandas as pd
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

    def evaluate_single_value(self, value: Any, task_ids: list[str] | None = None) -> dict:
        """Evaluate ONE value on this GPU's subset of problems.

        Args:
            value: The value to evaluate (temperature, coefficient, percentile, etc.)
            task_ids: Optional list of specific task_ids to process. If None,
                     use the GPU's pre-filtered data (legacy/sequential mode).

        Returns dict with at least:
        - 'value': the value tested
        - 'results': list of per-problem results (each must have 'task_id')
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
        all_task_ids: list[str] | None = None,
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
            all_task_ids: Optional list of all task_ids to process (discovered from evaluator if not provided)
        """
        self.evaluator_class = phase_evaluator_class
        self.config = config
        self.n_gpus = n_gpus
        self.values_to_test = values_to_test
        self.early_stop_fn = early_stop_fn or _no_early_stop
        self.merge_fn = merge_fn or _default_merge_fn
        self.timeout = timeout_per_iteration
        self.checkpoint_dir = checkpoint_dir
        self.all_task_ids = all_task_ids  # Will be discovered if not provided

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

        # Create a shared semaphore to prevent CPU contention during code evaluation.
        # When multiple GPU workers finish generation simultaneously and all try to
        # evaluate code, they compete for CPU resources, causing spurious timeouts.
        # This semaphore limits concurrent evaluations.
        import multiprocessing as mp
        manager = mp.Manager()
        self.eval_semaphore = manager.Semaphore(2)
        logger.info("Created shared evaluation semaphore (max 2 concurrent evals)")

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

        # Discover all task_ids if not provided
        if self.all_task_ids is None:
            logger.info("Discovering all task_ids from workers...")
            self.all_task_ids = self._discover_task_ids(task_queues, result_queues)
            logger.info(f"Discovered {len(self.all_task_ids)} total task_ids")

        # Track optimization progress
        history: list[dict] = []
        optimal_value = None
        optimal_score = float('-inf')

        # Load orchestrator state (completed values)
        completed_values = self._load_orchestrator_state()
        if completed_values:
            logger.info(f"Resuming from checkpoint, {len(completed_values)} values already completed")

        try:
            # Iterate through values
            for value in self.values_to_test:
                if value in completed_values:
                    # Check if there are actually remaining tasks (task set may have changed)
                    remaining_task_ids = self._get_remaining_tasks_for_value(value)
                    if not remaining_task_ids:
                        # Truly complete - skip
                        logger.info(f"Skipping value={value} (already completed, 0 remaining)")
                        merged = self._merge_value_results(value)
                        if merged:
                            history.append(merged)
                            score = merged.get('score', merged.get('net_benefit', 0.0))
                            if score > optimal_score:
                                optimal_score = score
                                optimal_value = value
                        continue
                    else:
                        # Task set changed - need to process remaining
                        logger.info(f"Value={value} marked complete but {len(remaining_task_ids)} tasks remain")
                        # Fall through to normal processing below

                logger.info(f"\n{'='*60}")
                logger.info(f"Evaluating value: {value}")
                logger.info(f"{'='*60}")

                # Check what's already done for this value
                remaining_task_ids = self._get_remaining_tasks_for_value(value)

                if not remaining_task_ids:
                    # All done for this value - just merge and continue
                    logger.info(f"All tasks complete for value={value}, merging results")
                    merged = self._merge_value_results(value)
                    if merged:
                        merged['value'] = value
                        score = merged.get('score', merged.get('net_benefit', 0.0))
                        merged['score'] = score
                        if score > optimal_score:
                            optimal_score = score
                            optimal_value = value
                            logger.info(f"New optimal: value={value}, score={score:.4f}")
                        history.append(merged)
                        self._save_orchestrator_state(value, merged)
                        if self.early_stop_fn(merged, history):
                            logger.info(f"Early stopping triggered at value={value}")
                            break
                    continue

                logger.info(f"Remaining tasks for value={value}: {len(remaining_task_ids)}")

                # Distribute remaining tasks across GPUs
                task_assignments = self._distribute_tasks(remaining_task_ids, self.n_gpus)

                # Send (value, task_ids) to workers
                for gpu_id, tq in enumerate(task_queues):
                    assigned_tasks = task_assignments.get(gpu_id, [])
                    tq.put(('evaluate', value, assigned_tasks))
                    logger.debug(f"Sent {len(assigned_tasks)} tasks to GPU {gpu_id}")

                # Collect results from all workers (workers save their own checkpoints)
                gpu_results = self._collect_results(result_queues, value)

                if gpu_results is None:
                    # One or more workers failed - but we have checkpoints, so continue
                    logger.warning(f"Some workers failed during value={value}, checking checkpoints")
                    # Check if we now have all tasks complete
                    remaining_after_failure = self._get_remaining_tasks_for_value(value)
                    if remaining_after_failure:
                        logger.error(f"Still {len(remaining_after_failure)} tasks incomplete for value={value}")
                        logger.error("Run will resume these on restart")
                        # Don't break - save what we have and continue to next value
                        # Or you could choose to retry here
                    else:
                        logger.info(f"Despite failures, all tasks are complete for value={value}")

                # Check if all tasks now complete
                remaining_check = self._get_remaining_tasks_for_value(value)
                if remaining_check:
                    logger.warning(f"Value {value}: {len(remaining_check)} tasks still incomplete")
                    # Continue to next value - incomplete values will be retried on restart
                    continue

                # Merge results from all GPU checkpoints
                merged = self._merge_value_results(value)
                if merged is None:
                    logger.error(f"Failed to merge results for value={value}")
                    continue

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

                # Save orchestrator state after each completed value
                self._save_orchestrator_state(value, merged)

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
        from common.dataset_utils import set_eval_semaphore
        worker_logger = get_logger(f"iterative_worker_{gpu_id}")

        worker_logger.info(f"Worker {gpu_id}: Initializing on GPU {gpu_id}")

        # Set the evaluation semaphore for this worker process
        # All evaluate_code_with_error_type() calls will use this semaphore
        if hasattr(self, 'eval_semaphore') and self.eval_semaphore is not None:
            set_eval_semaphore(self.eval_semaphore)
            worker_logger.info(f"Worker {gpu_id}: Evaluation semaphore configured")

        try:
            # Load model ONCE - pass n_gpus=1 so evaluator doesn't filter data
            # We'll handle task distribution explicitly via task_ids
            evaluator = self.evaluator_class(
                config=self.config,
                gpu_id=gpu_id,
                n_gpus=1  # Don't pre-filter - we pass task_ids explicitly
            )
            worker_logger.info(f"Worker {gpu_id}: Model loaded successfully")

            # Process tasks until shutdown
            while True:
                try:
                    msg = task_queue.get(timeout=self.timeout)

                    # Handle different message formats
                    if isinstance(msg, tuple) and len(msg) == 2:
                        cmd, value = msg
                        task_ids = None  # Legacy format
                    elif isinstance(msg, tuple) and len(msg) == 3:
                        cmd, value, task_ids = msg
                    else:
                        worker_logger.error(f"Worker {gpu_id}: Invalid message format: {msg}")
                        continue

                    if cmd == 'shutdown':
                        worker_logger.info(f"Worker {gpu_id}: Received shutdown signal")
                        break

                    if cmd == 'get_task_ids':
                        # Return all task_ids this evaluator knows about
                        all_ids = self._get_evaluator_task_ids(evaluator)
                        result_queue.put({
                            'status': 'task_ids',
                            'task_ids': all_ids,
                            'gpu_id': gpu_id
                        })
                        continue

                    if cmd == 'evaluate':
                        worker_logger.info(f"Worker {gpu_id}: Evaluating value={value} "
                                          f"with {len(task_ids) if task_ids else 'all'} tasks")
                        try:
                            # Pass task_ids to evaluator
                            result = evaluator.evaluate_single_value(value, task_ids=task_ids)
                            result['gpu_id'] = gpu_id
                            result['status'] = 'success'

                            # Save checkpoint immediately (before reporting to orchestrator)
                            self._save_gpu_checkpoint(value, gpu_id, result)

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

    def _get_evaluator_task_ids(self, evaluator) -> list[str]:
        """Extract task_ids from an evaluator's data.

        If the evaluator has a `get_relevant_task_ids()` method, use that
        (this allows evaluators to return only task_ids that will actually
        be evaluated based on experiment mode/steering type).
        """
        # Preferred: use evaluator's own method if available
        if hasattr(evaluator, 'get_relevant_task_ids'):
            return evaluator.get_relevant_task_ids()

        # Fallback: extract from common data attributes
        task_ids = []

        # Try common data attribute patterns
        for attr in ['analysis_data', 'dataset', 'baseline_data', 'data']:
            if hasattr(evaluator, attr):
                data = getattr(evaluator, attr)
                if hasattr(data, 'task_id'):
                    task_ids.extend(data['task_id'].tolist())
                    break

        # Also check split datasets (for phases that have correct/incorrect splits)
        for attr in ['incorrect_problems', 'correct_problems',
                     'initially_incorrect_data', 'initially_correct_data']:
            if hasattr(evaluator, attr):
                data = getattr(evaluator, attr)
                if hasattr(data, 'task_id'):
                    task_ids.extend(data['task_id'].tolist())

        return list(set(task_ids))  # Deduplicate

    def _discover_task_ids(self, task_queues: list[Queue], result_queues: list[Queue]) -> list[str]:
        """Discover all task_ids from workers."""
        # Ask first worker for task_ids
        task_queues[0].put(('get_task_ids', None, None))

        try:
            result = result_queues[0].get(timeout=self.timeout)
            if result.get('status') == 'task_ids':
                return result['task_ids']
        except queue.Empty:
            logger.warning("Timeout getting task_ids from worker")

        return []

    def _collect_results(self, result_queues: list[Queue], value: Any) -> list[dict] | None:
        """Collect results from all workers with timeout and error handling."""
        gpu_results = []
        had_errors = False

        for i, rq in enumerate(result_queues):
            try:
                result = rq.get(timeout=self.timeout)

                if result.get('status') == 'error':
                    logger.error(f"GPU {i} error on value={value}: {result.get('error')}")
                    if 'traceback' in result:
                        logger.error(f"Traceback: {result['traceback']}")
                    had_errors = True
                    continue  # Continue collecting other GPUs' results

                if result.get('status') == 'fatal_error':
                    logger.error(f"GPU {i} fatal error: {result.get('error')}")
                    had_errors = True
                    continue

                gpu_results.append(result)
                logger.info(f"GPU {i}: Received results for value={value}")

            except queue.Empty:
                logger.error(f"GPU {i} timed out on value={value}")
                had_errors = True
                continue

        if had_errors and not gpu_results:
            return None  # Total failure

        return gpu_results if gpu_results else None

    def _shutdown_workers(self, workers: list[Process], task_queues: list[Queue]):
        """Shutdown workers gracefully, with fallback to terminate."""
        logger.info("Shutting down workers...")

        # Send shutdown signal
        for q in task_queues:
            try:
                q.put(('shutdown', None, None), timeout=5)
            except Exception as e:
                logger.debug(f"Failed to send shutdown to worker queue: {e}")

        # Wait for workers to finish
        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Force terminating worker {p.pid}")
                p.terminate()
                p.join(timeout=5)

        logger.info("All workers shut down")

    # =========================================================================
    # Per-GPU-Per-Value Checkpoint Methods
    # =========================================================================

    def _get_value_checkpoint_dir(self, value: Any) -> Path:
        """Get checkpoint directory for a specific value."""
        if not self.checkpoint_dir:
            return None
        value_str = str(value).replace('.', '_').replace('-', 'neg')
        return self.checkpoint_dir / f"value_{value_str}"

    def _save_gpu_checkpoint(self, value: Any, gpu_id: int, result: dict):
        """Save per-GPU checkpoint, merging with any existing data."""
        if not self.checkpoint_dir:
            return

        value_dir = self._get_value_checkpoint_dir(value)
        value_dir.mkdir(parents=True, exist_ok=True)

        parquet_file = value_dir / f"gpu_{gpu_id}_results.parquet"
        meta_file = value_dir / f"gpu_{gpu_id}_results.meta.json"

        # Get new results
        new_results = result.get('results', [])
        new_task_ids = {r.get('task_id') for r in new_results if r.get('task_id')}

        # Load existing data (if any)
        existing_results = []
        existing_task_ids = set()

        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    old_meta = json.load(f)
                existing_task_ids = set(old_meta.get('processed_task_ids', []))
            except Exception as e:
                raise RuntimeError(
                    f"Corrupted checkpoint metadata at {meta_file}: {e}\n"
                    f"To recover, delete the metadata file and restart:\n"
                    f"  rm {meta_file}"
                )

        if parquet_file.exists() and existing_task_ids:
            try:
                old_df = pd.read_parquet(parquet_file)
                existing_results = old_df.to_dict('records')
            except Exception as e:
                logger.warning(f"Failed to load existing parquet: {e}")

        # Merge: old results + new results (deduplicate by task_id)
        combined_results = []
        seen_task_ids = set()

        # Add old results first (will be overwritten by new if duplicate)
        for r in existing_results:
            task_id = r.get('task_id')
            if task_id and task_id not in new_task_ids:
                combined_results.append(r)
                seen_task_ids.add(task_id)

        # Add new results (these take priority)
        for r in new_results:
            task_id = r.get('task_id')
            if task_id and task_id not in seen_task_ids:
                combined_results.append(r)
                seen_task_ids.add(task_id)

        # Merge task_ids
        combined_task_ids = list(existing_task_ids | new_task_ids)

        # Save merged parquet
        if combined_results:
            results_df = pd.DataFrame(combined_results)
            results_df.to_parquet(parquet_file, index=False)

        # Save merged metadata
        meta = {
            'gpu_id': gpu_id,
            'value': value,
            'processed_task_ids': combined_task_ids,
            'n_results': len(combined_results),
            'timestamp': datetime.now().isoformat()
        }
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        logger.debug(f"Saved checkpoint for GPU {gpu_id}, value={value}: "
                     f"{len(new_results)} new + {len(existing_results)} existing = "
                     f"{len(combined_results)} total results")

    def _get_remaining_tasks_for_value(self, value: Any) -> list[str]:
        """Load existing checkpoints and return unprocessed task_ids."""
        if not self.checkpoint_dir or not self.all_task_ids:
            return self.all_task_ids or []

        value_dir = self._get_value_checkpoint_dir(value)
        if not value_dir or not value_dir.exists():
            return self.all_task_ids

        # Collect all processed task_ids from existing GPU checkpoints
        processed_task_ids = set()

        for meta_file in value_dir.glob("gpu_*_results.meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                processed_task_ids.update(meta.get('processed_task_ids', []))
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata {meta_file}: {e}")

        # Return task_ids that haven't been processed
        remaining = [tid for tid in self.all_task_ids if tid not in processed_task_ids]
        return remaining

    def _distribute_tasks(self, task_ids: list[str], n_gpus: int) -> dict[int, list[str]]:
        """Distribute tasks across GPUs using round-robin."""
        assignments = {i: [] for i in range(n_gpus)}

        for idx, task_id in enumerate(task_ids):
            gpu_id = idx % n_gpus
            assignments[gpu_id].append(task_id)

        return assignments

    def _merge_value_results(self, value: Any) -> dict | None:
        """Merge all GPU checkpoint files for a value, deduplicating by task_id."""
        if not self.checkpoint_dir:
            return None

        value_dir = self._get_value_checkpoint_dir(value)
        if not value_dir or not value_dir.exists():
            return None

        # Load all parquet files for this value
        dfs = []
        for parquet_file in value_dir.glob("gpu_*_results.parquet"):
            try:
                dfs.append(pd.read_parquet(parquet_file))
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {parquet_file}: {e}")

        if not dfs:
            return None

        merged = pd.concat(dfs, ignore_index=True)
        if 'task_id' in merged.columns:
            merged = merged.drop_duplicates(subset=['task_id'], keep='last')
        all_results = merged.to_dict('records')

        if not all_results:
            return None

        # Use the custom merge function to compute metrics
        # Wrap results in expected format, including value for phase-specific merge functions
        # (e.g., temperature_runner expects 'temperature' key)
        merged = self.merge_fn([{'results': all_results, 'value': value, 'temperature': value}])
        merged['value'] = value
        merged['n_problems'] = len(all_results)

        return merged

    def _all_tasks_complete(self, value: Any) -> bool:
        """Check if all tasks are complete for a value."""
        remaining = self._get_remaining_tasks_for_value(value)
        return len(remaining) == 0

    def _load_orchestrator_state(self) -> set:
        """Load orchestrator state to get completed values."""
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            return set()

        state_file = self.checkpoint_dir / "orchestrator_state.json"
        if not state_file.exists():
            return set()

        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            completed = set(data.get('completed_values', []))
            return completed
        except Exception as e:
            logger.warning(f"Failed to load orchestrator state: {e}")
            return set()

    def _save_orchestrator_state(self, value: Any, result: dict):
        """Save orchestrator state after completing a value."""
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state_file = self.checkpoint_dir / "orchestrator_state.json"

        # Load existing state
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load orchestrator state ({e}), starting fresh")
                data = {'completed_values': [], 'results': {}}
        else:
            data = {'completed_values': [], 'results': {}}

        # Update with new completion
        if value not in data['completed_values']:
            data['completed_values'].append(value)

        # Store result summary (not full results to save space)
        data['results'][str(value)] = {
            'score': result.get('score', 0),
            'n_problems': result.get('n_problems', 0),
            'timestamp': datetime.now().isoformat()
        }

        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"Orchestrator state saved for value={value}")


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
