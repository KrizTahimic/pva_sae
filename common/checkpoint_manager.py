"""Unified checkpoint management for all phases.

This module provides a standardized way to save, load, and manage checkpoints
across all phases of the SAE-Code-Correctness project. It uses task ID tracking (not index)
for robustness against --start/--end argument variations.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from common.logging import get_logger
from common.utils import save_json, load_json

logger = get_logger(__name__)


@dataclass
class CheckpointData:
    """Data loaded from a checkpoint."""
    processed_task_ids: set[str]
    excluded_task_ids: set[str]
    results: list[dict]


class CheckpointManager:
    """Unified checkpoint management for all phases.

    Features:
    - Task ID-based tracking (survives --start/--end variations)
    - Version control (rejects incompatible checkpoints)
    - Automatic cleanup (keeps last N checkpoints)
    - Memory-aware saving (force save at high RAM usage)

    Usage:
        checkpoint_mgr = CheckpointManager(
            checkpoint_dir=output_dir / "checkpoints",
            experiment_name="correction",
            frequency=50
        )

        # Load existing checkpoint
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            results = checkpoint.results
            processed_ids = checkpoint.processed_task_ids
        else:
            results, processed_ids = [], set()

        # Process tasks
        for task in tasks:
            if task['task_id'] in processed_ids:
                continue

            result = process_task(task)
            results.append(result)
            processed_ids.add(task['task_id'])

            # Checkpoint if needed
            if checkpoint_mgr.should_save(len(results), memory_percent):
                checkpoint_mgr.save(results, processed_ids, excluded_ids)

        # Final cleanup
        checkpoint_mgr.cleanup_all()
    """

    VERSION = 2  # Bump when checkpoint format changes

    def __init__(
        self,
        checkpoint_dir: Path,
        experiment_name: str,
        frequency: int = 50,
        keep_last: int = 3,
        memory_threshold: float = 95.0,
        gpu_id: int = 0,
        n_gpus: int = 1
    ):
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
            experiment_name: Name of the experiment (e.g., "correction", "corruption")
            frequency: Save checkpoint every N processed items
            keep_last: Number of old checkpoints to keep
            memory_threshold: Force save when RAM usage exceeds this percentage
            gpu_id: GPU index for multi-GPU parallelization (default: 0)
            n_gpus: Total number of GPUs (default: 1, single-GPU mode)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.frequency = frequency
        self.keep_last = keep_last
        self.memory_threshold = memory_threshold
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_pattern(self, for_glob: bool = True) -> str:
        """Get the checkpoint filename pattern.

        Args:
            for_glob: If True, returns glob pattern with wildcard.
                      If False, returns format string for saving.

        Returns:
            Pattern string
        """
        if self.n_gpus > 1:
            if for_glob:
                return f"checkpoint_{self.experiment_name}_gpu{self.gpu_id}_*.json"
            else:
                return f"checkpoint_{self.experiment_name}_gpu{self.gpu_id}_{{timestamp}}.json"
        else:
            if for_glob:
                return f"checkpoint_{self.experiment_name}_*.json"
            else:
                return f"checkpoint_{self.experiment_name}_{{timestamp}}.json"

    def should_save(self, count: int, memory_percent: float = None) -> bool:
        """Check if we should save a checkpoint.

        Args:
            count: Number of items processed since last checkpoint
            memory_percent: Current RAM usage percentage (optional)

        Returns:
            True if checkpoint should be saved
        """
        if memory_percent is not None and memory_percent > self.memory_threshold:
            logger.warning(f"Memory at {memory_percent:.1f}%, forcing checkpoint save")
            return True
        return count > 0 and count % self.frequency == 0

    def save(
        self,
        results: list[dict],
        processed_ids: set[str],
        excluded_ids: set[str] = None
    ) -> Path:
        """Save checkpoint with version control.

        Args:
            results: List of result dictionaries
            processed_ids: Set of task IDs that have been processed
            excluded_ids: Set of task IDs that were excluded/failed (optional)

        Returns:
            Path to the saved checkpoint file
        """
        if excluded_ids is None:
            excluded_ids = set()

        checkpoint_data = {
            "version": self.VERSION,
            "experiment_name": self.experiment_name,
            "processed_task_ids": list(processed_ids),
            "excluded_task_ids": list(excluded_ids),
            "n_results": len(results),
            "n_processed": len(processed_ids),
            "n_excluded": len(excluded_ids),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pattern = self._get_checkpoint_pattern(for_glob=False)
        checkpoint_file = self.checkpoint_dir / pattern.format(timestamp=timestamp)

        save_json(checkpoint_data, checkpoint_file)
        logger.info(f"Saved checkpoint: {len(processed_ids)} processed, {len(excluded_ids)} excluded")

        # Clean up old checkpoints
        self._cleanup_old()
        return checkpoint_file

    def load(self) -> Optional[CheckpointData]:
        """Load most recent checkpoint, validating version.

        Returns:
            CheckpointData if a valid checkpoint exists, None otherwise

        Raises:
            ValueError: If checkpoint version doesn't match current VERSION
        """
        pattern = self._get_checkpoint_pattern(for_glob=True)
        files = sorted(self.checkpoint_dir.glob(pattern))

        if not files:
            return None

        latest = files[-1]
        logger.info(f"Loading checkpoint from {latest.name}")

        data = load_json(latest)

        # Version validation
        checkpoint_version = data.get("version")
        if checkpoint_version != self.VERSION:
            raise ValueError(
                f"Checkpoint version mismatch (got {checkpoint_version}, need {self.VERSION}). "
                f"Delete old checkpoints and restart:\n"
                f"  rm -rf {self.checkpoint_dir}/checkpoint_{self.experiment_name}_*.json"
            )

        processed_ids = set(str(tid) for tid in data.get("processed_task_ids", []))
        excluded_ids = set(str(tid) for tid in data.get("excluded_task_ids", []))
        results = data.get("results", [])

        logger.info(f"Resuming: {len(processed_ids)} processed, {len(excluded_ids)} excluded")

        return CheckpointData(
            processed_task_ids=processed_ids,
            excluded_task_ids=excluded_ids,
            results=results
        )

    def _cleanup_old(self) -> None:
        """Keep only last N checkpoints."""
        pattern = self._get_checkpoint_pattern(for_glob=True)
        files = sorted(self.checkpoint_dir.glob(pattern))

        if len(files) > self.keep_last:
            for old_file in files[:-self.keep_last]:
                old_file.unlink()
                logger.debug(f"Removed old checkpoint: {old_file.name}")

    def cleanup_all(self) -> None:
        """Remove all checkpoints after successful completion."""
        pattern = self._get_checkpoint_pattern(for_glob=True)
        files = list(self.checkpoint_dir.glob(pattern))

        for f in files:
            f.unlink()

        if files:
            logger.info(f"Cleaned up {len(files)} checkpoint file(s)")

    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists for this experiment."""
        pattern = self._get_checkpoint_pattern(for_glob=True)
        return any(self.checkpoint_dir.glob(pattern))
