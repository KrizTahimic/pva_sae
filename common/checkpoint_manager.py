"""Unified checkpoint management for all phases.

This module provides a standardized way to save, load, and manage checkpoints
across all phases of the SAE-Code-Correctness project. It uses task ID tracking (not index)
for robustness against --start/--end argument variations.

Supports two output formats:
- JSON: For phases with lightweight results (steering experiments, metrics)
- Parquet: For phases with heavy data (code generation, activations)
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

from common.logging import get_logger
from common.utils import save_json, load_json

logger = get_logger(__name__)


@dataclass
class CheckpointData:
    """Data loaded from a JSON checkpoint."""
    processed_task_ids: set[str]
    excluded_task_ids: set[str]
    results: list[dict]


@dataclass
class ParquetCheckpointData:
    """Data loaded from a parquet checkpoint."""
    processed_task_ids: set[str]
    excluded_task_ids: set[str]
    results_df: pd.DataFrame


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
        n_gpus: int = 1,
        output_format: Literal["json", "parquet"] = "json"
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
            output_format: Output format - "json" (default) or "parquet"
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.frequency = frequency
        self.keep_last = keep_last
        self.memory_threshold = memory_threshold
        self.gpu_id = gpu_id
        self.n_gpus = n_gpus
        self.output_format = output_format
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_pattern(self, for_glob: bool = True, file_format: str = None) -> str:
        """Get the checkpoint filename pattern.

        Args:
            for_glob: If True, returns glob pattern with wildcard.
                      If False, returns format string for saving.
            file_format: Override format extension ("json" or "parquet").
                        If None, uses self.output_format.

        Returns:
            Pattern string
        """
        ext = file_format or self.output_format
        if self.n_gpus > 1:
            if for_glob:
                return f"checkpoint_{self.experiment_name}_gpu{self.gpu_id}_*.{ext}"
            else:
                return f"checkpoint_{self.experiment_name}_gpu{self.gpu_id}_{{timestamp}}.{ext}"
        else:
            if for_glob:
                return f"checkpoint_{self.experiment_name}_*.{ext}"
            else:
                return f"checkpoint_{self.experiment_name}_{{timestamp}}.{ext}"

    def _get_exclusion_pattern(self, for_glob: bool = True) -> str:
        """Get the exclusion filename pattern (always JSON).

        Args:
            for_glob: If True, returns glob pattern with wildcard.
                      If False, returns format string for saving.

        Returns:
            Pattern string
        """
        if self.n_gpus > 1:
            if for_glob:
                return f"checkpoint_{self.experiment_name}_gpu{self.gpu_id}_*_exclusions.json"
            else:
                return f"checkpoint_{self.experiment_name}_gpu{self.gpu_id}_{{timestamp}}_exclusions.json"
        else:
            if for_glob:
                return f"checkpoint_{self.experiment_name}_*_exclusions.json"
            else:
                return f"checkpoint_{self.experiment_name}_{{timestamp}}_exclusions.json"

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
                old_file.unlink(missing_ok=True)
                logger.debug(f"Removed old checkpoint: {old_file.name}")

    def cleanup_all(self) -> None:
        """Remove all checkpoints after successful completion."""
        pattern = self._get_checkpoint_pattern(for_glob=True)
        files = list(self.checkpoint_dir.glob(pattern))

        for f in files:
            f.unlink(missing_ok=True)

        if files:
            logger.info(f"Cleaned up {len(files)} checkpoint file(s)")

    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists for this experiment."""
        pattern = self._get_checkpoint_pattern(for_glob=True)
        return any(self.checkpoint_dir.glob(pattern))

    # ==================== PARQUET FORMAT METHODS ====================

    def save_parquet(
        self,
        results_df: pd.DataFrame,
        processed_ids: set[str],
        excluded_ids: set[str] = None,
        excluded_tasks: list[dict] = None
    ) -> Path:
        """Save checkpoint in parquet format for data-heavy phases.

        Args:
            results_df: DataFrame with results (must have 'task_id' column)
            processed_ids: Set of task IDs that have been processed
            excluded_ids: Set of task IDs that were excluded/failed (optional)
            excluded_tasks: List of excluded task dicts with error info (optional)

        Returns:
            Path to the saved checkpoint file
        """
        if excluded_ids is None:
            excluded_ids = set()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main results as parquet
        pattern = self._get_checkpoint_pattern(for_glob=False, file_format="parquet")
        checkpoint_file = self.checkpoint_dir / pattern.format(timestamp=timestamp)
        results_df.to_parquet(checkpoint_file, index=False)

        # Save metadata as JSON sidecar (same name but .json)
        metadata = {
            "version": self.VERSION,
            "experiment_name": self.experiment_name,
            "processed_task_ids": list(processed_ids),
            "excluded_task_ids": list(excluded_ids),
            "n_results": len(results_df),
            "n_processed": len(processed_ids),
            "n_excluded": len(excluded_ids),
            "timestamp": datetime.now().isoformat()
        }
        metadata_file = checkpoint_file.with_suffix(".meta.json")
        save_json(metadata, metadata_file)

        # Save exclusions if provided
        if excluded_tasks:
            exclusion_pattern = self._get_exclusion_pattern(for_glob=False)
            exclusion_file = self.checkpoint_dir / exclusion_pattern.format(timestamp=timestamp)
            save_json(excluded_tasks, exclusion_file)

        logger.info(f"Saved parquet checkpoint: {len(results_df)} results, {len(excluded_ids)} excluded")

        # Clean up old checkpoints
        self._cleanup_old_parquet()
        return checkpoint_file

    def load_parquet(self) -> Optional[ParquetCheckpointData]:
        """Load most recent parquet checkpoint, validating version.

        Returns:
            ParquetCheckpointData if a valid checkpoint exists, None otherwise

        Raises:
            ValueError: If checkpoint version doesn't match current VERSION
        """
        pattern = self._get_checkpoint_pattern(for_glob=True, file_format="parquet")
        files = sorted(self.checkpoint_dir.glob(pattern))

        if not files:
            return None

        latest = files[-1]
        logger.info(f"Loading parquet checkpoint from {latest.name}")

        # Load metadata from JSON sidecar
        metadata_file = latest.with_suffix(".meta.json")
        if not metadata_file.exists():
            raise ValueError(
                f"Metadata file not found for checkpoint {latest.name}. "
                f"Expected: {metadata_file.name}"
            )

        metadata = load_json(metadata_file)

        # Version validation
        checkpoint_version = metadata.get("version")
        if checkpoint_version != self.VERSION:
            raise ValueError(
                f"Checkpoint version mismatch (got {checkpoint_version}, need {self.VERSION}). "
                f"Delete old checkpoints and restart:\n"
                f"  rm -rf {self.checkpoint_dir}/checkpoint_{self.experiment_name}_*.parquet"
            )

        # Load results DataFrame
        results_df = pd.read_parquet(latest)

        processed_ids = set(str(tid) for tid in metadata.get("processed_task_ids", []))
        excluded_ids = set(str(tid) for tid in metadata.get("excluded_task_ids", []))

        logger.info(f"Resuming: {len(processed_ids)} processed, {len(excluded_ids)} excluded")

        return ParquetCheckpointData(
            processed_task_ids=processed_ids,
            excluded_task_ids=excluded_ids,
            results_df=results_df
        )

    def load_all_parquet_checkpoints(self) -> Optional[ParquetCheckpointData]:
        """Load and merge all parquet checkpoints (for cross-run resumption).

        This is useful when resuming from multiple checkpoint files that may
        have been saved at different times (e.g., from previous partial runs).

        Returns:
            ParquetCheckpointData with merged results, None if no checkpoints exist
        """
        pattern = self._get_checkpoint_pattern(for_glob=True, file_format="parquet")
        files = sorted(self.checkpoint_dir.glob(pattern))

        if not files:
            return None

        all_dfs = []
        all_processed = set()
        all_excluded = set()

        for checkpoint_file in files:
            metadata_file = checkpoint_file.with_suffix(".meta.json")
            if not metadata_file.exists():
                logger.warning(f"Skipping checkpoint without metadata: {checkpoint_file.name}")
                continue

            metadata = load_json(metadata_file)
            df = pd.read_parquet(checkpoint_file)

            all_dfs.append(df)
            all_processed.update(str(tid) for tid in metadata.get("processed_task_ids", []))
            all_excluded.update(str(tid) for tid in metadata.get("excluded_task_ids", []))

        if not all_dfs:
            return None

        # Merge and deduplicate by task_id (keep latest)
        merged_df = pd.concat(all_dfs, ignore_index=True)
        if 'task_id' in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=['task_id'], keep='last')

        logger.info(f"Loaded {len(files)} checkpoint(s): {len(all_processed)} processed, {len(all_excluded)} excluded")

        return ParquetCheckpointData(
            processed_task_ids=all_processed,
            excluded_task_ids=all_excluded,
            results_df=merged_df
        )

    def load_excluded_tasks(self) -> list[dict]:
        """Load all excluded task records from checkpoint exclusion files.

        Returns:
            List of excluded task dicts with error info
        """
        pattern = self._get_exclusion_pattern(for_glob=True)
        files = sorted(self.checkpoint_dir.glob(pattern))

        all_excluded = []
        seen_ids = set()

        for exclusion_file in files:
            try:
                exclusions = load_json(exclusion_file)
                for excl in exclusions:
                    if excl['task_id'] not in seen_ids:
                        all_excluded.append(excl)
                        seen_ids.add(excl['task_id'])
            except Exception as e:
                logger.warning(f"Failed to load exclusion file {exclusion_file.name}: {e}")

        return all_excluded

    def _cleanup_old_parquet(self) -> None:
        """Keep only last N parquet checkpoints (and their metadata)."""
        pattern = self._get_checkpoint_pattern(for_glob=True, file_format="parquet")
        files = sorted(self.checkpoint_dir.glob(pattern))

        if len(files) > self.keep_last:
            for old_file in files[:-self.keep_last]:
                # Remove parquet file
                old_file.unlink(missing_ok=True)
                logger.debug(f"Removed old checkpoint: {old_file.name}")

                # Remove metadata sidecar
                metadata_file = old_file.with_suffix(".meta.json")
                if metadata_file.exists():
                    metadata_file.unlink(missing_ok=True)

        # Also cleanup old exclusion files
        excl_pattern = self._get_exclusion_pattern(for_glob=True)
        excl_files = sorted(self.checkpoint_dir.glob(excl_pattern))
        if len(excl_files) > self.keep_last:
            for old_excl in excl_files[:-self.keep_last]:
                old_excl.unlink(missing_ok=True)

    def cleanup_all_parquet(self) -> None:
        """Remove all parquet checkpoints and metadata after successful completion."""
        # Remove parquet files
        parquet_pattern = self._get_checkpoint_pattern(for_glob=True, file_format="parquet")
        parquet_files = list(self.checkpoint_dir.glob(parquet_pattern))

        for f in parquet_files:
            f.unlink(missing_ok=True)
            # Remove metadata sidecar
            metadata_file = f.with_suffix(".meta.json")
            if metadata_file.exists():
                metadata_file.unlink(missing_ok=True)

        # Remove exclusion files
        excl_pattern = self._get_exclusion_pattern(for_glob=True)
        excl_files = list(self.checkpoint_dir.glob(excl_pattern))
        for f in excl_files:
            f.unlink(missing_ok=True)

        total_cleaned = len(parquet_files) + len(excl_files)
        if total_cleaned:
            logger.info(f"Cleaned up {total_cleaned} parquet checkpoint file(s)")
