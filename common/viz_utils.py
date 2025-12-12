"""
Visualization utilities for --viz-only mode.

This module provides helpers for regenerating visualizations without
recomputing expensive phase data.
"""

from pathlib import Path
from typing import Callable, Any

from common.utils import load_json
from common.logging import get_logger


logger = get_logger(__name__)


def handle_viz_only_mode(
    runner,
    viz_data_file: str,
    viz_func: Callable[[dict], None]
) -> bool:
    """
    Handle --viz-only mode for a phase runner.

    If viz_only mode is enabled, loads saved data and regenerates visualizations
    without running the expensive computation.

    Args:
        runner: Phase runner instance (must have .config and .output_dir)
        viz_data_file: Filename of the saved visualization data (e.g., "metrics.json")
        viz_func: Function that creates visualizations from loaded data

    Returns:
        True if viz-only mode was handled (caller should return early)
        False if normal execution should continue

    Raises:
        FileNotFoundError: If viz_data_file doesn't exist (run phase normally first)

    Example:
        def run(self):
            if handle_viz_only_mode(self, "metrics.json", self._create_visualizations):
                return
            # ... normal computation ...
    """
    if not getattr(runner.config, 'viz_only', False):
        return False

    data_path = runner.output_dir / viz_data_file

    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot run --viz-only: {data_path} not found. "
            f"Run phase normally first to generate data."
        )

    logger.info(f"--viz-only mode: Loading data from {data_path}")
    data = load_json(data_path)

    logger.info("Regenerating visualizations...")
    viz_func(data)

    logger.info("Visualization regeneration complete")
    return True
