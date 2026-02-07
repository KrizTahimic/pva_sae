"""
Generic phase runner that eliminates boilerplate.

This module provides a unified way to run any phase by looking up
its metadata in the registry and dynamically importing/running it.
"""

import importlib
import sys
from pathlib import Path

from common.phase_registry import get_phase, PhaseInfo
from common.config import Config
from common.logging import get_logger


# Phases that require special handling and cannot use the generic runner
# Most phases have been refactored to follow the standard Runner(config).run() pattern
SPECIAL_PHASES = {
    "3": "placeholder - not implemented (skip this phase)",
}


def run_phase(phase_id: str, config: Config, device: str) -> any:
    """
    Generic runner for phases that follow the standard pattern.

    Standard patterns supported:
    1. Class with .run() method: Create instance, call run()
    2. Function: Import and call with config argument

    Args:
        phase_id: Phase ID as string (e.g., "3.5", "4.8")
        config: Config object
        device: Device string ("cuda", "cpu", etc.)

    Returns:
        Result from the phase runner

    Raises:
        ValueError: If phase requires special handling
    """
    # Check if phase requires special handling
    if phase_id in SPECIAL_PHASES:
        raise ValueError(
            f"Phase {phase_id} requires special handling: {SPECIAL_PHASES[phase_id]}. "
            f"Use the dedicated run_phase{phase_id.replace('.', '_')}() function in run.py"
        )

    phase = get_phase(phase_id)
    logger = get_logger("main")

    # Log phase start
    logger.info(f"Starting Phase {phase.id}: {phase.name}")
    logger.info("\n" + config.dump(phase=phase.id))

    # Dynamic import
    try:
        module = importlib.import_module(phase.module)
    except ImportError as e:
        logger.error(f"Failed to import phase module: {phase.module}")
        raise ImportError(f"Cannot import {phase.module}: {e}") from e

    # Get the runner (class or function)
    try:
        runner_obj = getattr(module, phase.runner)
    except AttributeError as e:
        logger.error(f"Runner '{phase.runner}' not found in module {phase.module}")
        raise AttributeError(f"Cannot find {phase.runner} in {phase.module}: {e}") from e

    # Execute based on runner type
    if phase.runner_type == "class":
        # Standard class pattern: instantiate and call .run()
        runner = runner_obj(config)
        result = runner.run()
    elif phase.runner_type == "function":
        # Function pattern: call with config
        result = runner_obj(config)
    else:
        raise ValueError(f"Unknown runner_type: {phase.runner_type}")

    logger.info(f"\n✅ Phase {phase.id} completed successfully")
    return result


def can_use_generic_runner(phase_id: str) -> bool:
    """
    Check if a phase can use the generic runner.

    Args:
        phase_id: Phase ID as string

    Returns:
        True if generic runner can be used, False if special handling needed
    """
    return phase_id not in SPECIAL_PHASES


def get_special_phases() -> dict[str, str]:
    """
    Get list of phases that require special handling.

    Returns:
        dict mapping phase_id to reason for special handling
    """
    return SPECIAL_PHASES.copy()
