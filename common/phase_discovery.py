"""
Phase discovery utilities for the SAE-Code-Correctness project.

This module provides functions for:
- Getting phase directories (with model/dataset suffixes)
- Auto-discovering phase outputs
- Writing and reading phase_output.json manifests
- Dataset range handling from config
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from common.config import Config

from .logging import get_logger

logger = get_logger("common.phase_discovery")


def get_phase_dir(phase: str) -> str:
    """
    Get the directory path for a given phase.

    Uses the phase registry as single source of truth.

    Args:
        phase: Phase string ("0", "0.1", "1", "2.2", "2.5", "3", "3.5", etc.)

    Returns:
        str: Directory path for the phase

    Examples:
        get_phase_dir("0") -> "data/phase0"
        get_phase_dir("1") -> "data/phase1_0"
        get_phase_dir("0.1") -> "data/phase0_1"
        get_phase_dir("2.5") -> "data/phase2_5"
        get_phase_dir("3.5") -> "data/phase3_5"
    """
    from common.phase_registry import get_phase_output_dir as registry_get_dir
    return registry_get_dir(phase)


def get_phase_output_dir(phase: str, config: 'Config') -> str:
    """
    Generate model/dataset-aware output directory for a phase.

    Uses the phase registry as single source of truth for base directory,
    then adds model/dataset suffixes for experiment separation.

    Args:
        phase: Phase string (e.g., "1", "2.5", "3.5")
        config: Config object with model_name and dataset_name

    Returns:
        str: Output directory path with appropriate suffix

    Examples:
        # Gemma + MBPP (default): "data/phase1_0"
        # LLAMA + MBPP: "data/phase1_0_llama"
        # Gemma + HumanEval: "data/phase1_0_humaneval"
        # LLAMA + HumanEval: "data/phase1_0_llama_humaneval"
    """
    # Get base directory from registry (single source of truth)
    from common.phase_registry import get_phase_output_dir as registry_get_dir
    from common.model_registry import get_model_suffix as registry_model_suffix
    from common.dataset_registry import get_dataset_suffix as registry_dataset_suffix

    base_dir = registry_get_dir(phase)

    # Build suffix based on model and dataset using registries
    suffix_parts = []

    # Phases 0, 0.1, 0.2, 0.3 are data preprocessing - no model suffix
    # Only phases that do model inference should have model suffixes
    data_preprocessing_phases = {'0', '0.1', '0.2', '0.3'}
    if phase not in data_preprocessing_phases:
        model_name = getattr(config, 'model_name', 'google/gemma-2-2b')
        model_suffix = registry_model_suffix(model_name)
        if model_suffix:
            # Strip leading underscore if present (registry returns "_llama", we want "llama")
            suffix_parts.append(model_suffix.lstrip('_'))

    # Add dataset suffix if not default MBPP (applies to all phases)
    dataset_name = getattr(config, 'dataset_name', 'mbpp')
    dataset_suffix = registry_dataset_suffix(dataset_name)
    if dataset_suffix:
        # Strip leading underscore if present (registry returns "_humaneval", we want "humaneval")
        suffix_parts.append(dataset_suffix.lstrip('_'))

    # Return base directory with suffixes
    if suffix_parts:
        return f"{base_dir}_{'_'.join(suffix_parts)}"
    return base_dir


def get_model_suffix(config: 'Config') -> str:
    """
    Get a short suffix string for the current model.

    Args:
        config: Config object with model_name

    Returns:
        str: Model suffix (e.g., "", "llama", "gemma9b")
    """
    from common.model_registry import get_model_suffix as registry_model_suffix
    model_name = getattr(config, 'model_name', 'google/gemma-2-2b')
    # Registry returns with leading underscore (e.g., "_llama"), strip it
    return registry_model_suffix(model_name).lstrip('_')


def get_dataset_suffix(config: 'Config') -> str:
    """
    Get a short suffix string for the current dataset.

    Args:
        config: Config object with dataset_name

    Returns:
        str: Dataset suffix (e.g., "", "humaneval")
    """
    from common.dataset_registry import get_dataset_suffix as registry_dataset_suffix
    dataset_name = getattr(config, 'dataset_name', 'mbpp')
    # Registry returns with leading underscore (e.g., "_humaneval"), strip it
    return registry_dataset_suffix(dataset_name).lstrip('_')


def discover_latest_phase_output(phase: str, phase_dir: Optional[str] = None, config=None) -> Optional[str]:
    """
    Discover the latest output file from any phase.

    Args:
        phase: Phase string ("0", "0.1", "1", "2.2", "2.5", etc.)
        phase_dir: Optional override for phase directory
        config: Optional config object for model/dataset-aware directory lookup

    Returns:
        str: Path to latest output file, or None if not found

    Raises:
        ValueError: If phase is invalid
    """
    from common.phase_registry import get_phase, get_phase_patterns
    from common.utils import find_latest_file

    # Get phase info from registry (single source of truth)
    phase_info = get_phase(phase)
    patterns = get_phase_patterns(phase)
    exclude_keywords = phase_info.exclude_keywords

    # Determine directory: explicit override > config-aware > registry default
    if phase_dir:
        directory = phase_dir
    elif config:
        # Use model/dataset-aware directory
        directory = get_phase_output_dir(phase, config)
    else:
        # Fall back to registry base directory
        directory = phase_info.output_dir

    return find_latest_file(directory, patterns, exclude_keywords)


def get_dataset_range(config, total_length: int) -> tuple[int, int]:
    """
    Get start/end indices from config for dataset slicing.

    Consolidates the common hasattr pattern used across 13+ phase runners
    into a single utility function.

    Args:
        config: Config object with optional dataset_start_idx and dataset_end_idx
        total_length: Total length of the dataset

    Returns:
        tuple[int, int]: (start_idx, end_idx) for slicing. end_idx is exclusive.

    Example:
        start_idx, end_idx = get_dataset_range(self.config, len(data))
        data = data.iloc[start_idx:end_idx]
    """
    start_idx = getattr(config, 'dataset_start_idx', None) or 0
    end_idx = getattr(config, 'dataset_end_idx', None)
    if end_idx is not None:
        # dataset_end_idx is inclusive, convert to exclusive for slicing
        end_idx = min(end_idx + 1, total_length)
    else:
        end_idx = total_length
    return start_idx, end_idx


def filter_by_range(
    data,
    config,
    description: str = "dataset"
):
    """
    Apply --start/--end filtering to data (DataFrame or sequence).

    Auto-detects data type and applies appropriate filtering.

    Args:
        data: DataFrame, list, or tuple to filter
        config: Config with optional dataset_start_idx and dataset_end_idx
        description: Name for logging (e.g., "validation dataset")

    Returns:
        Filtered data (same type as input)

    Example:
        data = filter_by_range(df, self.config, "validation dataset")
        texts = filter_by_range(texts, config, "pile samples")
    """
    import pandas as pd

    start_idx, end_idx = get_dataset_range(config, len(data))

    if start_idx > 0 or end_idx < len(data):
        logger.info(f"Filtering {description}: rows {start_idx}-{end_idx-1} (inclusive)")

        if isinstance(data, pd.DataFrame):
            return data.iloc[start_idx:end_idx].copy()
        else:
            return data[start_idx:end_idx]

    return data


# ============================================================================
# Phase Output Manifest (phase_output.json)
# ============================================================================

def write_phase_output(
    phase: str,
    outputs: dict[str, str],
    config,
    output_dir: Optional[str] = None,
    dependencies: Optional[dict[str, str]] = None,
    config_keys: Optional[list[str]] = None
) -> Path:
    """
    Write phase_output.json manifest for a completed phase.

    Args:
        phase: Phase ID (e.g., "2.5")
        outputs: dict mapping semantic names to filenames. Must include "primary".
        config: Config object (extracts relevant fields)
        output_dir: Optional override for output directory
        dependencies: Optional dict of phase_id -> file path used as input
        config_keys: Optional list of config keys to include (default: model_name, dataset_name)

    Returns:
        Path to the written phase_output.json

    Example:
        write_phase_output(
            phase="2.5",
            outputs={"primary": "top_20_latents.json"},
            config=self.config,
            dependencies={"1": "data/phase1_0/dataset_sae.parquet"}
        )
    """
    # Determine output directory
    directory = Path(output_dir) if output_dir else Path(get_phase_output_dir(phase, config))
    directory.mkdir(parents=True, exist_ok=True)

    # Extract config subset
    default_keys = ['model_name', 'dataset_name']
    keys_to_include = config_keys or default_keys
    config_subset = {k: getattr(config, k, None) for k in keys_to_include if hasattr(config, k)}

    # Build manifest
    manifest = {
        "phase": phase,
        "created_at": datetime.now().isoformat(),
        "config": config_subset,
        "outputs": outputs,
    }
    if dependencies:
        manifest["dependencies"] = dependencies

    # Write manifest
    manifest_path = directory / "phase_output.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def discover_phase_outputs(phase: str, phase_dir: Optional[str] = None, config=None) -> dict:
    """
    Discover phase outputs via phase_output.json manifest.

    Args:
        phase: Phase ID (e.g., "2.5")
        phase_dir: Optional override for phase directory
        config: Optional config for model/dataset-aware directory lookup

    Returns:
        dict with keys:
            - 'dir': Directory path
            - 'primary': Path to primary output file
            - 'outputs': dict of semantic_name -> Path
            - 'config': Config used to produce outputs
            - 'dependencies': dict of phase_id -> file path

    Raises:
        FileNotFoundError: If phase_output.json doesn't exist
    """
    # Determine directory
    if phase_dir:
        directory = Path(phase_dir)
    elif config:
        directory = Path(get_phase_output_dir(phase, config))
    else:
        from common.phase_registry import get_phase_output_dir as registry_get_dir
        directory = Path(registry_get_dir(phase))

    manifest_path = directory / "phase_output.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No phase_output.json in {directory}. Run phase {phase} first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build output paths
    outputs = {k: directory / v for k, v in manifest.get('outputs', {}).items()}

    return {
        'dir': str(directory),
        'primary': outputs.get('primary'),
        'outputs': outputs,
        'config': manifest.get('config', {}),
        'dependencies': manifest.get('dependencies', {}),
        'created_at': manifest.get('created_at'),
    }


def get_phase_output_file(phase: str, output_key: str = "primary", phase_dir: Optional[str] = None, config=None) -> Path:
    """
    Get a specific output file from a phase by semantic name.

    Args:
        phase: Phase ID (e.g., "2.5")
        output_key: Semantic name of output (default: "primary")
        phase_dir: Optional override for phase directory
        config: Optional config for model/dataset-aware directory lookup

    Returns:
        Path to the output file

    Raises:
        FileNotFoundError: If phase_output.json doesn't exist
        KeyError: If output_key not found in manifest

    Example:
        features_file = get_phase_output_file("2.5", "features")
    """
    phase_outputs = discover_phase_outputs(phase, phase_dir, config)

    if output_key not in phase_outputs['outputs']:
        available = list(phase_outputs['outputs'].keys())
        raise KeyError(
            f"Output '{output_key}' not found in phase {phase}. Available: {available}"
        )

    return phase_outputs['outputs'][output_key]


def discover_steering_coefficients(config: 'Config') -> dict[str, float]:
    """
    Load refined steering coefficients from Phase 4.6 via manifest system.

    Args:
        config: Config object for model/dataset-aware directory lookup

    Returns:
        dict with 'correct' and 'incorrect' coefficient values

    Raises:
        FileNotFoundError: If Phase 4.6 hasn't been run (no phase_output.json)
    """
    from common.utils import load_json

    coeff_file = get_phase_output_file("4.6", "refined_coefficients", config=config)
    data = load_json(coeff_file)
    return {
        "correct": data["correct"]["refined_coefficient"],
        "incorrect": data["incorrect"]["refined_coefficient"],
    }


def discover_optimal_percentile(config: 'Config') -> dict:
    """
    Load optimal percentile from Phase 8.2 via manifest system.

    Phase 8.2 runs a grid search across percentiles to find the one with
    the best net benefit (correction_rate - corruption_rate).

    Args:
        config: Config object for model/dataset-aware directory lookup

    Returns:
        dict with keys: 'percentile' (int), 'threshold' (float)

    Raises:
        FileNotFoundError: If Phase 8.2 hasn't been run (no phase_output.json)
    """
    from common.utils import load_json

    optimal_file = get_phase_output_file("8.2", "primary", config=config)
    data = load_json(optimal_file)
    # Data is nested under optimization_summary
    summary = data["optimization_summary"]
    return {
        "percentile": summary["optimal_percentile"],
        "threshold": summary["optimal_threshold"],
    }
