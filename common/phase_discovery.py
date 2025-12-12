"""
Phase discovery utilities for the PVA-SAE project.

This module provides functions for:
- Getting phase directories (with model/dataset suffixes)
- Auto-discovering phase outputs
- Writing and reading phase_output.json manifests
- Dataset range handling from config
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def get_phase_output_dir(phase: str, config) -> str:
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
    base_dir = registry_get_dir(phase)

    # Build suffix based on model and dataset
    suffixes = []

    # Add model suffix if not default Gemma
    model_name = getattr(config, 'model_name', 'google/gemma-2-2b')
    if 'llama' in model_name.lower():
        suffixes.append('llama')

    # Add dataset suffix if not default MBPP
    dataset_name = getattr(config, 'dataset_name', 'mbpp')
    if dataset_name.lower() != 'mbpp':
        suffixes.append(dataset_name.lower())

    # Return base directory with suffixes
    if suffixes:
        return f"{base_dir}_{'_'.join(suffixes)}"
    return base_dir


def get_model_suffix(config) -> str:
    """
    Get a short suffix string for the current model.

    Args:
        config: Config object with model_name

    Returns:
        str: Model suffix (e.g., "", "llama")
    """
    model_name = getattr(config, 'model_name', 'google/gemma-2-2b')
    if 'llama' in model_name.lower():
        return 'llama'
    return ''


def get_dataset_suffix(config) -> str:
    """
    Get a short suffix string for the current dataset.

    Args:
        config: Config object with dataset_name

    Returns:
        str: Dataset suffix (e.g., "", "humaneval")
    """
    dataset_name = getattr(config, 'dataset_name', 'mbpp')
    if dataset_name.lower() == 'humaneval':
        return 'humaneval'
    return ''


def discover_latest_phase_output(phase: str, phase_dir: Optional[str] = None) -> Optional[str]:
    """
    Discover the latest output file from any phase.

    Args:
        phase: Phase string ("0", "0.1", "1", "2.2", "2.5", etc.)
        phase_dir: Optional override for phase directory

    Returns:
        str: Path to latest output file, or None if not found

    Raises:
        ValueError: If phase is invalid
    """
    from common.phase_registry import get_phase, get_phase_patterns
    from common.utils import find_latest_file

    # Get phase info from registry (single source of truth)
    phase_info = get_phase(phase)
    directory = phase_dir or phase_info.output_dir
    patterns = get_phase_patterns(phase)
    exclude_keywords = phase_info.exclude_keywords

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
        outputs: Dict mapping semantic names to filenames. Must include "primary".
        config: Config object (extracts relevant fields)
        output_dir: Optional override for output directory
        dependencies: Optional dict of phase_id -> file path used as input
        config_keys: Optional list of config keys to include (default: model_name, dataset_name)

    Returns:
        Path to the written phase_output.json

    Example:
        write_phase_output(
            phase="2.5",
            outputs={"primary": "sae_analysis_results.json", "features": "top_20_features.json"},
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
        Dict with keys:
            - 'dir': Directory path
            - 'primary': Path to primary output file
            - 'outputs': Dict of semantic_name -> Path
            - 'config': Config used to produce outputs
            - 'dependencies': Dict of phase_id -> file path

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
