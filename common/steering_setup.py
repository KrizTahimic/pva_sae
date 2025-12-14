"""
Unified setup utilities for steering experiments.

Consolidates duplicated loading code from:
- phase4_5_coefficient_grid_search/steering_coefficient_selector.py
- phase4_8_steering_analysis/steering_effect_analyzer.py
- phase5_3_weight_orthogonalization/weight_orthogonalizer.py
- phase7_6_instruct_steering/instruct_steering_analyzer.py
"""

from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd

from common.config import Config
from common.phase_discovery import discover_latest_phase_output, get_dataset_range
from common.sae_loader import load_sae_for_config
from common.utils import load_json
from common.logging import get_logger

logger = get_logger("steering_setup")


@dataclass
class PVALatents:
    """Container for PVA latent data from Phase 2.5."""
    top_latents: dict
    best_correct_latent: dict
    best_incorrect_latent: dict
    phase_dir: str


@dataclass
class SAEDirections:
    """Container for SAE models and latent directions."""
    correct_sae: object
    incorrect_sae: object
    correct_direction: torch.Tensor
    incorrect_direction: torch.Tensor


def load_pva_latents(config: Config) -> PVALatents:
    """Load PVA latents from Phase 2.5.

    Returns:
        PVALatents dataclass with top_latents, best latents, and phase dir

    Raises:
        FileNotFoundError: If Phase 2.5 output not found
        ValueError: If latents file has invalid structure
    """
    logger.info("Loading PVA latents from Phase 2.5...")
    phase2_5_output = discover_latest_phase_output("2.5", config=config)
    if not phase2_5_output:
        raise FileNotFoundError("Phase 2.5 output not found. Run Phase 2.5 first.")

    phase_dir = str(Path(phase2_5_output).parent)
    logger.info(f"Using Phase 2.5 output: {phase2_5_output}")

    # Load top latents
    latents_file = Path(phase2_5_output).parent / "top_20_latents.json"
    if not latents_file.exists():
        raise FileNotFoundError(f"Top latents file not found: {latents_file}")

    top_latents = load_json(latents_file)

    # Validate structure
    if 'correct' not in top_latents or 'incorrect' not in top_latents:
        raise ValueError("Expected 'correct' and 'incorrect' keys in top_20_latents.json")

    if len(top_latents['correct']) == 0 or len(top_latents['incorrect']) == 0:
        raise ValueError("No latents found in correct or incorrect arrays")

    # Get the best (first) latent from each category
    best_correct = top_latents['correct'][0]
    best_incorrect = top_latents['incorrect'][0]

    logger.info(f"Best correct latent: Layer {best_correct['layer']}, "
               f"Index {best_correct['latent_idx']}, "
               f"Score {best_correct['separation_score']:.4f}")
    logger.info(f"Best incorrect latent: Layer {best_incorrect['layer']}, "
               f"Index {best_incorrect['latent_idx']}, "
               f"Score {best_incorrect['separation_score']:.4f}")

    return PVALatents(
        top_latents=top_latents,
        best_correct_latent=best_correct,
        best_incorrect_latent=best_incorrect,
        phase_dir=phase_dir
    )


def load_sae_and_directions(
    config: Config,
    device: torch.device,
    model: nn.Module,
    best_correct_latent: dict,
    best_incorrect_latent: dict
) -> SAEDirections:
    """Load SAE models and extract latent directions.

    Args:
        config: Configuration object
        device: Target device for SAE models
        model: The language model (for dtype matching)
        best_correct_latent: Dict with 'layer' and 'latent_idx' for correct latent
        best_incorrect_latent: Dict with 'layer' and 'latent_idx' for incorrect latent

    Returns:
        SAEDirections dataclass with SAEs and direction tensors
    """
    logger.info("Loading SAE models...")
    correct_sae = load_sae_for_config(
        config,
        best_correct_latent['layer'],
        device
    )
    incorrect_sae = load_sae_for_config(
        config,
        best_incorrect_latent['layer'],
        device
    )

    # Extract latent directions
    correct_direction = correct_sae.W_dec[best_correct_latent['latent_idx']].detach()
    incorrect_direction = incorrect_sae.W_dec[best_incorrect_latent['latent_idx']].detach()

    # Ensure latent directions are in the same dtype as the model
    model_dtype = next(model.parameters()).dtype
    correct_direction = correct_direction.to(dtype=model_dtype)
    incorrect_direction = incorrect_direction.to(dtype=model_dtype)

    logger.info(f"Latent directions converted to model dtype: {model_dtype}")

    return SAEDirections(
        correct_sae=correct_sae,
        incorrect_sae=incorrect_sae,
        correct_direction=correct_direction,
        incorrect_direction=incorrect_direction
    )


def load_baseline_data(
    config: Config,
    phase: str,
    filename: str = "dataset_temp_0_0.parquet"
) -> tuple[pd.DataFrame, str]:
    """Load baseline data from specified phase.

    Args:
        config: Configuration object
        phase: Phase number (e.g., "3.5", "3.6", "7.3")
        filename: Parquet filename (default: "dataset_temp_0_0.parquet")

    Returns:
        Tuple of (DataFrame with baseline data, phase_dir string)

    Raises:
        FileNotFoundError: If phase output or baseline file not found
    """
    logger.info(f"Loading baseline data from Phase {phase}...")
    phase_output = discover_latest_phase_output(phase, config=config)
    if not phase_output:
        raise FileNotFoundError(f"Phase {phase} output not found. Please run Phase {phase} first.")

    phase_dir = str(Path(phase_output).parent)
    logger.info(f"Using Phase {phase} output: {phase_dir}")

    # Load baseline dataset
    baseline_file = Path(phase_output).parent / filename
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")

    baseline_data = pd.read_parquet(baseline_file)
    logger.info(f"Loaded {len(baseline_data)} problems from Phase {phase} baseline")

    # Apply --start and --end arguments if provided
    start_idx, end_idx = get_dataset_range(config, len(baseline_data))
    if start_idx > 0 or end_idx < len(baseline_data):
        logger.info(f"Processing dataset rows {start_idx}-{end_idx-1} (inclusive)")
        baseline_data = baseline_data.iloc[start_idx:end_idx].copy()
        logger.info(f"Filtered to {len(baseline_data)} problems")

    return baseline_data, phase_dir


def split_by_correctness(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by baseline_passed column.

    Args:
        data: DataFrame with 'baseline_passed' column

    Returns:
        Tuple of (initially_correct, initially_incorrect) DataFrames
    """
    correct = data[data['baseline_passed'] == True].copy()
    incorrect = data[data['baseline_passed'] == False].copy()
    logger.info(f"Split baseline: {len(correct)} initially correct, "
               f"{len(incorrect)} initially incorrect problems")
    return correct, incorrect
