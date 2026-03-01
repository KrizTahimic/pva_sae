"""
Unified setup utilities for steering experiments.

Consolidates duplicated loading code from:
- phase4_5_coefficient_grid_search/steering_coefficient_selector.py
- phase4_8_steering_analysis/steering_effect_analyzer.py
- phase5_3_weight_orthogonalization/weight_orthogonalizer.py
- phase7_6_instruct_steering/instruct_steering_analyzer.py

Also provides probe direction loading for linear probe baseline comparison.
"""

from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from safetensors.torch import load_file

from common.config import Config
from common.phase_discovery import discover_latest_phase_output, filter_by_range
from common.sae_loader import load_sae_for_config
from common.utils import load_json
from common.logging import get_logger
from common.direction_utils import normalize_direction

logger = get_logger("steering_setup")


@dataclass
class PVALatents:
    """Container for PVA latent data from Phase 2.5 or 2.10."""
    top_latents: dict
    best_correct_latent: dict
    best_incorrect_latent: dict
    phase_dir: str
    source_phase: str  # "2.5" or "2.10"


@dataclass
class SAEDirections:
    """Container for SAE models and latent directions."""
    correct_sae: object
    incorrect_sae: object
    correct_direction: torch.Tensor
    incorrect_direction: torch.Tensor


@dataclass
class ProbeDirections:
    """Container for probe directions from Phase 2.6."""
    correct_direction: torch.Tensor  # For steering: mass_mean, for prediction: logreg
    incorrect_direction: torch.Tensor  # Negated direction for incorrect steering
    layer: int
    method: str  # "mass_mean" or "logreg"
    bias: float  # Only used for logreg
    phase_dir: str


def _load_probe_base(
    config: Config,
    device: torch.device,
    method: str
) -> tuple[torch.Tensor, int, float, str]:
    """Internal helper to load probe directions from Phase 2.6.

    Args:
        config: Configuration object
        device: Target device for tensors
        method: "mass_mean" (for steering) or "logreg" (for prediction)

    Returns:
        Tuple of (direction tensor, layer, bias, phase_dir)

    Raises:
        FileNotFoundError: If Phase 2.6 output not found
        ValueError: If invalid method specified
    """
    if method not in ("mass_mean", "logreg"):
        raise ValueError(f"Invalid probe method: {method}. Must be 'mass_mean' or 'logreg'")

    # Load Phase 2.6 output
    phase2_6_output = discover_latest_phase_output("2.6", config=config)
    if not phase2_6_output:
        raise FileNotFoundError(
            "Phase 2.6 output not found. Run Phase 2.6 first to compute probe directions."
        )
    phase_dir = str(Path(phase2_6_output).parent)
    logger.info(f"Loading probe directions from Phase 2.6: {phase_dir}")

    # Load best probe info
    best_probes = load_json(Path(phase_dir) / "best_probe_directions.json")
    best_layer = best_probes[method]['best_layer']
    bias = best_probes[method].get('bias', 0.0) if method == 'logreg' else 0.0

    logger.info(f"Best {method} probe at layer {best_layer}")

    # Load probe direction tensor
    probe_file = Path(phase_dir) / "probe_directions" / f"layer_{best_layer}_probes.safetensors"
    if not probe_file.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_file}")

    tensors = load_file(str(probe_file))
    direction_key = f"{method}_direction"
    direction = tensors[direction_key].to(device)

    return direction, best_layer, bias, phase_dir


def load_mass_mean_direction_for_layer(
    layer: int, phase2_6_dir, device: torch.device, model_dtype=None
) -> torch.Tensor:
    """Load and normalize mass_mean probe direction for a specific layer.

    Args:
        layer: Layer number
        phase2_6_dir: Phase 2.6 output directory (Path or str)
        device: Target device
        model_dtype: Optional model dtype to cast to

    Returns:
        Normalized mass_mean direction tensor [d_model]
    """
    probe_file = Path(phase2_6_dir) / "probe_directions" / f"layer_{layer}_probes.safetensors"
    if not probe_file.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_file}")
    tensors = load_file(str(probe_file))
    direction = tensors["mass_mean_direction"].to(device)
    direction = normalize_direction(direction)
    if model_dtype is not None:
        direction = direction.to(dtype=model_dtype)
    return direction


def load_probe_directions_for_predicting(
    config: Config,
    device: torch.device,
    method: str = "logreg"
) -> ProbeDirections:
    """Load probe directions for PREDICTION/DETECTION tasks (Phases 3.x, 8.2 prediction).

    Use this when scoring activations for AUROC/F1 metrics. Does NOT require
    the LLM model since no dtype matching is needed for dot-product scoring.

    Args:
        config: Configuration object
        device: Target device for tensors
        method: "logreg" (default, optimal for detection) or "mass_mean"

    Returns:
        ProbeDirections dataclass with direction tensors in float32

    Raises:
        FileNotFoundError: If Phase 2.6 output not found
        ValueError: If invalid method specified
    """
    direction, best_layer, bias, phase_dir = _load_probe_base(config, device, method)

    # No normalization needed for predicting: AUROC/F1 are threshold-independent metrics,
    # so direction magnitude doesn't affect ranking. Steering path normalizes for coefficient interpretation.
    correct_direction = direction
    incorrect_direction = -direction  # Negate for incorrect prediction

    logger.info(f"Probe direction loaded in dtype: {direction.dtype}")

    return ProbeDirections(
        correct_direction=correct_direction,
        incorrect_direction=incorrect_direction,
        layer=best_layer,
        method=method,
        bias=bias,
        phase_dir=phase_dir,
    )


def load_probe_directions_for_steering(
    config: Config,
    device: torch.device,
    model: nn.Module,
    method: str = "mass_mean"
) -> ProbeDirections:
    """Load probe directions for STEERING tasks (Phases 4.x, 5.x, 7.x, 8.x steering).

    Use this when modifying model activations via steering hooks. Requires
    the LLM model to match probe direction dtype with model dtype.

    Args:
        config: Configuration object
        device: Target device for tensors
        model: The language model (for dtype matching)
        method: "mass_mean" (default, optimal for steering) or "logreg"

    Returns:
        ProbeDirections dataclass with direction tensors matching model dtype

    Raises:
        FileNotFoundError: If Phase 2.6 output not found
        ValueError: If invalid method specified
    """
    direction, best_layer, bias, phase_dir = _load_probe_base(config, device, method)

    # Normalize to unit L2 norm (consistent coefficient interpretation, matches SAE path)
    direction = normalize_direction(direction, name=f"probe_{method}_direction")

    # Match model dtype for activation modification
    model_dtype = next(model.parameters()).dtype
    direction = direction.to(dtype=model_dtype)

    # For steering, we use the same direction for "correct" steering
    # and negate it for "incorrect" steering
    correct_direction = direction
    incorrect_direction = -direction  # Negate for incorrect steering

    logger.info(f"Probe direction normalized and converted to model dtype: {model_dtype}")

    return ProbeDirections(
        correct_direction=correct_direction,
        incorrect_direction=incorrect_direction,
        layer=best_layer,
        method=method,
        bias=bias,
        phase_dir=phase_dir,
    )



@dataclass
class DualProbeDirections:
    """Container for dual probe directions used by Phases 8.2 and 8.3.

    Predicting probe (logreg) is used for threshold decisions.
    Steering probe (mass_mean) is used for activation modification.
    """
    predicting_probe: ProbeDirections
    steering_probe: ProbeDirections
    predicting_direction: torch.Tensor
    predicting_bias: float
    predicting_layer: int
    correct_latent_direction: torch.Tensor
    incorrect_latent_direction: torch.Tensor
    steering_layer: int


def load_dual_probe_directions(
    config: Config,
    device: torch.device,
    model: torch.nn.Module
) -> DualProbeDirections:
    """Load both predicting and steering probe directions for Phases 8.2/8.3.

    These phases need two probes simultaneously:
    - LogReg probe for threshold prediction (AUROC/F1 optimal)
    - Mass-mean probe for steering intervention (causal optimal)

    Args:
        config: Configuration object
        device: Target device for tensors
        model: The language model (for dtype matching in steering probe)

    Returns:
        DualProbeDirections with both probe sets loaded
    """
    # Prediction: logreg (optimal for AUROC/F1)
    predicting_probe = load_probe_directions_for_predicting(
        config, device, method="logreg"
    )

    logger.info(f"Predicting probe: Layer {predicting_probe.layer}, "
               f"bias={predicting_probe.bias:.4f}")

    # Steering: mass_mean (optimal for causal intervention)
    steering_probe = load_probe_directions_for_steering(
        config, device, model, method="mass_mean"
    )

    logger.info(f"Steering probe: Layer {steering_probe.layer}")

    return DualProbeDirections(
        predicting_probe=predicting_probe,
        steering_probe=steering_probe,
        predicting_direction=predicting_probe.incorrect_direction,
        predicting_bias=predicting_probe.bias,
        predicting_layer=predicting_probe.layer,
        correct_latent_direction=steering_probe.correct_direction,
        incorrect_latent_direction=steering_probe.incorrect_direction,
        steering_layer=steering_probe.layer,
    )


def score_activation(
    activation: torch.Tensor,
    use_probe: bool,
    predicting_direction: torch.Tensor = None,
    predicting_bias: float = 0.0,
    predicting_sae: object = None,
    latent_idx: int = None,
    device: torch.device = None,
) -> float:
    """Score an activation using either probe or SAE mode.

    Used by Phases 8.2 and 8.3 for threshold checking during generation.

    Args:
        activation: Raw activation tensor, shape (hidden_dim,) or (1, hidden_dim)
        use_probe: If True, use probe dot-product scoring; if False, use SAE encoding
        predicting_direction: Probe direction vector (required if use_probe=True)
        predicting_bias: Probe bias term (used if use_probe=True)
        predicting_sae: SAE model (required if use_probe=False)
        latent_idx: SAE latent index (required if use_probe=False)
        device: Target device for SAE encoding

    Returns:
        Float activation score
    """
    with torch.no_grad():
        if use_probe:
            activation_float = activation.to(dtype=predicting_direction.dtype)
            score = (activation_float @ predicting_direction).item() + predicting_bias
            return score
        else:
            activation_bf16 = activation.to(dtype=predicting_sae.W_enc.dtype, device=device)
            if activation_bf16.ndim == 1:
                activation_bf16 = activation_bf16.unsqueeze(0)
            latent_activations = predicting_sae.encode(activation_bf16)
            return latent_activations[0, latent_idx].item()


def get_direction_source_info(config: Config) -> dict:
    """Get information about the configured direction source.

    Returns:
        Dictionary with 'source', 'is_probe', and 'probe_method' keys
    """
    source = getattr(config, 'direction_source', 'sae')

    if source == 'sae':
        return {'source': 'sae', 'is_probe': False, 'probe_method': None}
    elif source == 'probe_logreg':
        return {'source': 'probe', 'is_probe': True, 'probe_method': 'logreg'}
    elif source == 'probe_mass_mean':
        return {'source': 'probe', 'is_probe': True, 'probe_method': 'mass_mean'}
    else:
        raise ValueError(f"Unknown direction source: '{source}'. Valid options: 'sae', 'probe_logreg', 'probe_mass_mean'")


def _load_latents_from_phase(config: Config, phase: str, purpose: str) -> PVALatents:
    """Internal helper to load latents from a specific phase.

    Args:
        config: Configuration object
        phase: Phase number ("2.5" or "2.10")
        purpose: Description for error messages ("predicting" or "steering")

    Returns:
        PVALatents dataclass

    Raises:
        FileNotFoundError: If phase output not found
        ValueError: If latents file has invalid structure
    """
    phase_output = discover_latest_phase_output(phase, config=config)
    if not phase_output:
        raise FileNotFoundError(
            f"Phase {phase} output not found. Run Phase {phase} first.\n"
            f"Phase {phase} is required for {purpose} latents."
        )

    latents_file = Path(phase_output).parent / "top_20_latents.json"
    if not latents_file.exists():
        raise FileNotFoundError(
            f"Latents file not found: {latents_file}\n"
            f"Run Phase {phase} to generate top_20_latents.json"
        )

    phase_dir = str(Path(phase_output).parent)
    logger.info(f"Loading {purpose} latents from Phase {phase}...")
    logger.info(f"Using Phase {phase} output: {phase_dir}")

    # Load top latents
    top_latents = load_json(latents_file)

    # Validate structure
    if 'correct' not in top_latents or 'incorrect' not in top_latents:
        raise ValueError("Expected 'correct' and 'incorrect' keys in top_20_latents.json")

    if len(top_latents['correct']) == 0 or len(top_latents['incorrect']) == 0:
        raise ValueError("No latents found in correct or incorrect arrays")

    # Get the best (first) latent from each category
    best_correct = top_latents['correct'][0]
    best_incorrect = top_latents['incorrect'][0]

    # Handle different score field names between phases
    score_field = 't_statistic' if phase == "2.10" else 'separation_score'
    correct_score = best_correct.get(score_field, best_correct.get('t_statistic', best_correct.get('separation_score', 0)))
    incorrect_score = best_incorrect.get(score_field, best_incorrect.get('t_statistic', best_incorrect.get('separation_score', 0)))

    logger.info(f"Best correct latent: Layer {best_correct['layer']}, "
               f"Index {best_correct['latent_idx']}, "
               f"{score_field}={correct_score:.4f}")
    logger.info(f"Best incorrect latent: Layer {best_incorrect['layer']}, "
               f"Index {best_incorrect['latent_idx']}, "
               f"{score_field}={incorrect_score:.4f}")

    return PVALatents(
        top_latents=top_latents,
        best_correct_latent=best_correct,
        best_incorrect_latent=best_incorrect,
        phase_dir=phase_dir,
        source_phase=phase
    )


def load_predicting_latents(config: Config) -> PVALatents:
    """Load PREDICTING latents from Phase 2.10 (t-statistic selection).

    Predicting latents are selected by t-statistic which captures sensitivity
    to confidence gradients - appropriate for statistical validation (AUROC/F1).

    Used by: Phase 3.x (AUROC/F1 validation), Phase 8.x (prediction component)

    Returns:
        PVALatents dataclass with top_latents from Phase 2.10

    Raises:
        FileNotFoundError: If Phase 2.10 output not found
        ValueError: If latents file has invalid structure
    """
    return _load_latents_from_phase(config, "2.10", "predicting")


def load_steering_latents(config: Config) -> PVALatents:
    """Load STEERING latents from Phase 2.5 (separation score selection).

    Steering latents are selected by separation score which captures categorical
    exclusivity - appropriate for causal validation (steering, orthogonalization).

    Used by: Phases 4.x, 5.x, 6.x, 7.x (steering & orthogonalization)

    Returns:
        PVALatents dataclass with top_latents from Phase 2.5

    Raises:
        FileNotFoundError: If Phase 2.5 output not found
        ValueError: If latents file has invalid structure
    """
    return _load_latents_from_phase(config, "2.5", "steering")


def load_phase4_9_best_latent(config: Config) -> dict:
    """Load the best latent selection from Phase 4.9.

    Phase 4.9 selects the single best latent (per steering type) from
    Phase 4.8's top-N evaluation. Returns dict with 'correct' and 'incorrect'
    keys, each containing layer, latent_idx, refined_coefficient, and rank.

    Args:
        config: Configuration object

    Returns:
        dict with 'correct' and 'incorrect' entries from best_latent_selection.json

    Raises:
        FileNotFoundError: If Phase 4.9 output not found
    """
    phase4_9_output = discover_latest_phase_output("4.9", config=config)
    if not phase4_9_output:
        raise FileNotFoundError(
            "Phase 4.9 output not found. Run Phase 4.9 first.\n"
            "Phase 4.9 selects the best latent from Phase 4.8's top-N evaluation."
        )

    selection_file = Path(phase4_9_output).parent / "best_latent_selection.json"
    if not selection_file.exists():
        raise FileNotFoundError(
            f"best_latent_selection.json not found in {Path(phase4_9_output).parent}\n"
            "Run Phase 4.9 to generate best latent selection."
        )

    selection = load_json(selection_file)
    logger.info(f"Loaded Phase 4.9 best latent selection: "
               f"correct=L{selection['correct']['layer']}F{selection['correct']['latent_idx']} "
               f"(rank {selection['correct']['rank']}, coeff={selection['correct']['refined_coefficient']}), "
               f"incorrect=L{selection['incorrect']['layer']}F{selection['incorrect']['latent_idx']} "
               f"(rank {selection['incorrect']['rank']}, coeff={selection['incorrect']['refined_coefficient']})")

    return selection


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
        best_correct_latent: dict with 'layer' and 'latent_idx' for correct latent
        best_incorrect_latent: dict with 'layer' and 'latent_idx' for incorrect latent

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

    # Log original norms for diagnostics
    correct_norm = torch.norm(correct_direction).item()
    incorrect_norm = torch.norm(incorrect_direction).item()
    logger.info(f"Original W_dec norms - correct: {correct_norm:.4f}, incorrect: {incorrect_norm:.4f}")

    # Normalize to unit L2 norm (consistent coefficient interpretation across SAEs)
    correct_direction = normalize_direction(correct_direction, name="correct_direction")
    incorrect_direction = normalize_direction(incorrect_direction, name="incorrect_direction")

    # Ensure latent directions are in the same dtype as the model
    model_dtype = next(model.parameters()).dtype
    correct_direction = correct_direction.to(dtype=model_dtype)
    incorrect_direction = incorrect_direction.to(dtype=model_dtype)

    logger.info(f"Latent directions normalized to unit norm, converted to model dtype: {model_dtype}")

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

    # Load baseline dataset - try expected filename first, then merged pattern
    phase_path = Path(phase_output).parent
    baseline_file = phase_path / filename
    if not baseline_file.exists():
        # Try merged pattern from parallel execution (e.g., dataset_merged_*.parquet)
        merged_files = sorted(phase_path.glob("dataset_merged_*.parquet"))
        if merged_files:
            baseline_file = merged_files[-1]  # Use most recent
            logger.info(f"Using merged dataset: {baseline_file.name}")
        else:
            raise FileNotFoundError(f"Baseline dataset not found: {baseline_file}")

    baseline_data = pd.read_parquet(baseline_file)
    logger.info(f"Loaded {len(baseline_data)} problems from Phase {phase} baseline")

    # Apply --start and --end arguments if provided
    baseline_data = filter_by_range(baseline_data, config, "dataset")

    return baseline_data, phase_dir


def split_by_correctness(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by baseline_passed column.

    Delegates to dataset_utils.split_by_correctness for the actual implementation.

    Args:
        data: DataFrame with 'baseline_passed' column

    Returns:
        Tuple of (initially_correct, initially_incorrect) DataFrames
    """
    from common.dataset_utils import split_by_correctness as _split
    return _split(data, correctness_col='baseline_passed')
