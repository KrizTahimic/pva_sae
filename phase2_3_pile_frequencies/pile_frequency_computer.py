"""
Pile Frequency Computer for Phase 2.3.

Loads raw pile activations from Phase 2.2, encodes them through the SAE,
and computes per-latent activation frequencies for pile filtering.
"""

from pathlib import Path
from datetime import datetime

import torch
from einops import reduce

from common.config import Config
from common.logging import get_logger, tqdm_with_logging
from common.phase_discovery import get_phase_output_dir, write_phase_output
from common.sae_loader import load_sae_for_config
from common.tensor_utils import load_activation, save_activation

logger = get_logger("phase2_3", phase="2.3")


class PileFrequencyComputer:
    """Computes SAE latent activation frequencies on pile dataset."""

    def __init__(self, config: Config, device: str = "cuda"):
        """
        Initialize the pile frequency computer.

        Args:
            config: Configuration object
            device: Device for computation
        """
        self.config = config
        self.device = device
        self.pile_dir = Path(get_phase_output_dir("2.2", config)) / "pile_activations"

    def _load_pile_activations_for_layer(self, layer_idx: int) -> torch.Tensor | None:
        """
        Load pile activations for a specific layer from Phase 2.2.

        Args:
            layer_idx: Layer index to load activations for

        Returns:
            Stacked tensor of pile activations, or None if not found
        """
        if not self.pile_dir.exists():
            raise FileNotFoundError(
                f"Pile activation directory not found at {self.pile_dir}. "
                "Run Phase 2.2 first to cache pile activations."
            )

        pile_files = sorted(self.pile_dir.glob(f"*_layer_{layer_idx}.safetensors"))

        if not pile_files:
            logger.warning(f"No pile activations found for layer {layer_idx}")
            return None

        logger.debug(f"Loading {len(pile_files)} pile activations for layer {layer_idx}")

        activations = [load_activation(file_path, "cpu") for file_path in pile_files]

        if not activations:
            return None

        return torch.stack(activations).to(self.device)

    def compute_frequencies_for_layer(self, layer_idx: int) -> torch.Tensor | None:
        """
        Compute SAE latent activation frequencies for a single layer.

        Args:
            layer_idx: Layer index to process

        Returns:
            Frequency tensor (shape: [num_latents]), or None if no pile data
        """
        # Load raw pile activations
        pile_activations = self._load_pile_activations_for_layer(layer_idx)

        if pile_activations is None:
            return None

        # Load SAE for this layer
        sae = load_sae_for_config(self.config, layer_idx, self.device)

        # Ensure dtype matches SAE parameters
        pile_activations = pile_activations.to(sae.W_enc.dtype)

        # Encode pile activations through SAE and compute frequencies
        with torch.no_grad():
            pile_features = sae.encode(pile_activations)
            # Average over samples to get per-latent activation frequency
            frequencies = reduce((pile_features > 0).float(), 'n f -> f', 'mean')

        # Clean up
        del sae, pile_activations, pile_features
        torch.cuda.empty_cache()

        return frequencies.cpu()

    def run(self) -> dict:
        """
        Run pile frequency computation for all layers.

        Returns:
            dict with metadata about the computation
        """
        logger.info("Starting Phase 2.3: Pile SAE Frequency Computation")

        # Setup output directory
        output_dir = Path(get_phase_output_dir("2.3", self.config))
        output_dir.mkdir(parents=True, exist_ok=True)

        layers_processed = []
        layers_failed = []

        for layer_idx in tqdm_with_logging(
            self.config.activation_layers, logger, desc="Computing pile frequencies"
        ):
            try:
                frequencies = self.compute_frequencies_for_layer(layer_idx)

                if frequencies is not None:
                    # Save as safetensors
                    output_file = output_dir / f"layer_{layer_idx}_frequencies.safetensors"
                    save_activation(frequencies, output_file)
                    layers_processed.append(layer_idx)
                    logger.info(f"Saved pile frequencies for layer {layer_idx} to {output_file}")
                else:
                    layers_failed.append(layer_idx)
                    logger.warning(f"No pile data available for layer {layer_idx}")

            except Exception as e:
                logger.error(f"Failed to process layer {layer_idx}: {e}")
                layers_failed.append(layer_idx)
                continue

        # Prepare results
        results = {
            "creation_timestamp": datetime.now().isoformat(),
            "model_name": self.config.model_name,
            "pile_samples": self.config.pile_samples,
            "activation_layers": self.config.activation_layers,
            "layers_processed": layers_processed,
            "layers_failed": layers_failed,
        }

        # Write phase output manifest
        write_phase_output(
            phase="2.3",
            outputs={"primary": "layer_*_frequencies.safetensors"},
            config=self.config,
            output_dir=str(output_dir),
            config_keys=["model_name", "pile_samples", "activation_layers"],
        )

        logger.info(
            f"Phase 2.3 completed. Processed {len(layers_processed)} layers, "
            f"{len(layers_failed)} failed."
        )

        return results
