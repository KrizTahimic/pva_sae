"""
Phase 8.1: Percentile Threshold Calculator

Calculates percentile-based thresholds for selective steering from Phase 3.6
hyperparameter dataset activations. This ensures no data leakage - thresholds
are calculated on the hyperparameter tuning set, not the validation set.

Key Design:
- Loads Phase 3.6 data (hyperparams set with activations)
- Extracts incorrect-predicting feature activations (L19-5441)
- Calculates multiple percentiles (50, 75, 90, 95)
- Saves thresholds for use in Phase 8.3 selective steering
"""

import json
from pathlib import Path

from datetime import datetime

import numpy as np
import pandas as pd

import torch

from common.config import Config
from common.logging import get_logger
from common.utils import ensure_directory_exists, get_timestamp, detect_device, load_json, save_json
from common.tensor_utils import load_activation
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    write_phase_output
)
from common.sae_loader import load_sae_for_config

logger = get_logger(__name__)

class ThresholdCalculator:
    """
    Calculates percentile-based thresholds from Phase 3.6 hyperparameter dataset.

    Workflow:
    1. Load Phase 3.8 results to get incorrect-predicting feature info
    2. Load Phase 3.6 dataset (hyperparams set with activations)
    3. Extract L19-5441 activations
    4. Calculate multiple percentile thresholds
    5. Save results with metadata
    """

    def __init__(self, config: Config):
        """Initialize the threshold calculator."""
        self.config = config
        self.device = torch.device(detect_device())

        # Create output directory
        self.output_dir = Path(get_phase_output_dir("8.1", config))
        ensure_directory_exists(self.output_dir)

        logger.info(f"Initializing Threshold Calculator")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")

        # Load dependencies
        self._load_dependencies()

        logger.info("Initialization complete")

    def _load_dependencies(self):
        """Load Phase 3.8 feature info and Phase 3.6 activation data."""
        logger.info("Loading dependencies...")

        # === LOAD PHASE 3.8 FEATURE INFO ===
        logger.info("Loading incorrect-predicting feature info from Phase 3.8...")
        phase3_8_output = discover_latest_phase_output("3.8")
        if not phase3_8_output:
            raise FileNotFoundError("Phase 3.8 output not found. Run Phase 3.8 first.")

        phase3_8_results = load_json(Path(phase3_8_output).parent / "evaluation_results.json")

        # Extract incorrect-predicting latent info
        incorrect_pred_info = phase3_8_results['incorrect_predicting_latent']
        self.latent_layer = incorrect_pred_info['latent']['layer']  # 19
        self.latent_idx = incorrect_pred_info['latent']['idx']  # 5441
        self.phase3_8_threshold = incorrect_pred_info['threshold_optimization']['optimal_threshold']  # 15.5086

        logger.info(f"Incorrect-predicting latent: Layer {self.latent_layer}, Latent {self.latent_idx}")
        logger.info(f"Phase 3.8 optimal threshold (reference): {self.phase3_8_threshold:.4f}")

        # === LOAD PHASE 3.6 DATASET ===
        logger.info("Loading Phase 3.6 hyperparameter dataset...")
        phase3_6_output = discover_latest_phase_output("3.6")
        if not phase3_6_output:
            raise FileNotFoundError(
                "Phase 3.6 output not found. Run Phase 3.6 first.\n"
                "Phase 3.6 generates the hyperparameter dataset."
            )

        # Load dataset (task IDs)
        phase3_6_dir = Path(phase3_6_output).parent
        dataset_file = phase3_6_dir / "dataset_hyperparams_temp_0_0.parquet"

        if not dataset_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_file}\n"
                f"Phase 3.6 should generate dataset_hyperparams_temp_0_0.parquet"
            )

        self.dataset = pd.read_parquet(dataset_file)
        logger.info(f"Loaded {len(self.dataset)} samples from Phase 3.6")

        # Activation files are stored separately as safetensors files
        self.activation_dir = phase3_6_dir / "activations" / "task_activations"
        if not self.activation_dir.exists():
            raise FileNotFoundError(
                f"Activation directory not found: {self.activation_dir}\n"
                f"Phase 3.6 should store activations in activations/task_activations/"
            )

        logger.info(f"Activation directory: {self.activation_dir}")

        # === LOAD SAE FOR DECOMPOSITION ===
        logger.info(f"Loading SAE for Layer {self.latent_layer}...")
        self.sae = load_sae_for_config(self.config, self.latent_layer, self.device)
        logger.info(f"✓ SAE loaded for Layer {self.latent_layer}")

        logger.info("Dependencies loaded successfully")

    def calculate_thresholds(self) -> Dict:
        """
        Calculate percentile-based thresholds from Phase 3.6 activations.

        Returns:
            Dict containing thresholds, statistics, and metadata
        """
        logger.info("="*60)
        logger.info("Calculating Percentile Thresholds")
        logger.info("="*60)

        # === EXTRACT ACTIVATIONS FROM SAFETENSORS FILES ===
        logger.info(f"Extracting L{self.latent_layer}-{self.latent_idx} activations from safetensors files...")

        activations = []
        missing_files = []

        for idx, row in self.dataset.iterrows():
            task_id = row['task_id']

            # Construct activation filename
            activation_file = self.activation_dir / f"{task_id}_layer_{self.latent_layer}.safetensors"

            if not activation_file.exists():
                missing_files.append(task_id)
                continue

            try:
                # Load activation (preserves bfloat16)
                raw_tensor = load_activation(activation_file, self.device)

                # Apply SAE decomposition to get features (16384 dim)
                with torch.no_grad():
                    # Ensure dtype matches SAE parameters
                    raw_tensor = raw_tensor.to(dtype=self.sae.W_enc.dtype)

                    # Encode through SAE to get feature activations
                    latent_activations = self.sae.encode(raw_tensor)  # Shape: (1, 16384)

                    # Extract specific feature
                    latent_activation = latent_activations[0, self.latent_idx].item()
                    activations.append(float(latent_activation))

            except Exception as e:
                logger.warning(f"Task {task_id}: Error processing activation: {e}")
                continue

        if missing_files:
            logger.warning(f"Missing activation files for {len(missing_files)} tasks")
            logger.debug(f"Missing tasks: {missing_files[:10]}...")  # Show first 10

        if not activations:
            raise ValueError("No activations extracted! Check Phase 3.6 safetensors files.")

        logger.info(f"✓ Extracted {len(activations)} activations from {len(self.dataset)} samples")

        # === CALCULATE STATISTICS ===
        logger.info("Calculating activation statistics...")

        activations_array = np.array(activations)

        statistics = {
            'n_samples': len(activations),
            'mean': float(np.mean(activations_array)),
            'std': float(np.std(activations_array)),
            'min': float(np.min(activations_array)),
            'max': float(np.max(activations_array)),
            'median': float(np.median(activations_array))
        }

        logger.info(f"Activation statistics:")
        logger.info(f"  Samples: {statistics['n_samples']}")
        logger.info(f"  Mean: {statistics['mean']:.4f}")
        logger.info(f"  Std: {statistics['std']:.4f}")
        logger.info(f"  Min: {statistics['min']:.4f}")
        logger.info(f"  Max: {statistics['max']:.4f}")
        logger.info(f"  Median: {statistics['median']:.4f}")

        # === CALCULATE PERCENTILE THRESHOLDS ===
        logger.info("Calculating percentile thresholds...")

        percentiles = list(range(5, 100, 5))  # [5, 10, 15, ..., 95]
        thresholds = {}

        logger.info(f"\nPercentile Thresholds:")
        logger.info(f"{'Percentile':<15} {'Threshold':<12} {'Steer %':<10} Description")
        logger.info(f"{'-'*60}")

        for pct in percentiles:
            threshold = float(np.percentile(activations_array, pct))
            steer_pct = 100 - pct

            thresholds[f'p{pct}'] = {
                'percentile': pct,
                'threshold': threshold,
                'steer_percentage': steer_pct,
                'description': f'Steer top {steer_pct}% of cases'
            }

            logger.info(f"{pct}th{'':<11} {threshold:<12.4f} {steer_pct}%{'':<7} Steer top {steer_pct}%")

        # === CREATE SUMMARY ===
        summary = {
            'phase': '8.1',
            'timestamp': datetime.now().isoformat(),
            'source_phase': '3.6',
            'source_dataset': 'hyperparams',
            'latent_info': {
                'layer': self.latent_layer,
                'latent_idx': self.latent_idx,
                'description': 'Incorrect-predicting feature from Phase 3.8'
            },
            'activation_statistics': statistics,
            'percentile_thresholds': thresholds,
            'reference_thresholds': {
                'phase3_8_optimal': {
                    'threshold': self.phase3_8_threshold,
                    'description': 'Phase 3.8 AUROC/F1 optimal threshold (for classification)'
                }
            },
            'notes': [
                'Thresholds calculated on hyperparameter tuning set (Phase 3.6)',
                'No data leakage - validation set (Phase 3.5/8.3) not used',
                'Phase 3.8 threshold optimized for classification, not intervention',
                'Percentile thresholds designed for selective steering decisions'
            ]
        }

        logger.info(f"\n{'='*60}")
        logger.info("Threshold Calculation Complete")
        logger.info(f"{'='*60}")

        return summary

    def save_results(self, summary: Dict) -> None:
        """Save threshold calculation results."""
        logger.info("Saving results...")

        # Save main results
        results_file = self.output_dir / "percentile_thresholds.json"
        save_json(summary, results_file)
        logger.info(f"✓ Saved thresholds to {results_file.name}")

        # Create human-readable summary
        summary_lines = [
            "="*60,
            "PHASE 8.1: PERCENTILE THRESHOLD CALCULATOR",
            "="*60,
            "",
            f"Source: Phase 3.6 (hyperparameter dataset, {summary['activation_statistics']['n_samples']} samples)",
            f"Feature: Layer {summary['latent_info']['layer']}, Feature {summary['latent_info']['feature_idx']}",
            "",
            "PERCENTILE THRESHOLDS",
            "-"*60,
            f"{'Percentile':<15} {'Threshold':<12} {'Steer %':<10}",
            "-"*60,
        ]

        for key, info in summary['percentile_thresholds'].items():
            pct = info['percentile']
            threshold = info['threshold']
            steer_pct = info['steer_percentage']
            summary_lines.append(f"{pct}th{'':<11} {threshold:<12.4f} {steer_pct}%")

        summary_lines.extend([
            "",
            "ACTIVATION STATISTICS",
            "-"*60,
            f"Mean:   {summary['activation_statistics']['mean']:.4f}",
            f"Median: {summary['activation_statistics']['median']:.4f}",
            f"Std:    {summary['activation_statistics']['std']:.4f}",
            f"Range:  [{summary['activation_statistics']['min']:.4f}, {summary['activation_statistics']['max']:.4f}]",
            "",
            "REFERENCE",
            "-"*60,
            f"Phase 3.8 threshold: {summary['reference_thresholds']['phase3_8_optimal']['threshold']:.4f}",
            "  (Optimized for AUROC/F1 classification, not selective steering)",
            "",
            "="*60
        ])

        summary_text = "\n".join(summary_lines)
        summary_file = self.output_dir / "threshold_summary.txt"
        summary_file.write_text(summary_text)
        logger.info(f"✓ Saved summary to {summary_file.name}")

        logger.info(f"\nResults saved to: {self.output_dir}")

    def run(self) -> Dict:
        """Main execution: Calculate and save thresholds."""
        logger.info("="*60)
        logger.info("Starting Phase 8.1: Percentile Threshold Calculator")
        logger.info("="*60)

        summary = self.calculate_thresholds()
        self.save_results(summary)

        # Write phase output manifest
        write_phase_output(
            phase="8.1",
            outputs={
                "primary": "percentile_thresholds.json",
                "summary": "threshold_summary.txt"
            },
            config=self.config,
            output_dir=str(self.output_dir)
        )

        logger.info("\n✅ Phase 8.1 completed successfully")

        return summary
