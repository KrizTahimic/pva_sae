"""
Phase 2.6: Probe Direction Computation

Computes linear probe directions from Phase 1 activations for baseline comparison
with SAE-based correctness prediction and steering.

MOTIVATION: Recent work has shown that simple linear probes often match or outperform
SAE-based methods for both concept detection and steering:

- Kantamneni et al. (2025), "Are Sparse Autoencoders Useful? A Case Study in
  Sparse Probing" (arXiv:2502.16681) - Neel Nanda et al. at Google DeepMind
  found SAE probes underperform logistic regression across 113 datasets.

- AxBench (arXiv:2501.17148) found DiffMean and Linear Probe achieve ~0.94 AUROC
  vs vanilla SAE's 0.695 for concept detection.

This phase provides converging evidence: if probes and SAE directions identify
similar representations (high cosine similarity) and achieve comparable metrics,
this validates the linear representation hypothesis underlying both approaches.

Two probe methods:

1. Mass-Mean Probe (for steering, Phase 4.x)
   Reference: Marks & Tegmark (2023), "The Geometry of Truth"
              arXiv:2310.06824

   Formula: direction = Σ⁻¹ @ (μ₊ - μ₋)

   Equivalent to Fisher's Linear Discriminant direction. The covariance correction
   removes interference from correlated but irrelevant features, making it more
   suitable for causal interventions than plain mean-difference.

   Related: Contrastive Activation Addition (CAA) uses mean-difference without
   covariance correction (Rimsky et al. 2023).

2. Logistic Regression Probe (for prediction, Phase 3.8)
   Reference: Alain & Bengio (2016), "Understanding intermediate layers using
              linear classifier probes" arXiv:1610.01644

   Finds the discriminative hyperplane via L2-regularized logistic regression.
   Hyperparameter C selected via cross-validation on AUROC.

Usage:
    python3 run.py phase 2.6
"""

import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy import stats
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from joblib import Parallel, delayed

from common.config import Config
from common.logging import get_logger
from common.utils import ensure_directory_exists, save_json, load_json
from common.phase_discovery import (
    discover_latest_phase_output,
    get_phase_output_dir,
    write_phase_output,
)

logger = get_logger("phase2_6.probe_direction_computer")


def get_n_jobs(n_tasks: int) -> int:
    """Get optimal number of parallel jobs: min(CPUs, tasks).

    Args:
        n_tasks: Number of tasks to parallelize (e.g., number of layers)

    Returns:
        Optimal number of parallel workers
    """
    n_cpus = os.cpu_count() or 4
    return min(n_cpus, n_tasks)


@dataclass
class ProbeResult:
    """Results for a single layer's probe computation."""
    layer: int

    # Mass-mean probe
    mass_mean_cv_auroc: float
    mass_mean_cv_std: float
    mass_mean_full_auroc: float
    mass_mean_full_f1: float
    mass_mean_t_statistic: float
    mass_mean_separation: float

    # Logistic regression probe
    logreg_cv_auroc: float
    logreg_cv_std: float
    logreg_full_auroc: float
    logreg_full_f1: float
    logreg_t_statistic: float
    logreg_best_C: float
    logreg_bias: float

    # Sample counts
    n_samples: int
    n_correct: int
    n_incorrect: int


class ProbeDirectionComputer:
    """Compute linear probe directions for baseline comparison with SAE."""

    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.logger = get_logger("phase2_6.runner", phase="2.6")

        # Output directory
        self.output_dir = Path(get_phase_output_dir("2.6", config))
        ensure_directory_exists(self.output_dir)
        self.probe_dir = self.output_dir / "probe_directions"
        ensure_directory_exists(self.probe_dir)

        # Load Phase 1 output path
        self._load_phase1_path()

    def _load_phase1_path(self) -> None:
        """Load Phase 1 output directory."""
        phase1_output = discover_latest_phase_output("1", config=self.config)
        if not phase1_output:
            raise FileNotFoundError(
                "Phase 1 output not found. Run Phase 1 first to generate activations."
            )
        self.phase1_dir = Path(phase1_output).parent
        self.logger.info(f"Using Phase 1 output: {self.phase1_dir}")

    def load_activations_for_layer(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Load all activations for a given layer from Phase 1.

        Args:
            layer: Layer number to load activations from

        Returns:
            Tuple of (X, y) where X is activations [N, d_model] and y is labels
        """
        X = []
        y = []

        # Load correct samples
        correct_dir = self.phase1_dir / "activations" / "correct"
        if correct_dir.exists():
            for f in sorted(correct_dir.glob(f"*_layer_{layer}.safetensors")):
                tensor = load_file(str(f))
                act = list(tensor.values())[0]  # Get the tensor
                X.append(act.squeeze().float().numpy())
                y.append(1)  # correct = 1

        # Load incorrect samples
        incorrect_dir = self.phase1_dir / "activations" / "incorrect"
        if incorrect_dir.exists():
            for f in sorted(incorrect_dir.glob(f"*_layer_{layer}.safetensors")):
                tensor = load_file(str(f))
                act = list(tensor.values())[0]
                X.append(act.squeeze().float().numpy())
                y.append(0)  # incorrect = 0

        if len(X) == 0:
            raise ValueError(f"No activations found for layer {layer}")

        return np.array(X), np.array(y)

    def compute_mass_mean_direction(
        self, X: np.ndarray, y: np.ndarray, reg_lambda: Optional[float] = None
    ) -> np.ndarray:
        """Compute Mass-Mean probe direction (Fisher's LDA variant).

        Reference: Marks & Tegmark (2023), "The Geometry of Truth"
                   arXiv:2310.06824

        Formula: direction = Σ⁻¹ @ (μ₊ - μ₋)

        where Σ is the pooled covariance matrix (regularized for stability).

        This is equivalent to Fisher's Linear Discriminant (LDA) direction.
        Unlike plain mean-difference (CAA/DoM), this corrects for covariance
        structure, removing interference from correlated but irrelevant features.

        Why covariance correction matters for steering:
        - If feature A (correctness) is correlated with feature B (code style),
          plain mean-diff will partially encode B.
        - Σ⁻¹ decorrelates, isolating the direction that best separates classes.

        Args:
            X: Activations [N, d_model]
            y: Labels (1=correct, 0=incorrect)
            reg_lambda: Tikhonov regularization (λI added to Σ) for numerical
                        stability. Default 1e-4 prevents singular matrix issues.

        Returns:
            Mass-mean direction [d_model], L2-normalized to unit norm
        """
        if reg_lambda is None:
            reg_lambda = self.config.probe_mass_mean_reg_lambda

        # Mean difference
        mu_correct = X[y == 1].mean(axis=0)
        mu_incorrect = X[y == 0].mean(axis=0)
        mu_diff = mu_correct - mu_incorrect

        # Covariance matrix with regularization
        Sigma = np.cov(X.T)  # [d_model, d_model]
        Sigma_reg = Sigma + reg_lambda * np.eye(Sigma.shape[0])

        # Mass-Mean direction: Σ⁻¹ @ μ_diff
        direction = np.linalg.solve(Sigma_reg, mu_diff)

        # Normalize to unit L2 norm (matches SAE decoder direction convention)
        # This ensures same coefficient ranges work for both probe and SAE steering
        direction = direction / np.linalg.norm(direction)

        return direction

    def compute_logreg_direction(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Compute logistic regression probe direction with hyperparameter search.

        Reference: Alain & Bengio (2016), "Understanding intermediate layers
                   using linear classifier probes" arXiv:1610.01644

        Recent validation: Kantamneni et al. (2025) arXiv:2502.16681 showed
        logistic regression outperforms SAE probes across 113 datasets.

        Model: P(correct | x) = σ(w·x + b)

        The weight vector w defines the probe direction. L2 regularization
        (controlled by C = 1/λ) prevents overfitting. Best C selected via
        cross-validation on AUROC.

        Args:
            X: Activations [N, d_model]
            y: Labels (1=correct, 0=incorrect)

        Returns:
            Tuple of (direction [d_model], bias, best_C)
        """
        C_values = self.config.probe_logreg_C_values
        cv_folds = self.config.probe_cv_folds

        # Search for optimal regularization
        best_cv_auroc = 0
        best_C = C_values[0]

        for C in C_values:
            probe_tmp = LogisticRegression(
                C=C, max_iter=2000, random_state=42, solver='lbfgs'
            )
            cv_auroc = cross_val_score(
                probe_tmp, X, y, cv=cv_folds, scoring='roc_auc'
            ).mean()

            if cv_auroc > best_cv_auroc:
                best_cv_auroc = cv_auroc
                best_C = C

        # Fit final model with best C
        probe = LogisticRegression(
            C=best_C, max_iter=2000, random_state=42, solver='lbfgs'
        )
        probe.fit(X, y)

        return probe.coef_[0], probe.intercept_[0], best_C

    def compute_direction_metrics(
        self, X: np.ndarray, y: np.ndarray, direction: np.ndarray, bias: float = 0.0
    ) -> dict:
        """Compute all metrics for a given direction.

        Args:
            X: Activations [N, d_model]
            y: Labels (1=correct, 0=incorrect)
            direction: Weight vector [d_model]
            bias: Bias term (default 0.0 for non-logreg methods)

        Returns:
            Dictionary of metrics
        """
        # Compute logits: w·x + b
        logits = X @ direction + bias

        # Compute probabilities: sigmoid(logits)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

        # AUROC
        auroc = roc_auc_score(y, probs)

        # F1 at optimal threshold
        thresholds = np.linspace(0, 1, 100)
        best_f1 = 0
        for thresh in thresholds:
            preds = (probs > thresh).astype(int)
            f1 = f1_score(y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        # T-statistic (Welch's t-test on probabilities)
        correct_probs = probs[y == 1]
        incorrect_probs = probs[y == 0]
        t_stat, p_value = stats.ttest_ind(correct_probs, incorrect_probs, equal_var=False)

        # Separation: mean difference of probabilities
        separation = np.mean(correct_probs) - np.mean(incorrect_probs)

        return {
            'auroc': float(auroc),
            'f1': float(best_f1),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'separation': float(separation),
            'direction_norm': float(np.linalg.norm(direction)),
        }

    def compute_probes_for_layer(self, layer: int) -> ProbeResult:
        """Compute both probe types for a single layer.

        Args:
            layer: Layer number

        Returns:
            ProbeResult with all metrics
        """
        self.logger.info(f"Processing layer {layer}...")

        # Load activations
        X, y = self.load_activations_for_layer(layer)
        n_samples = len(X)
        n_correct = sum(y)
        n_incorrect = n_samples - n_correct

        self.logger.info(f"  Loaded {n_samples} samples: {n_correct} correct, {n_incorrect} incorrect")

        if n_samples < 20:
            raise ValueError(f"Not enough samples ({n_samples}) for reliable probe computation")

        # === Mass-Mean Probe ===
        # Cross-validation AUROC
        kf = KFold(n_splits=self.config.probe_cv_folds, shuffle=True, random_state=42)
        mass_mean_cv_aurocs = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            mm_dir = self.compute_mass_mean_direction(X_train, y_train)
            mm_scores = X_test @ mm_dir
            try:
                auroc = roc_auc_score(y_test, mm_scores)
                mass_mean_cv_aurocs.append(auroc)
            except ValueError:
                pass  # Skip if only one class in test set

        mass_mean_cv_mean = np.mean(mass_mean_cv_aurocs) if mass_mean_cv_aurocs else 0
        mass_mean_cv_std = np.std(mass_mean_cv_aurocs) if mass_mean_cv_aurocs else 0

        # Full data mass-mean
        mass_mean_dir = self.compute_mass_mean_direction(X, y)
        mass_mean_metrics = self.compute_direction_metrics(X, y, mass_mean_dir)

        # === Logistic Regression Probe ===
        logreg_dir, logreg_bias, best_C = self.compute_logreg_direction(X, y)

        # Cross-validation AUROC with best C
        probe = LogisticRegression(C=best_C, max_iter=2000, random_state=42, solver='lbfgs')
        cv_scores = cross_val_score(probe, X, y, cv=self.config.probe_cv_folds, scoring='roc_auc')
        logreg_cv_mean = cv_scores.mean()
        logreg_cv_std = cv_scores.std()

        # Full data metrics
        logreg_metrics = self.compute_direction_metrics(X, y, logreg_dir, logreg_bias)

        # Save probe directions for this layer
        probe_file = self.probe_dir / f"layer_{layer}_probes.safetensors"
        probe_tensors = {
            "mass_mean_direction": torch.tensor(mass_mean_dir, dtype=torch.float32),
            "logreg_direction": torch.tensor(logreg_dir, dtype=torch.float32),
            "logreg_bias": torch.tensor([logreg_bias], dtype=torch.float32),
        }
        save_file(probe_tensors, str(probe_file))
        self.logger.info(f"  Saved probe directions to {probe_file}")

        # Save layer metrics
        layer_metrics = {
            'layer': layer,
            'n_samples': n_samples,
            'n_correct': n_correct,
            'n_incorrect': n_incorrect,
            'mass_mean': {
                'cv_auroc': mass_mean_cv_mean,
                'cv_std': mass_mean_cv_std,
                **mass_mean_metrics,
            },
            'logreg': {
                'cv_auroc': logreg_cv_mean,
                'cv_std': logreg_cv_std,
                'best_C': best_C,
                'bias': logreg_bias,
                **logreg_metrics,
            }
        }
        save_json(layer_metrics, self.output_dir / f"layer_{layer}_metrics.json")

        return ProbeResult(
            layer=layer,
            mass_mean_cv_auroc=mass_mean_cv_mean,
            mass_mean_cv_std=mass_mean_cv_std,
            mass_mean_full_auroc=mass_mean_metrics['auroc'],
            mass_mean_full_f1=mass_mean_metrics['f1'],
            mass_mean_t_statistic=mass_mean_metrics['t_statistic'],
            mass_mean_separation=mass_mean_metrics['separation'],
            logreg_cv_auroc=logreg_cv_mean,
            logreg_cv_std=logreg_cv_std,
            logreg_full_auroc=logreg_metrics['auroc'],
            logreg_full_f1=logreg_metrics['f1'],
            logreg_t_statistic=logreg_metrics['t_statistic'],
            logreg_best_C=best_C,
            logreg_bias=logreg_bias,
            n_samples=n_samples,
            n_correct=n_correct,
            n_incorrect=n_incorrect,
        )

    def run(self) -> dict:
        """Run probe direction computation for all layers.

        Returns:
            Summary dictionary with best layer selections
        """
        start_time = datetime.now()
        self.logger.info("Starting Phase 2.6: Probe Direction Computation")
        self.logger.info("Computing linear probe directions for baseline comparison with SAE")
        self.logger.info("\n" + self.config.dump(phase="2.6"))

        # Get layers to process from config
        layers = self.config.activation_layers
        n_jobs = get_n_jobs(len(layers))
        self.logger.info(f"Processing {len(layers)} layers: {layers}")
        self.logger.info(f"Using {n_jobs} parallel workers (CPUs available: {os.cpu_count()})")

        # Compute probes for each layer in parallel with progress bar
        def process_layer_safe(layer: int) -> tuple[int, ProbeResult | None, str | None]:
            """Wrapper to catch exceptions and return (layer, result, error)."""
            try:
                result = self.compute_probes_for_layer(layer)
                return (layer, result, None)
            except Exception as e:
                return (layer, None, str(e))

        # Run parallel processing with tqdm progress
        parallel_results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_layer_safe)(layer)
            for layer in tqdm(layers, desc="Computing probes", unit="layer")
        )

        # Collect results
        results = {}
        for layer, result, error in parallel_results:
            if result is not None:
                results[layer] = result
            else:
                self.logger.warning(f"Failed to process layer {layer}: {error}")

        if not results:
            raise ValueError("No layers successfully processed")

        # Find best layers for each method
        # Mass-mean: best by CV AUROC (for steering)
        best_mass_mean = max(results.items(), key=lambda x: x[1].mass_mean_cv_auroc)

        # LogReg: best by CV AUROC (for prediction)
        best_logreg = max(results.items(), key=lambda x: x[1].logreg_cv_auroc)

        # Also track best by t-statistic
        best_mass_mean_t = max(results.items(), key=lambda x: abs(x[1].mass_mean_t_statistic))
        best_logreg_t = max(results.items(), key=lambda x: abs(x[1].logreg_t_statistic))

        # Log summary
        self.logger.info("\n" + "="*60)
        self.logger.info("PROBE COMPUTATION SUMMARY")
        self.logger.info("="*60)

        self.logger.info("\nBest layers by CV AUROC:")
        self.logger.info(f"  Mass-Mean: Layer {best_mass_mean[0]} (AUROC={best_mass_mean[1].mass_mean_cv_auroc:.3f})")
        self.logger.info(f"  LogReg:    Layer {best_logreg[0]} (AUROC={best_logreg[1].logreg_cv_auroc:.3f})")

        self.logger.info("\nBest layers by |t-statistic|:")
        self.logger.info(f"  Mass-Mean: Layer {best_mass_mean_t[0]} (t={best_mass_mean_t[1].mass_mean_t_statistic:.1f})")
        self.logger.info(f"  LogReg:    Layer {best_logreg_t[0]} (t={best_logreg_t[1].logreg_t_statistic:.1f})")

        # Create best probe directions summary
        best_probe_directions = {
            'mass_mean': {
                'best_layer': best_mass_mean[0],
                'cv_auroc': best_mass_mean[1].mass_mean_cv_auroc,
                'cv_std': best_mass_mean[1].mass_mean_cv_std,
                'full_auroc': best_mass_mean[1].mass_mean_full_auroc,
                'full_f1': best_mass_mean[1].mass_mean_full_f1,
                't_statistic': best_mass_mean[1].mass_mean_t_statistic,
                'separation': best_mass_mean[1].mass_mean_separation,
            },
            'logreg': {
                'best_layer': best_logreg[0],
                'cv_auroc': best_logreg[1].logreg_cv_auroc,
                'cv_std': best_logreg[1].logreg_cv_std,
                'full_auroc': best_logreg[1].logreg_full_auroc,
                'full_f1': best_logreg[1].logreg_full_f1,
                't_statistic': best_logreg[1].logreg_t_statistic,
                'best_C': best_logreg[1].logreg_best_C,
                'bias': best_logreg[1].logreg_bias,
            }
        }
        save_json(best_probe_directions, self.output_dir / "best_probe_directions.json")

        # Create layer-wise metrics summary
        layer_metrics = {}
        for layer, result in results.items():
            layer_metrics[str(layer)] = {
                'mass_mean_cv_auroc': result.mass_mean_cv_auroc,
                'logreg_cv_auroc': result.logreg_cv_auroc,
                'mass_mean_t_statistic': result.mass_mean_t_statistic,
                'logreg_t_statistic': result.logreg_t_statistic,
                'n_samples': result.n_samples,
            }

        # Create phase output summary
        duration = (datetime.now() - start_time).total_seconds()
        summary = {
            'phase': '2.6',
            'description': 'Probe Direction Computation',
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'config': {
                'model_name': self.config.model_name,
                'dataset_name': self.config.dataset_name,
                'mass_mean_reg_lambda': self.config.probe_mass_mean_reg_lambda,
                'logreg_C_values': self.config.probe_logreg_C_values,
                'cv_folds': self.config.probe_cv_folds,
            },
            'best_probe_directions': best_probe_directions,
            'layer_metrics': layer_metrics,
            'n_layers_processed': len(results),
        }
        save_json(summary, self.output_dir / "phase_2_6_summary.json")

        # Write phase_output.json manifest
        write_phase_output(
            phase="2.6",
            outputs={
                "primary": "best_probe_directions.json",
                "summary": "phase_2_6_summary.json",
            },
            config=self.config,
            output_dir=str(self.output_dir),
            dependencies={
                "1": str(self.phase1_dir),
            },
            config_keys=['model_name', 'dataset_name', 'probe_mass_mean_reg_lambda', 'probe_cv_folds']
        )

        self.logger.info(f"\nPhase 2.6 completed in {duration:.1f} seconds")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("="*60 + "\n")

        return summary
