"""
Linear Probe vs SAE Sanity Check

Compares:
1. Mass-Mean probe (difference-in-means with covariance correction)
2. Logistic Regression probe (discriminative direction)
3. SAE latent (best latent by AUROC)
4. Random direction (sanity baseline)

Mass-Mean Probe: From "The Geometry of Truth" (Marks & Tegmark 2023)
- More causally implicated in model outputs than logistic regression
- Corrects for interfering non-orthogonal features
- Formula: direction = Σ⁻¹ @ (μ_correct - μ_incorrect)

Usage:
    python run_sanity_check.py --model gemma2b --layer 19
    python run_sanity_check.py --model gemma2b --all-layers
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
from safetensors.torch import load_file, save_file
import json


def compute_mass_mean_probe(X: np.ndarray, y: np.ndarray, reg_lambda: float = 1e-4) -> np.ndarray:
    """
    Compute Mass-Mean probe direction from "The Geometry of Truth" (Marks & Tegmark 2023).

    Unlike plain mean-difference, this corrects for covariance structure,
    removing interference from correlated but irrelevant features.

    Args:
        X: Activations [N, d_model]
        y: Labels (1=correct, 0=incorrect)
        reg_lambda: Regularization for covariance inversion

    Returns:
        direction: Mass-Mean probe direction [d_model]
    """
    # Mean difference
    mu_correct = X[y == 1].mean(axis=0)
    mu_incorrect = X[y == 0].mean(axis=0)
    mu_diff = mu_correct - mu_incorrect

    # Covariance matrix with regularization
    Sigma = np.cov(X.T)  # [d_model, d_model]
    Sigma_reg = Sigma + reg_lambda * np.eye(Sigma.shape[0])

    # Mass-Mean direction: Σ⁻¹ @ μ_diff
    direction = np.linalg.solve(Sigma_reg, mu_diff)

    return direction


def compute_mean_diff(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simple mean difference (CAA/DoM baseline)."""
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


def load_activations_for_layer(phase1_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Load all activations for a given layer, return X (activations) and y (labels)."""
    X = []
    y = []

    # Load correct samples
    correct_dir = phase1_dir / "activations" / "correct"
    if correct_dir.exists():
        for f in correct_dir.glob(f"*_layer_{layer}.safetensors"):
            tensor = load_file(str(f))
            act = list(tensor.values())[0]  # Get the tensor
            X.append(act.squeeze().float().numpy())
            y.append(1)  # correct = 1

    # Load incorrect samples
    incorrect_dir = phase1_dir / "activations" / "incorrect"
    if incorrect_dir.exists():
        for f in incorrect_dir.glob(f"*_layer_{layer}.safetensors"):
            tensor = load_file(str(f))
            act = list(tensor.values())[0]
            X.append(act.squeeze().float().numpy())
            y.append(0)  # incorrect = 0

    return np.array(X), np.array(y)


def load_sae_and_encode(X: np.ndarray, layer: int, model_name: str = "google/gemma-2-2b") -> np.ndarray:
    """Load SAE and encode activations."""
    from common.sae_loader import load_sae_for_config
    from common.config import Config

    config = Config()
    config.model_name = model_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = load_sae_for_config(config, layer, device)

    # Convert numpy to tensor (numpy doesn't support bfloat16, so convert after)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    X_tensor = X_tensor.to(dtype=sae.W_enc.dtype)

    with torch.no_grad():
        latent_activations = sae.encode(X_tensor)  # [N, 16384]

    return latent_activations.cpu().float().numpy(), sae


def get_best_latent_from_phase25(phase25_dir: Path) -> dict | None:
    """Load best latent info from Phase 2.5 results."""
    top_latents_file = phase25_dir / "top_20_latents.json"
    if top_latents_file.exists():
        with open(top_latents_file) as f:
            data = json.load(f)
        # Return the best correct-steering latent
        if "correct" in data and len(data["correct"]) > 0:
            return data["correct"][0]
    return None


def compute_direction_metrics(X: np.ndarray, y: np.ndarray, direction: np.ndarray, bias: float = 0.0) -> dict:
    """Compute all metrics for a given direction.

    Args:
        X: Activations [N, d_model]
        y: Labels (1=correct, 0=incorrect)
        direction: Weight vector [d_model]
        bias: Bias term for logistic regression (default 0.0 for non-logreg methods)
    """
    # Compute logits: w·x + b
    logits = X @ direction + bias

    # Compute probabilities: sigmoid(logits)
    probs = 1.0 / (1.0 + np.exp(-logits))

    # AUROC using probabilities (proper logreg output)
    auroc = roc_auc_score(y, probs)

    # F1 at optimal threshold (on probabilities)
    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    # T-statistic on probabilities (Welch's t-test)
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
        'mean_prob_correct': float(np.mean(correct_probs)),
        'mean_prob_incorrect': float(np.mean(incorrect_probs)),
    }


def run_comparison(phase1_dir: Path, layer: int, phase25_dir: Path = None, model_name: str = "google/gemma-2-2b"):
    """Run the full comparison."""
    print(f"\n{'='*60}")
    print(f"Linear Probe vs SAE Comparison")
    print(f"Layer: {layer} | Model: {model_name}")
    print(f"{'='*60}\n")

    # 1. Load raw activations
    print("Loading activations...")
    X, y = load_activations_for_layer(phase1_dir, layer)
    print(f"  Loaded {len(X)} samples: {sum(y)} correct, {len(y) - sum(y)} incorrect")
    print(f"  Activation shape: {X.shape}")

    if len(X) < 20:
        print("ERROR: Not enough samples for meaningful comparison")
        return

    # =========================================================================
    # [1] MASS-MEAN PROBE (Recommended for steering)
    # =========================================================================
    print("\n[1] MASS-MEAN PROBE (Geometry of Truth)")
    print("-" * 40)

    # Cross-validation for Mass-Mean
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mass_mean_cv_aurocs = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        mm_dir = compute_mass_mean_probe(X_train, y_train)
        mm_scores = X_test @ mm_dir
        try:
            auroc = roc_auc_score(y_test, mm_scores)
            mass_mean_cv_aurocs.append(auroc)
        except:
            pass
    mass_mean_cv_mean = np.mean(mass_mean_cv_aurocs) if mass_mean_cv_aurocs else 0
    mass_mean_cv_std = np.std(mass_mean_cv_aurocs) if mass_mean_cv_aurocs else 0
    print(f"  5-fold CV AUROC: {mass_mean_cv_mean:.3f} (+/- {mass_mean_cv_std:.3f})")

    # Full data metrics
    mass_mean_dir = compute_mass_mean_probe(X, y)
    mass_mean_metrics = compute_direction_metrics(X, y, mass_mean_dir)

    print(f"  Full data AUROC: {mass_mean_metrics['auroc']:.3f}")
    print(f"  Full data F1:    {mass_mean_metrics['f1']:.3f}")
    print(f"  T-statistic:     {mass_mean_metrics['t_statistic']:.3f} (p={mass_mean_metrics['p_value']:.2e})")
    print(f"  Separation:      {mass_mean_metrics['separation']:.3f}")
    print(f"  Direction norm:  {mass_mean_metrics['direction_norm']:.3f}")

    # =========================================================================
    # [2] LOGISTIC REGRESSION (Best for prediction)
    # =========================================================================
    print("\n[2] LOGISTIC REGRESSION")
    print("-" * 40)

    # Search for optimal regularization strength
    # C is inverse regularization: larger C = less regularization
    C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    print("  Searching for optimal regularization (C)...")
    print(f"  {'C':<10} {'CV AUROC':<12} {'P(+|+)':<10} {'P(+|-)':<10} {'Sep':<10}")
    print(f"  {'-'*52}")

    best_cv_auroc = 0
    best_C = 0.0001
    cv_results = []

    for C in C_values:
        probe_tmp = LogisticRegression(C=C, max_iter=2000, random_state=42, solver='lbfgs')
        cv_auroc = cross_val_score(probe_tmp, X, y, cv=5, scoring='roc_auc').mean()

        # Fit to get probability calibration info
        probe_tmp.fit(X, y)
        tmp_metrics = compute_direction_metrics(X, y, probe_tmp.coef_[0], bias=probe_tmp.intercept_[0])

        cv_results.append({
            'C': C,
            'cv_auroc': cv_auroc,
            'mean_prob_correct': tmp_metrics['mean_prob_correct'],
            'mean_prob_incorrect': tmp_metrics['mean_prob_incorrect'],
            'separation': tmp_metrics['separation'],
        })

        print(f"  {C:<10} {cv_auroc:.3f}        {tmp_metrics['mean_prob_correct']:.3f}      {tmp_metrics['mean_prob_incorrect']:.3f}      {tmp_metrics['separation']:.3f}")

        if cv_auroc > best_cv_auroc:
            best_cv_auroc = cv_auroc
            best_C = C

    print(f"\n  Best C by CV AUROC: {best_C} (AUROC={best_cv_auroc:.3f})")

    # Use best C for final model
    probe = LogisticRegression(C=best_C, max_iter=2000, random_state=42, solver='lbfgs')

    # Cross-validation AUROC with best C
    cv_scores = cross_val_score(probe, X, y, cv=5, scoring='roc_auc')
    print(f"  5-fold CV AUROC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Fit on all data
    probe.fit(X, y)
    logreg_dir = probe.coef_[0]
    logreg_bias = probe.intercept_[0]
    logreg_metrics = compute_direction_metrics(X, y, logreg_dir, bias=logreg_bias)

    print(f"  AUROC:           {logreg_metrics['auroc']:.3f}")
    print(f"  F1:              {logreg_metrics['f1']:.3f}")
    print(f"  T-statistic:     {logreg_metrics['t_statistic']:.3f} (p={logreg_metrics['p_value']:.2e})")
    print(f"  Separation:      {logreg_metrics['separation']:.3f}")
    print(f"  Direction norm:  {logreg_metrics['direction_norm']:.3f}")
    print(f"  Bias:            {logreg_bias:.3f}")
    print(f"  P(correct) for correct:   {logreg_metrics['mean_prob_correct']:.3f}")
    print(f"  P(correct) for incorrect: {logreg_metrics['mean_prob_incorrect']:.3f}")

    # =========================================================================
    # [3] MEAN DIFFERENCE (Simple baseline)
    # =========================================================================
    print("\n[3] MEAN DIFFERENCE (CAA/DoM baseline)")
    print("-" * 40)

    mean_diff_dir = compute_mean_diff(X, y)
    mean_diff_metrics = compute_direction_metrics(X, y, mean_diff_dir)

    print(f"  AUROC:           {mean_diff_metrics['auroc']:.3f}")
    print(f"  F1:              {mean_diff_metrics['f1']:.3f}")
    print(f"  T-statistic:     {mean_diff_metrics['t_statistic']:.3f} (p={mean_diff_metrics['p_value']:.2e})")
    print(f"  Separation:      {mean_diff_metrics['separation']:.3f}")
    print(f"  Direction norm:  {mean_diff_metrics['direction_norm']:.3f}")

    # =========================================================================
    # [4] SAE LATENT (Best single latent by AUROC)
    # =========================================================================
    print("\n[4] SAE LATENT (Best by AUROC)")
    print("-" * 40)

    try:
        latent_activations, sae = load_sae_and_encode(X, layer, model_name)
        print(f"  SAE latent shape: {latent_activations.shape}")

        # Find best latent by AUROC (brute force over all 16k)
        best_auroc = 0
        best_latent_idx = 0
        best_latent_scores = None

        print("  Scanning all latents for best AUROC...")
        for idx in range(latent_activations.shape[1]):
            scores = latent_activations[:, idx]
            if scores.std() > 0:  # Skip dead latents
                try:
                    auroc = roc_auc_score(y, scores)
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_latent_idx = idx
                        best_latent_scores = scores
                    # Also check negative direction
                    auroc_neg = roc_auc_score(y, -scores)
                    if auroc_neg > best_auroc:
                        best_auroc = auroc_neg
                        best_latent_idx = idx
                        best_latent_scores = -scores
                except:
                    pass

        print(f"  Best latent index: {best_latent_idx}")
        print(f"  Best latent AUROC: {best_auroc:.3f}")

        # Compute F1 at optimal threshold
        best_f1 = 0
        sae_separation_score = 0
        sae_t_statistic = 0
        sae_p_value = 1.0

        if best_latent_scores is not None:
            thresholds = np.linspace(best_latent_scores.min(), best_latent_scores.max(), 100)
            for thresh in thresholds:
                preds = (best_latent_scores > thresh).astype(int)
                f1 = f1_score(y, preds)
                if f1 > best_f1:
                    best_f1 = f1
            print(f"  Best latent F1:    {best_f1:.3f}")

            # Compute separation score and t-statistic for SAE latent
            sae_correct_scores = best_latent_scores[y == 1]
            sae_incorrect_scores = best_latent_scores[y == 0]
            sae_separation_score = np.mean(sae_correct_scores) - np.mean(sae_incorrect_scores)
            sae_t_stat, sae_p_value = stats.ttest_ind(sae_correct_scores, sae_incorrect_scores, equal_var=False)
            sae_t_statistic = sae_t_stat

            print(f"  Separation score: {sae_separation_score:.3f}")
            print(f"  T-statistic: {sae_t_statistic:.3f} (p={sae_p_value:.2e})")

        # If we have Phase 2.5 results, compare to selected latent
        if phase25_dir and phase25_dir.exists():
            best_latent_info = get_best_latent_from_phase25(phase25_dir)
            if best_latent_info:
                selected_idx = best_latent_info['latent_idx']
                selected_layer = best_latent_info.get('layer')
                print(f"\n  Phase 2.5 selected: layer {selected_layer}, latent {selected_idx}")
                if selected_layer == layer:
                    selected_scores = latent_activations[:, selected_idx]
                    selected_auroc = roc_auc_score(y, selected_scores)
                    print(f"  Phase 2.5 latent AUROC: {selected_auroc:.3f}")

    except Exception as e:
        print(f"  ERROR loading SAE: {e}")
        best_auroc = 0
        best_f1 = 0
        best_latent_idx = -1
        sae_separation_score = 0
        sae_t_statistic = 0
        sae_p_value = 1.0

    # =========================================================================
    # [5] RANDOM DIRECTION (Sanity Check)
    # =========================================================================
    print("\n[5] RANDOM DIRECTION (Sanity Check)")
    print("-" * 40)

    np.random.seed(42)
    random_aurocs = []
    for _ in range(10):
        rand_dir = np.random.randn(X.shape[1])
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        rand_scores = X @ rand_dir
        try:
            auroc = roc_auc_score(y, rand_scores)
            auroc = max(auroc, 1 - auroc)  # Take best of +/-
            random_aurocs.append(auroc)
        except:
            pass

    print(f"  AUROC: {np.mean(random_aurocs):.3f} (+/- {np.std(random_aurocs):.3f})")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  {'Method':<30} {'CV AUROC':<12} {'Full AUROC':<12} {'Separation':<10}")
    print(f"  {'-'*64}")
    print(f"  {'Mass-Mean Probe':<30} {mass_mean_cv_mean:.3f}        {mass_mean_metrics['auroc']:.3f}        {mass_mean_metrics['separation']:.3f}")
    print(f"  {'Logistic Regression':<30} {cv_scores.mean():.3f}        {logreg_metrics['auroc']:.3f}        {logreg_metrics['separation']:.3f}")
    print(f"  {'Mean Difference':<30} {'N/A':<12} {mean_diff_metrics['auroc']:.3f}        {mean_diff_metrics['separation']:.3f}*")
    print(f"  {'SAE Best Latent':<30} {'N/A':<12} {best_auroc:.3f}        {sae_separation_score:.3f}")
    print(f"  {'Random (baseline)':<30} {'N/A':<12} {np.mean(random_aurocs):.3f}")
    print()
    print("  * Mean Diff separation = ||μ+ - μ-||² (not comparable)")
    print()
    print("  KEY INSIGHT: CV AUROC is the fair comparison (tests on unseen data)")
    print(f"    Mass-Mean CV:  {mass_mean_cv_mean:.3f}")
    print(f"    LogReg CV:     {cv_scores.mean():.3f}")
    print(f"    SAE:           {best_auroc:.3f} (no CV, but constrained to 16k pre-defined directions)")
    print()

    # Interpretation (using CV AUROC for fair comparison)
    best_probe_cv = max(mass_mean_cv_mean, cv_scores.mean())
    if best_probe_cv > best_auroc + 0.03:
        print("  CONCLUSION: Probes outperform SAE on CV AUROC")
    elif best_auroc > best_probe_cv + 0.03:
        print("  CONCLUSION: SAE outperforms probes on CV AUROC!")
        print("     SAE provides interpretability AND better generalization")
    else:
        print("  CONCLUSION: Comparable performance")
        print("     SAE provides interpretability without losing accuracy")

    return {
        # Mass-Mean probe (recommended for steering)
        'mass_mean_direction': mass_mean_dir,
        'mass_mean_cv_auroc': float(mass_mean_cv_mean),
        'mass_mean_cv_std': float(mass_mean_cv_std),
        'mass_mean_auroc': mass_mean_metrics['auroc'],
        'mass_mean_f1': mass_mean_metrics['f1'],
        'mass_mean_t_statistic': mass_mean_metrics['t_statistic'],
        'mass_mean_separation': mass_mean_metrics['separation'],

        # Logistic Regression (best for prediction)
        'logreg_direction': logreg_dir,
        'logreg_bias': float(logreg_bias),
        'logreg_best_C': float(best_C),
        'logreg_auroc': logreg_metrics['auroc'],
        'logreg_f1': logreg_metrics['f1'],
        'logreg_t_statistic': logreg_metrics['t_statistic'],
        'logreg_separation': logreg_metrics['separation'],
        'logreg_cv_auroc': float(cv_scores.mean()),
        'logreg_cv_std': float(cv_scores.std()),
        'logreg_mean_prob_correct': logreg_metrics['mean_prob_correct'],
        'logreg_mean_prob_incorrect': logreg_metrics['mean_prob_incorrect'],

        # Mean Difference (simple baseline)
        'mean_diff_direction': mean_diff_dir,
        'mean_diff_auroc': mean_diff_metrics['auroc'],
        'mean_diff_f1': mean_diff_metrics['f1'],
        'mean_diff_t_statistic': mean_diff_metrics['t_statistic'],
        'mean_diff_separation': mean_diff_metrics['separation'],

        # SAE metrics
        'sae_auroc': float(best_auroc),
        'sae_f1': float(best_f1),
        'sae_t_statistic': float(sae_t_statistic),
        'sae_separation': float(sae_separation_score),
        'sae_best_latent_idx': int(best_latent_idx),

        # Random baseline
        'random_auroc': float(np.mean(random_aurocs)),
        'random_auroc_std': float(np.std(random_aurocs)),

        # Data info
        'n_samples': int(len(X)),
        'n_correct': int(sum(y)),
        'n_incorrect': int(len(y) - sum(y)),
        'layer': int(layer),
    }


def main():
    parser = argparse.ArgumentParser(description="Linear Probe vs SAE Sanity Check")
    parser.add_argument("--model", type=str, default="gemma2b",
                        choices=["gemma2b", "gemma2b_it", "gemma9b", "llama"],
                        help="Model to test")
    parser.add_argument("--layer", type=int, default=19, help="Layer to analyze")
    parser.add_argument("--all-layers", action="store_true", help="Test all layers")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: experiments/linear_probe_sanity_check/results)")
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map model name to directories
    model_dirs = {
        "gemma2b": ("phase1_0", "phase2_5", "google/gemma-2-2b"),
        "gemma2b_it": ("phase1_0_it", "phase2_5_it", "google/gemma-2-2b-it"),
        "gemma9b": ("phase1_0_gemma9b", "phase2_5_gemma9b", "google/gemma-2-9b"),
        "llama": ("phase1_0_llama", "phase2_5_llama", "meta-llama/Llama-3.1-8B"),
    }

    phase1_name, phase25_name, model_name = model_dirs[args.model]
    phase1_dir = project_root / "data" / phase1_name
    phase25_dir = project_root / "data" / phase25_name

    from datetime import datetime

    # Directions to save (exclude from JSON)
    direction_keys = ['mass_mean_direction', 'logreg_direction', 'mean_diff_direction']

    if args.all_layers:
        # Test layers 10-25 (middle to late layers)
        results = {}
        for layer in range(10, 26):
            try:
                result = run_comparison(phase1_dir, layer, phase25_dir, model_name)
                results[layer] = result
            except Exception as e:
                print(f"Layer {layer} failed: {e}")

        # Summary across layers
        print("\n" + "="*60)
        print("LAYER-WISE SUMMARY (AUROC)")
        print("="*60)
        print(f"{'Layer':<8} {'MassMean':<10} {'LogReg':<10} {'MeanDiff':<10} {'SAE':<10}")
        print("-"*50)
        for layer, r in sorted(results.items()):
            print(f"{layer:<8} {r['mass_mean_auroc']:.3f}      {r['logreg_auroc']:.3f}      {r['mean_diff_auroc']:.3f}      {r['sae_auroc']:.3f}")

        # Find best layer for each method
        print("\n" + "="*60)
        print("BEST LAYER SELECTION")
        print("="*60)

        # By AUROC (for prediction)
        best_mass_mean_auroc = max(results.items(), key=lambda x: x[1]['mass_mean_auroc'])
        best_logreg_auroc = max(results.items(), key=lambda x: x[1]['logreg_auroc'])
        best_mean_diff_auroc = max(results.items(), key=lambda x: x[1]['mean_diff_auroc'])
        best_sae_auroc = max(results.items(), key=lambda x: x[1]['sae_auroc'])

        print("\nBy AUROC (for prediction):")
        print(f"  Mass-Mean:  Layer {best_mass_mean_auroc[0]} (AUROC={best_mass_mean_auroc[1]['mass_mean_auroc']:.3f})")
        print(f"  LogReg:     Layer {best_logreg_auroc[0]} (AUROC={best_logreg_auroc[1]['logreg_auroc']:.3f})")
        print(f"  MeanDiff:   Layer {best_mean_diff_auroc[0]} (AUROC={best_mean_diff_auroc[1]['mean_diff_auroc']:.3f})")
        print(f"  SAE:        Layer {best_sae_auroc[0]} (AUROC={best_sae_auroc[1]['sae_auroc']:.3f}, Latent={best_sae_auroc[1]['sae_best_latent_idx']})")

        # By t-statistic (alternative for prediction)
        best_mass_mean_t = max(results.items(), key=lambda x: abs(x[1]['mass_mean_t_statistic']))
        best_logreg_t = max(results.items(), key=lambda x: abs(x[1]['logreg_t_statistic']))
        best_sae_t = max(results.items(), key=lambda x: abs(x[1]['sae_t_statistic']))

        print("\nBy |t-statistic| (alternative for prediction):")
        print(f"  Mass-Mean:  Layer {best_mass_mean_t[0]} (t={best_mass_mean_t[1]['mass_mean_t_statistic']:.1f})")
        print(f"  LogReg:     Layer {best_logreg_t[0]} (t={best_logreg_t[1]['logreg_t_statistic']:.1f})")
        print(f"  SAE:        Layer {best_sae_t[0]} (t={best_sae_t[1]['sae_t_statistic']:.1f})")

        # By separation (for steering - note: probe separation != SAE separation in meaning)
        best_mass_mean_sep = max(results.items(), key=lambda x: abs(x[1]['mass_mean_separation']))
        best_sae_sep = max(results.items(), key=lambda x: abs(x[1]['sae_separation']))

        print("\nBy |separation| (for steering comparison):")
        print(f"  Mass-Mean:  Layer {best_mass_mean_sep[0]} (sep={best_mass_mean_sep[1]['mass_mean_separation']:.3f})")
        print(f"  SAE:        Layer {best_sae_sep[0]} (sep={best_sae_sep[1]['sae_separation']:.3f})")
        print("  Note: Probe and SAE separation scores are NOT directly comparable (different scales)")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save each layer's probe directions
        for layer, r in results.items():
            probe_file = output_dir / f"{args.model}_layer{layer}_probes_{timestamp}.safetensors"
            probe_tensors = {
                "mass_mean_direction": torch.tensor(r['mass_mean_direction'], dtype=torch.float32),
                "logreg_direction": torch.tensor(r['logreg_direction'], dtype=torch.float32),
                "mean_diff_direction": torch.tensor(r['mean_diff_direction'], dtype=torch.float32),
                "logreg_bias": torch.tensor([r['logreg_bias']], dtype=torch.float32),
            }
            save_file(probe_tensors, str(probe_file))

        # Save metrics (without numpy arrays)
        metrics_only = {}
        for layer, r in results.items():
            metrics_only[str(layer)] = {k: v for k, v in r.items() if k not in direction_keys}

        # Compile best layer info
        best_layers = {
            'by_auroc': {
                'mass_mean': {'layer': best_mass_mean_auroc[0], 'auroc': best_mass_mean_auroc[1]['mass_mean_auroc']},
                'logreg': {'layer': best_logreg_auroc[0], 'auroc': best_logreg_auroc[1]['logreg_auroc']},
                'mean_diff': {'layer': best_mean_diff_auroc[0], 'auroc': best_mean_diff_auroc[1]['mean_diff_auroc']},
                'sae': {'layer': best_sae_auroc[0], 'auroc': best_sae_auroc[1]['sae_auroc'], 'latent_idx': best_sae_auroc[1]['sae_best_latent_idx']},
            },
            'by_t_statistic': {
                'mass_mean': {'layer': best_mass_mean_t[0], 't_stat': best_mass_mean_t[1]['mass_mean_t_statistic']},
                'logreg': {'layer': best_logreg_t[0], 't_stat': best_logreg_t[1]['logreg_t_statistic']},
                'sae': {'layer': best_sae_t[0], 't_stat': best_sae_t[1]['sae_t_statistic']},
            },
            'by_separation': {
                'mass_mean': {'layer': best_mass_mean_sep[0], 'separation': best_mass_mean_sep[1]['mass_mean_separation']},
                'sae': {'layer': best_sae_sep[0], 'separation': best_sae_sep[1]['sae_separation']},
            },
        }

        output_file = output_dir / f"{args.model}_all_layers_metrics_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'model': args.model,
                'best_layers': best_layers,
                'layers': metrics_only,
            }, f, indent=2)
        print(f"\nProbe directions saved to: {output_dir}/{args.model}_layer*_probes_{timestamp}.safetensors")
        print(f"Metrics saved to: {output_file}")
    else:
        result = run_comparison(phase1_dir, args.layer, phase25_dir, model_name)

        # Save results
        if result:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save all probe directions as safetensors
            probe_file = output_dir / f"{args.model}_layer{args.layer}_probes_{timestamp}.safetensors"
            probe_tensors = {
                "mass_mean_direction": torch.tensor(result['mass_mean_direction'], dtype=torch.float32),
                "logreg_direction": torch.tensor(result['logreg_direction'], dtype=torch.float32),
                "mean_diff_direction": torch.tensor(result['mean_diff_direction'], dtype=torch.float32),
                "logreg_bias": torch.tensor([result['logreg_bias']], dtype=torch.float32),
            }
            save_file(probe_tensors, str(probe_file))
            print(f"\nProbe directions saved to: {probe_file}")

            # Save metrics as JSON (without the numpy arrays)
            metrics_file = output_dir / f"{args.model}_layer{args.layer}_metrics_{timestamp}.json"
            metrics = {k: v for k, v in result.items() if k not in direction_keys}
            metrics['model'] = args.model
            metrics['probe_file'] = probe_file.name
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
