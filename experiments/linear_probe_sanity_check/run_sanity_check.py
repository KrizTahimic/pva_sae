"""
Linear Probe vs SAE Sanity Check

Compares:
1. Linear probe (logistic regression on raw activations)
2. SAE latent (single latent from Phase 2.5)
3. Random direction (sanity baseline)

Usage:
    python run_sanity_check.py --model gemma2b --layer 19
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
from sklearn.model_selection import cross_val_score
from scipy import stats
from safetensors.torch import load_file, save_file
import json


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
    from common.config import ExperimentConfig

    config = ExperimentConfig()
    config.model_name = model_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = load_sae_for_config(config, layer, device)

    X_tensor = torch.tensor(X, dtype=sae.W_enc.dtype, device=device)

    with torch.no_grad():
        latent_activations = sae.encode(X_tensor)  # [N, 16384]

    return latent_activations.cpu().numpy(), sae


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

    # 2. Train linear probe with cross-validation
    print("\n[1] LINEAR PROBE (Logistic Regression)")
    print("-" * 40)

    probe = LogisticRegression(max_iter=1000, random_state=42)

    # Cross-validation AUROC
    cv_scores = cross_val_score(probe, X, y, cv=5, scoring='roc_auc')
    print(f"  5-fold CV AUROC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Fit on all data for direction extraction
    probe.fit(X, y)
    probe_scores = probe.predict_proba(X)[:, 1]
    probe_auroc = roc_auc_score(y, probe_scores)
    probe_preds = (probe_scores > 0.5).astype(int)
    probe_f1 = f1_score(y, probe_preds)
    probe_acc = accuracy_score(y, probe_preds)

    print(f"  Full data AUROC: {probe_auroc:.3f}")
    print(f"  Full data F1:    {probe_f1:.3f}")
    print(f"  Full data Acc:   {probe_acc:.3f}")

    # Extract the learned direction
    probe_direction = probe.coef_[0]
    probe_bias = probe.intercept_[0]
    print(f"  Probe direction norm: {np.linalg.norm(probe_direction):.3f}")
    print(f"  Probe bias: {probe_bias:.3f}")

    # Compute separation score and t-statistic for probe
    # (same metrics used for SAE latents in Phase 2.5 and 2.10)
    correct_scores = probe_scores[y == 1]
    incorrect_scores = probe_scores[y == 0]

    # Separation score: mean(correct) - mean(incorrect)
    probe_separation_score = np.mean(correct_scores) - np.mean(incorrect_scores)

    # T-statistic: Welch's t-test
    t_stat, p_value = stats.ttest_ind(correct_scores, incorrect_scores, equal_var=False)
    probe_t_statistic = t_stat

    print(f"  Separation score: {probe_separation_score:.3f}")
    print(f"  T-statistic: {probe_t_statistic:.3f} (p={p_value:.2e})")

    # 3. SAE-based prediction
    print("\n[2] SAE LATENT PREDICTION")
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
                selected_idx = best_latent_info.get('latent_idx', best_latent_info.get('feature_idx'))
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

    # 4. Random direction baseline
    print("\n[3] RANDOM DIRECTION (Sanity Check)")
    print("-" * 40)

    np.random.seed(42)
    random_directions = [np.random.randn(X.shape[1]) for _ in range(10)]
    random_aurocs = []

    for i, rand_dir in enumerate(random_directions):
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        rand_scores = X @ rand_dir
        try:
            auroc = roc_auc_score(y, rand_scores)
            # Take max of positive and negative direction
            auroc = max(auroc, 1 - auroc)
            random_aurocs.append(auroc)
        except:
            pass

    print(f"  Random direction AUROC: {np.mean(random_aurocs):.3f} (+/- {np.std(random_aurocs):.3f})")

    # 5. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Linear Probe AUROC:  {probe_auroc:.3f}")
    print(f"  Best SAE Latent:     {best_auroc:.3f}")
    print(f"  Random Baseline:     {np.mean(random_aurocs):.3f}")
    print()

    if probe_auroc > best_auroc + 0.05:
        print("  >> Linear probe significantly outperforms SAE latent")
        print("     This suggests SAE may be losing information")
    elif best_auroc > probe_auroc + 0.05:
        print("  >> SAE latent outperforms linear probe")
        print("     This suggests SAE captures meaningful structure")
    else:
        print("  >> Performance is comparable")
        print("     SAE provides interpretability without losing accuracy")

    return {
        # Probe metrics
        'probe_auroc': float(probe_auroc),
        'probe_f1': float(probe_f1),
        'probe_acc': float(probe_acc),
        'probe_separation_score': float(probe_separation_score),
        'probe_t_statistic': float(probe_t_statistic),
        'probe_p_value': float(p_value),
        'probe_direction': probe_direction,  # numpy array [d_model] - excluded from JSON
        'probe_bias': float(probe_bias),

        # SAE metrics
        'sae_auroc': float(best_auroc),
        'sae_f1': float(best_f1),
        'sae_separation_score': float(sae_separation_score),
        'sae_t_statistic': float(sae_t_statistic),
        'sae_p_value': float(sae_p_value),
        'sae_best_latent_idx': int(best_latent_idx),

        # Baseline
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
        print("LAYER-WISE SUMMARY")
        print("="*60)
        print(f"{'Layer':<8} {'Probe':<10} {'SAE':<10} {'Winner':<10}")
        print("-"*40)
        for layer, r in sorted(results.items()):
            winner = "Probe" if r['probe_auroc'] > r['sae_auroc'] + 0.02 else \
                     "SAE" if r['sae_auroc'] > r['probe_auroc'] + 0.02 else "Tie"
            print(f"{layer:<8} {r['probe_auroc']:.3f}      {r['sae_auroc']:.3f}      {winner}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save each layer's probe weights
        for layer, r in results.items():
            probe_file = output_dir / f"{args.model}_layer{layer}_probe_{timestamp}.safetensors"
            probe_tensors = {
                "direction": torch.tensor(r['probe_direction'], dtype=torch.float32),
                "bias": torch.tensor([r['probe_bias']], dtype=torch.float32),
            }
            save_file(probe_tensors, str(probe_file))

        # Save metrics (without numpy arrays)
        metrics_only = {}
        for layer, r in results.items():
            metrics_only[str(layer)] = {k: v for k, v in r.items() if k not in ['probe_direction']}

        output_file = output_dir / f"{args.model}_all_layers_metrics_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({'model': args.model, 'layers': metrics_only}, f, indent=2)
        print(f"\nProbe weights saved to: {output_dir}/{args.model}_layer*_probe_{timestamp}.safetensors")
        print(f"Metrics saved to: {output_file}")
    else:
        result = run_comparison(phase1_dir, args.layer, phase25_dir, model_name)

        # Save results
        if result:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save probe weights as safetensors
            probe_file = output_dir / f"{args.model}_layer{args.layer}_probe_{timestamp}.safetensors"
            probe_tensors = {
                "direction": torch.tensor(result['probe_direction'], dtype=torch.float32),
                "bias": torch.tensor([result['probe_bias']], dtype=torch.float32),
            }
            save_file(probe_tensors, str(probe_file))
            print(f"\nProbe weights saved to: {probe_file}")

            # Save metrics as JSON (without the numpy array)
            metrics_file = output_dir / f"{args.model}_layer{args.layer}_metrics_{timestamp}.json"
            metrics = {k: v for k, v in result.items() if k not in ['probe_direction']}
            metrics['model'] = args.model
            metrics['probe_file'] = probe_file.name
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
