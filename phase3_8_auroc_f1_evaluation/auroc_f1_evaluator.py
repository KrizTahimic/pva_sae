"""Phase 3.8: AUROC and F1 Evaluation for SAE-Code-Correctness Latents.

This script evaluates bidirectional SAE latents (correct-predicting and incorrect-predicting)
using AUROC and F1 metrics on the validation split from Phase 3.5 data.
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)

from common.logging import get_logger
from common.config import Config, PLOT_DPI, PLOT_STYLE
from common.utils import detect_device, ensure_directory_exists
from common.phase_discovery import discover_latest_phase_output
from common.viz_utils import handle_viz_only_mode
from common.utils import save_json, load_json
from common.sae_loader import load_sae_for_config
from common.tensor_utils import load_activation

logger = get_logger("phase3_8.auroc_f1_evaluator")

class Phase38Runner:
    """Standard runner for Phase 3.8: AUROC and F1 Evaluation."""

    def __init__(self, config):
        """Initialize with config object."""
        self.config = config
        self.logger = get_logger("phase3_8.runner", phase="3.8")

    def run(self):
        """Run Phase 3.8 AUROC and F1 evaluation."""
        self.logger.info("Starting Phase 3.8: AUROC and F1 Evaluation")
        self.logger.info("This phase evaluates bidirectional SAE latents using AUROC and F1 metrics")
        self.logger.info("\n" + self.config.dump(phase="3.8"))

        # Run the main evaluation logic with our config
        return run_evaluation(self.config)

def run_evaluation(config):
    """Core evaluation logic extracted from main()."""
    np.random.seed(config.evaluation_random_seed)
    torch.manual_seed(config.evaluation_random_seed)

    # Autodiscover Phase 3.5 (uses config for model/dataset-aware path)
    phase3_5_path = discover_latest_phase_output("3.5", config=config)
    if not phase3_5_path:
        raise FileNotFoundError("No Phase 3.5 output found. Please run Phase 3.5 first.")
    phase3_5_dir = Path(phase3_5_path).parent
    logger.info(f"Using Phase 3.5 output: {phase3_5_dir}")

    # Autodiscover Phase 3.6 (no dataset suffix - hyperparameters are model-specific, shared across datasets)
    phase3_6_path = discover_latest_phase_output("3.6", config=config)
    if not phase3_6_path:
        raise FileNotFoundError("No Phase 3.6 output found. Please run Phase 3.6 first.")
    phase3_6_dir = Path(phase3_6_path).parent
    logger.info(f"Using Phase 3.6 output: {phase3_6_dir}")

    # Setup output directory (with dataset suffix if needed)
    from common.phase_discovery import get_phase_output_dir
    output_dir = Path(get_phase_output_dir('3.8', config))
    ensure_directory_exists(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Handle --viz-only mode
    if config.viz_only:
        data_path = output_dir / "auroc_f1_results.json"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Cannot run --viz-only: {data_path} not found. "
                f"Run phase normally first to generate data."
            )
        logger.info(f"--viz-only mode: Loading data from {data_path}")
        results = load_json(data_path)

        # Reconstruct metrics for visualizations
        hp_metrics_correct = results['correct_predicting_latent']['hyperparameter_split']
        hp_metrics_incorrect = results['incorrect_predicting_latent']['hyperparameter_split']

        # Regenerate F1 threshold plot (uses saved f1_curve data)
        plot_combined_f1_thresholds(hp_metrics_correct, hp_metrics_incorrect, output_dir)

        # Regenerate comparative metrics (without ROC curves)
        # Need to restructure results for the plot function
        viz_results = {
            'correct_predicting_latent': {
                'validation_metrics': {
                    'metrics': results['correct_predicting_latent']['validation_split']
                }
            },
            'incorrect_predicting_latent': {
                'validation_metrics': {
                    'metrics': results['incorrect_predicting_latent']['validation_split']
                }
            }
        }
        plot_comparative_metrics(viz_results, output_dir)

        logger.info("Visualization regeneration complete")
        logger.info("Note: Confusion matrices and PR curves require full rerun to regenerate")
        return results

    # Phase 1: Load best latents from Phase 2.10 (t-statistic based selection)
    logger.info("Loading best latents from Phase 2.10...")

    # Auto-discover Phase 2.10 output
    phase2_10_dir = discover_latest_phase_output("2.10")
    if not phase2_10_dir:
        raise FileNotFoundError("No Phase 2.10 output found. Please run Phase 2.10 first.")
    phase2_10_dir = Path(phase2_10_dir).parent

    # Load best latents from Phase 2.10
    top_latents_file = phase2_10_dir / 'top_20_latents.json'
    if not top_latents_file.exists():
        raise FileNotFoundError(f"top_20_latents.json not found in {phase2_10_dir}. Please run Phase 2.10 first.")

    top_latents = load_json(top_latents_file)

    # Validate structure
    if 'correct' not in top_latents or 'incorrect' not in top_latents:
        raise ValueError("Missing 'correct' or 'incorrect' in top_20_latents.json")

    if not top_latents['correct'] or not top_latents['incorrect']:
        raise ValueError("Empty latent list in top_20_latents.json")

    # Get the best (index 0) latents
    best_correct = top_latents['correct'][0]
    best_incorrect = top_latents['incorrect'][0]

    correct_layer = best_correct['layer']
    correct_latent_idx = best_correct['latent_idx']
    incorrect_layer = best_incorrect['layer']
    incorrect_latent_idx = best_incorrect['latent_idx']

    logger.info(f"Best correct-predicting latent: idx {correct_latent_idx} at layer {correct_layer}")
    logger.info(f"Best incorrect-predicting latent: idx {incorrect_latent_idx} at layer {incorrect_layer}")

    # Phase 2: Evaluate Correct-Predicting Feature
    logger.info("\n" + "="*60)
    logger.info("EVALUATING CORRECT-PREDICTING FEATURE")
    logger.info("="*60)

    # Load tuning split for correct latent
    y_true_hp_correct, scores_hp_correct = load_split_activations(
        'tuning', correct_layer, correct_latent_idx, 'correct',
        phase3_5_dir, phase3_6_dir, config
    )

    logger.info(f"Correct-predicting feature (tuning split):")
    logger.info(f"  Total samples: {len(y_true_hp_correct)}")
    logger.info(f"  Positive class (correct code): {sum(y_true_hp_correct == 1)}")
    logger.info(f"  Negative class (incorrect code): {sum(y_true_hp_correct == 0)}")

    # Find optimal threshold
    optimal_threshold_correct, hp_metrics_correct = find_optimal_threshold(
        y_true_hp_correct,
        scores_hp_correct,
        'correct',
        output_dir
    )

    # Load analysis split
    y_true_val_correct, scores_val_correct = load_split_activations(
        'analysis', correct_layer, correct_latent_idx, 'correct',
        phase3_5_dir, phase3_6_dir, config
    )

    logger.info(f"\nCorrect-predicting feature (analysis split):")
    logger.info(f"  Total samples: {len(y_true_val_correct)}")
    logger.info(f"  Positive class (correct code): {sum(y_true_val_correct == 1)}")
    logger.info(f"  Negative class (incorrect code): {sum(y_true_val_correct == 0)}")

    # Evaluate on validation using hyperparameter threshold
    val_metrics_correct = calculate_metrics(
        y_true_val_correct, scores_val_correct,
        optimal_threshold_correct, 'correct_validation', output_dir
    )

    # Phase 3: Evaluate Incorrect-Predicting Feature
    logger.info("\n" + "="*60)
    logger.info("EVALUATING INCORRECT-PREDICTING FEATURE")
    logger.info("="*60)

    # Load tuning split for incorrect latent
    y_true_hp_incorrect, scores_hp_incorrect = load_split_activations(
        'tuning', incorrect_layer, incorrect_latent_idx, 'incorrect',
        phase3_5_dir, phase3_6_dir, config
    )

    logger.info(f"Incorrect-predicting feature (tuning split):")
    logger.info(f"  Total samples: {len(y_true_hp_incorrect)}")
    logger.info(f"  Positive class (incorrect code): {sum(y_true_hp_incorrect == 1)}")
    logger.info(f"  Negative class (correct code): {sum(y_true_hp_incorrect == 0)}")

    # Find optimal threshold (for incorrect-predicting, high activation = incorrect)
    optimal_threshold_incorrect, hp_metrics_incorrect = find_optimal_threshold(
        y_true_hp_incorrect,
        scores_hp_incorrect,
        'incorrect',
        output_dir
    )

    # Load analysis split
    y_true_val_incorrect, scores_val_incorrect = load_split_activations(
        'analysis', incorrect_layer, incorrect_latent_idx, 'incorrect',
        phase3_5_dir, phase3_6_dir, config
    )

    logger.info(f"\nIncorrect-predicting feature (analysis split):")
    logger.info(f"  Total samples: {len(y_true_val_incorrect)}")
    logger.info(f"  Positive class (incorrect code): {sum(y_true_val_incorrect == 1)}")
    logger.info(f"  Negative class (correct code): {sum(y_true_val_incorrect == 0)}")

    # Evaluate on validation using hyperparameter threshold
    val_metrics_incorrect = calculate_metrics(
        y_true_val_incorrect, scores_val_incorrect,
        optimal_threshold_incorrect, 'incorrect_validation', output_dir
    )

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'correct_predicting_latent': {
            'layer': correct_layer,
            'latent_idx': correct_latent_idx,
            'hyperparameter_split': hp_metrics_correct,
            'validation_split': val_metrics_correct
        },
        'incorrect_predicting_latent': {
            'layer': incorrect_layer,
            'latent_idx': incorrect_latent_idx,
            'hyperparameter_split': hp_metrics_incorrect,
            'validation_split': val_metrics_incorrect
        },
        'source_files': {
            'phase3_5_dir': str(phase3_5_dir),
            'phase3_6_dir': str(phase3_6_dir),
            'phase2_10_dir': str(phase2_10_dir)
        }
    }

    results_path = output_dir / 'auroc_f1_results.json'
    save_json(results, results_path)
    logger.info(f"\nResults saved to: {results_path}")

    # Print final summary
    def log_feature_summary(name: str, metrics: dict):
        logger.info(f"{name}:")
        logger.info(f"  AUROC: {metrics['auroc']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")

    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    log_feature_summary("Correct-predicting feature (validation)", val_metrics_correct)
    log_feature_summary("Incorrect-predicting feature (validation)", val_metrics_incorrect)

    # Write phase_output.json manifest
    from common.phase_discovery import write_phase_output

    write_phase_output(
        phase="3.8",
        outputs={
            "primary": "auroc_f1_results.json",
            "f1_plot": "f1_threshold_plot_combined.png",
            "confusion_correct": "confusion_matrix_correct.png",
            "confusion_incorrect": "confusion_matrix_incorrect.png",
            "comparative_metrics": "comparative_metrics.png",
        },
        config=config,
        output_dir=str(output_dir),
        dependencies={
            "3.5": str(phase3_5_dir),
            "3.6": str(phase3_6_dir),
            "2.10": str(phase2_10_dir),
        },
        config_keys=['model_name', 'dataset_name', 'evaluation_random_seed']
    )
    logger.info(f"Saved phase_output.json manifest to {output_dir}")

    return results

def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    latent_type: str,
    output_dir: Path
) -> dict[str, float]:
    """Calculate metrics for either correct or incorrect predicting latents.

    Args:
        y_true: Ground truth labels
        scores: Latent activation scores
        threshold: Binary classification threshold
        latent_type: 'correct' or 'incorrect'
        output_dir: Directory to save plots

    Returns:
        Dictionary of metrics including AUROC, F1, precision, recall
    """
    # Calculate AUROC - threshold independent
    auroc = roc_auc_score(y_true, scores)

    # Apply threshold for binary predictions
    y_pred = (scores > threshold).astype(int)

    # Calculate threshold-dependent metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"\nMetrics for {latent_type}-predicting latent:")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUROC: {auroc:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, latent_type, output_dir)
    
    return {
        'auroc': float(auroc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'threshold': float(threshold)
    }

def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    latent_type: str,
    output_dir: Path
) -> tuple[float, dict[str, float]]:
    """Find optimal threshold for a specific latent type.

    Args:
        y_true: Ground truth labels
        scores: Latent activation scores
        latent_type: 'correct' or 'incorrect'
        output_dir: Directory to save plots

    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    # Grid search for F1-Optimal Threshold
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    f1_scores = [
        f1_score(y_true, (scores >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]

    # Find threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds[optimal_idx]
    max_f1_score = f1_scores[optimal_idx]

    # Store threshold data for later combined plotting
    # Individual plots will be created after both latents are processed

    # Evaluate at optimal threshold
    logger.info(f'\nF1 optimal for {latent_type}-predicting latent:')
    metrics = calculate_metrics(y_true, scores, optimal_f1_threshold, latent_type, output_dir)

    # Return threshold data for combined plotting
    metrics['threshold_range'] = (float(scores.min()), float(scores.max()))
    metrics['f1_curve'] = {'thresholds': thresholds.tolist(), 'f1_scores': f1_scores}
    
    return optimal_f1_threshold, metrics

def plot_combined_f1_thresholds(
    correct_metrics: dict,
    incorrect_metrics: dict,
    output_dir: Path
) -> None:
    """Create combined F1 threshold plot for both features (Phase 3.11 style).

    Args:
        correct_metrics: Metrics dict for correct-predicting feature with f1_curve data
        incorrect_metrics: Metrics dict for incorrect-predicting feature with f1_curve data
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))

    # Extract data
    correct_thresholds = np.array(correct_metrics['f1_curve']['thresholds'])
    correct_f1s = np.array(correct_metrics['f1_curve']['f1_scores'])
    incorrect_thresholds = np.array(incorrect_metrics['f1_curve']['thresholds'])
    incorrect_f1s = np.array(incorrect_metrics['f1_curve']['f1_scores'])

    # Plot both curves
    plt.plot(correct_thresholds, correct_f1s, 'g-', linewidth=2, label='Correct-predicting')
    plt.plot(incorrect_thresholds, incorrect_f1s, 'r-', linewidth=2, label='Incorrect-predicting')

    # Mark optimal points
    correct_optimal_threshold = correct_metrics['threshold']
    correct_optimal_f1 = correct_metrics['f1']
    incorrect_optimal_threshold = incorrect_metrics['threshold']
    incorrect_optimal_f1 = incorrect_metrics['f1']

    plt.plot(correct_optimal_threshold, correct_optimal_f1, 'go', markersize=10,
             label=f'Correct optimal: {correct_optimal_f1:.3f}')
    plt.plot(incorrect_optimal_threshold, incorrect_optimal_f1, 'rs', markersize=10,
             label=f'Incorrect optimal: {incorrect_optimal_f1:.3f}')

    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_dir / 'f1_threshold_plot_combined.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved combined F1 threshold plot to {output_dir / 'f1_threshold_plot_combined.png'}")

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latent_type: str,
    output_dir: Path
) -> None:
    """Plot confusion matrix with appropriate labels for latent type.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        latent_type: 'correct' or 'incorrect'
        output_dir: Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    # Adjust labels based on what we're predicting
    if latent_type == 'correct':
        # Predicting correctness
        labels = ['Incorrect', 'Correct']
    else:
        # Predicting incorrectness
        labels = ['Correct', 'Incorrect']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {latent_type.capitalize()}-Predicting Latent')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save plot
    plt.savefig(output_dir / f'confusion_matrix_{latent_type}.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

def plot_comparative_metrics(
    results: dict, 
    output_dir: Path,
    y_true_val_correct: Optional[np.ndarray] = None,
    scores_val_correct: Optional[np.ndarray] = None,
    y_true_val_incorrect: Optional[np.ndarray] = None,
    scores_val_incorrect: Optional[np.ndarray] = None
) -> None:
    """Create side-by-side comparison of both feature performances.
    
    Args:
        results: Dictionary containing metrics for both features
        output_dir: Directory to save plot
        y_true_val_correct: True labels for correct feature validation
        scores_val_correct: Scores for correct feature validation
        y_true_val_incorrect: True labels for incorrect feature validation
        scores_val_incorrect: Scores for incorrect feature validation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract metrics
    metrics = ['AUROC', 'F1', 'Precision', 'Recall']
    correct_vals = [
        results['correct_predicting_latent']['validation_metrics']['metrics'][m.lower()]
        for m in metrics
    ]
    incorrect_vals = [
        results['incorrect_predicting_latent']['validation_metrics']['metrics'][m.lower()]
        for m in metrics
    ]

    # Plot bars
    bar_positions = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(bar_positions - width/2, correct_vals, width, label='Correct-predicting', color='green')
    bars2 = ax1.bar(bar_positions + width/2, incorrect_vals, width, label='Incorrect-predicting', color='red')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Feature Performance Comparison')
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot ROC curves if we have the data
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.5)')
    
    # Plot ROC curve for correct-predicting latent if data provided
    if y_true_val_correct is not None and scores_val_correct is not None:
        fpr_correct, tpr_correct, _ = roc_curve(y_true_val_correct, scores_val_correct)
        auc_correct = results['correct_predicting_latent']['validation_metrics']['metrics']['auroc']
        ax2.plot(fpr_correct, tpr_correct, color='green', linewidth=2,
                label=f'Correct-predicting (AUC = {auc_correct:.3f})')

    # Plot ROC curve for incorrect-predicting latent if data provided
    if y_true_val_incorrect is not None and scores_val_incorrect is not None:
        fpr_incorrect, tpr_incorrect, _ = roc_curve(y_true_val_incorrect, scores_val_incorrect)
        auc_incorrect = results['incorrect_predicting_latent']['validation_metrics']['metrics']['auroc']
        ax2.plot(fpr_incorrect, tpr_incorrect, color='red', linewidth=2,
                label=f'Incorrect-predicting (AUC = {auc_incorrect:.3f})')
    
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_metrics.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves(
    output_dir: Path,
    y_true_val_correct: np.ndarray,
    scores_val_correct: np.ndarray,
    y_true_val_incorrect: np.ndarray,
    scores_val_incorrect: np.ndarray
) -> None:
    """Create standalone precision-recall curves figure for both features.

    This generates a single, self-contained plot suitable for inclusion in papers.

    Args:
        output_dir: Directory to save plot
        y_true_val_correct: True labels for correct-predicting feature
        scores_val_correct: Prediction scores for correct-predicting feature
        y_true_val_incorrect: True labels for incorrect-predicting feature
        scores_val_incorrect: Prediction scores for incorrect-predicting feature
    """
    plt.figure(figsize=(8, 6))

    # Compute and plot PR curve for correct-predicting feature
    precision_correct, recall_correct, _ = precision_recall_curve(
        y_true_val_correct, scores_val_correct
    )
    ap_correct = auc(recall_correct, precision_correct)
    plt.plot(recall_correct, precision_correct, 'g-', linewidth=2,
            label=f'Correct-predicting (AP = {ap_correct:.3f})')

    # Compute and plot PR curve for incorrect-predicting feature
    precision_incorrect, recall_incorrect, _ = precision_recall_curve(
        y_true_val_incorrect, scores_val_incorrect
    )
    ap_incorrect = auc(recall_incorrect, precision_incorrect)
    plt.plot(recall_incorrect, precision_incorrect, 'r-', linewidth=2,
            label=f'Incorrect-predicting (AP = {ap_incorrect:.3f})')

    # Formatting
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.tight_layout()

    # Save as standalone PNG
    output_path = output_dir / 'precision_recall_curves.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved precision-recall curves to {output_path}")

def load_split_activations(
    split_name: str,
    layer_num: int,
    latent_idx: int,
    latent_type: str,
    phase3_5_dir: Path,
    phase3_6_dir: Path,
    config: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Load activations for a specific latent from appropriate phase data.

    Args:
        split_name: 'tuning' or 'analysis'
        layer_num: Layer number for the latent
        latent_idx: Index of the specific latent
        latent_type: 'correct' or 'incorrect'
        phase3_5_dir: Directory containing Phase 3.5 outputs (analysis split)
        phase3_6_dir: Directory containing Phase 3.6 outputs (tuning split)

    Returns:
        Tuple of (labels, activations)
    """
    # Select the correct directory based on split type
    if split_name == 'tuning':
        # Use Phase 3.6 directory for tuning split
        activation_dir = phase3_6_dir
        # Load temperature 0.0 dataset from Phase 3.6
        temp_data = pd.read_parquet(phase3_6_dir / 'dataset_hyperparams_temp_0_0.parquet')
    else:  # analysis
        # Use Phase 3.5 directory for analysis split
        activation_dir = phase3_5_dir
        # Load temperature 0.0 dataset from Phase 3.5
        temp_data = pd.read_parquet(phase3_5_dir / 'dataset_temp_0_0.parquet')

    # Detect device and load SAE for encoding
    device = detect_device()
    sae = load_sae_for_config(config, layer_num, device)
    logger.info(f"Loaded SAE for layer {layer_num} with 16,384 latents on {device}")

    activations = []
    labels = []
    missing_tasks = []

    # Iterate directly over temp_data (no Phase 0.1 needed)
    for _, row in temp_data.iterrows():
        task_id = row['task_id']
        baseline_passed = row['baseline_passed']

        # Load raw activations from appropriate phase (preserves bfloat16)
        act_file = activation_dir / f'activations/task_activations/{task_id}_layer_{layer_num}.safetensors'

        if not act_file.exists():
            missing_tasks.append(task_id)
            continue

        # Load activation (preserves bfloat16)
        raw_activation = load_activation(act_file, device)

        # Ensure dtype matches SAE parameters for matrix multiplication
        raw_activation = raw_activation.to(sae.W_enc.dtype)

        # Encode through SAE to get latent activations
        # Shape: (1, 16384) - SAE latent activations
        with torch.no_grad():
            latent_activations = sae.encode(raw_activation)

        # Extract specific latent value
        latent_activation = latent_activations[0, latent_idx].item()
        activations.append(latent_activation)

        # Create label based on what we're predicting
        if latent_type == 'correct':
            # Predicting correctness: 1=correct, 0=incorrect
            label = 1 if baseline_passed else 0
        else:
            # Predicting incorrectness: 1=incorrect, 0=correct
            label = 1 if not baseline_passed else 0

        labels.append(label)

    if missing_tasks:
        logger.warning(f"Missing activation files for {len(missing_tasks)} tasks: {missing_tasks[:5]}...")

    logger.info(f"Loaded {len(labels)} samples for {split_name} split")
    logger.info(f"Class distribution: {np.bincount(labels)}")

    # Clean up SAE to free memory
    del sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(labels), np.array(activations)

def main():
    """Legacy entry point for running directly. Uses Phase38Runner."""
    from common.config import Config
    config = Config()
    runner = Phase38Runner(config)
    runner.run()

if __name__ == "__main__":
    main()