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
from common.config import Config, PLOT_DPI, PLOT_STYLE, COLOR_CORRECT_PREDICTING, COLOR_INCORRECT_PREDICTING
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
    direction_source = getattr(config, 'direction_source', 'sae')
    output_dir = Path(get_phase_output_dir('3.8', config))

    # Add "_probe" suffix if using probe directions
    if direction_source == 'probe_logreg':
        output_dir = output_dir.parent / (output_dir.name + "_probe")
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

        # Regenerate candidate comparison plot if candidate data exists
        if results.get('candidate_evaluation') is not None:
            plot_candidate_comparison(results['candidate_evaluation'], output_dir)

        logger.info("Visualization regeneration complete")
        logger.info("Note: Confusion matrices and PR curves require full rerun to regenerate")
        return results

    # Determine direction source (SAE or probe)
    direction_source = getattr(config, 'direction_source', 'sae')
    use_probe = direction_source == 'probe_logreg'

    if use_probe:
        # === PROBE BASELINE MODE ===
        logger.info("=" * 60)
        logger.info("PROBE BASELINE MODE: Using LogReg probe from Phase 2.6")
        logger.info("=" * 60)

        # Load probe directions using shared utility
        from common.steering_setup import load_probe_directions_for_predicting
        probe = load_probe_directions_for_predicting(config, detect_device(), method="logreg")

        probe_layer = probe.layer
        probe_bias = probe.bias
        probe_direction = probe.correct_direction

        device = detect_device()
        probe_direction = probe_direction.to(device)

        logger.info(f"LogReg probe: layer {probe_layer}, bias {probe_bias:.4f}")

        # Set layer for logging (probe uses same layer for both)
        correct_layer = probe_layer
        incorrect_layer = probe_layer
        phase2_10_dir = Path(probe.phase_dir)  # For dependency tracking

    else:
        # === SAE MODE (default) — Top-N candidate evaluation ===
        from common.phase_discovery import discover_top_n_latents
        top_n = discover_top_n_latents(config, logger)

        phase2_10_dir = discover_latest_phase_output("2.10", config=config)
        if not phase2_10_dir:
            raise FileNotFoundError("No Phase 2.10 output found. Please run Phase 2.10 first.")
        phase2_10_dir = Path(phase2_10_dir).parent

    # Evaluation: separate paths for probe vs SAE top-N
    all_correct_candidates = None
    all_incorrect_candidates = None

    if use_probe:
        # === PROBE EVALUATION (single direction per category) ===
        logger.info("\n" + "="*60)
        logger.info("EVALUATING CORRECT-PREDICTING PROBE")
        logger.info("="*60)

        y_true_hp_correct, scores_hp_correct = load_split_probe_activations(
            'tuning', probe_layer, probe_direction, probe_bias, 'correct',
            phase3_5_dir, phase3_6_dir, config
        )

        logger.info(f"Correct-predicting probe (tuning split):")
        logger.info(f"  Total samples: {len(y_true_hp_correct)}")
        logger.info(f"  Positive class (correct code): {sum(y_true_hp_correct == 1)}")
        logger.info(f"  Negative class (incorrect code): {sum(y_true_hp_correct == 0)}")

        optimal_threshold_correct, hp_metrics_correct = find_optimal_threshold(
            y_true_hp_correct, scores_hp_correct, 'correct', output_dir
        )

        y_true_val_correct, scores_val_correct = load_split_probe_activations(
            'analysis', probe_layer, probe_direction, probe_bias, 'correct',
            phase3_5_dir, phase3_6_dir, config
        )

        logger.info(f"\nCorrect-predicting probe (analysis split):")
        logger.info(f"  Total samples: {len(y_true_val_correct)}")
        logger.info(f"  Positive class (correct code): {sum(y_true_val_correct == 1)}")
        logger.info(f"  Negative class (incorrect code): {sum(y_true_val_correct == 0)}")

        val_metrics_correct = calculate_metrics(
            y_true_val_correct, scores_val_correct,
            optimal_threshold_correct, 'correct_validation', output_dir
        )

        logger.info("\n" + "="*60)
        logger.info("EVALUATING INCORRECT-PREDICTING PROBE")
        logger.info("="*60)

        y_true_hp_incorrect, scores_hp_incorrect = load_split_probe_activations(
            'tuning', probe_layer, -probe_direction, -probe_bias, 'incorrect',
            phase3_5_dir, phase3_6_dir, config
        )

        logger.info(f"Incorrect-predicting probe (tuning split):")
        logger.info(f"  Total samples: {len(y_true_hp_incorrect)}")
        logger.info(f"  Positive class (incorrect code): {sum(y_true_hp_incorrect == 1)}")
        logger.info(f"  Negative class (correct code): {sum(y_true_hp_incorrect == 0)}")

        optimal_threshold_incorrect, hp_metrics_incorrect = find_optimal_threshold(
            y_true_hp_incorrect, scores_hp_incorrect, 'incorrect', output_dir
        )

        y_true_val_incorrect, scores_val_incorrect = load_split_probe_activations(
            'analysis', probe_layer, -probe_direction, -probe_bias, 'incorrect',
            phase3_5_dir, phase3_6_dir, config
        )

        logger.info(f"\nIncorrect-predicting probe (analysis split):")
        logger.info(f"  Total samples: {len(y_true_val_incorrect)}")
        logger.info(f"  Positive class (incorrect code): {sum(y_true_val_incorrect == 1)}")
        logger.info(f"  Negative class (correct code): {sum(y_true_val_incorrect == 0)}")

        val_metrics_incorrect = calculate_metrics(
            y_true_val_incorrect, scores_val_incorrect,
            optimal_threshold_incorrect, 'incorrect_validation', output_dir
        )

    else:
        # === SAE TOP-N CANDIDATE EVALUATION ===
        logger.info("\n" + "="*60)
        logger.info("EVALUATING CORRECT-PREDICTING CANDIDATES")
        logger.info("="*60)

        best_correct_result, all_correct_candidates = evaluate_candidates_and_select_best(
            top_n['correct'], 'correct', phase3_5_dir, phase3_6_dir, config
        )

        correct_layer = best_correct_result['layer']
        correct_latent_idx = best_correct_result['latent_idx']
        hp_metrics_correct = best_correct_result['hyperparameter_split']
        val_metrics_correct = best_correct_result['validation_split']
        y_true_val_correct = best_correct_result['_y_true_val']
        scores_val_correct = best_correct_result['_scores_val']

        # Generate confusion matrices for best correct candidate
        plot_confusion_matrix(
            best_correct_result['_y_true_hp'],
            (best_correct_result['_scores_hp'] >= hp_metrics_correct['threshold']).astype(int),
            'correct', output_dir
        )
        plot_confusion_matrix(
            y_true_val_correct,
            (scores_val_correct >= val_metrics_correct['threshold']).astype(int),
            'correct_validation', output_dir
        )

        logger.info(f"\nSelected best correct: L{correct_layer}-{correct_latent_idx} "
                     f"(val AUROC={val_metrics_correct['auroc']:.4f})")

        logger.info("\n" + "="*60)
        logger.info("EVALUATING INCORRECT-PREDICTING CANDIDATES")
        logger.info("="*60)

        best_incorrect_result, all_incorrect_candidates = evaluate_candidates_and_select_best(
            top_n['incorrect'], 'incorrect', phase3_5_dir, phase3_6_dir, config
        )

        incorrect_layer = best_incorrect_result['layer']
        incorrect_latent_idx = best_incorrect_result['latent_idx']
        hp_metrics_incorrect = best_incorrect_result['hyperparameter_split']
        val_metrics_incorrect = best_incorrect_result['validation_split']
        y_true_val_incorrect = best_incorrect_result['_y_true_val']
        scores_val_incorrect = best_incorrect_result['_scores_val']

        # Generate confusion matrices for best incorrect candidate
        plot_confusion_matrix(
            best_incorrect_result['_y_true_hp'],
            (best_incorrect_result['_scores_hp'] >= hp_metrics_incorrect['threshold']).astype(int),
            'incorrect', output_dir
        )
        plot_confusion_matrix(
            y_true_val_incorrect,
            (scores_val_incorrect >= val_metrics_incorrect['threshold']).astype(int),
            'incorrect_validation', output_dir
        )

        logger.info(f"\nSelected best incorrect: L{incorrect_layer}-{incorrect_latent_idx} "
                     f"(val AUROC={val_metrics_incorrect['auroc']:.4f})")

    # Generate combined F1 threshold plot
    plot_combined_f1_thresholds(hp_metrics_correct, hp_metrics_incorrect, output_dir)

    # Generate comparative metrics plot
    viz_results = {
        'correct_predicting_latent': {
            'validation_metrics': {
                'metrics': val_metrics_correct
            }
        },
        'incorrect_predicting_latent': {
            'validation_metrics': {
                'metrics': val_metrics_incorrect
            }
        }
    }
    plot_comparative_metrics(
        viz_results, output_dir,
        y_true_val_correct, scores_val_correct,
        y_true_val_incorrect, scores_val_incorrect
    )

    # Save results
    # (Build results dict first so candidate_evaluation is available for plotting)
    # For probe mode, latent_idx is None (not applicable)
    correct_latent_idx = None if use_probe else correct_latent_idx
    incorrect_latent_idx = None if use_probe else incorrect_latent_idx

    results = {
        'timestamp': datetime.now().isoformat(),
        'direction_source': direction_source,
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
            'phase2_10_dir' if not use_probe else 'phase2_6_dir': str(phase2_10_dir)
        },
        'candidate_evaluation': {
            'correct': all_correct_candidates,
            'incorrect': all_incorrect_candidates,
            'selection_criterion': 'validation_auroc',
            'n_candidates': getattr(config, 'phase3_8_n_candidates', 5)
        } if all_correct_candidates is not None else None
    }
    if use_probe:
        results['probe_info'] = {
            'method': 'logreg',
            'layer': probe_layer,
            'bias': probe_bias,
        }

    # Plot candidate comparison chart (SAE mode only)
    if results.get('candidate_evaluation') is not None:
        plot_candidate_comparison(results['candidate_evaluation'], output_dir)

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
            "candidate_comparison": "candidate_comparison.png",
        },
        config=config,
        output_dir=str(output_dir),
        dependencies={
            "3.5": str(phase3_5_dir),
            "3.6": str(phase3_6_dir),
            "2.6" if use_probe else "2.10": str(phase2_10_dir),
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
    y_pred = (scores >= threshold).astype(int)

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
    thresholds = np.linspace(scores.min(), scores.max(), 102)[1:-1]
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

    bars1 = ax1.bar(bar_positions - width/2, correct_vals, width, label='Correct-predicting', color=COLOR_CORRECT_PREDICTING)
    bars2 = ax1.bar(bar_positions + width/2, incorrect_vals, width, label='Incorrect-predicting', color=COLOR_INCORRECT_PREDICTING)

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
        ax2.plot(fpr_correct, tpr_correct, color=COLOR_CORRECT_PREDICTING, linewidth=2,
                label=f'Correct-predicting (AUC = {auc_correct:.3f})')

    # Plot ROC curve for incorrect-predicting latent if data provided
    if y_true_val_incorrect is not None and scores_val_incorrect is not None:
        fpr_incorrect, tpr_incorrect, _ = roc_curve(y_true_val_incorrect, scores_val_incorrect)
        auc_incorrect = results['incorrect_predicting_latent']['validation_metrics']['metrics']['auroc']
        ax2.plot(fpr_incorrect, tpr_incorrect, color=COLOR_INCORRECT_PREDICTING, linewidth=2,
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


def plot_candidate_comparison(
    candidate_evaluation: dict,
    output_dir: Path
) -> None:
    """Create grouped bar chart comparing all evaluated candidates (AUROC + F1).

    Produces a 1x2 figure: left panel for correct-predicting candidates,
    right panel for incorrect-predicting candidates. The selected (best)
    candidate is highlighted with a bold edge.

    Args:
        candidate_evaluation: The 'candidate_evaluation' dict from results JSON,
            containing 'correct' and 'incorrect' lists with per-candidate metrics.
        output_dir: Directory to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = sns.color_palette()
    color_auroc = palette[0]
    color_f1 = palette[1]

    categories = [
        ('correct', 'Correct-Predicting Candidates', axes[0]),
        ('incorrect', 'Incorrect-Predicting Candidates', axes[1]),
    ]

    for cat_key, title, ax in categories:
        candidates = candidate_evaluation.get(cat_key)
        if not candidates:
            ax.set_title(title)
            ax.text(0.5, 0.5, 'No candidates', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        labels = [
            f"#{c['rank']+1}: L{c['layer']}-{c['latent_idx']}\n(t = {c['t_statistic']:.2f})"
            if c.get('t_statistic') is not None
            else f"#{c['rank']+1}: L{c['layer']}-{c['latent_idx']}"
            for c in candidates
        ]
        aurocs = [c['validation_split']['auroc'] for c in candidates]
        f1s = [c['validation_split']['f1'] for c in candidates]
        selected_flags = [c.get('selected', False) for c in candidates]

        x = np.arange(len(labels))
        width = 0.35

        bars_auroc = ax.bar(x - width / 2, aurocs, width, label='AUROC', color=color_auroc)
        bars_f1 = ax.bar(x + width / 2, f1s, width, label='F1', color=color_f1)

        # Highlight selected candidate with bold edge
        for i, is_selected in enumerate(selected_flags):
            if is_selected:
                for bar_group in [bars_auroc, bars_f1]:
                    bar_group[i].set_edgecolor('black')
                    bar_group[i].set_linewidth(2.5)
                # Star marker above the taller bar
                peak = max(aurocs[i], f1s[i])
                ax.plot(x[i], peak + 0.04, '*', color='black', markersize=14,
                        zorder=5)

        ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.6, linewidth=1)
        ax.set_xlabel('Candidate Latent')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'candidate_comparison.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved candidate comparison plot to {output_path}")


def load_split_probe_activations(
    split_name: str,
    layer_num: int,
    probe_direction: torch.Tensor,
    probe_bias: float,
    latent_type: str,
    phase3_5_dir: Path,
    phase3_6_dir: Path,
    config: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Load activations and compute probe scores (for probe baseline comparison).

    Args:
        split_name: 'tuning' or 'analysis'
        layer_num: Layer number for the probe
        probe_direction: Probe direction tensor [d_model]
        probe_bias: Probe bias term (for logreg, 0 for mass_mean)
        latent_type: 'correct' or 'incorrect'
        phase3_5_dir: Directory containing Phase 3.5 outputs (analysis split)
        phase3_6_dir: Directory containing Phase 3.6 outputs (tuning split)
        config: Configuration object

    Returns:
        Tuple of (labels, scores)
    """
    # Select the correct directory based on split type
    if split_name == 'tuning':
        activation_dir = phase3_6_dir
        # Try merged file first (parallel runs), then original pattern
        merged_files = list(phase3_6_dir.glob('dataset_merged_*.parquet'))
        if merged_files:
            temp_data = pd.read_parquet(sorted(merged_files)[-1])  # Latest merged file
        else:
            temp_data = pd.read_parquet(phase3_6_dir / 'dataset_hyperparams_temp_0_0.parquet')
    else:
        activation_dir = phase3_5_dir
        temp_data = pd.read_parquet(phase3_5_dir / 'dataset_temp_0_0.parquet')

    device = detect_device()
    scores = []
    labels = []
    missing_tasks = []

    for _, row in temp_data.iterrows():
        task_id = row['task_id']
        baseline_passed = row['baseline_passed']

        act_file = activation_dir / f'activations/task_activations/{task_id}_layer_{layer_num}.safetensors'

        if not act_file.exists():
            missing_tasks.append(task_id)
            continue

        # Load raw activation
        raw_activation = load_activation(act_file, device)
        raw_activation = raw_activation.to(probe_direction.dtype)

        # Compute probe score: w @ x + b
        with torch.no_grad():
            score = (raw_activation @ probe_direction).item() + probe_bias

        scores.append(score)

        # Create label
        if latent_type == 'correct':
            label = 1 if baseline_passed else 0
        else:
            label = 1 if not baseline_passed else 0

        labels.append(label)

    if missing_tasks:
        logger.warning(f"Missing activation files for {len(missing_tasks)} tasks")

    logger.info(f"Loaded {len(labels)} samples for {split_name} split (probe)")
    logger.info(f"Class distribution: {np.bincount(labels)}")

    return np.array(labels), np.array(scores)


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
        # Try merged file first (parallel runs), then original pattern
        merged_files = list(phase3_6_dir.glob('dataset_merged_*.parquet'))
        if merged_files:
            temp_data = pd.read_parquet(sorted(merged_files)[-1])
        else:
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

    try:
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
    finally:
        # Ensure SAE is freed from GPU memory even on exception
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.array(labels), np.array(activations)

def evaluate_single_latent(
    layer: int,
    latent_idx: int,
    latent_type: str,
    phase3_5_dir: Path,
    phase3_6_dir: Path,
    config: Config
) -> Optional[dict]:
    """Evaluate one latent candidate on tuning + analysis splits (no plots).

    Args:
        layer: SAE layer number
        latent_idx: Latent index within the SAE
        latent_type: 'correct' or 'incorrect'
        phase3_5_dir: Phase 3.5 output directory (analysis split)
        phase3_6_dir: Phase 3.6 output directory (tuning split)
        config: Configuration object

    Returns:
        dict with metrics and raw data, or None if evaluation fails
    """
    try:
        # Tuning split: find optimal threshold
        y_true_hp, scores_hp = load_split_activations(
            'tuning', layer, latent_idx, latent_type,
            phase3_5_dir, phase3_6_dir, config
        )

        if len(y_true_hp) == 0:
            logger.warning(f"No tuning samples for L{layer}-{latent_idx}")
            return None

        # Grid search for F1-optimal threshold (same logic as find_optimal_threshold)
        thresholds = np.linspace(scores_hp.min(), scores_hp.max(), 102)[1:-1]
        f1_scores = [
            f1_score(y_true_hp, (scores_hp >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[optimal_idx])

        # Tuning metrics
        hp_auroc = float(roc_auc_score(y_true_hp, scores_hp))
        y_pred_hp = (scores_hp >= optimal_threshold).astype(int)
        hp_metrics = {
            'auroc': hp_auroc,
            'f1': float(f1_score(y_true_hp, y_pred_hp, zero_division=0)),
            'precision': float(precision_score(y_true_hp, y_pred_hp, zero_division=0)),
            'recall': float(recall_score(y_true_hp, y_pred_hp, zero_division=0)),
            'threshold': optimal_threshold,
            'threshold_range': (float(scores_hp.min()), float(scores_hp.max())),
            'f1_curve': {'thresholds': thresholds.tolist(), 'f1_scores': f1_scores}
        }

        # Analysis (validation) split
        y_true_val, scores_val = load_split_activations(
            'analysis', layer, latent_idx, latent_type,
            phase3_5_dir, phase3_6_dir, config
        )

        if len(y_true_val) == 0:
            logger.warning(f"No analysis samples for L{layer}-{latent_idx}")
            return None

        val_auroc = float(roc_auc_score(y_true_val, scores_val))
        y_pred_val = (scores_val >= optimal_threshold).astype(int)
        val_metrics = {
            'auroc': val_auroc,
            'f1': float(f1_score(y_true_val, y_pred_val, zero_division=0)),
            'precision': float(precision_score(y_true_val, y_pred_val, zero_division=0)),
            'recall': float(recall_score(y_true_val, y_pred_val, zero_division=0)),
            'threshold': optimal_threshold
        }

        logger.info(f"  L{layer}-{latent_idx}: val AUROC={val_auroc:.4f}, val F1={val_metrics['f1']:.4f}")

        return {
            'layer': layer,
            'latent_idx': latent_idx,
            'hyperparameter_split': hp_metrics,
            'validation_split': val_metrics,
            '_y_true_hp': y_true_hp,
            '_scores_hp': scores_hp,
            '_y_true_val': y_true_val,
            '_scores_val': scores_val,
        }
    except Exception as e:
        logger.warning(f"Failed to evaluate L{layer}-{latent_idx} ({latent_type}): {e}")
        return None


def evaluate_candidates_and_select_best(
    candidates: list[dict],
    latent_type: str,
    phase3_5_dir: Path,
    phase3_6_dir: Path,
    config: Config
) -> tuple[dict, list[dict]]:
    """Evaluate top-N candidates and select best by validation AUROC.

    Args:
        candidates: List of candidate dicts from Phase 2.10 (layer, latent_idx, ...)
        latent_type: 'correct' or 'incorrect'
        phase3_5_dir: Phase 3.5 output directory (analysis split)
        phase3_6_dir: Phase 3.6 output directory (tuning split)
        config: Configuration object

    Returns:
        Tuple of (best_result, all_results_for_json)

    Raises:
        RuntimeError: If all candidates fail evaluation
    """
    logger.info(f"Evaluating {len(candidates)} {latent_type}-predicting candidates:")

    results = []
    for rank, candidate in enumerate(candidates):
        layer = candidate['layer']
        latent_idx = candidate['latent_idx']
        logger.info(f"  Candidate {rank}: L{layer}-{latent_idx}")

        result = evaluate_single_latent(
            layer, latent_idx, latent_type,
            phase3_5_dir, phase3_6_dir, config
        )

        if result is not None:
            result['rank'] = rank
            result['t_statistic'] = candidate.get('t_statistic')
            results.append(result)
        else:
            logger.warning(f"  Candidate {rank} (L{layer}-{latent_idx}): SKIPPED (missing layer data)")

    if not results:
        raise RuntimeError(
            f"All {len(candidates)} {latent_type}-predicting candidates failed. "
            f"Re-run Phases 3.5/3.6 to extract activations for the needed layers."
        )

    # Select best by validation AUROC
    best = max(results, key=lambda r: r['validation_split']['auroc'])

    # Log comparison table
    logger.info(f"\n{latent_type.upper()}-PREDICTING CANDIDATE COMPARISON:")
    logger.info(f"{'Rank':<6} {'Latent':<14} {'Val AUROC':<12} {'Val F1':<10} {'HP AUROC':<12} {'HP F1':<10} {'Selected'}")
    logger.info("-" * 78)
    for r in results:
        selected = " <-- BEST" if r is best else ""
        logger.info(
            f"{r['rank']:<6} L{r['layer']}-{r['latent_idx']:<6} "
            f"{r['validation_split']['auroc']:<12.4f} {r['validation_split']['f1']:<10.4f} "
            f"{r['hyperparameter_split']['auroc']:<12.4f} {r['hyperparameter_split']['f1']:<10.4f}"
            f"{selected}"
        )

    # Build JSON-serializable candidate list (without numpy arrays)
    all_for_json = [
        {
            'rank': r['rank'],
            'layer': r['layer'],
            'latent_idx': r['latent_idx'],
            't_statistic': r.get('t_statistic'),
            'hyperparameter_split': r['hyperparameter_split'],
            'validation_split': r['validation_split'],
            'selected': (r is best),
        }
        for r in results
    ]

    return best, all_for_json


def main():
    """Legacy entry point for running directly. Uses Phase38Runner."""
    from common.config import Config
    config = Config()
    runner = Phase38Runner(config)
    runner.run()

if __name__ == "__main__":
    main()