"""Phase 3.8: AUROC and F1 Evaluation for PVA-SAE Features.

This script evaluates bidirectional SAE features (correct-predicting and incorrect-predicting)
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
from typing import Dict, Tuple, Optional
import argparse
import os

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)

from common.logging import get_logger
from common.utils import detect_device, ensure_directory_exists, discover_latest_phase_output
from common_simplified.helpers import save_json, load_json
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase3_8.auroc_f1_evaluator")


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    feature_type: str,
    output_dir: Path
) -> Dict[str, float]:
    """Calculate metrics for either correct or incorrect predicting features.
    
    Args:
        y_true: Ground truth labels
        scores: Feature activation scores
        threshold: Binary classification threshold
        feature_type: 'correct' or 'incorrect'
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
    
    logger.info(f"\nMetrics for {feature_type}-predicting feature:")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUROC: {auroc:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, feature_type, output_dir)
    
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
    feature_type: str, 
    output_dir: Path
) -> Tuple[float, Dict[str, float]]:
    """Find optimal threshold for a specific feature type.
    
    Args:
        y_true: Ground truth labels
        scores: Feature activation scores
        feature_type: 'correct' or 'incorrect'
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    # Grid search for F1-Optimal Threshold
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    # Find threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds[optimal_idx]
    max_f1_score = f1_scores[optimal_idx]
    
    # Store threshold data for later combined plotting
    # Individual plots will be created after both features are processed

    # Evaluate at optimal threshold
    logger.info(f'\nF1 optimal for {feature_type}-predicting feature:')
    metrics = calculate_metrics(y_true, scores, optimal_f1_threshold, feature_type, output_dir)

    # Return threshold data for combined plotting
    metrics['threshold_range'] = (float(scores.min()), float(scores.max()))
    metrics['f1_curve'] = {'thresholds': thresholds.tolist(), 'f1_scores': f1_scores}
    
    return optimal_f1_threshold, metrics


def plot_combined_f1_thresholds(
    correct_metrics: Dict,
    incorrect_metrics: Dict,
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
    plt.plot(correct_thresholds, correct_f1s, 'b-', linewidth=2, label='Correct-predicting')
    plt.plot(incorrect_thresholds, incorrect_f1s, 'r-', linewidth=2, label='Incorrect-predicting')

    # Mark optimal points
    correct_optimal_threshold = correct_metrics['threshold']
    correct_optimal_f1 = correct_metrics['f1']
    incorrect_optimal_threshold = incorrect_metrics['threshold']
    incorrect_optimal_f1 = incorrect_metrics['f1']

    plt.plot(correct_optimal_threshold, correct_optimal_f1, 'bo', markersize=10,
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
    plt.savefig(output_dir / 'f1_threshold_plot_combined.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved combined F1 threshold plot to {output_dir / 'f1_threshold_plot_combined.png'}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_type: str,
    output_dir: Path
) -> None:
    """Plot confusion matrix with appropriate labels for feature type.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        feature_type: 'correct' or 'incorrect'
        output_dir: Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Adjust labels based on what we're predicting
    if feature_type == 'correct':
        # Predicting correctness
        labels = ['Incorrect', 'Correct']
    else:
        # Predicting incorrectness
        labels = ['Correct', 'Incorrect']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {feature_type.capitalize()}-Predicting Feature')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot
    plt.savefig(output_dir / f'confusion_matrix_{feature_type}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparative_metrics(
    results: Dict, 
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
        results['correct_predicting_feature']['validation_metrics']['metrics'][m.lower()]
        for m in metrics
    ]
    incorrect_vals = [
        results['incorrect_predicting_feature']['validation_metrics']['metrics'][m.lower()]
        for m in metrics
    ]

    # Plot bars
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, correct_vals, width, label='Correct-predicting', color='blue')
    bars2 = ax1.bar(x + width/2, incorrect_vals, width, label='Incorrect-predicting', color='red')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Feature Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot ROC curves if we have the data
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.5)')
    
    # Plot ROC curve for correct-predicting feature if data provided
    if y_true_val_correct is not None and scores_val_correct is not None:
        fpr_correct, tpr_correct, _ = roc_curve(y_true_val_correct, scores_val_correct)
        auc_correct = results['correct_predicting_feature']['validation_metrics']['metrics']['auroc']
        ax2.plot(fpr_correct, tpr_correct, color='blue', linewidth=2,
                label=f'Correct-predicting (AUC = {auc_correct:.3f})')

    # Plot ROC curve for incorrect-predicting feature if data provided
    if y_true_val_incorrect is not None and scores_val_incorrect is not None:
        fpr_incorrect, tpr_incorrect, _ = roc_curve(y_true_val_incorrect, scores_val_incorrect)
        auc_incorrect = results['incorrect_predicting_feature']['validation_metrics']['metrics']['auroc']
        ax2.plot(fpr_incorrect, tpr_incorrect, color='red', linewidth=2,
                label=f'Incorrect-predicting (AUC = {auc_incorrect:.3f})')
    
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_metrics.png', dpi=150, bbox_inches='tight')
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
    plt.plot(recall_correct, precision_correct, 'b-', linewidth=2,
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved precision-recall curves to {output_path}")


def load_split_activations(
    split_name: str, 
    layer_num: int, 
    feature_idx: int, 
    feature_type: str,
    phase0_1_dir: Path,
    phase3_5_dir: Path,
    phase3_6_dir: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Load activations for a specific feature from appropriate phase data.
    
    Args:
        split_name: 'hyperparams' or 'validation'
        layer_num: Layer number for the feature
        feature_idx: Index of the specific feature
        feature_type: 'correct' or 'incorrect'
        phase0_1_dir: Directory containing Phase 0.1 outputs
        phase3_5_dir: Directory containing Phase 3.5 outputs (validation split)
        phase3_6_dir: Directory containing Phase 3.6 outputs (hyperparams split)
        
    Returns:
        Tuple of (labels, activations)
    """
    # Load split data
    split_data = pd.read_parquet(phase0_1_dir / f'{split_name}_mbpp.parquet')
    
    # Select the correct directory based on split type
    if split_name == 'hyperparams':
        # Use Phase 3.6 directory for hyperparameter split
        activation_dir = phase3_6_dir
        # Load temperature 0.0 dataset from Phase 3.6
        temp_data = pd.read_parquet(phase3_6_dir / 'dataset_hyperparams_temp_0_0.parquet')
    else:  # validation
        # Use Phase 3.5 directory for validation split
        activation_dir = phase3_5_dir
        # Load temperature 0.0 dataset from Phase 3.5
        temp_data = pd.read_parquet(phase3_5_dir / 'dataset_temp_0_0.parquet')
    
    # Detect device and load SAE for encoding
    device = detect_device()
    sae = load_gemma_scope_sae(layer_num, device)
    logger.info(f"Loaded SAE for layer {layer_num} with 16,384 features on {device}")
    
    activations = []
    labels = []
    missing_tasks = []
    
    for _, row in split_data.iterrows():
        task_id = row['task_id']
        
        # Load raw activations from appropriate phase
        act_file = activation_dir / f'activations/task_activations/{task_id}_layer_{layer_num}.npz'
        
        if not act_file.exists():
            missing_tasks.append(task_id)
            continue
        
        # Get test result from temperature 0.0 dataset FIRST
        task_results = temp_data[temp_data['task_id'] == task_id]['test_passed'].values
        if len(task_results) == 0:
            logger.warning(f"No test results found for task {task_id}")
            continue
            
        act_data = np.load(act_file)
        
        # Get raw activation from Phase 3.5 (stored as 'arr_0')
        # Shape: (1, 2304) - raw residual stream activation
        raw_activation = torch.from_numpy(act_data['arr_0']).to(device)
        
        # Ensure dtype matches SAE parameters for matrix multiplication
        raw_activation = raw_activation.to(sae.W_enc.dtype)
        
        # Encode through SAE to get features
        # Shape: (1, 16384) - SAE feature activations
        with torch.no_grad():
            sae_features = sae.encode(raw_activation)
        
        # Extract specific feature value
        feature_activation = sae_features[0, feature_idx].item()
        activations.append(feature_activation)
            
        # Use the first sample at temperature 0.0
        test_passed = task_results[0]
        
        # Create label based on what we're predicting
        if feature_type == 'correct':
            # Predicting correctness: 1=correct, 0=incorrect
            label = 1 if test_passed else 0
        else:
            # Predicting incorrectness: 1=incorrect, 0=correct
            label = 1 if not test_passed else 0
        
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
    parser = argparse.ArgumentParser(description="Phase 3.8: AUROC and F1 Evaluation")
    parser.add_argument("--phase0-1-dir", type=str, help="Path to Phase 0.1 output directory")
    parser.add_argument("--phase3-5-dir", type=str, help="Path to Phase 3.5 output directory")
    parser.add_argument("--phase3-6-dir", type=str, help="Path to Phase 3.6 output directory")
    parser.add_argument("--output-dir", type=str, default=None, 
                       help="Output directory for results")
    args = parser.parse_args()
    
    # Use seed from config
    from common.config import Config
    config = Config()
    np.random.seed(config.evaluation_random_seed)
    torch.manual_seed(config.evaluation_random_seed)
    
    # Auto-discover phase outputs if not provided
    if not args.phase0_1_dir:
        latest_output = discover_latest_phase_output("0.1")
        if latest_output:
            phase0_1_dir = Path(latest_output).parent
            logger.info(f"Auto-discovered Phase 0.1 output: {phase0_1_dir}")
        else:
            raise FileNotFoundError("No Phase 0.1 output found. Please run Phase 0.1 first.")
    else:
        phase0_1_dir = Path(args.phase0_1_dir)
        
    if not args.phase3_5_dir:
        latest_output = discover_latest_phase_output("3.5")
        if latest_output:
            phase3_5_dir = Path(latest_output).parent
            logger.info(f"Auto-discovered Phase 3.5 output: {phase3_5_dir}")
        else:
            raise FileNotFoundError("No Phase 3.5 output found. Please run Phase 3.5 first.")
    else:
        phase3_5_dir = Path(args.phase3_5_dir)
    
    if not args.phase3_6_dir:
        latest_output = discover_latest_phase_output("3.6")
        if latest_output:
            phase3_6_dir = Path(latest_output).parent
            logger.info(f"Auto-discovered Phase 3.6 output: {phase3_6_dir}")
        else:
            raise FileNotFoundError("No Phase 3.6 output found. Please run Phase 3.6 first.")
    else:
        phase3_6_dir = Path(args.phase3_6_dir)
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from common.utils import get_phase_dir
        output_dir = Path(get_phase_dir('3.8'))
    ensure_directory_exists(output_dir)
    
    # Phase 1: Load best features from Phase 2.10 (t-statistic based selection)
    logger.info("Loading best features from Phase 2.10...")
    
    # Auto-discover Phase 2.10 output
    phase2_10_dir = discover_latest_phase_output("2.10")
    if not phase2_10_dir:
        raise FileNotFoundError("No Phase 2.10 output found. Please run Phase 2.10 first.")
    phase2_10_dir = Path(phase2_10_dir).parent
    
    # Load best features from Phase 2.10
    top_features_file = phase2_10_dir / 'top_20_features.json'
    if not top_features_file.exists():
        raise FileNotFoundError(f"top_20_features.json not found in {phase2_10_dir}. Please run Phase 2.10 first.")

    top_features = load_json(top_features_file)

    # Validate structure
    if 'correct' not in top_features or 'incorrect' not in top_features:
        raise ValueError("Missing 'correct' or 'incorrect' in top_20_features.json")

    if not top_features['correct'] or not top_features['incorrect']:
        raise ValueError("Empty feature list in top_20_features.json")

    # Get the best (index 0) features
    best_correct = top_features['correct'][0]
    best_incorrect = top_features['incorrect'][0]

    correct_layer = best_correct['layer']
    correct_feature_idx = best_correct['feature_idx']
    incorrect_layer = best_incorrect['layer']
    incorrect_feature_idx = best_incorrect['feature_idx']
    
    logger.info(f"Best correct-predicting feature: idx {correct_feature_idx} at layer {correct_layer}")
    logger.info(f"Best incorrect-predicting feature: idx {incorrect_feature_idx} at layer {incorrect_layer}")

    # Phase 2: Evaluate Correct-Predicting Feature
    logger.info("\n" + "="*60)
    logger.info("EVALUATING CORRECT-PREDICTING FEATURE")
    logger.info("="*60)
    
    # Load hyperparameter split for correct feature
    y_true_hp_correct, scores_hp_correct = load_split_activations(
        'hyperparams', correct_layer, correct_feature_idx, 'correct',
        phase0_1_dir, phase3_5_dir, phase3_6_dir
    )
    
    print(f"\nCorrect-predicting feature (hyperparameter split):")
    print(f"Total samples: {len(y_true_hp_correct)}")
    print(f"Positive class (correct code): {sum(y_true_hp_correct == 1)}")
    print(f"Negative class (incorrect code): {sum(y_true_hp_correct == 0)}")
    
    # Find optimal threshold
    optimal_threshold_correct, hp_metrics_correct = find_optimal_threshold(
        y_true_hp_correct, 
        scores_hp_correct,
        'correct',
        output_dir
    )
    
    # Load validation split
    y_true_val_correct, scores_val_correct = load_split_activations(
        'validation', correct_layer, correct_feature_idx, 'correct',
        phase0_1_dir, phase3_5_dir, phase3_6_dir
    )
    
    print(f"\nCorrect-predicting feature (validation split):")
    print(f"Total samples: {len(y_true_val_correct)}")
    
    # Evaluate on validation set
    val_metrics_correct = calculate_metrics(
        y_true_val_correct, 
        scores_val_correct, 
        optimal_threshold_correct,
        'correct',
        output_dir
    )
    
    # Phase 3: Evaluate Incorrect-Predicting Feature
    logger.info("\n" + "="*60)
    logger.info("EVALUATING INCORRECT-PREDICTING FEATURE")
    logger.info("="*60)
    
    # Load hyperparameter split for incorrect feature
    y_true_hp_incorrect, scores_hp_incorrect = load_split_activations(
        'hyperparams', incorrect_layer, incorrect_feature_idx, 'incorrect',
        phase0_1_dir, phase3_5_dir, phase3_6_dir
    )
    
    print(f"\nIncorrect-predicting feature (hyperparameter split):")
    print(f"Total samples: {len(y_true_hp_incorrect)}")
    print(f"Positive class (incorrect code): {sum(y_true_hp_incorrect == 1)}")
    print(f"Negative class (correct code): {sum(y_true_hp_incorrect == 0)}")
    
    # Find optimal threshold
    optimal_threshold_incorrect, hp_metrics_incorrect = find_optimal_threshold(
        y_true_hp_incorrect, 
        scores_hp_incorrect,
        'incorrect',
        output_dir
    )
    
    # Load validation split
    y_true_val_incorrect, scores_val_incorrect = load_split_activations(
        'validation', incorrect_layer, incorrect_feature_idx, 'incorrect',
        phase0_1_dir, phase3_5_dir, phase3_6_dir
    )
    
    print(f"\nIncorrect-predicting feature (validation split):")
    print(f"Total samples: {len(y_true_val_incorrect)}")
    
    # Evaluate on validation set
    val_metrics_incorrect = calculate_metrics(
        y_true_val_incorrect, 
        scores_val_incorrect, 
        optimal_threshold_incorrect,
        'incorrect',
        output_dir
    )
    
    # Phase 4: Generate Combined F1 Threshold Plot
    logger.info("\n" + "="*60)
    logger.info("GENERATING COMBINED F1 THRESHOLD PLOT")
    logger.info("="*60)

    plot_combined_f1_thresholds(hp_metrics_correct, hp_metrics_incorrect, output_dir)

    # Phase 5: Save Combined Results
    logger.info("\n" + "="*60)
    logger.info("SAVING RESULTS")
    logger.info("="*60)

    # Compile results for both features
    results = {
        'phase': '3.8',
        'correct_predicting_feature': {
            'feature': {
                'idx': int(correct_feature_idx),
                'layer': int(correct_layer)
            },
            'threshold_optimization': {
                'split': 'hyperparameter',
                'n_samples': int(len(y_true_hp_correct)),
                'optimal_threshold': float(optimal_threshold_correct),
                'metrics': hp_metrics_correct
            },
            'validation_metrics': {
                'split': 'validation',
                'n_samples': int(len(y_true_val_correct)),
                'metrics': val_metrics_correct
            }
        },
        'incorrect_predicting_feature': {
            'feature': {
                'idx': int(incorrect_feature_idx),
                'layer': int(incorrect_layer)
            },
            'threshold_optimization': {
                'split': 'hyperparameter',
                'n_samples': int(len(y_true_hp_incorrect)),
                'optimal_threshold': float(optimal_threshold_incorrect),
                'metrics': hp_metrics_incorrect
            },
            'validation_metrics': {
                'split': 'validation',
                'n_samples': int(len(y_true_val_incorrect)),
                'metrics': val_metrics_incorrect
            }
        },
        'creation_timestamp': datetime.now().isoformat()
    }
    
    # Save comprehensive results
    save_json(results, output_dir / 'evaluation_results.json')
    
    # Generate comparative visualization
    plot_comparative_metrics(
        results, output_dir,
        y_true_val_correct, scores_val_correct,
        y_true_val_incorrect, scores_val_incorrect
    )

    # Generate precision-recall curves (standalone figure for paper)
    logger.info("\n" + "="*60)
    logger.info("GENERATING PRECISION-RECALL CURVES")
    logger.info("="*60)

    plot_precision_recall_curves(
        output_dir,
        y_true_val_correct, scores_val_correct,
        y_true_val_incorrect, scores_val_incorrect
    )

    # Generate summary
    summary_lines = [
        "=" * 60,
        "PHASE 3.8 FINAL RESULTS SUMMARY",
        "=" * 60,
        f"\nCorrect-Predicting Feature (Layer {correct_layer}, Feature {correct_feature_idx}):",
        f"  Hyperparameter Optimal Threshold: {optimal_threshold_correct:.4f}",
        f"  Validation AUROC: {val_metrics_correct['auroc']:.4f}",
        f"  Validation F1: {val_metrics_correct['f1']:.4f}",
        f"  Validation Precision: {val_metrics_correct['precision']:.4f}",
        f"  Validation Recall: {val_metrics_correct['recall']:.4f}",
        f"\nIncorrect-Predicting Feature (Layer {incorrect_layer}, Feature {incorrect_feature_idx}):",
        f"  Hyperparameter Optimal Threshold: {optimal_threshold_incorrect:.4f}",
        f"  Validation AUROC: {val_metrics_incorrect['auroc']:.4f}",
        f"  Validation F1: {val_metrics_incorrect['f1']:.4f}",
        f"  Validation Precision: {val_metrics_incorrect['precision']:.4f}",
        f"  Validation Recall: {val_metrics_incorrect['recall']:.4f}",
        "\n" + "=" * 60
    ]
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save summary to file
    with open(output_dir / 'evaluation_summary.txt', 'w') as f:
        f.write(summary_text)
    
    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()