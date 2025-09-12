"""Phase 3.8: AUROC and F1 Evaluation for PVA-SAE Features.

This script evaluates bidirectional SAE features (correct-preferring and incorrect-preferring)
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
    confusion_matrix, roc_curve
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
    """Calculate metrics for either correct or incorrect preferring features.
    
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
    
    logger.info(f"\nMetrics for {feature_type}-preferring feature:")
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
    
    # Plot F1 scores against thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Scores vs Thresholds - {feature_type.capitalize()}-Preferring Feature')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=optimal_f1_threshold, color='r', linestyle='--', 
               label=f'Optimal F1 Threshold: {optimal_f1_threshold:.3f}')
    plt.axhline(y=max_f1_score, color='g', linestyle='--', 
               label=f'Max F1 Score: {max_f1_score:.3f}')
    plt.legend()
    
    # Save plot
    plt.savefig(output_dir / f'f1_threshold_plot_{feature_type}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Evaluate at optimal threshold
    logger.info(f'\nF1 optimal for {feature_type}-preferring feature:')
    metrics = calculate_metrics(y_true, scores, optimal_f1_threshold, feature_type, output_dir)
    
    return optimal_f1_threshold, metrics


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
    plt.title(f'Confusion Matrix - {feature_type.capitalize()}-Preferring Feature')
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
        results['correct_preferring_feature']['validation_metrics']['metrics'][m.lower()]
        for m in metrics
    ]
    incorrect_vals = [
        results['incorrect_preferring_feature']['validation_metrics']['metrics'][m.lower()]
        for m in metrics
    ]
    
    # Plot bars
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, correct_vals, width, label='Correct-Preferring', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, incorrect_vals, width, label='Incorrect-Preferring', color='#e74c3c')
    
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
    
    # Plot ROC curve for correct-preferring feature if data provided
    if y_true_val_correct is not None and scores_val_correct is not None:
        fpr_correct, tpr_correct, _ = roc_curve(y_true_val_correct, scores_val_correct)
        auc_correct = results['correct_preferring_feature']['validation_metrics']['metrics']['auroc']
        ax2.plot(fpr_correct, tpr_correct, color='#2ecc71', linewidth=2, 
                label=f'Correct-Preferring (AUC = {auc_correct:.3f})')
    
    # Plot ROC curve for incorrect-preferring feature if data provided
    if y_true_val_incorrect is not None and scores_val_incorrect is not None:
        fpr_incorrect, tpr_incorrect, _ = roc_curve(y_true_val_incorrect, scores_val_incorrect)
        auc_incorrect = results['incorrect_preferring_feature']['validation_metrics']['metrics']['auroc']
        ax2.plot(fpr_incorrect, tpr_incorrect, color='#e74c3c', linewidth=2,
                label=f'Incorrect-Preferring (AUC = {auc_incorrect:.3f})')
    
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_dir / 'comparative_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


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
    best_layer_file = phase2_10_dir / 'best_layer.json'
    if not best_layer_file.exists():
        raise FileNotFoundError(f"best_layer.json not found in {phase2_10_dir}. Please run Phase 2.10 first.")
    
    best_features = load_json(best_layer_file)
    correct_layer = best_features['correct']
    correct_feature_idx = best_features['correct_feature_idx']
    incorrect_layer = best_features['incorrect']
    incorrect_feature_idx = best_features['incorrect_feature_idx']
    
    logger.info(f"Best correct-preferring feature: idx {correct_feature_idx} at layer {correct_layer}")
    logger.info(f"Best incorrect-preferring feature: idx {incorrect_feature_idx} at layer {incorrect_layer}")
    
    # Phase 2: Evaluate Correct-Preferring Feature
    logger.info("\n" + "="*60)
    logger.info("EVALUATING CORRECT-PREFERRING FEATURE")
    logger.info("="*60)
    
    # Load hyperparameter split for correct feature
    y_true_hp_correct, scores_hp_correct = load_split_activations(
        'hyperparams', correct_layer, correct_feature_idx, 'correct',
        phase0_1_dir, phase3_5_dir, phase3_6_dir
    )
    
    print(f"\nCorrect-preferring feature (hyperparameter split):")
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
    
    print(f"\nCorrect-preferring feature (validation split):")
    print(f"Total samples: {len(y_true_val_correct)}")
    
    # Evaluate on validation set
    val_metrics_correct = calculate_metrics(
        y_true_val_correct, 
        scores_val_correct, 
        optimal_threshold_correct,
        'correct',
        output_dir
    )
    
    # Phase 3: Evaluate Incorrect-Preferring Feature
    logger.info("\n" + "="*60)
    logger.info("EVALUATING INCORRECT-PREFERRING FEATURE")
    logger.info("="*60)
    
    # Load hyperparameter split for incorrect feature
    y_true_hp_incorrect, scores_hp_incorrect = load_split_activations(
        'hyperparams', incorrect_layer, incorrect_feature_idx, 'incorrect',
        phase0_1_dir, phase3_5_dir, phase3_6_dir
    )
    
    print(f"\nIncorrect-preferring feature (hyperparameter split):")
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
    
    print(f"\nIncorrect-preferring feature (validation split):")
    print(f"Total samples: {len(y_true_val_incorrect)}")
    
    # Evaluate on validation set
    val_metrics_incorrect = calculate_metrics(
        y_true_val_incorrect, 
        scores_val_incorrect, 
        optimal_threshold_incorrect,
        'incorrect',
        output_dir
    )
    
    # Phase 4: Save Combined Results
    logger.info("\n" + "="*60)
    logger.info("SAVING RESULTS")
    logger.info("="*60)
    
    # Compile results for both features
    results = {
        'phase': '3.8',
        'correct_preferring_feature': {
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
        'incorrect_preferring_feature': {
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
    
    # Generate summary
    summary_lines = [
        "=" * 60,
        "PHASE 3.8 FINAL RESULTS SUMMARY",
        "=" * 60,
        f"\nCorrect-Preferring Feature (Layer {correct_layer}, Feature {correct_feature_idx}):",
        f"  Hyperparameter Optimal Threshold: {optimal_threshold_correct:.4f}",
        f"  Validation AUROC: {val_metrics_correct['auroc']:.4f}",
        f"  Validation F1: {val_metrics_correct['f1']:.4f}",
        f"  Validation Precision: {val_metrics_correct['precision']:.4f}",
        f"  Validation Recall: {val_metrics_correct['recall']:.4f}",
        f"\nIncorrect-Preferring Feature (Layer {incorrect_layer}, Feature {incorrect_feature_idx}):",
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