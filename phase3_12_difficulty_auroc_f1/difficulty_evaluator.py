"""Phase 3.12: Difficulty-Based AUROC Analysis for PVA-SAE Features.

This script evaluates bidirectional SAE features across different problem
difficulty levels (Easy/Medium/Hard) using cyclomatic complexity stratification.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import argparse

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)

# Note: We implement our own calculate_difficulty_metrics function
# rather than reusing calculate_metrics from Phase 3.8 due to different requirements

from common.logging import get_logger
from common.utils import detect_device, ensure_directory_exists, discover_latest_phase_output
from common_simplified.helpers import save_json, load_json
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae

logger = get_logger("phase3_12.difficulty_evaluator")


def group_by_difficulty(validation_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group validation tasks by cyclomatic complexity into Easy/Medium/Hard.
    
    Args:
        validation_data: DataFrame with cyclomatic complexity annotations
        
    Returns:
        Dictionary with 'easy', 'medium', 'hard' groups
    """
    # Define difficulty thresholds based on spec
    difficulty_groups = {
        'easy': validation_data[validation_data['cyclomatic_complexity'] == 1],
        'medium': validation_data[validation_data['cyclomatic_complexity'].between(2, 3)],
        'hard': validation_data[validation_data['cyclomatic_complexity'] >= 4]
    }
    
    # Log group sizes
    for group_name, group_data in difficulty_groups.items():
        logger.info(f"{group_name.capitalize()} group: {len(group_data)} tasks "
                   f"(complexity range: {group_data['cyclomatic_complexity'].min()}-"
                   f"{group_data['cyclomatic_complexity'].max()})")
    
    return difficulty_groups


def load_group_activations(
    group_data: pd.DataFrame,
    layer_num: int,
    feature_idx: int,
    feature_type: str,
    sae: torch.nn.Module,
    device: torch.device,
    temp_data: pd.DataFrame,
    phase3_5_dir: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Load activations for a specific difficulty group.
    
    Args:
        group_data: DataFrame with tasks for this difficulty group
        layer_num: Layer number for SAE
        feature_idx: Feature index to extract
        feature_type: 'correct' or 'incorrect'
        sae: Pre-loaded SAE model
        device: Device for computation
        temp_data: Pre-loaded temperature 0.0 dataset
        phase3_5_dir: Directory containing Phase 3.5 outputs
        
    Returns:
        Tuple of (labels, activations)
    """
    activations = []
    labels = []
    missing_tasks = []
    
    for _, row in group_data.iterrows():
        task_id = row['task_id']
        
        # Load raw activations from Phase 3.5
        act_file = phase3_5_dir / f'activations/task_activations/{task_id}_layer_{layer_num}.npz'
        
        if not act_file.exists():
            missing_tasks.append(task_id)
            continue
            
        # Load and encode through SAE
        act_data = np.load(act_file)
        raw_activation = torch.from_numpy(act_data['arr_0']).to(device)
        
        # Ensure dtype matches SAE parameters for matrix multiplication
        raw_activation = raw_activation.to(sae.W_enc.dtype)
        
        with torch.no_grad():
            sae_features = sae.encode(raw_activation)
        
        # Extract specific feature value
        feature_activation = sae_features[0, feature_idx].item()
        activations.append(feature_activation)
        
        # Get test result and create label
        task_results = temp_data[temp_data['task_id'] == task_id]['test_passed'].values
        if len(task_results) == 0:
            continue
            
        test_passed = task_results[0]  # Use first sample at temperature 0.0
        
        # Create label based on feature type
        if feature_type == 'correct':
            label = 1 if test_passed else 0  # Flipped for correct-preferring
        else:
            label = 0 if test_passed else 1  # Standard for incorrect-preferring
        
        labels.append(label)
    
    if missing_tasks:
        logger.warning(f"Missing activation files for {len(missing_tasks)} tasks")
    
    # Check for edge cases
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        logger.warning(f"WARNING: {feature_type}-preferring feature has imbalanced classes - "
                      f"positive: {n_positive}, negative: {n_negative}")
    elif n_positive < 5 or n_negative < 5:
        logger.warning(f"WARNING: {feature_type}-preferring feature has very few samples in one class - "
                      f"positive: {n_positive}, negative: {n_negative}")
    
    return np.array(labels), np.array(activations)


def calculate_difficulty_metrics(
    difficulty_groups: Dict[str, pd.DataFrame],
    best_features: Dict,
    global_threshold: float,
    feature_type: str,
    output_dir: Path,
    sae: torch.nn.Module,
    device: torch.device,
    temp_data: pd.DataFrame,
    phase3_5_dir: Path
) -> Dict[str, Dict]:
    """Calculate AUROC and F1 for each difficulty group for a specific feature type.
    
    Args:
        difficulty_groups: Dict of difficulty groups
        best_features: Best feature information
        global_threshold: F1-optimal threshold from Phase 3.8
        feature_type: 'correct' or 'incorrect'
        output_dir: Output directory for plots
        sae: Pre-loaded SAE model
        device: Device for computation
        temp_data: Pre-loaded temperature 0.0 dataset
        phase3_5_dir: Directory containing Phase 3.5 outputs
        
    Returns:
        Dictionary of results per difficulty group
    """
    results = {}
    
    for group_name, group_data in difficulty_groups.items():
        logger.info(f"\nEvaluating {feature_type}-preferring feature on {group_name} group:")
        
        # Get feature info
        if feature_type == 'correct':
            layer = best_features['correct']
            feature_idx = best_features['correct_feature_idx']
        else:
            layer = best_features['incorrect']
            feature_idx = best_features['incorrect_feature_idx']
        
        # Load activations for this group
        y_true, scores = load_group_activations(
            group_data, layer, feature_idx, feature_type,
            sae, device, temp_data, phase3_5_dir
        )
        
        # Calculate AUROC (threshold-independent)
        auroc = roc_auc_score(y_true, scores)
        
        # Calculate F1 using global threshold from Phase 3.8
        y_pred = (scores > global_threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        
        # Save ROC curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {feature_type.capitalize()}-Preferring Feature ({group_name.capitalize()} Group)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'roc_curve_{feature_type}_{group_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot confusion matrix for this group
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        
        # Adjust labels based on feature type
        if feature_type == 'correct':
            labels = ['Incorrect', 'Correct']
        else:
            labels = ['Correct', 'Incorrect']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {feature_type.capitalize()}-Preferring Feature ({group_name.capitalize()} Group)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_dir / f'confusion_matrix_{feature_type}_{group_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        results[group_name] = {
            'auroc': float(auroc),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'n_samples': int(len(y_true)),
            'n_positive': int(sum(y_true)),
            'n_negative': int(len(y_true) - sum(y_true)),
            'complexity_range': [
                int(group_data['cyclomatic_complexity'].min()),
                int(group_data['cyclomatic_complexity'].max())
            ]
        }
        
        logger.info(f"  AUROC: {auroc:.4f}")
        logger.info(f"  F1: {f1:.4f} (using global threshold: {global_threshold:.4f})")
        logger.info(f"  Samples: {len(y_true)} (pos: {sum(y_true)}, neg: {len(y_true) - sum(y_true)})")
    
    return results


def plot_difficulty_distribution(
    difficulty_groups: Dict[str, pd.DataFrame],
    output_dir: Path
) -> None:
    """Visualize the distribution of tasks across difficulty levels."""
    plt.figure(figsize=(10, 6))
    
    group_sizes = [len(group) for group in difficulty_groups.values()]
    group_names = [name.capitalize() for name in difficulty_groups.keys()]
    
    bars = plt.bar(group_names, group_sizes, color=['lightgreen', 'orange', 'lightcoral'])
    plt.xlabel('Difficulty Level')
    plt.ylabel('Number of Tasks')
    plt.title('Task Distribution by Difficulty Level (Cyclomatic Complexity)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels and percentages
    total_tasks = sum(group_sizes)
    for i, (bar, size) in enumerate(zip(bars, group_sizes)):
        percentage = size / total_tasks * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{size}\n({percentage:.1f}%)', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'difficulty_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves_by_difficulty(
    difficulty_groups: Dict[str, pd.DataFrame],
    feature_type: str,
    results: Dict,
    output_dir: Path,
    best_features: Dict,
    sae: torch.nn.Module,
    device: torch.device,
    temp_data: pd.DataFrame,
    phase3_5_dir: Path
) -> None:
    """Plot ROC curves for each difficulty group on the same plot."""
    plt.figure(figsize=(10, 8))
    
    colors = ['green', 'orange', 'red']
    layer = best_features[feature_type]
    feature_idx = best_features[f'{feature_type}_feature_idx']
    
    for i, (group_name, group_result) in enumerate(results.items()):
        # Re-calculate ROC curve points for plotting
        y_true, scores = load_group_activations(
            difficulty_groups[group_name], 
            layer,
            feature_idx, 
            feature_type,
            sae, device, temp_data, phase3_5_dir
        )
        
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = group_result['auroc']
        
        plt.plot(fpr, tpr, linewidth=2, color=colors[i],
                label=f'{group_name.capitalize()} (AUC = {auroc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves by Difficulty - {feature_type.capitalize()}-Preferring Feature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_curves_by_difficulty_{feature_type}.png', dpi=150, bbox_inches='tight')
    plt.close()


def calculate_trend(values: list) -> str:
    """Calculate trend from a list of values, handling NaN."""
    valid_values = [v for v in values if not np.isnan(v)]
    if len(valid_values) < 2:
        return 'undefined'
    
    # Get first and last valid values
    first_valid = valid_values[0]
    last_valid = valid_values[-1]
    
    if first_valid > last_valid:
        return 'decreasing'
    elif first_valid < last_valid:
        return 'increasing'
    else:
        return 'stable'


def find_max_excluding_nan(results: Dict, metric: str) -> str:
    """Find the key with maximum value for a metric, excluding NaN."""
    valid_items = [(k, v[metric]) for k, v in results.items() if not np.isnan(v[metric])]
    
    if not valid_items:
        return 'undefined'
    
    return max(valid_items, key=lambda x: x[1])[0]


def plot_auroc_trends(
    correct_results: Dict,
    incorrect_results: Dict,
    output_dir: Path
) -> None:
    """Plot AUROC trends across difficulty levels for both feature types."""
    plt.figure(figsize=(12, 6))
    
    difficulties = list(correct_results.keys())
    correct_aurocs = [correct_results[d]['auroc'] for d in difficulties]
    incorrect_aurocs = [incorrect_results[d]['auroc'] for d in difficulties]
    
    # Handle NaN values for correct-preferring
    valid_correct_idx = [i for i, val in enumerate(correct_aurocs) if not np.isnan(val)]
    valid_correct_diff = [difficulties[i] for i in valid_correct_idx]
    valid_correct_aurocs = [correct_aurocs[i] for i in valid_correct_idx]
    
    # Handle NaN values for incorrect-preferring
    valid_incorrect_idx = [i for i, val in enumerate(incorrect_aurocs) if not np.isnan(val)]
    valid_incorrect_diff = [difficulties[i] for i in valid_incorrect_idx]
    valid_incorrect_aurocs = [incorrect_aurocs[i] for i in valid_incorrect_idx]
    
    # Plot valid points
    if valid_correct_aurocs:
        plt.plot(valid_correct_diff, valid_correct_aurocs, marker='o', linewidth=2, markersize=8,
                 label='Correct-Preferring Feature', color='blue')
    if valid_incorrect_aurocs:
        plt.plot(valid_incorrect_diff, valid_incorrect_aurocs, marker='s', linewidth=2, markersize=8,
                 label='Incorrect-Preferring Feature', color='red')
    
    # Mark NaN points
    for i, (c_auroc, i_auroc) in enumerate(zip(correct_aurocs, incorrect_aurocs)):
        if np.isnan(c_auroc):
            plt.plot(i, 0.5, 'x', markersize=10, color='lightblue')
            plt.text(i, 0.48, 'N/A', ha='center', va='top', color='lightblue', fontsize=8)
        if np.isnan(i_auroc):
            plt.plot(i, 0.5, 'x', markersize=10, color='lightcoral')
            plt.text(i, 0.52, 'N/A', ha='center', va='bottom', color='lightcoral', fontsize=8)
    
    plt.xlabel('Difficulty Level')
    plt.ylabel('AUROC')
    plt.title('AUROC Performance Trends Across Difficulty Levels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xticks(range(len(difficulties)), [d.capitalize() for d in difficulties])
    
    # Add value labels for valid points
    for i, (c_auroc, i_auroc) in enumerate(zip(correct_aurocs, incorrect_aurocs)):
        if not np.isnan(c_auroc):
            plt.text(i, c_auroc + 0.02, f'{c_auroc:.3f}', ha='center', va='bottom', color='blue')
        if not np.isnan(i_auroc):
            plt.text(i, i_auroc - 0.05, f'{i_auroc:.3f}', ha='center', va='top', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'auroc_trends_by_difficulty.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 3.12: Difficulty-Based AUROC Analysis")
    parser.add_argument("--phase3-5-dir", type=str, help="Path to Phase 3.5 output directory")
    parser.add_argument("--phase3-8-dir", type=str, help="Path to Phase 3.8 output directory")
    parser.add_argument("--output-dir", type=str, default=None, 
                       help="Output directory for results")
    args = parser.parse_args()
    
    # Use seed from config
    from common.config import Config
    config = Config()
    np.random.seed(config.evaluation_random_seed)
    torch.manual_seed(config.evaluation_random_seed)
    
    # Auto-discover phase outputs if not provided
    if not args.phase3_5_dir:
        latest_output = discover_latest_phase_output("3.5")
        if latest_output:
            phase3_5_dir = Path(latest_output).parent
            logger.info(f"Auto-discovered Phase 3.5 output: {phase3_5_dir}")
        else:
            raise FileNotFoundError("No Phase 3.5 output found. Please run Phase 3.5 first.")
    else:
        phase3_5_dir = Path(args.phase3_5_dir)
    
    if not args.phase3_8_dir:
        latest_output = discover_latest_phase_output("3.8")
        if latest_output:
            phase3_8_dir = Path(latest_output).parent
            logger.info(f"Auto-discovered Phase 3.8 output: {phase3_8_dir}")
        else:
            raise FileNotFoundError("No Phase 3.8 output found. Please run Phase 3.8 first.")
    else:
        phase3_8_dir = Path(args.phase3_8_dir)
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from common.utils import get_phase_dir
        output_dir = Path(get_phase_dir('3.12'))
    ensure_directory_exists(output_dir)
    
    # Phase 1: Load Dependencies and Setup
    logger.info("="*60)
    logger.info("PHASE 3.12: DIFFICULTY-BASED AUROC ANALYSIS")
    logger.info("="*60)
    
    # Load Phase 3.8 results to get best features and thresholds
    logger.info("\nLoading Phase 3.8 results...")
    phase3_8_results = load_json(phase3_8_dir / 'evaluation_results.json')
    best_features = {
        'correct': phase3_8_results['correct_preferring_feature']['feature']['layer'],
        'correct_feature_idx': phase3_8_results['correct_preferring_feature']['feature']['idx'],
        'incorrect': phase3_8_results['incorrect_preferring_feature']['feature']['layer'],
        'incorrect_feature_idx': phase3_8_results['incorrect_preferring_feature']['feature']['idx']
    }
    
    # Extract global F1-optimal thresholds from Phase 3.8
    global_thresholds = {
        'correct': phase3_8_results['correct_preferring_feature']['threshold_optimization']['optimal_threshold'],
        'incorrect': phase3_8_results['incorrect_preferring_feature']['threshold_optimization']['optimal_threshold']
    }
    
    logger.info(f"Best correct-preferring feature: idx {best_features['correct_feature_idx']} "
               f"at layer {best_features['correct']} (threshold: {global_thresholds['correct']:.4f})")
    logger.info(f"Best incorrect-preferring feature: idx {best_features['incorrect_feature_idx']} "
               f"at layer {best_features['incorrect']} (threshold: {global_thresholds['incorrect']:.4f})")
    
    # Load temperature 0.0 dataset which includes cyclomatic complexity
    logger.info("\nLoading validation dataset from Phase 3.5...")
    temp_data = pd.read_parquet(phase3_5_dir / 'dataset_temp_0_0.parquet')
    
    # Get unique tasks (since there are multiple samples per task)
    validation_data = temp_data.drop_duplicates(subset=['task_id'])[['task_id', 'cyclomatic_complexity']]
    logger.info(f"Validation dataset loaded: {len(validation_data)} unique tasks")
    logger.info(f"Cyclomatic complexity range: {validation_data['cyclomatic_complexity'].min()}-"
               f"{validation_data['cyclomatic_complexity'].max()}")
    
    # Phase 2: Group by Difficulty
    logger.info("\nGrouping tasks by difficulty...")
    difficulty_groups = group_by_difficulty(validation_data)
    
    # Generate difficulty distribution visualization
    plot_difficulty_distribution(difficulty_groups, output_dir)
    
    # Detect device once
    device = detect_device()
    logger.info(f"Using device: {device}")
    
    # Phase 3: Evaluate Correct-Preferring Feature
    logger.info("\n" + "="*60)
    logger.info("EVALUATING CORRECT-PREFERRING FEATURE ACROSS DIFFICULTY LEVELS")
    logger.info("="*60)
    
    # Load SAE for correct-preferring feature
    correct_layer = best_features['correct']
    sae_correct = load_gemma_scope_sae(correct_layer, device)
    logger.info(f"Loaded SAE for layer {correct_layer} on {device}")
    
    correct_results = calculate_difficulty_metrics(
        difficulty_groups, 
        best_features, 
        global_thresholds['correct'],
        'correct', 
        output_dir,
        sae_correct,
        device,
        temp_data,
        phase3_5_dir
    )
    
    # Clean up SAE after use
    del sae_correct
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Generate comparative visualization for correct-preferring
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    difficulties = list(correct_results.keys())
    aurocs = [correct_results[d]['auroc'] for d in difficulties]
    f1s = [correct_results[d]['f1'] for d in difficulties]
    
    # AUROC plot - handle NaN values
    valid_indices = [i for i, val in enumerate(aurocs) if not np.isnan(val)]
    valid_difficulties = [difficulties[i] for i in valid_indices]
    valid_aurocs = [aurocs[i] for i in valid_indices]
    
    # Plot only valid points
    if valid_aurocs:
        ax1.plot(valid_difficulties, valid_aurocs, marker='o', linewidth=2, markersize=8, 
                 label='Correct-Preferring Feature', color='blue')
    
    # Mark NaN points with a different marker
    nan_indices = [i for i, val in enumerate(aurocs) if np.isnan(val)]
    for idx in nan_indices:
        ax1.plot(idx, 0.5, 'x', markersize=10, color='red', 
                label='Undefined (single class)' if idx == nan_indices[0] else '')
        ax1.text(idx, 0.45, 'N/A', ha='center', va='top', color='red', fontsize=9)
    
    ax1.set_xlabel('Difficulty Level')
    ax1.set_ylabel('AUROC')
    ax1.set_title('AUROC Performance by Difficulty Level - Correct-Preferring Feature')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(range(len(difficulties)))
    ax1.set_xticklabels([d.capitalize() for d in difficulties])
    ax1.legend()
    
    # Add value labels for valid points
    for i, auroc in enumerate(aurocs):
        if not np.isnan(auroc):
            ax1.text(i, auroc + 0.02, f'{auroc:.3f}', ha='center', va='bottom')
    
    # F1 plot
    ax2.plot(difficulties, f1s, marker='o', linewidth=2, markersize=8, 
             label='Correct-Preferring Feature', color='green')
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Performance by Difficulty Level - Correct-Preferring Feature')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    # Add value labels
    for i, f1 in enumerate(f1s):
        ax2.text(i, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_difficulty_correct.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Phase 4: Evaluate Incorrect-Preferring Feature
    logger.info("\n" + "="*60)
    logger.info("EVALUATING INCORRECT-PREFERRING FEATURE ACROSS DIFFICULTY LEVELS")
    logger.info("="*60)
    
    # Load SAE for incorrect-preferring feature
    incorrect_layer = best_features['incorrect']
    sae_incorrect = load_gemma_scope_sae(incorrect_layer, device)
    logger.info(f"Loaded SAE for layer {incorrect_layer} on {device}")
    
    incorrect_results = calculate_difficulty_metrics(
        difficulty_groups, 
        best_features, 
        global_thresholds['incorrect'],
        'incorrect', 
        output_dir,
        sae_incorrect,
        device,
        temp_data,
        phase3_5_dir
    )
    
    # Clean up SAE after use
    del sae_incorrect
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Generate comparative visualization for incorrect-preferring
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    difficulties = list(incorrect_results.keys())
    aurocs = [incorrect_results[d]['auroc'] for d in difficulties]
    f1s = [incorrect_results[d]['f1'] for d in difficulties]
    
    # AUROC plot - handle NaN values
    valid_indices = [i for i, val in enumerate(aurocs) if not np.isnan(val)]
    valid_difficulties = [difficulties[i] for i in valid_indices]
    valid_aurocs = [aurocs[i] for i in valid_indices]
    
    # Plot only valid points
    if valid_aurocs:
        ax1.plot(valid_difficulties, valid_aurocs, marker='s', linewidth=2, markersize=8, 
                 label='Incorrect-Preferring Feature', color='red')
    
    # Mark NaN points with a different marker
    nan_indices = [i for i, val in enumerate(aurocs) if np.isnan(val)]
    for idx in nan_indices:
        ax1.plot(idx, 0.5, 'x', markersize=10, color='darkred', 
                label='Undefined (single class)' if idx == nan_indices[0] else '')
        ax1.text(idx, 0.45, 'N/A', ha='center', va='top', color='darkred', fontsize=9)
    
    ax1.set_xlabel('Difficulty Level')
    ax1.set_ylabel('AUROC')
    ax1.set_title('AUROC Performance by Difficulty Level - Incorrect-Preferring Feature')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(range(len(difficulties)))
    ax1.set_xticklabels([d.capitalize() for d in difficulties])
    ax1.legend()
    
    # Add value labels for valid points
    for i, auroc in enumerate(aurocs):
        if not np.isnan(auroc):
            ax1.text(i, auroc + 0.02, f'{auroc:.3f}', ha='center', va='bottom')
    
    # F1 plot
    ax2.plot(difficulties, f1s, marker='s', linewidth=2, markersize=8, 
             label='Incorrect-Preferring Feature', color='orange')
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Performance by Difficulty Level - Incorrect-Preferring Feature')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    # Add value labels
    for i, f1 in enumerate(f1s):
        ax2.text(i, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_difficulty_incorrect.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Phase 5: Comparative Analysis and Results
    logger.info("\n" + "="*60)
    logger.info("GENERATING COMPARATIVE ANALYSIS")
    logger.info("="*60)
    
    # Generate side-by-side comparison
    difficulties = list(correct_results.keys())
    correct_aurocs = [correct_results[d]['auroc'] for d in difficulties]
    incorrect_aurocs = [incorrect_results[d]['auroc'] for d in difficulties]
    correct_f1s = [correct_results[d]['f1'] for d in difficulties]
    incorrect_f1s = [incorrect_results[d]['f1'] for d in difficulties]
    
    # Create side-by-side comparison for both metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(difficulties))
    width = 0.35
    
    # AUROC comparison - handle NaN values
    # Replace NaN with 0 for bar height, but mark them differently
    correct_aurocs_plot = [val if not np.isnan(val) else 0 for val in correct_aurocs]
    incorrect_aurocs_plot = [val if not np.isnan(val) else 0 for val in incorrect_aurocs]
    
    bars1 = ax1.bar(x - width/2, correct_aurocs_plot, width, label='Correct-Preferring', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, incorrect_aurocs_plot, width, label='Incorrect-Preferring', color='red', alpha=0.7)
    
    # Mark NaN bars with hatching
    for i, (c_auroc, i_auroc) in enumerate(zip(correct_aurocs, incorrect_aurocs)):
        if np.isnan(c_auroc):
            bars1[i].set_hatch('///')
            bars1[i].set_alpha(0.3)
        if np.isnan(i_auroc):
            bars2[i].set_hatch('///')
            bars2[i].set_alpha(0.3)
    
    ax1.set_xlabel('Difficulty Level')
    ax1.set_ylabel('AUROC')
    ax1.set_title('AUROC Performance Comparison Across Difficulty Levels')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in difficulties])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add value labels for AUROC
    for i, (c_auroc, i_auroc) in enumerate(zip(correct_aurocs, incorrect_aurocs)):
        if not np.isnan(c_auroc):
            ax1.text(i - width/2, c_auroc + 0.01, f'{c_auroc:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(i - width/2, 0.05, 'N/A', ha='center', va='bottom', fontsize=8, color='darkblue')
        
        if not np.isnan(i_auroc):
            ax1.text(i + width/2, i_auroc + 0.01, f'{i_auroc:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(i + width/2, 0.05, 'N/A', ha='center', va='bottom', fontsize=8, color='darkred')
    
    # F1 comparison
    ax2.bar(x - width/2, correct_f1s, width, label='Correct-Preferring', color='green', alpha=0.7)
    ax2.bar(x + width/2, incorrect_f1s, width, label='Incorrect-Preferring', color='orange', alpha=0.7)
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Performance Comparison Across Difficulty Levels')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.capitalize() for d in difficulties])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add value labels for F1
    for i, (c_f1, i_f1) in enumerate(zip(correct_f1s, incorrect_f1s)):
        ax2.text(i - width/2, c_f1 + 0.01, f'{c_f1:.3f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, i_f1 + 0.01, f'{i_f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison_by_difficulty.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate additional comparative visualizations
    # Note: We need to reload SAEs since they were deleted after individual analyses
    sae_correct = load_gemma_scope_sae(best_features['correct'], device)
    plot_roc_curves_by_difficulty(difficulty_groups, 'correct', correct_results, output_dir, 
                                   best_features, sae_correct, device, temp_data, phase3_5_dir)
    del sae_correct
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    sae_incorrect = load_gemma_scope_sae(best_features['incorrect'], device)
    plot_roc_curves_by_difficulty(difficulty_groups, 'incorrect', incorrect_results, output_dir,
                                   best_features, sae_incorrect, device, temp_data, phase3_5_dir)
    del sae_incorrect
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    plot_auroc_trends(correct_results, incorrect_results, output_dir)
    
    # Compile comprehensive results
    results = {
        'phase': '3.12',
        'analysis_type': 'difficulty_based_auroc',
        'difficulty_groups': {
            group_name: {
                'complexity_min': int(group_data['cyclomatic_complexity'].min()),
                'complexity_max': int(group_data['cyclomatic_complexity'].max()),
                'n_tasks': int(len(group_data)),
                'percentage': float(len(group_data) / len(validation_data) * 100)
            }
            for group_name, group_data in difficulty_groups.items()
        },
        'best_features': {
            'correct': int(best_features['correct']),
            'correct_feature_idx': int(best_features['correct_feature_idx']),
            'incorrect': int(best_features['incorrect']),
            'incorrect_feature_idx': int(best_features['incorrect_feature_idx'])
        },
        'global_thresholds': {
            'correct': float(global_thresholds['correct']),
            'incorrect': float(global_thresholds['incorrect'])
        },
        'correct_preferring_results': correct_results,
        'incorrect_preferring_results': incorrect_results,
        'insights': {
            'correct_feature_trend': calculate_trend(correct_aurocs),
            'incorrect_feature_trend': calculate_trend(incorrect_aurocs),
            'most_effective_difficulty': {
                'correct': find_max_excluding_nan(correct_results, 'auroc'),
                'incorrect': find_max_excluding_nan(incorrect_results, 'auroc')
            },
            'most_effective_difficulty_f1': {
                'correct': max(correct_results.keys(), key=lambda k: correct_results[k]['f1']),
                'incorrect': max(incorrect_results.keys(), key=lambda k: incorrect_results[k]['f1'])
            }
        },
        'creation_timestamp': datetime.now().isoformat()
    }
    
    # Save results
    save_json(results, output_dir / 'difficulty_analysis_results.json')
    
    # Generate human-readable summary
    summary_lines = [
        "=" * 60,
        "PHASE 3.12: DIFFICULTY-BASED AUROC ANALYSIS SUMMARY",
        "=" * 60,
        f"\nDataset: {len(validation_data)} validation tasks",
        f"Difficulty Groups: Easy ({len(difficulty_groups['easy'])}), "
        f"Medium ({len(difficulty_groups['medium'])}), "
        f"Hard ({len(difficulty_groups['hard'])})",
        f"\nCorrect-Preferring Feature (Layer {best_features['correct']}, "
        f"Feature {best_features['correct_feature_idx']}):"
    ]
    
    for difficulty, result in correct_results.items():
        summary_lines.append(
            f"  {difficulty.capitalize()}: AUROC = {result['auroc']:.4f}, "
            f"F1 = {result['f1']:.4f} (n={result['n_samples']})"
        )
    
    summary_lines.append(
        f"\nIncorrect-Preferring Feature (Layer {best_features['incorrect']}, "
        f"Feature {best_features['incorrect_feature_idx']}):"
    )
    
    for difficulty, result in incorrect_results.items():
        summary_lines.append(
            f"  {difficulty.capitalize()}: AUROC = {result['auroc']:.4f}, "
            f"F1 = {result['f1']:.4f} (n={result['n_samples']})"
        )
    
    summary_lines.extend([
        "\nInsights:",
        f"  Correct-preferring feature trend: {results['insights']['correct_feature_trend']}",
        f"  Incorrect-preferring feature trend: {results['insights']['incorrect_feature_trend']}",
        f"  Most effective difficulty (AUROC):",
        f"    Correct-preferring: {results['insights']['most_effective_difficulty']['correct']}",
        f"    Incorrect-preferring: {results['insights']['most_effective_difficulty']['incorrect']}",
        f"  Most effective difficulty (F1):",
        f"    Correct-preferring: {results['insights']['most_effective_difficulty_f1']['correct']}",
        f"    Incorrect-preferring: {results['insights']['most_effective_difficulty_f1']['incorrect']}",
        "\n" + "=" * 60
    ])
    
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save summary to file
    with open(output_dir / 'difficulty_summary.txt', 'w') as f:
        f.write(summary_text)
    
    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()