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

import argparse

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve
)

# Note: We implement our own calculate_difficulty_metrics function
# rather than reusing calculate_metrics from Phase 3.8 due to different requirements

from common.logging import get_logger
from common.utils import detect_device, ensure_directory_exists
from common.phase_discovery import discover_latest_phase_output
from common.viz_utils import handle_viz_only_mode
from common.utils import save_json, load_json
from common.sae_loader import load_sae_for_config
from common.tensor_utils import load_activation

logger = get_logger("phase3_12.difficulty_evaluator")

class Phase312Runner:
    """Standard runner for Phase 3.12: Difficulty-Based AUROC Analysis."""

    def __init__(self, config):
        """Initialize with config object."""
        self.config = config
        self.logger = get_logger("phase3_12.runner", phase="3.12")

    def run(self):
        """Run Phase 3.12 difficulty-based AUROC analysis."""
        self.logger.info("Starting Phase 3.12: Difficulty-Based AUROC Analysis for PVA-SAE")
        self.logger.info("This phase evaluates PVA features across different problem difficulty levels")
        self.logger.info("\n" + self.config.dump(phase="3.12"))

        # main() creates its own Config internally, so just call it
        return main()

def group_by_difficulty(validation_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
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
    latent_idx: int,
    latent_type: str,
    sae: torch.nn.Module,
    device: torch.device,
    temp_data: pd.DataFrame,
    phase3_5_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load activations for a specific difficulty group.

    Args:
        group_data: DataFrame with tasks for this difficulty group
        layer_num: Layer number for SAE
        latent_idx: Latent index to extract
        latent_type: 'correct' or 'incorrect'
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

        # Load raw activations from Phase 3.5 (preserves bfloat16)
        act_file = phase3_5_dir / f'activations/task_activations/{task_id}_layer_{layer_num}.safetensors'

        if not act_file.exists():
            missing_tasks.append(task_id)
            continue

        # Load activation (preserves bfloat16)
        raw_activation = load_activation(act_file, device)

        # Ensure dtype matches SAE parameters for matrix multiplication
        raw_activation = raw_activation.to(sae.W_enc.dtype)

        with torch.no_grad():
            latent_activations = sae.encode(raw_activation)

        # Extract specific latent value
        latent_activation = latent_activations[0, latent_idx].item()
        activations.append(latent_activation)
        
        # Get test result and create label
        task_results = temp_data[temp_data['task_id'] == task_id]['test_passed'].values
        if len(task_results) == 0:
            continue
            
        test_passed = task_results[0]  # Use first sample at temperature 0.0
        
        # Create label based on feature type
        if latent_type == 'correct':
            label = 1 if test_passed else 0  # Flipped for correct-predicting
        else:
            label = 0 if test_passed else 1  # Standard for incorrect-predicting
        
        labels.append(label)
    
    if missing_tasks:
        logger.warning(f"Missing activation files for {len(missing_tasks)} tasks")
    
    # Check for edge cases
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        logger.warning(f"WARNING: {latent_type}-predicting feature has imbalanced classes - "
                      f"positive: {n_positive}, negative: {n_negative}")
    elif n_positive < 5 or n_negative < 5:
        logger.warning(f"WARNING: {latent_type}-predicting feature has very few samples in one class - "
                      f"positive: {n_positive}, negative: {n_negative}")
    
    return np.array(labels), np.array(activations)

def calculate_difficulty_metrics(
    difficulty_groups: dict[str, pd.DataFrame],
    best_latents: Dict,
    global_threshold: float,
    latent_type: str,
    output_dir: Path,
    sae: torch.nn.Module,
    device: torch.device,
    temp_data: pd.DataFrame,
    phase3_5_dir: Path
) -> dict[str, Dict]:
    """Calculate AUROC and F1 for each difficulty group for a specific feature type.
    
    Args:
        difficulty_groups: Dict of difficulty groups
        best_latents: Best feature information
        global_threshold: F1-optimal threshold from Phase 3.8
        latent_type: 'correct' or 'incorrect'
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
        logger.info(f"\nEvaluating {latent_type}-predicting feature on {group_name} group:")
        
        # Get feature info
        if latent_type == 'correct':
            layer = best_latents['correct']
            latent_idx = best_latents['correct_latent_idx']
        else:
            layer = best_latents['incorrect']
            latent_idx = best_latents['incorrect_latent_idx']
        
        # Load activations for this group
        y_true, scores = load_group_activations(
            group_data, layer, latent_idx, latent_type,
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
        plt.title(f'ROC Curve - {latent_type.capitalize()}-Predicting Feature ({group_name.capitalize()} Group)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'roc_curve_{latent_type}_{group_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot confusion matrix for this group
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        
        # Adjust labels based on feature type
        if latent_type == 'correct':
            labels = ['Incorrect', 'Correct']
        else:
            labels = ['Correct', 'Incorrect']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {latent_type.capitalize()}-Predicting Feature ({group_name.capitalize()} Group)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_dir / f'confusion_matrix_{latent_type}_{group_name}.png', dpi=150, bbox_inches='tight')
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
    difficulty_groups: dict[str, pd.DataFrame],
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
    difficulty_groups: dict[str, pd.DataFrame],
    latent_type: str,
    results: Dict,
    output_dir: Path,
    best_latents: Dict,
    sae: torch.nn.Module,
    device: torch.device,
    temp_data: pd.DataFrame,
    phase3_5_dir: Path
) -> None:
    """Plot ROC curves for each difficulty group on the same plot."""
    plt.figure(figsize=(10, 8))
    
    colors = ['green', 'orange', 'red']
    layer = best_latents[latent_type]
    latent_idx = best_latents[f'{latent_type}_latent_idx']
    
    for i, (group_name, group_result) in enumerate(results.items()):
        # Re-calculate ROC curve points for plotting
        y_true, scores = load_group_activations(
            difficulty_groups[group_name], 
            layer,
            latent_idx, 
            latent_type,
            sae, device, temp_data, phase3_5_dir
        )
        
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = group_result['auroc']
        
        plt.plot(fpr, tpr, linewidth=2, color=colors[i],
                label=f'{group_name.capitalize()} (AUC = {auroc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves by Difficulty - {latent_type.capitalize()}-Predicting Feature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_curves_by_difficulty_{latent_type}.png', dpi=150, bbox_inches='tight')
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
    
    # Handle NaN values for correct-predicting
    valid_correct_idx = [i for i, val in enumerate(correct_aurocs) if not np.isnan(val)]
    valid_correct_diff = [difficulties[i] for i in valid_correct_idx]
    valid_correct_aurocs = [correct_aurocs[i] for i in valid_correct_idx]

    # Handle NaN values for incorrect-predicting
    valid_incorrect_idx = [i for i, val in enumerate(incorrect_aurocs) if not np.isnan(val)]
    valid_incorrect_diff = [difficulties[i] for i in valid_incorrect_idx]
    valid_incorrect_aurocs = [incorrect_aurocs[i] for i in valid_incorrect_idx]
    
    # Plot valid points
    if valid_correct_aurocs:
        plt.plot(valid_correct_diff, valid_correct_aurocs, marker='o', linewidth=2, markersize=8,
                 label='Correct-predicting', color='blue')
    if valid_incorrect_aurocs:
        plt.plot(valid_incorrect_diff, valid_incorrect_aurocs, marker='s', linewidth=2, markersize=8,
                 label='Incorrect-predicting', color='red')
    
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

    # Handle --viz-only mode
    if config.viz_only:
        data_path = output_dir / "difficulty_analysis_results.json"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Cannot run --viz-only: {data_path} not found. "
                f"Run phase normally first to generate data."
            )
        logger.info(f"--viz-only mode: Loading data from {data_path}")
        results = load_json(data_path)

        # Regenerate plots that don't require SAE loading
        correct_results = results['correct_predicting_results']
        incorrect_results = results['incorrect_predicting_results']

        # Regenerate metrics comparison plot
        difficulties = list(correct_results.keys())
        correct_aurocs = [correct_results[d]['auroc'] for d in difficulties]
        incorrect_aurocs = [incorrect_results[d]['auroc'] for d in difficulties]
        correct_f1s = [correct_results[d]['f1'] for d in difficulties]
        incorrect_f1s = [incorrect_results[d]['f1'] for d in difficulties]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(difficulties, correct_aurocs, 'b-o', label='Correct-predicting', markersize=8)
        ax1.plot(difficulties, incorrect_aurocs, 'r-s', label='Incorrect-predicting', markersize=8)
        ax1.set_xlabel('Difficulty Level')
        ax1.set_ylabel('AUROC')
        ax1.set_title('AUROC vs Difficulty Level')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        ax1.set_xticks(range(len(difficulties)))
        ax1.set_xticklabels([d.capitalize() for d in difficulties])
        ax2.plot(difficulties, correct_f1s, 'b-o', label='Correct-predicting', markersize=8)
        ax2.plot(difficulties, incorrect_f1s, 'r-s', label='Incorrect-predicting', markersize=8)
        ax2.set_xlabel('Difficulty Level')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score vs Difficulty Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        ax2.set_xticks(range(len(difficulties)))
        ax2.set_xticklabels([d.capitalize() for d in difficulties])
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_comparison_by_difficulty.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Regenerate AUROC trends plot
        plot_auroc_trends(correct_results, incorrect_results, output_dir)

        logger.info("Visualization regeneration complete")
        logger.info("Note: ROC curves and confusion matrices require full rerun to regenerate")
        return results

    # Phase 1: Load Dependencies and Setup
    logger.info("="*60)
    logger.info("PHASE 3.12: DIFFICULTY-BASED AUROC ANALYSIS")
    logger.info("="*60)
    
    # Load Phase 3.8 results to get best latents and thresholds
    logger.info("\nLoading Phase 3.8 results...")
    phase3_8_results = load_json(phase3_8_dir / 'evaluation_results.json')
    best_latents = {
        'correct': phase3_8_results['correct_predicting_latent']['latent']['layer'],
        'correct_latent_idx': phase3_8_results['correct_predicting_latent']['latent']['idx'],
        'incorrect': phase3_8_results['incorrect_predicting_latent']['latent']['layer'],
        'incorrect_latent_idx': phase3_8_results['incorrect_predicting_latent']['latent']['idx']
    }

    # Extract global F1-optimal thresholds from Phase 3.8
    global_thresholds = {
        'correct': phase3_8_results['correct_predicting_latent']['threshold_optimization']['optimal_threshold'],
        'incorrect': phase3_8_results['incorrect_predicting_latent']['threshold_optimization']['optimal_threshold']
    }

    logger.info(f"Best correct-predicting latent: idx {best_latents['correct_latent_idx']} "
               f"at layer {best_latents['correct']} (threshold: {global_thresholds['correct']:.4f})")
    logger.info(f"Best incorrect-predicting latent: idx {best_latents['incorrect_latent_idx']} "
               f"at layer {best_latents['incorrect']} (threshold: {global_thresholds['incorrect']:.4f})")
    
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
    
    # Phase 3: Evaluate Correct-Predicting Feature
    logger.info("\n" + "="*60)
    logger.info("EVALUATING CORRECT-PREDICTING FEATURE ACROSS DIFFICULTY LEVELS")
    logger.info("="*60)

    # Load SAE for correct-predicting feature
    correct_layer = best_latents['correct']
    sae_correct = load_sae_for_config(config, correct_layer, device)
    logger.info(f"Loaded SAE for layer {correct_layer} on {device}")
    
    correct_results = calculate_difficulty_metrics(
        difficulty_groups, 
        best_latents, 
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
    
    # Store correct results for later combined visualization
    
    # Phase 4: Evaluate Incorrect-Predicting Feature
    logger.info("\n" + "="*60)
    logger.info("EVALUATING INCORRECT-PREDICTING FEATURE ACROSS DIFFICULTY LEVELS")
    logger.info("="*60)

    # Load SAE for incorrect-predicting feature
    incorrect_layer = best_latents['incorrect']
    sae_incorrect = load_sae_for_config(config, incorrect_layer, device)
    logger.info(f"Loaded SAE for layer {incorrect_layer} on {device}")
    
    incorrect_results = calculate_difficulty_metrics(
        difficulty_groups, 
        best_latents, 
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
    
    # Store incorrect results for later combined visualization
    
    # Phase 5: Comparative Analysis and Results
    logger.info("\n" + "="*60)
    logger.info("GENERATING COMPARATIVE ANALYSIS")
    logger.info("="*60)

    # Generate combined line plot (like Phase 3.11 style)
    difficulties = list(correct_results.keys())
    correct_aurocs = [correct_results[d]['auroc'] for d in difficulties]
    incorrect_aurocs = [incorrect_results[d]['auroc'] for d in difficulties]
    correct_f1s = [correct_results[d]['f1'] for d in difficulties]
    incorrect_f1s = [incorrect_results[d]['f1'] for d in difficulties]

    # Create combined line plot for both metrics (Phase 3.11 style)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # AUROC plot - combined lines
    ax1.plot(difficulties, correct_aurocs, 'b-o', label='Correct-predicting', markersize=8)
    ax1.plot(difficulties, incorrect_aurocs, 'r-s', label='Incorrect-predicting', markersize=8)
    ax1.set_xlabel('Difficulty Level')
    ax1.set_ylabel('AUROC')
    ax1.set_title('AUROC vs Difficulty Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(range(len(difficulties)))
    ax1.set_xticklabels([d.capitalize() for d in difficulties])

    # F1 plot - combined lines
    ax2.plot(difficulties, correct_f1s, 'b-o', label='Correct-predicting', markersize=8)
    ax2.plot(difficulties, incorrect_f1s, 'r-s', label='Incorrect-predicting', markersize=8)
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Difficulty Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(range(len(difficulties)))
    ax2.set_xticklabels([d.capitalize() for d in difficulties])

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison_by_difficulty.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate additional comparative visualizations
    # Note: We need to reload SAEs since they were deleted after individual analyses
    sae_correct = load_sae_for_config(config, best_latents['correct'], device)
    plot_roc_curves_by_difficulty(difficulty_groups, 'correct', correct_results, output_dir, 
                                   best_latents, sae_correct, device, temp_data, phase3_5_dir)
    del sae_correct
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    sae_incorrect = load_sae_for_config(config, best_latents['incorrect'], device)
    plot_roc_curves_by_difficulty(difficulty_groups, 'incorrect', incorrect_results, output_dir,
                                   best_latents, sae_incorrect, device, temp_data, phase3_5_dir)
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
        'best_latents': {
            'correct': int(best_latents['correct']),
            'correct_latent_idx': int(best_latents['correct_latent_idx']),
            'incorrect': int(best_latents['incorrect']),
            'incorrect_latent_idx': int(best_latents['incorrect_latent_idx'])
        },
        'global_thresholds': {
            'correct': float(global_thresholds['correct']),
            'incorrect': float(global_thresholds['incorrect'])
        },
        'correct_predicting_results': correct_results,
        'incorrect_predicting_results': incorrect_results,
        'insights': {
            'correct_latent_trend': calculate_trend(correct_aurocs),
            'incorrect_latent_trend': calculate_trend(incorrect_aurocs),
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
        f"\nCorrect-Predicting Latent (Layer {best_latents['correct']}, "
        f"Latent {best_latents['correct_latent_idx']}):"
    ]

    for difficulty, result in correct_results.items():
        summary_lines.append(
            f"  {difficulty.capitalize()}: AUROC = {result['auroc']:.4f}, "
            f"F1 = {result['f1']:.4f} (n={result['n_samples']})"
        )

    summary_lines.append(
        f"\nIncorrect-Predicting Latent (Layer {best_latents['incorrect']}, "
        f"Latent {best_latents['incorrect_latent_idx']}):"
    )
    
    for difficulty, result in incorrect_results.items():
        summary_lines.append(
            f"  {difficulty.capitalize()}: AUROC = {result['auroc']:.4f}, "
            f"F1 = {result['f1']:.4f} (n={result['n_samples']})"
        )
    
    summary_lines.extend([
        "\nInsights:",
        f"  Correct-predicting latent trend: {results['insights']['correct_latent_trend']}",
        f"  Incorrect-predicting latent trend: {results['insights']['incorrect_latent_trend']}",
        f"  Most effective difficulty (AUROC):",
        f"    Correct-predicting: {results['insights']['most_effective_difficulty']['correct']}",
        f"    Incorrect-predicting: {results['insights']['most_effective_difficulty']['incorrect']}",
        f"  Most effective difficulty (F1):",
        f"    Correct-predicting: {results['insights']['most_effective_difficulty_f1']['correct']}",
        f"    Incorrect-predicting: {results['insights']['most_effective_difficulty_f1']['incorrect']}",
        "\n" + "=" * 60
    ])
    
    summary_text = "\n".join(summary_lines)
    logger.info(summary_text)

    # Save summary to file
    with open(output_dir / 'difficulty_summary.txt', 'w') as f:
        f.write(summary_text)

    logger.info(f"\nAll results saved to {output_dir}")

    # Write phase_output.json manifest
    from common.phase_discovery import write_phase_output

    write_phase_output(
        phase="3.12",
        outputs={
            "primary": "difficulty_analysis_results.json",
            "summary": "difficulty_summary.txt",
            "distribution_plot": "difficulty_distribution.png",
            "metrics_comparison": "metrics_comparison_by_difficulty.png",
        },
        config=config,
        output_dir=str(output_dir),
        dependencies={
            "3.5": str(phase3_5_dir),
        },
        config_keys=['model_name', 'dataset_name']
    )
    logger.info(f"Saved phase_output.json manifest to {output_dir}")

if __name__ == "__main__":
    main()