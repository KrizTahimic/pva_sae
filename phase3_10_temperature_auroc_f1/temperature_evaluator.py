"""
Temperature-Based AUROC Analysis for PVA-SAE (Phase 3.10).

Analyzes how PVA feature effectiveness varies across different temperature settings
in Python code generation using per-sample analysis.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from common.config import Config
from common.logging import get_logger
from common.utils import detect_device, discover_latest_phase_output, format_duration
from common_simplified.helpers import save_json, load_json
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae


class TemperatureAUROCEvaluator:
    """Evaluates PVA feature performance across different temperatures using per-sample analysis."""
    
    def __init__(self, config: Config):
        """Initialize evaluator with configuration."""
        self.config = config
        self.logger = get_logger("phase3_10", phase="3.10")
        self.device = detect_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Temperature levels to analyze from config
        self.temperatures = config.phase3_10_temperatures
        self.logger.info(f"Will analyze {len(self.temperatures)} temperature levels: {self.temperatures}")
        
        # Discover dependencies
        self._discover_dependencies()
        
        # Output directory
        self.output_dir = Path(config.phase3_10_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # No aggregation tracking needed for per-sample analysis
    
    def _discover_dependencies(self) -> None:
        """Discover and load dependencies from previous phases."""
        # Phase 3.8: Best features and thresholds
        phase3_8_path = discover_latest_phase_output("3.8")
        if not phase3_8_path:
            raise ValueError("Phase 3.8 output not found. Please run Phase 3.8 first.")
        
        # If path is a file, get its parent directory
        phase3_8_path = Path(phase3_8_path)
        if phase3_8_path.is_file():
            phase3_8_dir = phase3_8_path.parent
        else:
            phase3_8_dir = phase3_8_path
            
        self.phase3_8_results_path = phase3_8_dir / "evaluation_results.json"
        if not self.phase3_8_results_path.exists():
            raise FileNotFoundError(f"Phase 3.8 results not found at {self.phase3_8_results_path}")
        
        self.logger.info(f"Found Phase 3.8 results: {self.phase3_8_results_path}")
        
        # Phase 3.5: Temperature datasets
        phase3_5_path = discover_latest_phase_output("3.5")
        if not phase3_5_path:
            raise ValueError("Phase 3.5 output not found. Please run Phase 3.5 first.")
        
        # If path is a file, get its parent directory
        phase3_5_path = Path(phase3_5_path)
        if phase3_5_path.is_file():
            self.phase3_5_dir = phase3_5_path.parent
        else:
            self.phase3_5_dir = phase3_5_path
            
        self.logger.info(f"Found Phase 3.5 data: {self.phase3_5_dir}")
        
        # Verify temperature datasets exist
        for temp in self.temperatures:
            temp_str = f"{temp:.1f}".replace(".", "_")
            temp_file = self.phase3_5_dir / f"dataset_temp_{temp_str}.parquet"
            if not temp_file.exists():
                raise FileNotFoundError(f"Temperature dataset not found: {temp_file}")
    
    def load_best_features(self) -> Dict[str, Dict]:
        """Load best features and thresholds from Phase 3.8."""
        self.logger.info("Loading Phase 3.8 best features and thresholds")
        
        results = load_json(self.phase3_8_results_path)
        
        # Extract best features for correct and incorrect
        best_features = {
            'correct': {
                'layer': results['correct_preferring_feature']['feature']['layer'],
                'feature_idx': results['correct_preferring_feature']['feature']['idx'],
                'threshold': results['correct_preferring_feature']['threshold_optimization']['optimal_threshold']
            },
            'incorrect': {
                'layer': results['incorrect_preferring_feature']['feature']['layer'],
                'feature_idx': results['incorrect_preferring_feature']['feature']['idx'],
                'threshold': results['incorrect_preferring_feature']['threshold_optimization']['optimal_threshold']
            }
        }
        
        self.logger.info(f"Best correct feature: Layer {best_features['correct']['layer']}, "
                        f"Index {best_features['correct']['feature_idx']}")
        self.logger.info(f"Best incorrect feature: Layer {best_features['incorrect']['layer']}, "
                        f"Index {best_features['incorrect']['feature_idx']}")
        
        return best_features
    
    def load_temperature_dataset(self, temperature: float) -> pd.DataFrame:
        """Load dataset for a specific temperature."""
        temp_str = f"{temperature:.1f}".replace(".", "_")
        temp_file = self.phase3_5_dir / f"dataset_temp_{temp_str}.parquet"
        
        self.logger.info(f"Loading temperature {temperature} dataset from {temp_file}")
        df = pd.read_parquet(temp_file)
        
        # Verify expected columns
        required_cols = ['task_id', 'test_passed']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.logger.info(f"Loaded {len(df)} samples for temperature {temperature}")
        return df
    
    def process_temperature_data(
        self, 
        temp_dataset: pd.DataFrame, 
        best_features: Dict[str, Any], 
        sae: Any,
        temperature: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process data for a single temperature using per-sample analysis."""
        sample_features = []
        sample_labels = []
        
        # Cache for activation values to avoid redundant loading
        activation_cache = {}
        
        # Process each row as an individual sample
        for _, row in tqdm(temp_dataset.iterrows(), total=len(temp_dataset), desc="Processing samples"):
            task_id = row['task_id']
            
            # Check cache first
            if task_id in activation_cache:
                feature_value = activation_cache[task_id]
            else:
                # Load pre-saved activation from Phase 3.5
                activation_path = (self.phase3_5_dir / "activations" / "task_activations" / 
                                 f"{task_id}_layer_{best_features['layer']}.npz")
                
                # Fail fast if activation file missing
                if not activation_path.exists():
                    raise FileNotFoundError(
                        f"Missing activation file for task {task_id} at {activation_path}. "
                        f"Phase 3.10 requires all activations from Phase 3.5 to be present."
                    )
                
                try:
                    # Load activation
                    raw_activation = np.load(activation_path)['arr_0']
                    
                    # Encode through SAE to get feature value
                    with torch.no_grad():
                        # Get SAE dtype to ensure compatibility
                        sae_dtype = next(sae.parameters()).dtype
                        raw_tensor = torch.from_numpy(raw_activation).to(dtype=sae_dtype, device=self.device)
                        # Ensure correct shape
                        if raw_tensor.ndim == 1:
                            raw_tensor = raw_tensor.unsqueeze(0)
                        sae_features = sae.encode(raw_tensor)
                        feature_value = sae_features[0, best_features['feature_idx']].item()
                    
                    # Cache the feature value for this task
                    activation_cache[task_id] = feature_value
                    
                except Exception as e:
                    self.logger.error(f"Error processing {task_id}: {str(e)}")
                    raise
            
            # Each sample has its own label
            label = int(row['test_passed'])
            
            sample_features.append(feature_value)
            sample_labels.append(label)
        
        return np.array(sample_features), np.array(sample_labels)
    
    def evaluate_across_temperatures(self, best_features: Dict[str, Dict]) -> Dict:
        """Evaluate feature performance at each temperature."""
        results = {}
        
        # Load SAEs once for reuse
        self.logger.info("Loading SAEs for feature encoding")
        sae_correct = load_gemma_scope_sae(best_features['correct']['layer'], self.device)
        sae_incorrect = load_gemma_scope_sae(best_features['incorrect']['layer'], self.device)
        
        for temp in self.temperatures:
            self.logger.info(f"\nProcessing temperature {temp}")
            
            # Load temperature dataset
            temp_data = self.load_temperature_dataset(temp)
            results[temp] = {}
            
            # Process for both feature types
            for feature_type in ['correct', 'incorrect']:
                self.logger.info(f"Evaluating {feature_type}-preferring feature")
                
                sae = sae_correct if feature_type == 'correct' else sae_incorrect
                
                features, labels = self.process_temperature_data(
                    temp_data, 
                    best_features[feature_type],
                    sae,
                    temp
                )
                
                # Count original distribution (before any flipping)
                n_correct = sum(labels)  # test_passed = 1
                n_incorrect = len(labels) - n_correct  # test_passed = 0
                
                # For AUROC: we want high feature values to predict class 1
                # Correct-preferring: high activation = correct code (already label=1)
                # Incorrect-preferring: high activation = incorrect code (need to flip)
                if feature_type == 'incorrect':
                    labels = 1 - labels  # Flip so high activation → 1 (incorrect)
                
                # Calculate metrics with edge case handling
                n_positive = sum(labels)  # After flip for AUROC
                n_negative = len(labels) - n_positive
                
                if n_positive < 2 or n_negative < 2:
                    # Not enough samples for AUROC
                    self.logger.warning(f"Class imbalance at temp {temp} for {feature_type}: "
                                      f"pos={n_positive}, neg={n_negative}")
                    auroc = float('nan')
                    f1 = float('nan')
                    fpr = np.array([0, 1])
                    tpr = np.array([0, 1])
                    precision = np.array([0, 1])
                    recall = np.array([0, 1])
                else:
                    auroc = roc_auc_score(labels, features)
                    threshold = best_features[feature_type]['threshold']
                    predictions = (features > threshold).astype(int)
                    f1 = f1_score(labels, predictions)
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(labels, features)
                    # Calculate Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(labels, features)
                
                results[temp][feature_type] = {
                    'auroc': auroc,
                    'f1': f1,
                    'n_samples': len(features),  # Changed from n_tasks to n_samples
                    'n_correct': n_correct,  # Original correct samples
                    'n_incorrect': n_incorrect,  # Original incorrect samples
                    'n_positive': n_positive,  # For AUROC (after flip if correct-preferring)
                    'n_negative': n_negative,  # For AUROC (after flip if correct-preferring)
                    'feature_values': features.tolist(),
                    'labels': labels.tolist(),
                    'fpr': fpr.tolist(),  # Store for ROC curve plotting
                    'tpr': tpr.tolist(),   # Store for ROC curve plotting
                    'precision': precision.tolist(),  # Store for PR curve plotting
                    'recall': recall.tolist()         # Store for PR curve plotting
                }
                
                self.logger.info(f"Temperature {temp}, {feature_type}: "
                               f"AUROC={auroc:.3f}, F1={f1:.3f}")
        
        # Clean up SAEs
        del sae_correct, sae_incorrect
        torch.cuda.empty_cache()
        
        return results
    
    def plot_temperature_trends(self, results: Dict) -> None:
        """Create temperature vs metric plots with enhanced visualizations."""
        temperatures = sorted(results.keys())
        
        # Extract metrics, handling NaN values
        correct_aurocs = [results[t]['correct']['auroc'] for t in temperatures]
        incorrect_aurocs = [results[t]['incorrect']['auroc'] for t in temperatures]
        correct_f1s = [results[t]['correct']['f1'] for t in temperatures]
        incorrect_f1s = [results[t]['incorrect']['f1'] for t in temperatures]
        
        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # AUROC plot
        ax1.plot(temperatures, correct_aurocs, 'b-o', label='Correct-preferring', markersize=8)
        ax1.plot(temperatures, incorrect_aurocs, 'r-s', label='Incorrect-preferring', markersize=8)
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('AUROC')
        ax1.set_title('AUROC vs Temperature')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        ax1.set_xticks(temperatures)
        
        # F1 plot
        ax2.plot(temperatures, correct_f1s, 'b-o', label='Correct-preferring', markersize=8)
        ax2.plot(temperatures, incorrect_f1s, 'r-s', label='Incorrect-preferring', markersize=8)
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score vs Temperature')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        ax2.set_xticks(temperatures)
        
        # Sample distribution plot (using original correct/incorrect counts)
        n_correct = [results[t]['correct']['n_correct'] for t in temperatures]
        n_incorrect = [results[t]['correct']['n_incorrect'] for t in temperatures]
        
        x = np.arange(len(temperatures))
        width = 0.35
        
        ax3.bar(x - width/2, n_correct, width, label='Correct (test passed)', alpha=0.7, color='green')
        ax3.bar(x + width/2, n_incorrect, width, label='Incorrect (test failed)', alpha=0.7, color='red')
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel('Number of Samples')
        ax3.set_title('Original Sample Distribution')
        ax3.set_xticks(x)
        ax3.set_xticklabels(temperatures)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Feature value distribution
        feature_data = []
        for temp in temperatures:
            feature_vals = results[temp]['correct']['feature_values']
            feature_data.append(feature_vals)
        
        ax4.boxplot(feature_data, positions=range(len(temperatures)), widths=0.6)
        ax4.set_xticklabels(temperatures)
        ax4.set_xlabel('Temperature')
        ax4.set_ylabel('Feature Value')
        ax4.set_title('Feature Value Distribution (Correct-preferring)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'temperature_trends.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        self.logger.info(f"Saved temperature trends plot to {output_path}")
    
    def plot_roc_curves(self, results: Dict) -> None:
        """Plot ROC curves for all temperatures."""
        temperatures = sorted(results.keys())
        
        # Create figure with 1x2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Create color map for temperatures (blue to red gradient)
        colors = cm.coolwarm(np.linspace(0.2, 0.8, len(temperatures)))
        
        # Plot for correct-preferring feature
        for i, temp in enumerate(temperatures):
            fpr = np.array(results[temp]['correct']['fpr'])
            tpr = np.array(results[temp]['correct']['tpr'])
            auroc = results[temp]['correct']['auroc']
            
            if not np.isnan(auroc):
                ax1.plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f'Temp {temp:.1f} (AUC={auroc:.3f})')
        
        # Add diagonal reference line
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random (AUC=0.5)')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Correct-Preferring Feature')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([-0.01, 1.01])
        ax1.set_ylim([-0.01, 1.01])
        
        # Plot for incorrect-preferring feature
        for i, temp in enumerate(temperatures):
            fpr = np.array(results[temp]['incorrect']['fpr'])
            tpr = np.array(results[temp]['incorrect']['tpr'])
            auroc = results[temp]['incorrect']['auroc']
            
            if not np.isnan(auroc):
                ax2.plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f'Temp {temp:.1f} (AUC={auroc:.3f})')
        
        # Add diagonal reference line
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random (AUC=0.5)')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves - Incorrect-Preferring Feature')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-0.01, 1.01])
        ax2.set_ylim([-0.01, 1.01])
        
        # Add overall title
        fig.suptitle('ROC Curves Across Different Temperatures', fontsize=14, y=1.02)
        
        # Add color bar to show temperature gradient
        sm = cm.ScalarMappable(cmap=cm.coolwarm, 
                               norm=plt.Normalize(vmin=min(temperatures), vmax=max(temperatures)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', 
                           pad=0.1, aspect=50, shrink=0.5)
        cbar.set_label('Temperature', fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / 'roc_curves_by_temperature.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved ROC curves plot to {output_path}")
    
    def plot_precision_recall_curves(self, results: Dict) -> None:
        """Plot Precision-Recall curves for all temperatures."""
        from sklearn.metrics import average_precision_score
        
        temperatures = sorted(results.keys())
        
        # Create figure with 1x2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Create color map for temperatures (blue to red gradient)
        colors = cm.coolwarm(np.linspace(0.2, 0.8, len(temperatures)))
        
        # Plot for correct-preferring feature
        for i, temp in enumerate(temperatures):
            precision = np.array(results[temp]['correct']['precision'])
            recall = np.array(results[temp]['correct']['recall'])
            labels = np.array(results[temp]['correct']['labels'])
            features = np.array(results[temp]['correct']['feature_values'])
            
            # Calculate AUC-PR (Average Precision Score)
            if len(np.unique(labels)) > 1:  # Check if we have both classes
                auc_pr = average_precision_score(labels, features)
                ax1.plot(recall, precision, color=colors[i], linewidth=2,
                        label=f'Temp {temp:.1f} (AUC-PR={auc_pr:.3f})')
        
        # Add baseline (random performance = positive class ratio)
        if temperatures:
            # Use first temperature to get baseline positive ratio
            first_temp = temperatures[0]
            n_positive = results[first_temp]['correct']['n_positive']
            n_total = results[first_temp]['correct']['n_samples']
            baseline = n_positive / n_total if n_total > 0 else 0.5
            ax1.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, linewidth=1,
                       label=f'Random (AP={baseline:.3f})')
        
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curves - Correct-Preferring Feature')
        ax1.legend(loc='lower left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([-0.01, 1.01])
        ax1.set_ylim([-0.01, 1.01])
        
        # Plot for incorrect-preferring feature
        for i, temp in enumerate(temperatures):
            precision = np.array(results[temp]['incorrect']['precision'])
            recall = np.array(results[temp]['incorrect']['recall'])
            labels = np.array(results[temp]['incorrect']['labels'])
            features = np.array(results[temp]['incorrect']['feature_values'])
            
            # Calculate AUC-PR (Average Precision Score)
            if len(np.unique(labels)) > 1:  # Check if we have both classes
                auc_pr = average_precision_score(labels, features)
                ax2.plot(recall, precision, color=colors[i], linewidth=2,
                        label=f'Temp {temp:.1f} (AUC-PR={auc_pr:.3f})')
        
        # Add baseline (random performance = positive class ratio)
        if temperatures:
            # Use first temperature to get baseline positive ratio
            first_temp = temperatures[0]
            n_positive = results[first_temp]['incorrect']['n_positive']
            n_total = results[first_temp]['incorrect']['n_samples']
            baseline = n_positive / n_total if n_total > 0 else 0.5
            ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, linewidth=1,
                       label=f'Random (AP={baseline:.3f})')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves - Incorrect-Preferring Feature')
        ax2.legend(loc='lower left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-0.01, 1.01])
        ax2.set_ylim([-0.01, 1.01])
        
        # Add overall title
        fig.suptitle('Precision-Recall Curves Across Different Temperatures', fontsize=14, y=1.02)
        
        # Add color bar to show temperature gradient
        sm = cm.ScalarMappable(cmap=cm.coolwarm, 
                               norm=plt.Normalize(vmin=min(temperatures), vmax=max(temperatures)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', 
                           pad=0.1, aspect=50, shrink=0.5)
        cbar.set_label('Temperature', fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / 'precision_recall_curves_by_temperature.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved Precision-Recall curves plot to {output_path}")
    
    def generate_summary(self, results: Dict, best_features: Dict[str, Dict]) -> str:
        """Generate human-readable summary of results."""
        lines = ["=" * 60]
        lines.append("PHASE 3.10: TEMPERATURE-BASED AUROC ANALYSIS")
        lines.append("=" * 60)
        lines.append("")
        
        # Feature information
        lines.append("BEST FEATURES ANALYZED:")
        lines.append(f"  Correct-preferring: Layer {best_features['correct']['layer']}, "
                    f"Feature {best_features['correct']['feature_idx']}")
        lines.append(f"  Incorrect-preferring: Layer {best_features['incorrect']['layer']}, "
                    f"Feature {best_features['incorrect']['feature_idx']}")
        lines.append("")
        
        # Methodology information
        lines.append("METHODOLOGY:")
        lines.append(f"  Analysis method: Per-sample (no aggregation)")
        lines.append(f"  Total samples per temperature: Varies by dataset")
        lines.append("")
        
        # Temperature analysis
        lines.append("TEMPERATURE ANALYSIS:")
        lines.append("")
        
        temperatures = sorted(results.keys())
        
        # Find critical temperature thresholds
        auroc_threshold = 0.7
        correct_critical_temp = None
        incorrect_critical_temp = None
        
        for temp in temperatures:
            correct_auroc = results[temp]['correct']['auroc']
            incorrect_auroc = results[temp]['incorrect']['auroc']
            
            lines.append(f"Temperature {temp}:")
            lines.append(f"  Correct-preferring:   AUROC={correct_auroc:.3f}, "
                        f"F1={results[temp]['correct']['f1']:.3f}")
            lines.append(f"  Incorrect-preferring: AUROC={incorrect_auroc:.3f}, "
                        f"F1={results[temp]['incorrect']['f1']:.3f}")
            lines.append(f"  Sample distribution: {results[temp]['correct']['n_correct']} correct (test passed), "
                        f"{results[temp]['correct']['n_incorrect']} incorrect (test failed)")
            lines.append("")
            
            # Track critical temperatures
            if correct_critical_temp is None and not np.isnan(correct_auroc) and correct_auroc < auroc_threshold:
                correct_critical_temp = temp
            if incorrect_critical_temp is None and not np.isnan(incorrect_auroc) and incorrect_auroc < auroc_threshold:
                incorrect_critical_temp = temp
        
        # Critical temperature analysis
        lines.append("CRITICAL TEMPERATURE ANALYSIS:")
        lines.append(f"(Threshold: AUROC < {auroc_threshold})")
        
        if correct_critical_temp is not None:
            lines.append(f"  Correct-preferring feature degrades at temperature {correct_critical_temp}")
        else:
            lines.append(f"  Correct-preferring feature maintains AUROC > {auroc_threshold} across all temperatures")
        
        if incorrect_critical_temp is not None:
            lines.append(f"  Incorrect-preferring feature degrades at temperature {incorrect_critical_temp}")
        else:
            lines.append(f"  Incorrect-preferring feature maintains AUROC > {auroc_threshold} across all temperatures")
        
        lines.append("")
        
        # Performance degradation analysis
        lines.append("PERFORMANCE DEGRADATION:")
        
        # Get min and max temperatures from results
        min_temp = min(temperatures)
        max_temp = max(temperatures)
        
        # Calculate degradation percentages
        correct_degradation = ((results[min_temp]['correct']['auroc'] - results[max_temp]['correct']['auroc']) / 
                             results[min_temp]['correct']['auroc'] * 100)
        incorrect_degradation = ((results[min_temp]['incorrect']['auroc'] - results[max_temp]['incorrect']['auroc']) / 
                               results[min_temp]['incorrect']['auroc'] * 100)
        
        lines.append(f"  Correct-preferring:   {correct_degradation:.1f}% degradation from temp {min_temp} to {max_temp}")
        lines.append(f"  Incorrect-preferring: {incorrect_degradation:.1f}% degradation from temp {min_temp} to {max_temp}")
        
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        
        # Determine recommended temperature range
        if correct_critical_temp is None and incorrect_critical_temp is None:
            lines.append(f"  - Both features remain reliable across all tested temperatures ({min_temp} - {max_temp})")
        elif correct_critical_temp is not None and incorrect_critical_temp is not None:
            recommended_max = min(correct_critical_temp, incorrect_critical_temp) - 0.3
            lines.append(f"  - Recommend using temperatures <= {recommended_max} for reliable PVA analysis")
        else:
            critical = correct_critical_temp or incorrect_critical_temp
            recommended_max = critical - 0.3
            lines.append(f"  - Recommend using temperatures <= {recommended_max} for reliable PVA analysis")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_results(self, results: Dict, best_features: Dict[str, Dict]) -> None:
        """Save all results to output directory."""
        # Save comprehensive JSON results
        output_data = {
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'phase': '3.10',
            'description': 'Temperature-Based AUROC Analysis',
            'temperatures_analyzed': sorted(results.keys()),
            'best_features': best_features,
            'results_by_temperature': results,
            'methodology': {
                'analysis_method': 'per_sample',
                'description': 'Each generated sample analyzed independently'
            }
        }
        
        json_path = self.output_dir / 'temperature_analysis_results.json'
        save_json(output_data, json_path)
        self.logger.info(f"Saved results to {json_path}")
        
        # Save human-readable summary
        summary = self.generate_summary(results, best_features)
        summary_path = self.output_dir / 'temperature_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)
        self.logger.info(f"Saved summary to {summary_path}")
    
    def run(self) -> Dict:
        """Run the complete temperature-based AUROC analysis."""
        self.logger.info("Starting Phase 3.10: Temperature-Based AUROC Analysis")
        self.logger.info("Using per-sample analysis (no aggregation)")
        
        # Load best features from Phase 3.8
        best_features = self.load_best_features()
        
        # Evaluate across all temperatures
        results = self.evaluate_across_temperatures(best_features)
        
        # Generate visualizations
        self.plot_temperature_trends(results)
        self.plot_roc_curves(results)
        self.plot_precision_recall_curves(results)
        
        # Save all results
        self.save_results(results, best_features)
        
        self.logger.info("Phase 3.10 completed successfully")
        
        return results