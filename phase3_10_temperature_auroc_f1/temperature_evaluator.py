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
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

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
                        raw_tensor = torch.from_numpy(raw_activation).to(self.device)
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
                
                # Flip labels for correct-preferring features
                if feature_type == 'correct':
                    labels = 1 - labels
                
                # Calculate metrics with edge case handling
                n_positive = sum(labels)
                n_negative = len(labels) - n_positive
                
                if n_positive < 2 or n_negative < 2:
                    # Not enough samples for AUROC
                    self.logger.warning(f"Class imbalance at temp {temp} for {feature_type}: "
                                      f"pos={n_positive}, neg={n_negative}")
                    auroc = float('nan')
                    f1 = float('nan')
                else:
                    auroc = roc_auc_score(labels, features)
                    threshold = best_features[feature_type]['threshold']
                    predictions = (features > threshold).astype(int)
                    f1 = f1_score(labels, predictions)
                
                results[temp][feature_type] = {
                    'auroc': auroc,
                    'f1': f1,
                    'n_samples': len(features),  # Changed from n_tasks to n_samples
                    'n_positive': n_positive,
                    'n_negative': n_negative,
                    'feature_values': features.tolist(),
                    'labels': labels.tolist()
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
        
        # Sample distribution plot
        n_positive_correct = [results[t]['correct']['n_positive'] for t in temperatures]
        n_negative_correct = [results[t]['correct']['n_negative'] for t in temperatures]
        
        x = np.arange(len(temperatures))
        width = 0.35
        
        ax3.bar(x - width/2, n_positive_correct, width, label='Positive', alpha=0.7, color='green')
        ax3.bar(x + width/2, n_negative_correct, width, label='Negative', alpha=0.7, color='red')
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel('Number of Samples')
        ax3.set_title('Sample Distribution (Correct-preferring Feature)')
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
            lines.append(f"  Sample distribution: {results[temp]['correct']['n_positive']} positive, "
                        f"{results[temp]['correct']['n_negative']} negative")
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
        
        # Calculate degradation percentages
        correct_degradation = ((results[0.0]['correct']['auroc'] - results[1.2]['correct']['auroc']) / 
                             results[0.0]['correct']['auroc'] * 100)
        incorrect_degradation = ((results[0.0]['incorrect']['auroc'] - results[1.2]['incorrect']['auroc']) / 
                               results[0.0]['incorrect']['auroc'] * 100)
        
        lines.append(f"  Correct-preferring:   {correct_degradation:.1f}% degradation from temp 0.0 to 1.2")
        lines.append(f"  Incorrect-preferring: {incorrect_degradation:.1f}% degradation from temp 0.0 to 1.2")
        
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        
        # Determine recommended temperature range
        if correct_critical_temp is None and incorrect_critical_temp is None:
            lines.append("  - Both features remain reliable across all tested temperatures (0.0 - 1.2)")
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
        
        # Save all results
        self.save_results(results, best_features)
        
        self.logger.info("Phase 3.10 completed successfully")
        
        return results