"""
Attention Pattern Analyzer for Phase 6.3.

Analyzes how attention patterns change when steering model behavior with SAE features.
Compares attention captured at the final prompt token before and after applying SAE steering.
Generates visualizations and statistical analyses to understand mechanistic changes.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from common.logging import get_logger
from common.utils import (
    discover_latest_phase_output, 
    ensure_directory_exists,
    detect_device
)
from common.config import Config
from common_simplified.helpers import load_json, save_json

logger = get_logger("phase6_3.attention_analyzer")


class AttentionAnalyzer:
    """Analyze attention patterns from baseline (Phase 3.5) and steered (Phase 4.8) conditions."""
    
    def __init__(self, config: Config):
        """Initialize with configuration, discover dependencies."""
        self.config = config
        self.device = detect_device()
        
        # Output directories
        self.output_dir = Path(config.phase6_3_output_dir)
        ensure_directory_exists(self.output_dir)
        
        self.visualizations_dir = self.output_dir / "visualizations"
        ensure_directory_exists(self.visualizations_dir)
        
        self.statistical_dir = self.output_dir / "statistical_results"
        ensure_directory_exists(self.statistical_dir)
        
        # Load dependencies
        self._load_pva_features()
        self._discover_phase_directories()
        
        # Model configuration (Gemma-2-2b)
        self.n_heads = 8  # From gemma_config.txt
        
        logger.info("AttentionAnalyzer initialized successfully")
        
    def _load_pva_features(self) -> None:
        """Load best PVA features from Phase 2.5."""
        # Discover Phase 2.5 output
        phase2_5_output = discover_latest_phase_output("2.5")
        if not phase2_5_output:
            raise FileNotFoundError("Phase 2.5 output not found. Please run Phase 2.5 first.")
        
        phase2_5_dir = Path(phase2_5_output).parent
        
        # Load best features
        best_features_path = phase2_5_dir / "top_20_features.json"
        if not best_features_path.exists():
            raise FileNotFoundError(f"Best features not found at {best_features_path}")
        
        best_features = load_json(best_features_path)
        self.best_correct_feature = best_features['correct'][0]  # First element has highest score
        self.best_incorrect_feature = best_features['incorrect'][0]  # First element has highest score
        
        # Extract layer indices
        self.best_correct_layer = self.best_correct_feature['layer']
        self.best_incorrect_layer = self.best_incorrect_feature['layer']
        
        logger.info(f"Loaded best features - Correct: Layer {self.best_correct_layer}, Incorrect: Layer {self.best_incorrect_layer}")
        
    def _discover_phase_directories(self) -> None:
        """Discover Phase 3.5 and Phase 4.8 output directories."""
        # Discover Phase 3.5 (baseline attention)
        phase3_5_output = discover_latest_phase_output("3.5")
        if not phase3_5_output:
            raise FileNotFoundError("Phase 3.5 output not found. Please run Phase 3.5 first.")
        
        self.phase3_5_dir = Path(phase3_5_output).parent
        self.baseline_attention_dir = self.phase3_5_dir / "activations" / "attention_patterns"
        
        if not self.baseline_attention_dir.exists():
            raise FileNotFoundError(f"Baseline attention patterns not found at {self.baseline_attention_dir}")
        
        # Discover Phase 4.8 (steered attention)
        phase4_8_output = discover_latest_phase_output("4.8")
        if not phase4_8_output:
            raise FileNotFoundError("Phase 4.8 output not found. Please run Phase 4.8 first.")
        
        self.phase4_8_dir = Path(phase4_8_output).parent
        self.steered_attention_dir = self.phase4_8_dir / "attention_patterns"
        
        if not self.steered_attention_dir.exists():
            logger.warning(f"Steered attention patterns not found at {self.steered_attention_dir}")
            # Check alternative location
            self.steered_attention_dir = self.phase4_8_dir
        
        logger.info(f"Discovered Phase 3.5 dir: {self.phase3_5_dir}")
        logger.info(f"Discovered Phase 4.8 dir: {self.phase4_8_dir}")
        
    def run(self) -> Dict[str, Any]:
        """Main analysis pipeline."""
        logger.info("Starting Phase 6.3: Attention Pattern Analysis")
        
        # 1. Load attention data
        logger.info("Loading attention data from Phases 3.5 and 4.8...")
        attention_data = self.load_attention_data()
        
        # 2. Compute attention differences
        logger.info("Computing attention differences...")
        differences_correct = self.compute_differences(attention_data, 'correct')
        differences_incorrect = self.compute_differences(attention_data, 'incorrect')
        
        # 3. Statistical analysis
        logger.info("Performing statistical analysis...")
        stats_correct = self.compute_statistical_significance(differences_correct)
        stats_incorrect = self.compute_statistical_significance(differences_incorrect)
        
        # 4. Generate visualizations
        logger.info("Creating visualizations...")
        self.create_all_visualizations(attention_data, differences_correct, differences_incorrect)
        
        # 5. Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_tasks_analyzed': len(attention_data),
            'layers_analyzed': {
                'correct': self.best_correct_layer,
                'incorrect': self.best_incorrect_layer
            },
            'statistical_results': {
                'correct_steering': stats_correct,
                'incorrect_steering': stats_incorrect
            }
        }
        
        self.save_analysis_results(results)
        
        logger.info("✅ Phase 6.3 completed successfully")
        return results
        
    def load_attention_data(self) -> Dict[str, Dict]:
        """Load attention patterns from Phase 3.5 and Phase 4.8."""
        attention_data = {}
        
        # Get task IDs from validation split
        phase0_1_output = discover_latest_phase_output("0.1")
        if not phase0_1_output:
            raise FileNotFoundError("Phase 0.1 output not found")
        
        validation_path = Path(phase0_1_output).parent / "validation_mbpp.parquet"
        validation_df = pd.read_parquet(validation_path)
        
        # Load attention for each task
        for task_id in tqdm(validation_df['task_id'], desc="Loading attention data"):
            task_data = {}
            
            # Load baseline attention from Phase 3.5
            baseline_attention = self._load_task_attention(
                self.baseline_attention_dir, 
                task_id, 
                'baseline'
            )
            if baseline_attention:
                task_data['baseline'] = baseline_attention
            
            # Load steered attention from Phase 4.8 (both correct and incorrect)
            correct_dir = self.steered_attention_dir / "correct_steering"
            if correct_dir.exists():
                steered_correct = self._load_task_attention(
                    correct_dir,
                    task_id,
                    'correct'
                )
                if steered_correct:
                    task_data['steered_correct'] = steered_correct
            
            incorrect_dir = self.steered_attention_dir / "incorrect_steering"
            if incorrect_dir.exists():
                steered_incorrect = self._load_task_attention(
                    incorrect_dir,
                    task_id,
                    'incorrect'
                )
                if steered_incorrect:
                    task_data['steered_incorrect'] = steered_incorrect
            
            # Only include tasks with complete data
            if 'baseline' in task_data:
                attention_data[task_id] = task_data
        
        logger.info(f"Loaded attention data for {len(attention_data)} tasks")
        return attention_data
        
    def _load_task_attention(self, attention_dir: Path, task_id: str, condition: str) -> Optional[Dict]:
        """Load attention patterns for a specific task."""
        # Try different file naming patterns
        patterns = [
            f"{task_id}_layer_*_attention.npz",
            f"{task_id}_attention.npz",
            f"task_{task_id}_layer_*_attention.npz"
        ]
        
        for pattern in patterns:
            files = list(attention_dir.glob(pattern))
            if files:
                # Load the first matching file
                data = np.load(files[0], allow_pickle=True)
                
                # Get raw attention data
                raw_attention = data['raw_attention'] if 'raw_attention' in data else data['attention']
                
                # Convert float16 to float32 for compatibility with linalg operations
                if raw_attention.dtype == np.float16:
                    raw_attention = raw_attention.astype(np.float32)
                
                return {
                    'raw': raw_attention,
                    'boundaries': data['boundaries'].item() if 'boundaries' in data else {},
                    'layer': data['layer'].item() if 'layer' in data else None
                }
        
        return None
        
    def aggregate_to_3_bins(self, attention_tensor: np.ndarray, boundaries: Dict) -> Dict:
        """
        Aggregate raw attention into 3 bins based on section boundaries.
        
        Args:
            attention_tensor: [n_heads, sequence_length] - raw attention
            boundaries: Dict with 'problem_end', 'test_end' indices
        
        Returns:
            Dict with aggregated attention per section
        """
        # Handle missing boundaries
        if not boundaries or 'problem_end' not in boundaries:
            # Fallback: divide sequence into thirds
            seq_len = attention_tensor.shape[1]
            boundaries = {
                'problem_end': seq_len // 3,
                'test_end': 2 * seq_len // 3,
                'total_length': seq_len
            }
        
        # Sum attention within each section
        section_attention = {
            'problem': attention_tensor[:, :boundaries['problem_end']].sum(axis=-1),
            'tests': attention_tensor[:, boundaries['problem_end']:boundaries['test_end']].sum(axis=-1),
            'solution_marker': attention_tensor[:, boundaries['test_end']:].sum(axis=-1)
        }
        
        # Compute normalized percentages
        total_per_head = np.array([
            section_attention['problem'],
            section_attention['tests'],
            section_attention['solution_marker']
        ]).sum(axis=0)
        
        # Avoid division by zero
        total_per_head = np.where(total_per_head > 0, total_per_head, 1.0)
        
        section_percentages = {
            section: (attention / total_per_head * 100)
            for section, attention in section_attention.items()
        }
        
        return {
            'raw_sums': section_attention,
            'percentages': section_percentages,
            'normalized': {  # Length-normalized attention
                'problem': section_attention['problem'] / max(boundaries.get('problem_end', 1), 1),
                'tests': section_attention['tests'] / max(boundaries.get('test_end', boundaries.get('problem_end', 1)) - boundaries.get('problem_end', 0), 1),
                'solution_marker': section_attention['solution_marker'] / max(boundaries.get('total_length', 1) - boundaries.get('test_end', 0), 1)
            }
        }
        
    def compute_differences(self, attention_data: Dict, steering_type: str) -> Dict[str, np.ndarray]:
        """
        Compare attention patterns between baseline and steered generations.
        
        Args:
            attention_data: Dict with baseline and steered attention
            steering_type: 'correct' or 'incorrect'
        
        Returns:
            Dict mapping task_id to attention differences
        """
        differences = {}
        
        for task_id, task_data in attention_data.items():
            if 'baseline' not in task_data:
                continue
            
            steered_key = f'steered_{steering_type}'
            if steered_key not in task_data:
                continue
            
            # Aggregate both baseline and steered
            baseline_agg = self.aggregate_to_3_bins(
                task_data['baseline']['raw'],
                task_data['baseline']['boundaries']
            )
            
            steered_agg = self.aggregate_to_3_bins(
                task_data[steered_key]['raw'],
                task_data[steered_key]['boundaries']
            )
            
            # Compute differences in percentages
            diff = {}
            for section in ['problem', 'tests', 'solution_marker']:
                diff[section] = (steered_agg['percentages'][section] - 
                               baseline_agg['percentages'][section])
            
            differences[task_id] = diff
        
        return differences
        
    def compute_statistical_significance(self, attention_differences: Dict[str, Dict]) -> Dict:
        """
        Test if steering produces statistically significant attention changes.
        
        Uses paired t-tests to compare baseline vs steered for each section.
        """
        results = {}
        
        # Organize differences by section
        sections = ['problem', 'tests', 'solution_marker']
        
        for section in sections:
            # Collect differences across all tasks and heads
            all_diffs = []
            for task_diffs in attention_differences.values():
                if section in task_diffs:
                    # Flatten across heads
                    all_diffs.extend(task_diffs[section].flatten())
            
            if len(all_diffs) > 0:
                # Test if mean difference is significantly different from 0
                all_diffs = np.array(all_diffs)
                t_stat, p_value = stats.ttest_1samp(all_diffs, 0)
                
                results[section] = {
                    'mean_change': float(np.mean(all_diffs)),
                    'std_change': float(np.std(all_diffs)),
                    'median_change': float(np.median(all_diffs)),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'n_samples': len(all_diffs)
                }
            else:
                results[section] = {
                    'mean_change': 0.0,
                    'std_change': 0.0,
                    'median_change': 0.0,
                    't_statistic': 0.0,
                    'p_value': 1.0,
                    'significant': False,
                    'n_samples': 0
                }
        
        return results
        
    def create_all_visualizations(self, attention_data: Dict, 
                                 differences_correct: Dict, 
                                 differences_incorrect: Dict) -> None:
        """Create all 8 main visualizations."""
        
        # 1. Attention distribution stacked bar chart
        self.create_attention_distribution_chart(attention_data)
        
        # 2. Head-level attention heatmap
        self.create_head_attention_heatmap(attention_data)
        
        # 3. Attention change delta plots
        self.create_attention_delta_plots(differences_correct, differences_incorrect)
        
        # 4. Statistical significance table
        self.create_significance_table(differences_correct, differences_incorrect)
        
        # 5. Attention transformation scatter plots
        self.create_attention_transformation_scatter(attention_data, 'correct')
        self.create_attention_transformation_scatter(attention_data, 'incorrect')
        
        # 6. Head-specific transformation plot
        self.create_head_specific_transformation_plot(attention_data, 'correct')
        self.create_head_specific_transformation_plot(attention_data, 'incorrect')
        
        # 7. Test last token transformation plot (new)
        self.create_test_last_token_transformation_plot(attention_data, 'correct')
        self.create_test_last_token_transformation_plot(attention_data, 'incorrect')
        
        # 8. Head attention change bar charts
        self.create_head_attention_change_bars(attention_data, 'correct')
        self.create_head_attention_change_bars(attention_data, 'incorrect')
        
        # 9. Comparative head changes
        self.create_comparative_head_changes(attention_data)
        
        logger.info(f"Created all visualizations in {self.visualizations_dir}")
        
    def create_attention_distribution_chart(self, attention_data: Dict) -> None:
        """Create stacked bar chart showing attention distribution across 3 sections."""
        # Aggregate distributions across all tasks
        distributions = self._aggregate_distributions(attention_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        conditions = ['Baseline', 'Correct\nSteering', 'Incorrect\nSteering']
        condition_keys = ['baseline', 'correct', 'incorrect']
        x = np.arange(len(conditions))
        
        # Get heights for each section
        problem_heights = []
        test_heights = []
        solution_heights = []
        
        for key in condition_keys:
            if key in distributions:
                problem_heights.append(distributions[key]['problem'])
                test_heights.append(distributions[key]['tests'])
                solution_heights.append(distributions[key]['solution_marker'])
            else:
                problem_heights.append(0)
                test_heights.append(0)
                solution_heights.append(0)
        
        # Stack the bars
        ax.bar(x, problem_heights, label='Problem Description', color='#8dd3c7')
        ax.bar(x, test_heights, bottom=problem_heights, label='Test Cases', color='#ff8c00')
        ax.bar(x, solution_heights, 
               bottom=np.array(problem_heights) + np.array(test_heights),
               label='Solution Marker', color='#bebada')
        
        ax.set_ylabel('Attention Distribution (%)')
        ax.set_title('Attention Focus Across Prompt Sections')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'attention_distribution.png', dpi=150)
        plt.close()
        
    def create_head_attention_heatmap(self, attention_data: Dict) -> None:
        """Heatmap showing each head's attention to each section."""
        # Create matrix [n_heads, 9] for 3 sections × 3 conditions
        attention_matrix = self._build_head_attention_matrix(attention_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(attention_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        
        # Labels and formatting
        ax.set_xticks(np.arange(9))
        ax.set_xticklabels(['Prob', 'Test', 'Sol'] * 3)
        ax.set_yticks(np.arange(self.n_heads))
        ax.set_yticklabels([f'Head {i}' for i in range(self.n_heads)])
        
        # Add condition separators
        ax.axvline(x=2.5, color='white', linewidth=2)
        ax.axvline(x=5.5, color='white', linewidth=2)
        
        # Add condition labels
        ax.text(1, -1, 'Baseline', ha='center', fontweight='bold')
        ax.text(4, -1, 'Correct Steering', ha='center', fontweight='bold')
        ax.text(7, -1, 'Incorrect Steering', ha='center', fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Attention Score (%)')
        plt.title('Per-Head Attention Distribution')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'head_attention_heatmap.png', dpi=150)
        plt.close()
        
    def create_attention_delta_plots(self, differences_correct: Dict, differences_incorrect: Dict) -> None:
        """Show how steering changes attention to each section."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate average changes
        correct_deltas = self._calculate_average_deltas(differences_correct)
        incorrect_deltas = self._calculate_average_deltas(differences_incorrect)
        
        sections = ['Problem\nDesc.', 'Test\nCases', 'Solution\nMarker']
        x = np.arange(len(sections))
        
        # Calculate common y-axis limits for both plots
        all_values = []
        all_values.extend(correct_deltas['means'])
        all_values.extend(incorrect_deltas['means'])
        # Add error bars to determine full range
        all_values.extend([m + s for m, s in zip(correct_deltas['means'], correct_deltas['stds'])])
        all_values.extend([m - s for m, s in zip(correct_deltas['means'], correct_deltas['stds'])])
        all_values.extend([m + s for m, s in zip(incorrect_deltas['means'], incorrect_deltas['stds'])])
        all_values.extend([m - s for m, s in zip(incorrect_deltas['means'], incorrect_deltas['stds'])])
        
        # Set symmetric limits around zero with some padding
        y_max = max(abs(min(all_values)), abs(max(all_values))) * 1.1
        y_limits = (-y_max, y_max)
        
        # Plot correct steering effects
        colors = ['blue' if d > 0 else 'red' for d in correct_deltas['means']]
        bars1 = ax1.bar(x, correct_deltas['means'], yerr=correct_deltas['stds'],
                        capsize=5, color=colors, alpha=0.6)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Attention Change (%)')
        ax1.set_title('Correct Steering Effect')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sections)
        ax1.set_ylim(y_limits)  # Apply common y-axis limits
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot incorrect steering effects
        colors = ['blue' if d > 0 else 'red' for d in incorrect_deltas['means']]
        bars2 = ax2.bar(x, incorrect_deltas['means'], yerr=incorrect_deltas['stds'],
                        capsize=5, color=colors, alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Attention Change (%)')
        ax2.set_title('Incorrect Steering Effect')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sections)
        ax2.set_ylim(y_limits)  # Apply common y-axis limits
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Attention Redistribution Due to Steering')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'attention_delta_plots.png', dpi=150)
        plt.close()
        
    def create_significance_table(self, differences_correct: Dict, differences_incorrect: Dict) -> None:
        """Table showing statistical significance of attention changes."""
        # Compute statistics
        stats_correct = self.compute_statistical_significance(differences_correct)
        stats_incorrect = self.compute_statistical_significance(differences_incorrect)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # Build table data
        headers = ['Section', 'Correct Δ (%)', 'p-value', 'Sig?', 'Incorrect Δ (%)', 'p-value', 'Sig?']
        table_data = []
        
        for section in ['problem', 'tests', 'solution_marker']:
            section_name = section.replace('_', ' ').title()
            row = [
                section_name,
                f"{stats_correct[section]['mean_change']:.2f}",
                f"{stats_correct[section]['p_value']:.4f}",
                '✓' if stats_correct[section]['significant'] else '✗',
                f"{stats_incorrect[section]['mean_change']:.2f}",
                f"{stats_incorrect[section]['p_value']:.4f}",
                '✓' if stats_incorrect[section]['significant'] else '✗'
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight significant results
        for i, row in enumerate(table_data, 1):
            if row[3] == '✓':  # Correct steering significant
                table[(i, 1)].set_facecolor('#c6efce')
                table[(i, 2)].set_facecolor('#c6efce')
                table[(i, 3)].set_facecolor('#c6efce')
            if row[6] == '✓':  # Incorrect steering significant
                table[(i, 4)].set_facecolor('#ffc7ce')
                table[(i, 5)].set_facecolor('#ffc7ce')
                table[(i, 6)].set_facecolor('#ffc7ce')
        
        plt.title('Statistical Significance of Attention Changes', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'significance_table.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def create_attention_transformation_scatter(self, attention_data: Dict, steering_type: str) -> None:
        """Scatter plot comparing baseline vs steered attention scores."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sections = ['problem', 'tests', 'solution_marker']
        section_labels = ['Problem Description', 'Test Cases', 'Solution Marker']
        colors = ['#8dd3c7', '#ff8c00', '#bebada']
        
        for idx, (ax, section, label, color) in enumerate(zip(axes, sections, section_labels, colors)):
            # Collect all attention scores for this section
            baseline_scores = []
            steered_scores = []
            
            steered_key = f'steered_{steering_type}'
            
            for task_id, task_data in attention_data.items():
                if 'baseline' not in task_data or steered_key not in task_data:
                    continue
                
                # Get aggregated attention
                baseline_agg = self.aggregate_to_3_bins(
                    task_data['baseline']['raw'], 
                    task_data['baseline']['boundaries']
                )
                steered_agg = self.aggregate_to_3_bins(
                    task_data[steered_key]['raw'],
                    task_data[steered_key]['boundaries']
                )
                
                # Collect scores for all heads
                baseline_scores.extend(baseline_agg['percentages'][section])
                steered_scores.extend(steered_agg['percentages'][section])
            
            if len(baseline_scores) == 0:
                continue
            
            # Create scatter plot
            ax.scatter(baseline_scores, steered_scores, alpha=0.5, s=20, c=color)
            
            # Add diagonal reference line (y=x)
            max_val = max(max(baseline_scores), max(steered_scores))
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='No change')
            
            # Add trend line
            if len(baseline_scores) > 1:
                z = np.polyfit(baseline_scores, steered_scores, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(baseline_scores), max(baseline_scores), 100)
                ax.plot(x_trend, p(x_trend), 'r-', alpha=0.5, linewidth=2, 
                       label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            
            # Formatting
            ax.set_xlabel('Baseline Attention (%)')
            ax.set_ylabel(f'{steering_type.title()} Steered Attention (%)')
            ax.set_title(label)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Attention Transformation: Baseline → {steering_type.title()} Steering')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f'transformation_scatter_{steering_type}.png', dpi=150)
        plt.close()
        
    def create_head_specific_transformation_plot(self, attention_data: Dict, steering_type: str) -> None:
        """Detailed scatter plot showing each head's transformation separately."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        steered_key = f'steered_{steering_type}'
        
        for head_idx in range(self.n_heads):
            ax = axes[head_idx]
            
            # Collect scores for this specific head
            baseline_problem, baseline_tests, baseline_solution = [], [], []
            steered_problem, steered_tests, steered_solution = [], [], []
            
            for task_id, task_data in attention_data.items():
                if 'baseline' not in task_data or steered_key not in task_data:
                    continue
                
                baseline_agg = self.aggregate_to_3_bins(
                    task_data['baseline']['raw'],
                    task_data['baseline']['boundaries']
                )
                steered_agg = self.aggregate_to_3_bins(
                    task_data[steered_key]['raw'],
                    task_data[steered_key]['boundaries']
                )
                
                # Get scores for this head only
                if head_idx < len(baseline_agg['percentages']['problem']):
                    baseline_problem.append(baseline_agg['percentages']['problem'][head_idx])
                    baseline_tests.append(baseline_agg['percentages']['tests'][head_idx])
                    baseline_solution.append(baseline_agg['percentages']['solution_marker'][head_idx])
                    
                    steered_problem.append(steered_agg['percentages']['problem'][head_idx])
                    steered_tests.append(steered_agg['percentages']['tests'][head_idx])
                    steered_solution.append(steered_agg['percentages']['solution_marker'][head_idx])
            
            # Plot all three sections with different colors
            if baseline_problem:
                ax.scatter(baseline_problem, steered_problem, alpha=0.6, s=15, 
                          c='#8dd3c7', label='Problem')
            if baseline_tests:
                ax.scatter(baseline_tests, steered_tests, alpha=0.6, s=15,
                          c='#ff8c00', label='Tests')
            if baseline_solution:
                ax.scatter(baseline_solution, steered_solution, alpha=0.6, s=15,
                          c='#bebada', label='Solution')
            
            # Add diagonal
            max_val = 100
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
            
            ax.set_xlabel('Baseline %')
            ax.set_ylabel('Steered %')
            ax.set_title(f'Head {head_idx}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            
            if head_idx == 0:
                ax.legend(loc='upper left', fontsize=7)
        
        plt.suptitle(f'Per-Head Attention Transformation Analysis - {steering_type.title()} Steering')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f'head_specific_transformations_{steering_type}.png', dpi=150)
        plt.close()
        
    def create_test_last_token_transformation_plot(self, attention_data: Dict, steering_type: str) -> None:
        """Scatter plot showing attention to last token of test cases for each head."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        steered_key = f'steered_{steering_type}'
        
        # Better color palette (avoiding yellow)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                  '#9467bd', '#8c564b', '#e377c2', '#17becf']
        
        for head_idx in range(self.n_heads):
            ax = axes[head_idx]
            
            # Collect attention values at last test token for this head
            baseline_values = []
            steered_values = []
            
            for task_id, task_data in attention_data.items():
                if 'baseline' not in task_data or steered_key not in task_data:
                    continue
                
                # Get boundaries
                baseline_boundaries = task_data['baseline']['boundaries']
                steered_boundaries = task_data[steered_key]['boundaries']
                
                # Skip if boundaries are missing
                if not baseline_boundaries or 'test_end' not in baseline_boundaries:
                    continue
                if not steered_boundaries or 'test_end' not in steered_boundaries:
                    continue
                
                # Get raw attention tensors
                baseline_raw = task_data['baseline']['raw']  # [n_heads, seq_len]
                steered_raw = task_data[steered_key]['raw']   # [n_heads, seq_len]
                
                # Extract attention at last test token position
                baseline_test_end = baseline_boundaries['test_end'] - 1  # Last token of tests
                steered_test_end = steered_boundaries['test_end'] - 1
                
                # Ensure indices are valid
                if baseline_test_end >= 0 and baseline_test_end < baseline_raw.shape[1]:
                    if head_idx < baseline_raw.shape[0]:
                        baseline_values.append(baseline_raw[head_idx, baseline_test_end])
                
                if steered_test_end >= 0 and steered_test_end < steered_raw.shape[1]:
                    if head_idx < steered_raw.shape[0]:
                        steered_values.append(steered_raw[head_idx, steered_test_end])
            
            # Only plot if we have matching pairs
            n_points = min(len(baseline_values), len(steered_values))
            if n_points > 0:
                baseline_plot = baseline_values[:n_points]
                steered_plot = steered_values[:n_points]
                
                # Plot all individual data points
                ax.scatter(baseline_plot, steered_plot, alpha=0.6, s=20, 
                          c=colors[head_idx], label=f'Head {head_idx}')
                
                # Add diagonal reference line
                all_values = baseline_plot + steered_plot
                if all_values:
                    max_val = max(all_values) * 1.1
                    min_val = min(all_values) * 0.9
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1)
                
                # Add trend line (optional - using numpy polyfit)
                if len(baseline_plot) > 1:
                    z = np.polyfit(baseline_plot, steered_plot, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(baseline_plot), max(baseline_plot), 100)
                    ax.plot(x_trend, p(x_trend), color=colors[head_idx], alpha=0.5, linewidth=2)
            
            # Formatting
            ax.set_xlabel('Baseline Attention', fontsize=9)
            ax.set_ylabel('Steered Attention', fontsize=9)
            ax.set_title(f'Head {head_idx}', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add count of data points
            ax.text(0.05, 0.95, f'n={n_points}', transform=ax.transAxes,
                   fontsize=8, verticalalignment='top')
        
        plt.suptitle(f'Attention to Last Test Token: {steering_type.title()} Steering\n(All individual data points shown)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f'test_last_token_transformation_{steering_type}.png', dpi=150)
        plt.close()
        
    def create_head_attention_change_bars(self, attention_data: Dict, steering_type: str) -> None:
        """Bar charts showing attention score differences for each head."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        sections = ['problem', 'tests', 'solution_marker']
        section_labels = ['Problem Description', 'Test Cases', 'Solution Marker']
        colors = ['#8dd3c7', '#ff8c00', '#bebada']
        
        steered_key = f'steered_{steering_type}'
        
        # First pass: calculate all means and stds to determine common y-axis range
        all_section_data = {}
        y_min_global = float('inf')
        y_max_global = float('-inf')
        
        for section in sections:
            # Calculate differences for each head across all tasks
            head_differences = [[] for _ in range(self.n_heads)]
            
            for task_id, task_data in attention_data.items():
                if 'baseline' not in task_data or steered_key not in task_data:
                    continue
                
                # Get aggregated attention
                baseline_agg = self.aggregate_to_3_bins(
                    task_data['baseline']['raw'],
                    task_data['baseline']['boundaries']
                )
                steered_agg = self.aggregate_to_3_bins(
                    task_data[steered_key]['raw'],
                    task_data[steered_key]['boundaries']
                )
                
                # Calculate difference for each head
                for head_idx in range(min(self.n_heads, len(baseline_agg['percentages'][section]))):
                    diff = (steered_agg['percentages'][section][head_idx] - 
                           baseline_agg['percentages'][section][head_idx])
                    head_differences[head_idx].append(diff)
            
            # Calculate means and standard deviations
            means = []
            stds = []
            
            for head_idx in range(self.n_heads):
                if head_differences[head_idx]:
                    means.append(np.mean(head_differences[head_idx]))
                    stds.append(np.std(head_differences[head_idx]))
                else:
                    means.append(0)
                    stds.append(0)
            
            # Store data for this section
            all_section_data[section] = {'means': means, 'stds': stds}
            
            # Update global min/max including error bars
            for mean, std in zip(means, stds):
                y_min_global = min(y_min_global, mean - std)
                y_max_global = max(y_max_global, mean + std)
        
        # Add padding to y-axis range
        y_padding = (y_max_global - y_min_global) * 0.1
        y_limits = (y_min_global - y_padding, y_max_global + y_padding)
        
        # Second pass: create plots with common y-axis
        for ax, section, label, color in zip(axes, sections, section_labels, colors):
            means = all_section_data[section]['means']
            stds = all_section_data[section]['stds']
            
            # Create bar chart with error bars
            x_labels = [f'H{i}' for i in range(self.n_heads)]
            x_pos = np.arange(len(x_labels))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                          color=color, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Color bars based on positive/negative change
            for bar, mean in zip(bars, means):
                if mean > 0:
                    bar.set_facecolor('blue')
                    bar.set_alpha(0.6)
                else:
                    bar.set_facecolor('red')
                    bar.set_alpha(0.6)
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Formatting
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels)
            ax.set_ylabel('Attention Score Difference (%)')
            ax.set_title(f'{label} - {steering_type.title()} Steering Effect')
            ax.set_ylim(y_limits)  # Apply common y-axis limits
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add significance markers for heads with substantial changes
            for i, (mean, std) in enumerate(zip(means, stds)):
                # Mark as significant if |mean| > 2*std (roughly 95% confidence)
                if std > 0 and abs(mean) > 2 * std:
                    y_pos = mean + np.sign(mean) * (std + 1)
                    ax.text(i, y_pos, '*', ha='center', va='bottom' if mean > 0 else 'top',
                           fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Per-Head Attention Changes: {steering_type.title()} Steering\n(* indicates statistically significant change)')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f'head_attention_changes_{steering_type}.png', dpi=150)
        plt.close()
        
    def create_comparative_head_changes(self, attention_data: Dict) -> None:
        """Side-by-side comparison of head-specific changes for correct vs incorrect steering."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        sections = ['problem', 'tests', 'solution_marker']
        section_labels = ['Problem Description', 'Test Cases', 'Solution Marker']
        
        for row_idx, (section, label) in enumerate(zip(sections, section_labels)):
            # Calculate differences for both steering types
            for col_idx, steering_type in enumerate(['correct', 'incorrect']):
                ax = axes[row_idx, col_idx]
                steered_key = f'steered_{steering_type}'
                
                # Calculate differences
                head_differences = []
                head_stds = []
                
                for head_idx in range(self.n_heads):
                    diffs = []
                    for task_id, task_data in attention_data.items():
                        if 'baseline' not in task_data or steered_key not in task_data:
                            continue
                        
                        baseline_agg = self.aggregate_to_3_bins(
                            task_data['baseline']['raw'],
                            task_data['baseline']['boundaries']
                        )
                        steered_agg = self.aggregate_to_3_bins(
                            task_data[steered_key]['raw'],
                            task_data[steered_key]['boundaries']
                        )
                        
                        if head_idx < len(baseline_agg['percentages'][section]):
                            diff = (steered_agg['percentages'][section][head_idx] - 
                                   baseline_agg['percentages'][section][head_idx])
                            diffs.append(diff)
                    
                    if diffs:
                        head_differences.append(np.mean(diffs))
                        head_stds.append(np.std(diffs))
                    else:
                        head_differences.append(0)
                        head_stds.append(0)
                
                # Create grouped bar chart
                x_pos = np.arange(self.n_heads)
                bars = ax.bar(x_pos, head_differences, yerr=head_stds, capsize=5,
                             alpha=0.7, edgecolor='black', linewidth=1)
                
                # Color based on expected behavior
                for bar, diff in zip(bars, head_differences):
                    if section == 'tests':
                        # Color test section bars based on actual change direction
                        bar.set_facecolor('blue' if diff > 0 else 'red')
                    else:
                        bar.set_facecolor('gray')
                    bar.set_alpha(0.6)
                
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'H{i}' for i in range(self.n_heads)])
                ax.set_ylabel('Δ Attention (%)' if col_idx == 0 else '')
                ax.set_xlabel('Head Index')
                ax.set_title(f'{label}\n{steering_type.title()} Steering')
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim(-15, 15)  # Fixed scale for comparison
        
        plt.suptitle('Head-Specific Attention Changes: Correct vs Incorrect Steering Comparison')
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'comparative_head_changes.png', dpi=150)
        plt.close()
        
    def save_analysis_results(self, results: Dict) -> None:
        """Save analysis results to JSON files."""
        # Main results file
        save_json(results, self.output_dir / 'attention_analysis_results.json')
        
        # Summary file
        summary = {
            'timestamp': results['timestamp'],
            'n_tasks_analyzed': results['n_tasks_analyzed'],
            'layers_analyzed': results['layers_analyzed'],
            'key_findings': self._extract_key_findings(results['statistical_results'])
        }
        save_json(summary, self.output_dir / 'phase_6_3_summary.json')
        
        # Save statistical results separately
        save_json(
            results['statistical_results'],
            self.statistical_dir / 'paired_t_tests.json'
        )
        
        logger.info(f"Saved results to {self.output_dir}")
        
    def _aggregate_distributions(self, attention_data: Dict) -> Dict:
        """Aggregate attention distributions across all tasks."""
        distributions = {}
        
        # Process each condition
        conditions = ['baseline', 'correct', 'incorrect']
        condition_keys = ['baseline', 'steered_correct', 'steered_incorrect']
        
        for cond, key in zip(conditions, condition_keys):
            all_problem = []
            all_tests = []
            all_solution = []
            
            for task_data in attention_data.values():
                if key in task_data:
                    agg = self.aggregate_to_3_bins(
                        task_data[key]['raw'],
                        task_data[key]['boundaries']
                    )
                    
                    # Average across heads
                    all_problem.append(np.mean(agg['percentages']['problem']))
                    all_tests.append(np.mean(agg['percentages']['tests']))
                    all_solution.append(np.mean(agg['percentages']['solution_marker']))
            
            if all_problem:
                distributions[cond] = {
                    'problem': np.mean(all_problem),
                    'tests': np.mean(all_tests),
                    'solution_marker': np.mean(all_solution)
                }
        
        return distributions
        
    def _build_head_attention_matrix(self, attention_data: Dict) -> np.ndarray:
        """Build matrix for head attention heatmap."""
        # Initialize matrix [n_heads, 9] for 3 sections × 3 conditions
        matrix = np.zeros((self.n_heads, 9))
        
        # Map conditions to column indices
        condition_offsets = {'baseline': 0, 'steered_correct': 3, 'steered_incorrect': 6}
        section_indices = {'problem': 0, 'tests': 1, 'solution_marker': 2}
        
        # Aggregate across tasks
        for condition, offset in condition_offsets.items():
            head_sums = {section: [[] for _ in range(self.n_heads)] 
                        for section in section_indices.keys()}
            
            for task_data in attention_data.values():
                if condition in task_data:
                    agg = self.aggregate_to_3_bins(
                        task_data[condition]['raw'],
                        task_data[condition]['boundaries']
                    )
                    
                    for section in section_indices.keys():
                        for h in range(min(self.n_heads, len(agg['percentages'][section]))):
                            head_sums[section][h].append(agg['percentages'][section][h])
            
            # Fill matrix
            for section, idx in section_indices.items():
                for h in range(self.n_heads):
                    if head_sums[section][h]:
                        matrix[h, offset + idx] = np.mean(head_sums[section][h])
        
        return matrix
        
    def _calculate_average_deltas(self, differences: Dict) -> Dict:
        """Calculate average attention changes across tasks."""
        all_problem = []
        all_tests = []
        all_solution = []
        
        for task_diffs in differences.values():
            # Average across heads for each task
            all_problem.append(np.mean(task_diffs['problem']))
            all_tests.append(np.mean(task_diffs['tests']))
            all_solution.append(np.mean(task_diffs['solution_marker']))
        
        return {
            'means': [np.mean(all_problem), np.mean(all_tests), np.mean(all_solution)],
            'stds': [np.std(all_problem), np.std(all_tests), np.std(all_solution)]
        }
        
    def _extract_key_findings(self, statistical_results: Dict) -> Dict:
        """Extract key findings from statistical results."""
        findings = {}
        
        for steering_type in ['correct_steering', 'incorrect_steering']:
            if steering_type not in statistical_results:
                continue
            
            results = statistical_results[steering_type]
            
            # Find most significant changes
            significant_sections = []
            for section, stats in results.items():
                if stats['significant']:
                    significant_sections.append({
                        'section': section,
                        'mean_change': stats['mean_change'],
                        'p_value': stats['p_value']
                    })
            
            findings[steering_type] = {
                'significant_sections': significant_sections,
                'test_case_change': results.get('tests', {}).get('mean_change', 0.0),
                'test_case_significant': results.get('tests', {}).get('significant', False)
            }
        
        return findings