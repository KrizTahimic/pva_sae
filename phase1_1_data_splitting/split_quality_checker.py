"""
Split quality checking and reporting for Phase 1.1.

This module validates the quality of dataset splits, ensuring:
- Accurate split ratios
- Similar complexity distributions across splits
- Balanced correct/incorrect rates

Different from:
- Phase 1.0 solution evaluation (checks code correctness)
- Phase 3 validation (validates SAE findings)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import logging
logger = logging.getLogger(__name__)


def check_split_quality(
    splits: Dict[str, List[int]],
    df: pd.DataFrame,
    tolerance: float = 0.02
) -> Dict[str, Any]:
    """
    Check quality of dataset splits.
    
    Performs comprehensive validation including:
    - Ratio accuracy check (against fixed 50%/10%/40% ratios)
    - Complexity distribution similarity tests
    - Correct/incorrect balance verification
    - Statistical significance tests
    
    Args:
        splits: Dictionary of split names to index lists
        df: Original dataframe with complexity_score and optionally test_passed
        tolerance: Maximum allowed deviation from target ratios
        
    Returns:
        Dictionary with detailed quality metrics and test results
    """
    total_samples = len(df)
    complexity_scores = df['complexity_score'].values
    
    # Fixed target ratios: 50% SAE, 10% hyperparams, 40% validation
    target_ratios = [0.5, 0.1, 0.4]
    
    results = {
        'total_samples': total_samples,
        'split_sizes': {},
        'ratio_accuracy': {},
        'complexity_stats': {},
        'correctness_balance': {},
        'distribution_tests': {},
        'overall_quality': True,
        'quality_summary': []
    }
    
    # Check each split
    split_names = list(splits.keys())
    
    for i, (name, indices) in enumerate(splits.items()):
        split_size = len(indices)
        actual_ratio = split_size / total_samples
        
        # Basic size and ratio metrics
        results['split_sizes'][name] = split_size
        
        ratio_error = abs(actual_ratio - target_ratios[i])
        within_tolerance = ratio_error <= tolerance
        
        results['ratio_accuracy'][name] = {
            'target': target_ratios[i],
            'actual': actual_ratio,
            'error': ratio_error,
            'within_tolerance': within_tolerance
        }
        
        if not within_tolerance:
            results['quality_summary'].append(
                f"Split '{name}' ratio {actual_ratio:.3f} exceeds tolerance "
                f"(target: {target_ratios[i]:.3f}, error: {ratio_error:.3f})"
            )
        
        # Complexity statistics
        split_complexity = complexity_scores[indices]
        results['complexity_stats'][name] = {
            'mean': float(np.mean(split_complexity)),
            'std': float(np.std(split_complexity)),
            'min': float(np.min(split_complexity)),
            'max': float(np.max(split_complexity)),
            'median': float(np.median(split_complexity))
        }
        
        # Correctness balance (if available)
        if 'test_passed' in df.columns:
            split_df = df.iloc[indices]
            correct_count = split_df['test_passed'].sum()
            total_count = len(split_df)
            
            results['correctness_balance'][name] = {
                'correct_count': int(correct_count),
                'incorrect_count': int(total_count - correct_count),
                'correct_rate': float(correct_count / total_count) if total_count > 0 else 0.0,
                'total': total_count
            }
    
    # Statistical tests for distribution similarity
    results['distribution_tests'] = run_distribution_tests(splits, complexity_scores)
    
    # Check correctness balance similarity if available
    if 'test_passed' in df.columns:
        correctness_test = check_correctness_balance(results['correctness_balance'])
        results['distribution_tests']['correctness_balance'] = correctness_test
        
        if not correctness_test['balanced']:
            results['quality_summary'].append(
                "Correctness rates vary significantly across splits"
            )
    
    # Overall quality assessment
    ratio_check = all(r['within_tolerance'] for r in results['ratio_accuracy'].values())
    distribution_check = results['distribution_tests']['all_similar']
    
    results['overall_quality'] = ratio_check and distribution_check
    
    if not distribution_check:
        results['quality_summary'].append(
            "Complexity distributions are significantly different across splits"
        )
    
    if results['overall_quality']:
        results['quality_summary'].append("All quality checks passed!")
    
    return results


def run_distribution_tests(
    splits: Dict[str, List[int]],
    complexity_scores: np.ndarray
) -> Dict[str, Any]:
    """
    Run statistical tests to verify distribution similarity across splits.
    
    Uses multiple statistical tests:
    - Kolmogorov-Smirnov: Distribution shape similarity
    - Kruskal-Wallis: Median similarity (non-parametric ANOVA)
    - Levene's test: Variance homogeneity
    
    Args:
        splits: Dictionary of split indices
        complexity_scores: Array of complexity scores
        
    Returns:
        Dictionary with test results and interpretation
    """
    split_complexities = {
        name: complexity_scores[indices] 
        for name, indices in splits.items()
    }
    
    tests = {
        'ks_tests': {},
        'kruskal_wallis': None,
        'levene_test': None,
        'all_similar': True,
        'summary': []
    }
    
    split_names = list(splits.keys())
    
    # Pairwise Kolmogorov-Smirnov tests
    ks_failures = []
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            name1, name2 = split_names[i], split_names[j]
            
            ks_stat, p_value = stats.ks_2samp(
                split_complexities[name1],
                split_complexities[name2]
            )
            
            similar = p_value > 0.05
            tests['ks_tests'][f"{name1}_vs_{name2}"] = {
                'statistic': float(ks_stat),
                'p_value': float(p_value),
                'similar': similar
            }
            
            if not similar:
                ks_failures.append(f"{name1} vs {name2}")
                tests['all_similar'] = False
    
    if ks_failures:
        tests['summary'].append(
            f"KS test failures: {', '.join(ks_failures)}"
        )
    
    # Kruskal-Wallis test (non-parametric ANOVA)
    if len(split_names) > 2:
        complexity_lists = [split_complexities[name] for name in split_names]
        h_stat, p_value = stats.kruskal(*complexity_lists)
        
        similar = p_value > 0.05
        tests['kruskal_wallis'] = {
            'statistic': float(h_stat),
            'p_value': float(p_value),
            'similar': similar,
            'interpretation': 'No significant difference in medians' if similar 
                            else 'Significant difference in medians'
        }
        
        if not similar:
            tests['all_similar'] = False
            tests['summary'].append("Kruskal-Wallis test indicates different medians")
    
    # Levene's test for variance homogeneity
    complexity_lists = [split_complexities[name] for name in split_names]
    levene_stat, p_value = stats.levene(*complexity_lists)
    
    similar = p_value > 0.05
    tests['levene_test'] = {
        'statistic': float(levene_stat),
        'p_value': float(p_value),
        'similar': similar,
        'interpretation': 'Equal variances' if similar else 'Unequal variances'
    }
    
    if not similar:
        tests['summary'].append("Levene's test indicates unequal variances")
    
    if tests['all_similar']:
        tests['summary'].append("All distribution tests passed")
    
    return tests


def check_correctness_balance(
    correctness_stats: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Check if correct/incorrect ratios are balanced across splits.
    
    Args:
        correctness_stats: Dictionary with correctness statistics per split
        
    Returns:
        Dictionary with balance test results
    """
    # Extract correct rates
    correct_rates = [
        stats['correct_rate'] 
        for stats in correctness_stats.values()
    ]
    
    # Calculate statistics
    mean_rate = np.mean(correct_rates)
    std_rate = np.std(correct_rates)
    max_deviation = max(abs(rate - mean_rate) for rate in correct_rates)
    
    # Check if balanced (within 5% of mean)
    balanced = max_deviation <= 0.05
    
    return {
        'mean_correct_rate': float(mean_rate),
        'std_correct_rate': float(std_rate),
        'max_deviation': float(max_deviation),
        'balanced': balanced,
        'split_rates': {
            name: float(stats['correct_rate'])
            for name, stats in correctness_stats.items()
        }
    }


def generate_quality_report(
    splits: Dict[str, List[int]],
    df: pd.DataFrame,
    quality_results: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Generate comprehensive HTML report with visualizations.
    
    Creates visualizations and detailed HTML report including:
    - Split size and ratio accuracy
    - Complexity distribution plots
    - Statistical test results
    - Quality summary
    
    Args:
        splits: Split indices
        df: Original dataframe
        quality_results: Results from check_split_quality
        output_dir: Directory to save report
        
    Returns:
        Path to generated HTML report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    plot_paths = create_split_visualizations(splits, df, output_path)
    
    # Generate HTML report
    html_content = generate_html_content(quality_results, plot_paths)
    
    report_path = output_path / "split_quality_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Generated quality report: {report_path}")
    return str(report_path)


def create_split_visualizations(
    splits: Dict[str, List[int]],
    df: pd.DataFrame,
    output_path: Path
) -> Dict[str, str]:
    """
    Create and save visualization plots for the report.
    
    Args:
        splits: Split indices
        df: Original dataframe
        output_path: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to filenames
    """
    complexity_scores = df['complexity_score'].values
    plot_paths = {}
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Complexity distribution comparison (overlapping histograms)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for name, indices in splits.items():
        split_complexity = complexity_scores[indices]
        ax.hist(split_complexity, bins=20, alpha=0.6, label=name, density=True)
    
    ax.set_xlabel('Complexity Score')
    ax.set_ylabel('Density')
    ax.set_title('Complexity Distribution Across Splits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'complexity_distributions.png'
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['distributions'] = filename
    
    # 2. Box plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    split_data = [complexity_scores[indices] for indices in splits.values()]
    box_plot = ax.boxplot(split_data, labels=list(splits.keys()), patch_artist=True)
    
    # Color the boxes
    colors = sns.color_palette("husl", len(splits))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Complexity Score')
    ax.set_title('Complexity Distribution Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add mean markers
    means = [np.mean(data) for data in split_data]
    ax.scatter(range(1, len(means) + 1), means, 
              color='red', marker='D', s=50, zorder=3, label='Mean')
    ax.legend()
    
    plt.tight_layout()
    filename = 'complexity_boxplots.png'
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['boxplots'] = filename
    
    # 3. Split size comparison (bar chart)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    split_sizes = [len(indices) for indices in splits.values()]
    split_names = list(splits.keys())
    
    bars = ax.bar(split_names, split_sizes, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, size in zip(bars, split_sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{size}', ha='center', va='bottom')
    
    ax.set_ylabel('Number of Samples')
    ax.set_title('Split Sizes')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = 'split_sizes.png'
    plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['sizes'] = filename
    
    # 4. Correctness balance (if available)
    if 'test_passed' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Stacked bar chart for counts
        correct_counts = []
        incorrect_counts = []
        
        for name, indices in splits.items():
            split_df = df.iloc[indices]
            correct = split_df['test_passed'].sum()
            incorrect = len(split_df) - correct
            correct_counts.append(correct)
            incorrect_counts.append(incorrect)
        
        x = np.arange(len(split_names))
        width = 0.6
        
        p1 = ax1.bar(x, correct_counts, width, label='Correct', color='green', alpha=0.7)
        p2 = ax1.bar(x, incorrect_counts, width, bottom=correct_counts, 
                    label='Incorrect', color='red', alpha=0.7)
        
        ax1.set_ylabel('Count')
        ax1.set_title('Correct/Incorrect Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(split_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Correct rate comparison
        correct_rates = [c / (c + i) for c, i in zip(correct_counts, incorrect_counts)]
        bars = ax2.bar(split_names, correct_rates, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, rate in zip(bars, correct_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # Add mean line
        mean_rate = np.mean(correct_rates)
        ax2.axhline(y=mean_rate, color='red', linestyle='--', 
                   label=f'Mean: {mean_rate:.1%}')
        
        ax2.set_ylabel('Correct Rate')
        ax2.set_title('Code Correctness Rate by Split')
        ax2.set_ylim(0, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = 'correctness_balance.png'
        plt.savefig(output_path / filename, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['correctness'] = filename
    
    return plot_paths


def generate_html_content(
    quality_results: Dict[str, Any],
    plot_paths: Dict[str, str]
) -> str:
    """Generate HTML content for the quality report."""
    
    # Determine overall status
    overall_status = 'PASS' if quality_results['overall_quality'] else 'FAIL'
    status_class = 'pass' if quality_results['overall_quality'] else 'fail'
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Phase 1.1 Split Quality Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
            margin-top: 30px;
        }}
        h1 {{
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .pass {{
            color: #28a745;
            font-weight: bold;
        }}
        .fail {{
            color: #dc3545;
            font-weight: bold;
        }}
        .warning {{
            color: #ffc107;
            font-weight: bold;
        }}
        .metric {{
            display: inline-block;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 4px;
            background-color: #e9ecef;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 20px 0;
        }}
        .summary-box.fail {{
            border-left-color: #dc3545;
        }}
        .summary-box.pass {{
            border-left-color: #28a745;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Phase 1.1 Dataset Split Quality Report</h1>
        
        <div class="summary-box {status_class}">
            <h2>Overall Quality: <span class="{status_class}">{overall_status}</span></h2>
            <p>Total samples: <strong>{quality_results['total_samples']}</strong></p>
            <h3>Summary:</h3>
            <ul>
"""
    
    # Add summary points
    for point in quality_results['quality_summary']:
        html += f"                <li>{point}</li>\n"
    
    html += """            </ul>
        </div>
        
        <h2>Split Sizes and Ratio Accuracy</h2>
        <table>
            <tr>
                <th>Split</th>
                <th>Size</th>
                <th>Target Ratio</th>
                <th>Actual Ratio</th>
                <th>Error</th>
                <th>Status</th>
            </tr>
"""
    
    # Add ratio accuracy table
    for split_name, accuracy in quality_results['ratio_accuracy'].items():
        status = 'PASS' if accuracy['within_tolerance'] else 'FAIL'
        status_class = 'pass' if accuracy['within_tolerance'] else 'fail'
        
        html += f"""            <tr>
                <td><strong>{split_name}</strong></td>
                <td>{quality_results['split_sizes'][split_name]}</td>
                <td>{accuracy['target']:.1%}</td>
                <td>{accuracy['actual']:.1%}</td>
                <td>{accuracy['error']:.4f}</td>
                <td class="{status_class}">{status}</td>
            </tr>
"""
    
    html += """        </table>
        
        <h2>Complexity Statistics</h2>
        <table>
            <tr>
                <th>Split</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Median</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
"""
    
    # Add complexity statistics
    for split_name, stats in quality_results['complexity_stats'].items():
        html += f"""            <tr>
                <td><strong>{split_name}</strong></td>
                <td>{stats['mean']:.2f}</td>
                <td>{stats['std']:.2f}</td>
                <td>{stats['median']:.2f}</td>
                <td>{stats['min']:.0f}</td>
                <td>{stats['max']:.0f}</td>
            </tr>
"""
    
    html += """        </table>
"""
    
    # Add correctness balance if available
    if quality_results['correctness_balance']:
        html += """
        <h2>Code Correctness Balance</h2>
        <table>
            <tr>
                <th>Split</th>
                <th>Correct</th>
                <th>Incorrect</th>
                <th>Total</th>
                <th>Correct Rate</th>
            </tr>
"""
        
        for split_name, stats in quality_results['correctness_balance'].items():
            html += f"""            <tr>
                <td><strong>{split_name}</strong></td>
                <td>{stats['correct_count']}</td>
                <td>{stats['incorrect_count']}</td>
                <td>{stats['total']}</td>
                <td>{stats['correct_rate']:.1%}</td>
            </tr>
"""
        
        html += """        </table>
"""
    
    # Add distribution tests
    html += """
        <h2>Statistical Distribution Tests</h2>
        <p>These tests verify that complexity distributions are similar across splits.</p>
        
        <h3>Kolmogorov-Smirnov Tests</h3>
        <p>Tests if two samples come from the same distribution (p > 0.05 indicates similarity)</p>
        <table>
            <tr>
                <th>Comparison</th>
                <th>KS Statistic</th>
                <th>P-value</th>
                <th>Result</th>
            </tr>
"""
    
    # Add KS test results
    for comparison, test in quality_results['distribution_tests']['ks_tests'].items():
        result = 'Similar' if test['similar'] else 'Different'
        result_class = 'pass' if test['similar'] else 'fail'
        
        html += f"""            <tr>
                <td>{comparison.replace('_', ' ')}</td>
                <td>{test['statistic']:.4f}</td>
                <td>{test['p_value']:.4f}</td>
                <td class="{result_class}">{result}</td>
            </tr>
"""
    
    html += """        </table>
"""
    
    # Add other tests if available
    if quality_results['distribution_tests']['kruskal_wallis']:
        kw_test = quality_results['distribution_tests']['kruskal_wallis']
        result_class = 'pass' if kw_test['similar'] else 'fail'
        
        html += f"""
        <h3>Kruskal-Wallis Test</h3>
        <p>Tests if all splits have the same median (non-parametric ANOVA)</p>
        <div class="metric">
            <strong>H-statistic:</strong> {kw_test['statistic']:.4f} | 
            <strong>P-value:</strong> {kw_test['p_value']:.4f} | 
            <strong>Result:</strong> <span class="{result_class}">{kw_test['interpretation']}</span>
        </div>
"""
    
    if quality_results['distribution_tests']['levene_test']:
        lev_test = quality_results['distribution_tests']['levene_test']
        result_class = 'pass' if lev_test['similar'] else 'warning'
        
        html += f"""
        <h3>Levene's Test</h3>
        <p>Tests if all splits have equal variances</p>
        <div class="metric">
            <strong>Statistic:</strong> {lev_test['statistic']:.4f} | 
            <strong>P-value:</strong> {lev_test['p_value']:.4f} | 
            <strong>Result:</strong> <span class="{result_class}">{lev_test['interpretation']}</span>
        </div>
"""
    
    # Add visualizations
    html += """
        <h2>Visualizations</h2>
        
        <h3>Split Sizes</h3>
        <img src="split_sizes.png" alt="Split Sizes">
        
        <h3>Complexity Distributions</h3>
        <img src="complexity_distributions.png" alt="Complexity Distributions">
        
        <h3>Complexity Box Plots</h3>
        <img src="complexity_boxplots.png" alt="Complexity Box Plots">
"""
    
    if 'correctness' in plot_paths:
        html += """
        <h3>Correctness Balance</h3>
        <img src="correctness_balance.png" alt="Correctness Balance">
"""
    
    # Close HTML
    html += """
    </div>
</body>
</html>
"""
    
    return html