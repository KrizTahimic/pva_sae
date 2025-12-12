"""Phase 4.16: Difficulty-Stratified Steering Analysis.

Analyzes whether problem difficulty (cyclomatic complexity) is a significant
factor in steering success rates (correction, corruption, preservation).

Uses chi-square tests to determine statistical significance.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from scipy.stats import chi2_contingency

from common.logging import get_logger
from common.config import Config
from common.utils import ensure_directory_exists
from common.phase_discovery import get_phase_output_dir
from common.viz_utils import handle_viz_only_mode
from phase3_12_difficulty_auroc_f1.difficulty_evaluator import group_by_difficulty

logger = get_logger("phase4_16.difficulty_steering_analyzer")


class Phase416Runner:
    """Standard runner for Phase 4.16: Difficulty-Stratified Steering Analysis."""

    def __init__(self, config):
        """Initialize with config object."""
        self.config = config
        self.logger = get_logger("phase4_16.runner", phase="4.16")

    def run(self):
        """Run Phase 4.16 difficulty steering analysis."""
        self.logger.info("Starting Phase 4.16: Difficulty-Stratified Steering Analysis")
        self.logger.info("Analyzing whether problem difficulty affects steering success rates")
        self.logger.info("\n" + self.config.dump(phase="4.16"))

        # main() creates its own Config internally
        return main()


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_steering_results(config: Config) -> Dict[str, List[Dict]]:
    """Load steering results from Phase 4.8 with fallback for preservation.

    Returns:
        Dictionary with keys 'correction', 'corruption', 'preservation'
    """
    phase4_8_dir = Path(get_phase_output_dir("4.8", config))
    if not phase4_8_dir.exists():
        raise FileNotFoundError(f"Phase 4.8 output not found at {phase4_8_dir}. Run Phase 4.8 first.")
    phase4_8_preserve_dir = Path("data/phase4_8_preserve_only")

    results = {}

    # Load correction results
    correction_path = phase4_8_dir / "all_correction_results.json"
    if correction_path.exists():
        results['correction'] = load_json(correction_path)
        logger.info(f"Loaded {len(results['correction'])} correction results from {phase4_8_dir}")
    else:
        raise FileNotFoundError(f"Correction results not found: {correction_path}")

    # Load corruption results
    corruption_path = phase4_8_dir / "all_corruption_results.json"
    if corruption_path.exists():
        results['corruption'] = load_json(corruption_path)
        logger.info(f"Loaded {len(results['corruption'])} corruption results")
    else:
        raise FileNotFoundError(f"Corruption results not found: {corruption_path}")

    # Load preservation results with fallback
    preservation_path = phase4_8_dir / "all_preservation_results.json"
    preservation_fallback_path = phase4_8_preserve_dir / "all_preservation_results.json"

    if preservation_path.exists():
        preservation_data = load_json(preservation_path)

        # Check if steered_passed has NaN values (represented as None in JSON)
        has_nan = any(
            r.get('steered_passed') is None or
            (isinstance(r.get('steered_passed'), float) and np.isnan(r.get('steered_passed')))
            for r in preservation_data
        )

        if has_nan and preservation_fallback_path.exists():
            logger.info(f"Primary preservation has NaN values, using fallback: {preservation_fallback_path}")
            results['preservation'] = load_json(preservation_fallback_path)
        else:
            results['preservation'] = preservation_data

        logger.info(f"Loaded {len(results['preservation'])} preservation results")
    else:
        raise FileNotFoundError(f"Preservation results not found: {preservation_path}")

    return results


def load_validation_with_difficulty(config: Config) -> pd.DataFrame:
    """Load validation dataset with cyclomatic complexity."""
    phase0_1_dir = Path(get_phase_output_dir("0.1", config))
    if not phase0_1_dir.exists():
        raise FileNotFoundError(f"Phase 0.1 output not found at {phase0_1_dir}. Run Phase 0.1 first.")
    validation_path = phase0_1_dir / "validation_mbpp.parquet"

    if not validation_path.exists():
        # Try alternative naming
        validation_path = phase0_1_dir / "validation.parquet"
        if not validation_path.exists():
            raise FileNotFoundError(f"Validation dataset not found in: {phase0_1_dir}")

    df = pd.read_parquet(validation_path)
    logger.info(f"Loaded validation dataset: {len(df)} problems")
    logger.info(f"Cyclomatic complexity range: {df['cyclomatic_complexity'].min()}-{df['cyclomatic_complexity'].max()}")

    return df


def calculate_difficulty_metrics(
    steering_results: List[Dict],
    difficulty_groups: Dict[str, pd.DataFrame],
    experiment_type: str
) -> Dict[str, Dict]:
    """Calculate steering success metrics per difficulty group.

    Args:
        steering_results: List of steering result dictionaries
        difficulty_groups: Dict with 'easy', 'medium', 'hard' DataFrames
        experiment_type: 'correction', 'corruption', or 'preservation'

    Returns:
        Dictionary with metrics per difficulty group
    """
    # Create task_id to steering result mapping
    results_by_task = {r['task_id']: r for r in steering_results}

    metrics = {}

    for group_name, group_data in difficulty_groups.items():
        group_task_ids = set(group_data['task_id'].values)

        # Filter steering results for this difficulty group
        group_results = [
            r for r in steering_results
            if r['task_id'] in group_task_ids
        ]

        if not group_results:
            logger.warning(f"No steering results found for {group_name} group")
            metrics[group_name] = {
                'n_total': 0,
                'n_success': 0,
                'n_failure': 0,
                'rate': 0.0
            }
            continue

        n_total = len(group_results)

        if experiment_type == 'correction':
            # Success = initially incorrect, became correct (flipped from False to True)
            n_success = sum(
                1 for r in group_results
                if not r['test_passed'] and r.get('steered_passed', False)
            )
        elif experiment_type == 'corruption':
            # Success = initially correct, became incorrect (flipped from True to False)
            n_success = sum(
                1 for r in group_results
                if r['test_passed'] and not r.get('steered_passed', True)
            )
        elif experiment_type == 'preservation':
            # Success = initially correct, stayed correct
            n_success = sum(
                1 for r in group_results
                if r['test_passed'] and r.get('steered_passed', False)
            )
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        n_failure = n_total - n_success
        rate = (n_success / n_total * 100) if n_total > 0 else 0.0

        metrics[group_name] = {
            'n_total': n_total,
            'n_success': n_success,
            'n_failure': n_failure,
            'rate': rate
        }

        logger.info(f"  {group_name.capitalize()}: {n_success}/{n_total} = {rate:.2f}%")

    return metrics


def run_chi_square_test(metrics: Dict[str, Dict]) -> Dict:
    """Run chi-square test for independence.

    Tests whether the distribution of success/failure differs
    significantly across difficulty groups.

    Args:
        metrics: Per-difficulty metrics dictionary

    Returns:
        Dictionary with chi-square test results
    """
    # Build contingency table
    # Rows: Success, Failure
    # Columns: Easy, Medium, Hard
    groups = ['easy', 'medium', 'hard']

    success_row = [metrics[g]['n_success'] for g in groups]
    failure_row = [metrics[g]['n_failure'] for g in groups]

    contingency_table = np.array([success_row, failure_row])

    # Check if we have enough data for chi-square test
    if contingency_table.sum() == 0:
        return {
            'statistic': None,
            'p_value': None,
            'dof': None,
            'expected': None,
            'significant': None,
            'error': 'No data for chi-square test'
        }

    # Check for zero rows/columns which would make chi-square invalid
    if np.any(contingency_table.sum(axis=0) == 0) or np.any(contingency_table.sum(axis=1) == 0):
        return {
            'statistic': None,
            'p_value': None,
            'dof': None,
            'expected': None,
            'significant': None,
            'error': 'Zero row or column in contingency table - chi-square test not applicable'
        }

    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        return {
            'statistic': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'expected': expected.tolist(),
            'contingency_table': contingency_table.tolist(),
            'significant': bool(p_value < 0.05)
        }
    except Exception as e:
        return {
            'statistic': None,
            'p_value': None,
            'dof': None,
            'expected': None,
            'significant': None,
            'error': str(e)
        }


def plot_steering_trends(
    correction_metrics: Dict[str, Dict],
    corruption_metrics: Dict[str, Dict],
    preservation_metrics: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Plot steering success rate trends across difficulty levels."""
    plt.figure(figsize=(12, 6))

    difficulties = ['easy', 'medium', 'hard']
    x_labels = [d.capitalize() for d in difficulties]
    x_pos = range(len(difficulties))

    # Extract rates
    correction_rates = [correction_metrics[d]['rate'] for d in difficulties]
    corruption_rates = [corruption_metrics[d]['rate'] for d in difficulties]
    preservation_rates = [preservation_metrics[d]['rate'] for d in difficulties]

    # Plot lines
    plt.plot(x_pos, correction_rates, 'b-o', linewidth=2, markersize=10,
             label='Correction Rate')
    plt.plot(x_pos, corruption_rates, 'r-s', linewidth=2, markersize=10,
             label='Corruption Rate')
    plt.plot(x_pos, preservation_rates, color='orange', marker='^', linestyle='-', linewidth=2, markersize=10,
             label='Preservation Rate')

    # Add value labels
    for i, (c, cr, p) in enumerate(zip(correction_rates, corruption_rates, preservation_rates)):
        plt.text(i, c + 2, f'{c:.1f}%', ha='center', va='bottom', color='blue', fontweight='bold')
        plt.text(i, cr + 2, f'{cr:.1f}%', ha='center', va='bottom', color='red', fontweight='bold')
        plt.text(i, p - 4, f'{p:.1f}%', ha='center', va='top', color='orange', fontweight='bold')

    plt.xlabel('Difficulty Level (Cyclomatic Complexity)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Steering Success Rates by Problem Difficulty', fontsize=14)
    plt.xticks(x_pos, x_labels)
    plt.ylim(0, 110)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / 'steering_by_difficulty_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved trend plot: {output_dir / 'steering_by_difficulty_trends.png'}")


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
    for bar, size in zip(bars, group_sizes):
        percentage = size / total_tasks * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{size}\n({percentage:.1f}%)', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'difficulty_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distribution plot: {output_dir / 'difficulty_distribution.png'}")


def generate_summary_text(
    difficulty_groups: Dict[str, pd.DataFrame],
    correction_metrics: Dict[str, Dict],
    corruption_metrics: Dict[str, Dict],
    preservation_metrics: Dict[str, Dict],
    correction_chi2: Dict,
    corruption_chi2: Dict,
    preservation_chi2: Dict
) -> str:
    """Generate human-readable summary."""
    lines = [
        "=" * 70,
        "PHASE 4.16: DIFFICULTY-STRATIFIED STEERING ANALYSIS",
        "=" * 70,
        "",
        "OBJECTIVE: Determine if problem difficulty affects steering success",
        "",
        "DIFFICULTY GROUPS:",
        f"  Easy (complexity = 1):    {len(difficulty_groups['easy'])} tasks",
        f"  Medium (complexity 2-3):  {len(difficulty_groups['medium'])} tasks",
        f"  Hard (complexity >= 4):   {len(difficulty_groups['hard'])} tasks",
        "",
        "-" * 70,
        "STEERING SUCCESS RATES BY DIFFICULTY",
        "-" * 70,
        "",
        "                    Easy        Medium      Hard",
        "                    ----        ------      ----",
    ]

    # Format rates as table
    corr_line = f"  Correction:       {correction_metrics['easy']['rate']:5.1f}%      {correction_metrics['medium']['rate']:5.1f}%      {correction_metrics['hard']['rate']:5.1f}%"
    corr_counts = f"                    ({correction_metrics['easy']['n_success']}/{correction_metrics['easy']['n_total']})       ({correction_metrics['medium']['n_success']}/{correction_metrics['medium']['n_total']})       ({correction_metrics['hard']['n_success']}/{correction_metrics['hard']['n_total']})"

    lines.append(corr_line)
    lines.append(corr_counts)
    lines.append("")

    corrup_line = f"  Corruption:       {corruption_metrics['easy']['rate']:5.1f}%      {corruption_metrics['medium']['rate']:5.1f}%      {corruption_metrics['hard']['rate']:5.1f}%"
    corrup_counts = f"                    ({corruption_metrics['easy']['n_success']}/{corruption_metrics['easy']['n_total']})       ({corruption_metrics['medium']['n_success']}/{corruption_metrics['medium']['n_total']})       ({corruption_metrics['hard']['n_success']}/{corruption_metrics['hard']['n_total']})"

    lines.append(corrup_line)
    lines.append(corrup_counts)
    lines.append("")

    pres_line = f"  Preservation:     {preservation_metrics['easy']['rate']:5.1f}%      {preservation_metrics['medium']['rate']:5.1f}%      {preservation_metrics['hard']['rate']:5.1f}%"
    pres_counts = f"                    ({preservation_metrics['easy']['n_success']}/{preservation_metrics['easy']['n_total']})       ({preservation_metrics['medium']['n_success']}/{preservation_metrics['medium']['n_total']})       ({preservation_metrics['hard']['n_success']}/{preservation_metrics['hard']['n_total']})"

    lines.append(pres_line)
    lines.append(pres_counts)
    lines.append("")

    lines.extend([
        "-" * 70,
        "CHI-SQUARE TESTS FOR INDEPENDENCE",
        "-" * 70,
        "",
        "H0: Difficulty level and steering success are independent",
        "H1: Difficulty level affects steering success",
        "Significance level: α = 0.05",
        "",
    ])

    # Chi-square results
    for name, chi2_result in [("Correction", correction_chi2),
                               ("Corruption", corruption_chi2),
                               ("Preservation", preservation_chi2)]:
        if chi2_result.get('error'):
            lines.append(f"  {name}: {chi2_result['error']}")
        elif chi2_result['statistic'] is not None:
            sig_str = "YES" if chi2_result['significant'] else "NO"
            lines.append(f"  {name}: χ² = {chi2_result['statistic']:.4f}, p = {chi2_result['p_value']:.4f}, Significant: {sig_str}")
        else:
            lines.append(f"  {name}: Unable to compute chi-square test")
        lines.append("")

    lines.extend([
        "-" * 70,
        "CONCLUSION",
        "-" * 70,
        "",
    ])

    # Generate conclusion
    significant_experiments = []
    for name, chi2_result in [("Correction", correction_chi2),
                               ("Corruption", corruption_chi2),
                               ("Preservation", preservation_chi2)]:
        if chi2_result.get('significant'):
            significant_experiments.append(name.lower())

    if not significant_experiments:
        lines.append("  Difficulty level is NOT a statistically significant factor in")
        lines.append("  any of the steering experiments (correction, corruption, preservation).")
        lines.append("")
        lines.append("  This suggests that the steering effect is consistent across")
        lines.append("  problem complexity levels.")
    else:
        lines.append(f"  Difficulty level IS a statistically significant factor for:")
        for exp in significant_experiments:
            lines.append(f"    - {exp.capitalize()}")
        lines.append("")
        lines.append("  Further investigation is recommended to understand why difficulty")
        lines.append("  affects these specific steering experiments.")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Main entry point for Phase 4.16."""
    config = Config()

    # Set random seed for reproducibility
    np.random.seed(config.evaluation_random_seed)

    # Create output directory
    output_dir = Path(get_phase_output_dir("4.16", config))
    ensure_directory_exists(output_dir)

    # Handle --viz-only mode
    if config.viz_only:
        data_path = output_dir / "difficulty_steering_analysis.json"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Cannot run --viz-only: {data_path} not found. "
                f"Run phase normally first to generate data."
            )
        logger.info(f"--viz-only mode: Loading data from {data_path}")
        data = load_json(data_path)

        # Reconstruct metrics for visualization
        correction_metrics = data['correction_analysis']['counts']
        corruption_metrics = data['corruption_analysis']['counts']
        preservation_metrics = data['preservation_analysis']['counts']

        # For difficulty distribution we need validation data
        validation_data = load_validation_with_difficulty(config)
        difficulty_groups = group_by_difficulty(validation_data)

        plot_steering_trends(correction_metrics, corruption_metrics, preservation_metrics, output_dir)
        plot_difficulty_distribution(difficulty_groups, output_dir)
        logger.info("Visualization regeneration complete")
        return

    logger.info("=" * 60)
    logger.info("PHASE 4.16: DIFFICULTY-STRATIFIED STEERING ANALYSIS")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    steering_results = load_steering_results(config)
    validation_data = load_validation_with_difficulty(config)

    # Step 2: Group by difficulty
    logger.info("\nStep 2: Grouping tasks by difficulty...")
    difficulty_groups = group_by_difficulty(validation_data)

    # Step 3: Calculate metrics per difficulty group
    logger.info("\nStep 3: Calculating correction metrics by difficulty...")
    correction_metrics = calculate_difficulty_metrics(
        steering_results['correction'], difficulty_groups, 'correction'
    )

    logger.info("\nCalculating corruption metrics by difficulty...")
    corruption_metrics = calculate_difficulty_metrics(
        steering_results['corruption'], difficulty_groups, 'corruption'
    )

    logger.info("\nCalculating preservation metrics by difficulty...")
    preservation_metrics = calculate_difficulty_metrics(
        steering_results['preservation'], difficulty_groups, 'preservation'
    )

    # Step 4: Run chi-square tests
    logger.info("\nStep 4: Running chi-square tests...")

    logger.info("  Testing correction experiment...")
    correction_chi2 = run_chi_square_test(correction_metrics)
    if correction_chi2.get('significant') is not None:
        sig = "significant" if correction_chi2['significant'] else "not significant"
        logger.info(f"    χ² = {correction_chi2['statistic']:.4f}, p = {correction_chi2['p_value']:.4f} ({sig})")

    logger.info("  Testing corruption experiment...")
    corruption_chi2 = run_chi_square_test(corruption_metrics)
    if corruption_chi2.get('significant') is not None:
        sig = "significant" if corruption_chi2['significant'] else "not significant"
        logger.info(f"    χ² = {corruption_chi2['statistic']:.4f}, p = {corruption_chi2['p_value']:.4f} ({sig})")

    logger.info("  Testing preservation experiment...")
    preservation_chi2 = run_chi_square_test(preservation_metrics)
    if preservation_chi2.get('significant') is not None:
        sig = "significant" if preservation_chi2['significant'] else "not significant"
        logger.info(f"    χ² = {preservation_chi2['statistic']:.4f}, p = {preservation_chi2['p_value']:.4f} ({sig})")

    # Step 5: Generate visualizations
    logger.info("\nStep 5: Generating visualizations...")
    plot_steering_trends(correction_metrics, corruption_metrics, preservation_metrics, output_dir)
    plot_difficulty_distribution(difficulty_groups, output_dir)

    # Step 6: Compile and save results
    logger.info("\nStep 6: Saving results...")

    results = {
        'phase': '4.16',
        'description': 'Difficulty-stratified steering analysis',
        'creation_timestamp': datetime.now().isoformat(),
        'difficulty_groups': {
            group_name: {
                'complexity_range': [
                    int(group_data['cyclomatic_complexity'].min()),
                    int(group_data['cyclomatic_complexity'].max())
                ],
                'n_tasks': len(group_data)
            }
            for group_name, group_data in difficulty_groups.items()
        },
        'correction_analysis': {
            'rates': {g: m['rate'] for g, m in correction_metrics.items()},
            'counts': correction_metrics,
            'chi_square': correction_chi2
        },
        'corruption_analysis': {
            'rates': {g: m['rate'] for g, m in corruption_metrics.items()},
            'counts': corruption_metrics,
            'chi_square': corruption_chi2
        },
        'preservation_analysis': {
            'rates': {g: m['rate'] for g, m in preservation_metrics.items()},
            'counts': preservation_metrics,
            'chi_square': preservation_chi2
        },
        'conclusion': {
            'correction_significant': correction_chi2.get('significant'),
            'corruption_significant': corruption_chi2.get('significant'),
            'preservation_significant': preservation_chi2.get('significant'),
            'overall': 'Difficulty is a significant factor' if any([
                correction_chi2.get('significant'),
                corruption_chi2.get('significant'),
                preservation_chi2.get('significant')
            ]) else 'Difficulty is NOT a significant factor'
        }
    }

    save_json(results, output_dir / 'difficulty_steering_analysis.json')

    # Generate and save summary
    summary_text = generate_summary_text(
        difficulty_groups,
        correction_metrics,
        corruption_metrics,
        preservation_metrics,
        correction_chi2,
        corruption_chi2,
        preservation_chi2
    )

    logger.info("\n" + summary_text)

    with open(output_dir / 'difficulty_steering_summary.txt', 'w') as f:
        f.write(summary_text)

    logger.info(f"\nAll results saved to: {output_dir}")

    # Write phase_output.json manifest
    from common.phase_discovery import write_phase_output

    write_phase_output(
        phase="4.16",
        outputs={
            "primary": "difficulty_steering_analysis.json",
            "summary": "difficulty_steering_summary.txt",
        },
        config=config,
        output_dir=str(output_dir),
        dependencies={
            "4.8": str(phase4_8_dir),
        },
        config_keys=['model_name', 'dataset_name']
    )
    logger.info(f"Saved phase_output.json manifest to {output_dir}")

    logger.info("✅ Phase 4.16 completed successfully")


if __name__ == "__main__":
    main()
