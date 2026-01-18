"""
Phase registry - single source of truth for all phase metadata.

This module eliminates the need to edit 5+ locations when adding a new phase.
All phase metadata (name, output_dir, runner, patterns) is defined here.
"""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class PhaseInfo:
    """Metadata for a single phase."""
    id: str                     # "3.5", "4.8", etc. (string to avoid float precision issues)
    name: str                   # "Temperature Robustness"
    output_dir: str             # "data/phase3_5"
    module: str                 # "phase3_5_temperature_robustness.temperature_runner"
    runner: str                 # "TemperatureRobustnessRunner" (class) or "run_func" (function)
    runner_type: str = "class"  # "class" (call .run()) or "function" (call directly)
    category: str = "other"     # "data_prep", "feature_discovery", "validation", "steering", etc.
    patterns: Union[str, list[str]] = ""  # File patterns for auto-discovery
    exclude_keywords: Optional[list[str]] = None  # Keywords to exclude in file search
    experiment_modes: Optional[dict] = None  # CLI flag -> config attr mapping for experiment modes


# =============================================================================
# PHASE REGISTRY - Single source of truth for all 35 phases
# =============================================================================

PHASES: dict[str, PhaseInfo] = {
    # =========================================================================
    # Phase 0.x: Data Preparation
    # =========================================================================
    "0": PhaseInfo(
        id="0",
        name="Difficulty Analysis",
        output_dir="data/phase0",
        module="phase0_difficulty_analysis.mbpp_preprocessor",
        runner="MBPPPreprocessor",
        runner_type="class",
        category="data_prep",
        patterns="mbpp_with_complexity_*.parquet",
    ),
    "0.1": PhaseInfo(
        id="0.1",
        name="Problem Splitting",
        output_dir="data/phase0_1",
        module="phase0_1_problem_splitting.problem_splitter",
        runner="Phase01Runner",
        runner_type="class",
        category="data_prep",
        patterns="split_metadata.json",
    ),
    "0.2": PhaseInfo(
        id="0.2",
        name="HumanEval to MBPP Conversion",
        output_dir="data/phase0_2_humaneval",
        module="phase0_2_humaneval_preprocessing.runner",
        runner="run_phase_0_2",
        runner_type="function",
        category="data_prep",
        patterns="humaneval.parquet",
    ),
    "0.3": PhaseInfo(
        id="0.3",
        name="HumanEval Import Scanning",
        output_dir="data/phase0_3_humaneval",
        module="phase0_3_humaneval_imports.runner",
        runner="run_phase_0_3",
        runner_type="function",
        category="data_prep",
        patterns="required_imports.json",
    ),

    # =========================================================================
    # Phase 1: Dataset Building
    # =========================================================================
    "1": PhaseInfo(
        id="1",
        name="Dataset Building",
        output_dir="data/phase1_0",
        module="phase1_latent_selection_dataset.runner",
        runner="Phase1Runner",
        runner_type="class",
        category="dataset",
        patterns="dataset_*.parquet",
        exclude_keywords=["checkpoint", "autosave", "emergency"],
    ),

    # =========================================================================
    # Phase 2.x: Feature Discovery
    # =========================================================================
    "2.2": PhaseInfo(
        id="2.2",
        name="Pile Activation Caching",
        output_dir="data/phase2_2",
        module="phase2_2_pile_caching.runner",
        runner="run_phase2_2_caching",
        runner_type="function",
        category="feature_discovery",
        patterns="pile_activations/*.safetensors",
    ),
    "2.3": PhaseInfo(
        id="2.3",
        name="Pile SAE Frequency Computation",
        output_dir="data/phase2_3",
        module="phase2_3_pile_frequencies.runner",
        runner="run_phase_2_3",
        runner_type="function",
        category="feature_discovery",
        patterns="layer_*_frequencies.safetensors",
    ),
    "2.5": PhaseInfo(
        id="2.5",
        name="SAE Analysis with Pile Filtering",
        output_dir="data/phase2_5",
        module="phase2_5_separation_score_analysis.sae_analyzer",
        runner="SimplifiedSAEAnalyzer",
        runner_type="class",
        category="feature_discovery",
        patterns=["top_20_latents.json"],
    ),
    "2.10": PhaseInfo(
        id="2.10",
        name="T-Statistic Latent Selection",
        output_dir="data/phase2_10",
        module="phase2_10_t_statistic_latent_selector.t_statistic_selector",
        runner="TStatisticSelector",
        runner_type="class",
        category="feature_discovery",
        patterns=["top_20_latents.json"],
    ),
    "2.15": PhaseInfo(
        id="2.15",
        name="Layer-wise Analysis Visualization",
        output_dir="data/phase2_15",
        module="phase2_15_layerwise_visualization.layerwise_visualizer",
        runner="LayerwiseVisualizer",
        runner_type="class",
        category="feature_discovery",
        patterns="*.png",
    ),
    "2.20": PhaseInfo(
        id="2.20",
        name="Latent Landscape Visualization",
        output_dir="data/phase2_20",
        module="phase2_20_latent_landscape.latent_landscape_visualizer",
        runner="Phase220Runner",
        runner_type="class",
        category="feature_discovery",
        patterns=["latent_landscape_scatter.png", "separation_score_distribution.png", "t_statistic_distribution.png"],
    ),

    # =========================================================================
    # Phase 3.x: Statistical Validation
    # =========================================================================
    "3": PhaseInfo(
        id="3",
        name="Validation",
        output_dir="data/phase3",
        module="",  # Not implemented
        runner="",
        runner_type="placeholder",
        category="validation",
        patterns=["validation_results_*.json", "steering_results_*.json"],
    ),
    "3.5": PhaseInfo(
        id="3.5",
        name="Temperature Robustness",
        output_dir="data/phase3_5",
        module="phase3_5_temperature_robustness.temperature_runner",
        runner="TemperatureRobustnessRunner",
        runner_type="class",
        category="validation",
        patterns=["dataset_temp_*.parquet", "metadata.json"],
    ),
    "3.6": PhaseInfo(
        id="3.6",
        name="Hyperparameter Tuning Set Processing",
        output_dir="data/phase3_6",
        module="phase3_6_hyperparameter_baseline.hyperparameter_runner",
        runner="HyperparameterDataRunner",
        runner_type="class",
        category="validation",
        patterns=["dataset_hyperparams_temp_0_0.parquet", "metadata.json"],
    ),
    "3.8": PhaseInfo(
        id="3.8",
        name="AUROC and F1 Evaluation",
        output_dir="data/phase3_8",
        module="phase3_8_auroc_f1_evaluation.auroc_f1_evaluator",
        runner="Phase38Runner",
        runner_type="class",
        category="validation",
        patterns=["evaluation_results.json", "*.png", "evaluation_summary.txt"],
    ),
    "3.10": PhaseInfo(
        id="3.10",
        name="Temperature-Based AUROC Analysis",
        output_dir="data/phase3_10",
        module="phase3_10_temperature_auroc_f1.temperature_evaluator",
        runner="TemperatureAUROCEvaluator",
        runner_type="class",
        category="validation",
        patterns=["temperature_analysis_results.json", "*.png", "temperature_summary.txt"],
    ),
    "3.11": PhaseInfo(
        id="3.11",
        name="Temperature Trends Visualization Update",
        output_dir="data/phase3_11",
        module="phase3_11_temperature_trends_updated.temperature_trends_visualizer",
        runner="TemperatureTrendsVisualizer",
        runner_type="class",
        category="validation",
        patterns="*.png",
    ),
    "3.12": PhaseInfo(
        id="3.12",
        name="Difficulty-Based AUROC Analysis",
        output_dir="data/phase3_12",
        module="phase3_12_difficulty_auroc_f1.difficulty_evaluator",
        runner="Phase312Runner",
        runner_type="class",
        category="validation",
        patterns=["difficulty_analysis_results.json", "*.png", "difficulty_summary.txt"],
    ),

    # =========================================================================
    # Phase 4.x: Causal Validation (Steering)
    # =========================================================================
    "4.5": PhaseInfo(
        id="4.5",
        name="Steering Coefficient Selection",
        output_dir="data/phase4_5",
        module="phase4_5_coefficient_grid_search.steering_coefficient_selector",
        runner="SteeringCoefficientSelector",
        runner_type="class",
        category="steering",
        patterns=["selected_coefficients.json", "phase_4_5_summary.json"],
        experiment_modes={
            "config_attr": "phase4_5_experiment_mode",
            "flags": [("correction_only", "correction"), ("corruption_only", "corruption")],
        },
    ),
    "4.6": PhaseInfo(
        id="4.6",
        name="Golden Section Search Coefficient Refinement",
        output_dir="data/phase4_6",
        module="phase4_6_golden_section_refinement.golden_section_refiner",
        runner="GoldenSectionCoefficientRefiner",
        runner_type="class",
        category="steering",
        patterns="refined_coefficients.json",
        experiment_modes={
            "config_attr": "phase4_6_experiment_mode",
            "flags": [("correction_only", "correction"), ("corruption_only", "corruption")],
        },
    ),
    "4.7": PhaseInfo(
        id="4.7",
        name="Coefficient Optimization Visualization",
        output_dir="data/phase4_7",
        module="phase4_7_coefficient_visualization.coefficient_plotter",
        runner="Phase47Runner",
        runner_type="class",
        category="steering",
        patterns="*.png",
    ),
    "4.8": PhaseInfo(
        id="4.8",
        name="Steering Effect Analysis",
        output_dir="data/phase4_8",
        module="phase4_8_steering_analysis.steering_effect_analyzer",
        runner="SteeringEffectAnalyzer",
        runner_type="class",
        category="steering",
        patterns=["steering_effect_analysis.json", "phase_4_8_summary.json"],
        experiment_modes={
            "config_attr": "phase4_8_experiment_mode",
            "flags": [("preservation_only", "preservation"), ("correction_only", "correction"), ("corruption_only", "corruption")],
        },
    ),
    "4.10": PhaseInfo(
        id="4.10",
        name="Zero-Discrimination Feature Selection",
        output_dir="data/phase4_10",
        module="phase4_10_zero_discrimination.zero_discrimination_selector",
        runner="ZeroDiscriminationSelector",
        runner_type="class",
        category="steering",
        patterns=["zero_discrimination_features.json", "zero_discrimination_summary.json"],
    ),
    "4.12": PhaseInfo(
        id="4.12",
        name="Zero-Discrimination Steering Generation",
        output_dir="data/phase4_12",
        module="phase4_12_zero_disc_steering.zero_disc_steering_generator",
        runner="ZeroDiscSteeringGenerator",
        runner_type="class",
        category="steering",
        patterns=["zero_disc_steering_results.json"],
    ),
    "4.14": PhaseInfo(
        id="4.14",
        name="Statistical Significance Testing",
        output_dir="data/phase4_14",
        module="phase4_14_statistical_significance.significance_tester",
        runner="SignificanceTester",
        runner_type="class",
        category="steering",
        patterns=["significance_test_results.json", "significance_summary.json", "triangulation_analysis.json"],
    ),
    "4.16": PhaseInfo(
        id="4.16",
        name="Difficulty-Stratified Steering Analysis",
        output_dir="data/phase4_16",
        module="phase4_16_difficulty_steering.difficulty_steering_analyzer",
        runner="Phase416Runner",
        runner_type="class",
        category="steering",
        patterns="difficulty_steering_results.json",
    ),

    # =========================================================================
    # Phase 5.x: Weight Orthogonalization
    # =========================================================================
    "5.3": PhaseInfo(
        id="5.3",
        name="Weight Orthogonalization Analysis",
        output_dir="data/phase5_3",
        module="phase5_3_weight_orthogonalization.weight_orthogonalizer",
        runner="WeightOrthogonalizer",
        runner_type="class",
        category="orthogonalization",
        patterns=["orthogonalization_results.json", "phase_5_3_summary.json"],
    ),
    "5.6": PhaseInfo(
        id="5.6",
        name="Zero-Discrimination Weight Orthogonalization",
        output_dir="data/phase5_6",
        module="phase5_6_zero_disc_orthogonalization.zero_disc_weight_orthogonalizer",
        runner="ZeroDiscWeightOrthogonalizer",
        runner_type="class",
        category="orthogonalization",
        patterns=["zero_disc_orthogonalization_results.json", "phase_5_6_summary.json"],
    ),
    "5.9": PhaseInfo(
        id="5.9",
        name="Weight Orthogonalization Statistical Significance",
        output_dir="data/phase5_9",
        module="phase5_9_orthogonalization_significance.orthogonalization_significance_tester",
        runner="OrthogonalizationSignificanceTester",
        runner_type="class",
        category="orthogonalization",
        patterns=["orthogonalization_triangulation.json", "phase_5_9_summary.json"],
    ),

    # =========================================================================
    # Phase 6.x: Attention Analysis
    # =========================================================================
    "6.3": PhaseInfo(
        id="6.3",
        name="Attention Pattern Analysis",
        output_dir="data/phase6_3",
        module="phase6_3_attention_analysis.attention_analyzer",
        runner="AttentionAnalyzer",
        runner_type="class",
        category="attention",
        patterns=["attention_analysis_results.json", "phase_6_3_summary.json"],
    ),

    # =========================================================================
    # Phase 7.x: Model Comparison (Instruction-Tuned)
    # =========================================================================
    "7.3": PhaseInfo(
        id="7.3",
        name="Instruction-Tuned Model Baseline",
        output_dir="data/phase7_3",
        module="phase7_3_instruct_baseline.instruct_baseline_runner",
        runner="InstructBaselineRunner",
        runner_type="class",
        category="instruct",
        patterns=["dataset_instruct_temp_0_0.parquet", "metadata.json"],
    ),
    "7.6": PhaseInfo(
        id="7.6",
        name="Instruction-Tuned Model Steering Analysis",
        output_dir="data/phase7_6",
        module="phase7_6_instruct_steering.instruct_steering_analyzer",
        runner="InstructSteeringAnalyzer",
        runner_type="class",
        category="instruct",
        patterns=["steering_effect_analysis.json", "phase_7_6_summary.json"],
    ),
    "7.9": PhaseInfo(
        id="7.9",
        name="Universality Analysis",
        output_dir="data/phase7_9",
        module="phase7_9_universality_analysis.universality_analysis",
        runner="Phase79Runner",
        runner_type="class",
        category="instruct",
        patterns=["universality_metrics.json", "phase_7_9_summary.json"],
    ),
    "7.12": PhaseInfo(
        id="7.12",
        name="Instruction-Tuned Model AUROC/F1 Evaluation",
        output_dir="data/phase7_12",
        module="phase7_12_instruct_auroc_f1.instruct_auroc_f1_evaluator",
        runner="Phase712Runner",
        runner_type="class",
        category="instruct",
        patterns=["evaluation_results.json", "evaluation_summary.txt"],
    ),

    # =========================================================================
    # Phase 8.x: Selective Steering
    # =========================================================================
    "8.1": PhaseInfo(
        id="8.1",
        name="Percentile Threshold Calculator",
        output_dir="data/phase8_1",
        module="phase8_1_threshold_calculator.runner",
        runner="run_phase_8_1",
        runner_type="function",
        category="selective",
        patterns=["percentile_thresholds.json", "threshold_summary.txt"],
    ),
    "8.2": PhaseInfo(
        id="8.2",
        name="Percentile Threshold Optimizer",
        output_dir="data/phase8_2",
        module="phase8_2_threshold_optimizer.runner",
        runner="run_phase_8_2",
        runner_type="function",
        category="selective",
        patterns=["optimal_percentile.json", "threshold_comparison.json"],
    ),
    "8.3": PhaseInfo(
        id="8.3",
        name="Selective Steering Based on Threshold Analysis",
        output_dir="data/phase8_3",
        module="phase8_3_selective_steering.selective_steering_analyzer",
        runner="SelectiveSteeringAnalyzer",
        runner_type="class",
        category="selective",
        patterns=["selective_steering_summary.json", "selective_correction_results.json", "selective_preservation_results.json"],
    ),
}


# =============================================================================
# Registry API Functions
# =============================================================================

def get_phase(phase_id: str) -> PhaseInfo:
    """
    Get phase info by ID.

    Args:
        phase_id: Phase ID as string (e.g., "3.5", "4.8")

    Returns:
        PhaseInfo for the requested phase

    Raises:
        ValueError: If phase_id is not found in registry
    """
    if phase_id not in PHASES:
        valid_phases = ", ".join(sorted(PHASES.keys(), key=lambda x: float(x)))
        raise ValueError(f"Unknown phase: {phase_id}. Valid phases: {valid_phases}")
    return PHASES[phase_id]


def get_all_phase_ids() -> list[str]:
    """
    Get all valid phase IDs for CLI choices.

    Returns:
        List of phase IDs sorted numerically (e.g., ["0", "0.1", "0.2", "1", "2.2", ...])
    """
    return sorted(PHASES.keys(), key=lambda x: float(x))


def get_phase_choices_help() -> str:
    """
    Generate help text for argparse showing all phases.

    Returns:
        Formatted string like "0=Difficulty Analysis, 0.1=Problem Splitting, ..."
    """
    sorted_ids = get_all_phase_ids()
    return ", ".join(f"{pid}={PHASES[pid].name}" for pid in sorted_ids)


def get_phases_by_category(category: str) -> list[PhaseInfo]:
    """
    Get all phases in a specific category.

    Args:
        category: Category name (e.g., "validation", "steering")

    Returns:
        List of PhaseInfo objects in that category
    """
    return [p for p in PHASES.values() if p.category == category]


def get_phase_output_dir(phase_id: str) -> str:
    """
    Get output directory for a phase.

    Args:
        phase_id: Phase ID as string

    Returns:
        Output directory path
    """
    return get_phase(phase_id).output_dir


def get_phase_patterns(phase_id: str) -> Union[str, list[str]]:
    """
    Get file patterns for auto-discovery.

    Args:
        phase_id: Phase ID as string

    Returns:
        Glob pattern(s) for finding phase output files
    """
    return get_phase(phase_id).patterns
