#!/usr/bin/env python3
"""

Usage:
    python3 run.py phase 0                                      # Difficulty analysis
    python3 run.py phase 1 --model google/gemma-2-2b           # Dataset building (single GPU)
    python3 run.py phase 2.5                                    # SAE analysis with pile filtering (auto-discovers input)
    python3 run.py phase 3                                      # Validation (auto-discovers input)
    
Manual input override:
    python3 run.py phase 1 --input data/phase0/mapping.parquet  # Use specific difficulty mapping
    python3 run.py phase 2.5 --input data/phase1_0/dataset.parquet  # Use specific dataset
    python3 run.py phase 3 --input data/phase2/results.json       # Use specific SAE results
    
For multi-GPU dataset building, use:
    python3 multi_gpu_launcher.py --phase 1 --num-gpus 3 --model google/gemma-2-2b
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
from os import environ
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common.logging import set_logging_phase, get_logger
from common.gpu_utils import cleanup_gpu_memory, ensure_gpu_available, setup_cuda_environment
from common import MAX_NEW_TOKENS
from common.config import Config

# Import centralized phase directory function
from common.utils import get_phase_dir


def setup_argument_parser():
    """Setup command line argument parser with phase-specific argument groups"""
    parser = ArgumentParser(
        description="Run phases",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test GPU command
    test_gpu_parser = subparsers.add_parser('test-gpu', help='Test GPU detection and memory')
    test_gpu_parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed GPU information'
    )
    
    # GPU cleanup command
    cleanup_gpu_parser = subparsers.add_parser('cleanup-gpu', help='Clean GPU memory and zombie contexts')
    cleanup_gpu_parser.add_argument(
        '--aggressive',
        action='store_true',
        help='Use aggressive cleanup methods'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show current system status')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate system dependencies and setup')
    
    # Test phase commands
    test_phase1_parser = subparsers.add_parser('test-phase1', help='Quick test of Phase 1 with 10 records')
    
    test_phase2_parser = subparsers.add_parser('test-phase2', help='Quick test of Phase 2 SAE analysis with small sample')
    test_phase2_parser.add_argument(
        '--layer',
        type=int,
        default=13,
        help='Layer to test (default: 13 for Gemma-2B)'
    )
    test_phase2_parser.add_argument(
        '--samples',
        type=int, 
        default=5,
        help='Number of samples per class to test (default: 5)'
    )
    
    # Phase command (existing functionality)
    phase_parser = subparsers.add_parser('phase', help='Run a specific phase')
    
    # Required phase selection
    phase_parser.add_argument(
        'phase',
        type=float,
        choices=[0, 0.1, 0.2, 0.3, 1, 2.2, 2.5, 2.10, 2.15, 3, 3.5, 3.6, 3.8, 3.10, 3.11, 3.12, 4.5, 4.6, 4.7, 4.8, 4.10, 4.12, 4.14, 5.3, 5.6, 5.9, 6.3, 7.3, 7.6, 7.9, 7.12, 8.1, 8.2, 8.3],
        help='Phase to run: 0=Difficulty Analysis, 0.1=Problem Splitting, 0.2=HumanEval to MBPP Conversion, 0.3=HumanEval Import Scanning, 1=Dataset Building, 2.2=Pile Caching, 2.5=SAE Analysis with Filtering, 2.10=T-Statistic Latent Selection, 2.15=Layer-wise Analysis Visualization, 3=Validation, 3.5=Temperature Robustness, 3.6=Hyperparameter Tuning Set Processing, 3.8=AUROC and F1 Evaluation, 3.10=Temperature-Based AUROC Analysis, 3.11=Temperature Trends Visualization Update, 3.12=Difficulty-Based AUROC Analysis, 4.5=Steering Coefficient Selection, 4.6=Golden Section Search Coefficient Refinement, 4.7=Coefficient Optimization Visualization, 4.8=Steering Effect Analysis, 4.10=Zero-Discrimination Feature Selection, 4.12=Zero-Discrimination Steering Generation, 4.14=Statistical Significance Testing, 5.3=Weight Orthogonalization, 5.6=Zero-Discrimination Weight Orthogonalization, 5.9=Weight Orthogonalization Statistical Significance, 6.3=Attention Pattern Analysis, 7.3=Instruction-Tuned Model Baseline, 7.6=Instruction-Tuned Model Steering Analysis, 7.9=Universality Analysis, 7.12=Instruction-Tuned Model AUROC/F1 Evaluation, 8.1=Percentile Threshold Calculator, 8.2=Percentile Threshold Optimizer, 8.3=Selective Steering Based on Threshold Analysis'
    )
    
    # Global arguments (add to phase parser)
    phase_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Config management arguments
    phase_parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show the final configuration and exit'
    )
    
    # Universal input argument for all phases
    phase_parser.add_argument(
        '--input',
        type=str,
        help='Input file from previous phase (overrides auto-discovery). '
             'Phase 1: difficulty mapping (.parquet), '
             'Phase 2.5: dataset (.parquet), '
             'Phase 3: SAE results (.json)'
    )

    # Common dataset range arguments for Phase 1 and Phase 3.5
    phase_parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Starting index for dataset (Phase 1: MBPP dataset, Phase 3.5: validation dataset)'
    )
    phase_parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='Ending index for dataset (inclusive). If not specified, processes to end of dataset'
    )
    
    # Common steering experiment mode arguments for Phases 4.5, 4.6, and 4.8
    phase_parser.add_argument(
        '--correction-only',
        action='store_true',
        help='Run only correction experiment (applies to phases 4.5, 4.6, 4.8)'
    )
    phase_parser.add_argument(
        '--corruption-only',
        action='store_true',
        help='Run only corruption experiment (applies to phases 4.5, 4.6, 4.8)'
    )
    phase_parser.add_argument(
        '--preservation-only',
        action='store_true',
        help='Run only preservation experiment (applies to phase 4.8)'
    )
    
    # Phase 0: Difficulty Analysis arguments
    phase0_group = phase_parser.add_argument_group('Phase 0: Difficulty Analysis')
    phase0_group.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save difficulty mapping (default: data/phase0)'
    )
    phase0_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Run analysis without saving difficulty mapping to file'
    )
    
    # Phase 1: Dataset Building arguments
    phase1_group = phase_parser.add_argument_group('Phase 1: Dataset Building')
    phase1_group.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name to use for dataset building (defaults to config.py setting)'
    )
    phase1_group.add_argument(
        '--dataset-dir',
        type=str,
        default=get_phase_dir('1'),
        help='Directory for dataset files'
    )
    
    # Phase 2.2: Pile Activation Caching arguments
    phase2_2_group = phase_parser.add_argument_group('Phase 2.2: Pile Activation Caching')
    phase2_2_group.add_argument(
        '--run-count',
        type=int,
        default=10000,
        help='Number of pile samples to process (default: 10000, use small value for testing)'
    )
    
    # Phase 0.1: Problem Splitting arguments
    phase0_1_group = phase_parser.add_argument_group('Phase 0.1: Problem Splitting')
    phase0_1_group.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible splitting (default: 42)'
    )
    phase0_1_group.add_argument(
        '--n-strata',
        type=int,
        default=10,
        help='Number of complexity strata for stratified sampling (default: 10)'
    )
    phase0_1_group.add_argument(
        '--split-output-dir',
        type=str,
        default=get_phase_dir('0.1'),
        help='Directory to save split task IDs'
    )
    phase0_1_group.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate split metadata after splitting'
    )
    
    # Phase 3.5: Temperature Robustness arguments
    phase3_5_group = phase_parser.add_argument_group('Phase 3.5: Temperature Robustness')
    
    # Phase 2.5: SAE Analysis arguments
    phase2_5_group = phase_parser.add_argument_group('Phase 2.5: SAE Analysis with Pile Filtering')
    phase2_5_group.add_argument(
        '--sae-model',
        type=str,
        help='Path to SAE model'
    )
    phase2_5_group.add_argument(
        '--latent-threshold',
        type=float,
        default=0.02,
        help='Activation threshold for latent filtering'
    )
    phase2_5_group.add_argument(
        '--no-pile-filter',
        action='store_true',
        help='Disable Pile dataset filtering (enabled by default)'
    )
    phase2_5_group.add_argument(
        '--pile-threshold',
        type=float,
        default=0.02,
        help='Maximum Pile activation frequency (default: 2%%)'
    )
    phase2_5_group.add_argument(
        '--pile-samples',
        type=int,
        default=10000,
        help='Number of Pile samples to use (default: 10000)'
    )
    
    # Phase 2.10: T-Statistic Latent Selection
    phase2_10_group = phase_parser.add_argument_group('Phase 2.10: T-Statistic Latent Selection')
    phase2_10_group.add_argument(
        '--t-min-samples',
        type=int,
        default=10,
        help='Minimum samples for reliable t-test (default: 10)'
    )
    # Note: Phase 2.10 reuses pile filtering args from Phase 2.5
    
    # Phase 3: Validation arguments
    phase3_group = phase_parser.add_argument_group('Phase 3: Validation')
    phase3_group.add_argument(
        '--temperatures',
        nargs='+',
        type=float,
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help='Temperature values for robustness testing'
    )
    phase3_group.add_argument(
        '--steering-coeffs',
        nargs='+',
        type=float,
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
        help='Steering coefficients'
    )
    
    # Phase 4.5: Steering Coefficient Selection arguments
    # (experiment mode arguments are defined at top level)
    phase4_5_group = phase_parser.add_argument_group('Phase 4.5: Steering Coefficient Selection')
    
    # Phase 4.6: Golden Section Search Refinement arguments  
    # (experiment mode arguments are defined at top level)
    phase4_6_group = phase_parser.add_argument_group('Phase 4.6: Golden Section Search Refinement')
    
    # Phase 4.8: Steering Effect Analysis arguments
    # (experiment mode arguments are defined at top level)
    phase4_8_group = phase_parser.add_argument_group('Phase 4.8: Steering Effect Analysis')
    
    return parser


def validate_phase_arguments(args):
    """Validate phase-specific argument requirements"""
    # Phase 1 model is now optional - defaults to config.py setting

    # Validate input file if provided
    if args.input and not Path(args.input).exists():
        raise ValueError(f"Input file not found: {args.input}")


def run_phase0(config: Config, logger, device: str, dry_run: bool = False):
    """Run Phase 0: Difficulty Analysis"""
    from phase0_difficulty_analysis.mbpp_preprocessor import MBPPPreprocessor
    
    logger.info("Starting Phase 0: MBPP difficulty analysis")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="0"))
    
    # Initialize preprocessor with output directory
    preprocessor = MBPPPreprocessor(output_dir=config.phase0_output_dir)
    
    # COMPUTATION PATH: Analyze all 974 MBPP problems from scratch
    # This loops through dataset, calculates cyclomatic complexity, test metrics
    # Creates new timestamped files: mbpp_difficulty_mapping_{timestamp}.parquet + .summary.json
    try:
        enriched_df = preprocessor.preprocess_dataset(save_mapping=not dry_run)
        
        logger.info("✅ Phase 0 completed successfully")
        logger.info(f"Analyzed {len(enriched_df)} MBPP problems")
        
        # FILE CREATION: Show path to newly created parquet file (only if files were saved)
        if not dry_run:
            # Gets most recent enriched dataset file
            latest_enriched = preprocessor.get_latest_enriched_dataset_path()
            if latest_enriched:
                logger.info(f"Enriched dataset available at: {latest_enriched}")
            else:
                logger.error("Failed to find saved enriched dataset after Phase 0 completion")
            
    except Exception as e:
        logger.error(f"❌ Phase 0 failed: {str(e)}")
        sys.exit(1)


def run_phase1(config: Config, logger, device: str):
    """Run Phase 1: Dataset Building using SAE split"""
    # Import simplified implementation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))  # Add project root to path
    from phase1_simplified.runner import Phase1Runner
    
    logger.info("Starting Phase 1: Dataset Building")
    logger.info(f"Model: {config.model_name}, Split: sae")
    logger.info("Processing mode: Sequential (use multi_gpu_launcher.py for parallel processing)")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="1"))
    
    # Verify Phase 0.1 has been run (split files must exist)
    sae_split_path = Path(config.phase0_1_output_dir) / "sae_mbpp.parquet"
    if not sae_split_path.exists():
        logger.error(f"SAE split not found at {sae_split_path}")
        logger.error("Phase 1 requires Phase 0.1 to be completed first.")
        logger.error("Please run: python3 run.py phase 0.1")
        sys.exit(1)
    
    logger.info(f"Found SAE split: {sae_split_path}")
    
    # Setup CUDA environment and cleanup GPUs before starting
    if torch.cuda.is_available() and device == "cuda":
        logger.info("Setting up CUDA environment and cleaning GPU memory...")
        setup_cuda_environment()
        cleanup_gpu_memory()
    
    # Create and run simplified runner
    runner = Phase1Runner(config)
    final_df = runner.run(split_name='sae')
    
    logger.info("✅ Phase 1 completed successfully")


def run_phase0_1(config: Config, logger, device: str):
    """Run Phase 0.1: Problem Splitting"""
    import pandas as pd
    from pathlib import Path
    from phase0_1_problem_splitting import split_problems
    from common.utils import discover_latest_phase_output
    
    logger.info("Starting Phase 0.1: Problem Splitting")
    logger.info("Split ratios: 50% SAE, 10% hyperparameters, 40% validation")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="0.1"))
    
    # Auto-discover or use provided difficulty mapping
    if hasattr(config, '_input_file') and config._input_file:
        mapping_path = config._input_file
        logger.info(f"Using provided difficulty mapping: {mapping_path}")
    else:
        logger.info("Auto-discovering difficulty mapping from Phase 0...")
        mapping_path = discover_latest_phase_output("0", phase_dir=config.phase0_output_dir)
        
        if not mapping_path:
            logger.error("No Phase 0 difficulty mapping found! Please run Phase 0 first.")
            sys.exit(1)
        
        logger.info(f"Found difficulty mapping: {mapping_path}")
    
    # Load difficulty mapping
    try:
        df = pd.read_parquet(mapping_path)
        logger.info(f"Loaded difficulty mapping with {len(df)} problems")
        
        # Check minimum size
        if len(df) < 10:
            logger.error(f"Too few problems for splitting: {len(df)} problems (minimum 10 required)")
            sys.exit(1)
        
        # Verify required columns
        required_columns = ['task_id', 'cyclomatic_complexity']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Difficulty mapping missing required columns: {required_columns}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to load difficulty mapping: {e}")
        sys.exit(1)
    
    # Perform splitting with unified config
    logger.info("Performing stratified randomized splitting...")
    splits = split_problems(df, config)
    
    # Log split results
    for split_name, task_ids in splits.items():
        logger.info(f"Split '{split_name}': {len(task_ids)} problems ({len(task_ids)/len(df):.1%})")
    
    logger.info("✅ Phase 0.1 completed successfully")
    logger.info(f"Splits saved to: {config.phase0_1_output_dir}")


def run_phase0_2(config: Config, logger, device: str):
    """Run Phase 0.2: HumanEval to MBPP Conversion"""
    from phase0_2_humaneval_preprocessing.runner import run_phase_0_2

    logger.info("Starting Phase 0.2: HumanEval to MBPP Conversion")
    logger.info("Converting HumanEval dataset to MBPP format for seamless integration")

    # Log configuration
    logger.info("\n" + config.dump(phase="0.2"))

    # Run the conversion
    run_phase_0_2(config)

    logger.info("✅ Phase 0.2 completed successfully")
    logger.info(f"Converted dataset saved to: {config.phase0_2_output_dir}/humaneval.parquet")


def run_phase0_3(config: Config, logger, device: str):
    """Run Phase 0.3: HumanEval Import Scanning"""
    from phase0_3_humaneval_imports.runner import run_phase_0_3

    logger.info("Starting Phase 0.3: HumanEval Import Scanning")
    logger.info("Scanning all HumanEval prompts for required import statements")

    # Log configuration
    logger.info("\n" + config.dump(phase="0.3"))

    # Run the scanner
    run_phase_0_3(config)

    logger.info("✅ Phase 0.3 completed successfully")
    logger.info(f"Import list saved to: {config.phase0_3_output_dir}/required_imports.json")


def run_phase2_2(config: Config, logger, device: str):
    """Run Phase 2.2: Cache Pile Activations"""
    from phase2_2_pile_caching.runner import run_phase2_2_caching
    
    logger.info("Starting Phase 2.2: Pile Activation Caching")
    logger.info("This phase extracts activations from diverse text at random word positions")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="2.2"))
    
    # Check if we have a run count override
    run_count = getattr(config, '_run_count', config.pile_samples)
    logger.info(f"Will process {run_count} pile samples")
    
    # Run the pile caching
    run_phase2_2_caching(config, device)
    
    logger.info("✅ Phase 2.2 completed successfully")


def run_phase2_5(config: Config, logger, device: str):
    """Run Phase 2.5: SAE Analysis with Pile Filtering using simplified implementation"""
    from phase2_5_simplified.sae_analyzer import SimplifiedSAEAnalyzer
    
    logger.info("Starting Phase 2.5: SAE Analysis with Pile Filtering")
    logger.info("Using simplified implementation")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="2"))
    
    # Create and run analyzer
    analyzer = SimplifiedSAEAnalyzer(config)
    results = analyzer.run()
    
    logger.info("\n✅ Phase 2.5 completed successfully")


def run_phase2_10(config: Config, logger, device: str):
    """Run Phase 2.10: T-Statistic based latent selection"""
    from phase2_10_t_statistic_latent_selector.t_statistic_selector import TStatisticSelector

    logger.info("Starting Phase 2.10: T-Statistic Based Latent Selection")
    logger.info("Using Welch's t-test for feature selection")

    # Log configuration
    logger.info("\n" + config.dump(phase="2.10"))

    # Create and run selector
    selector = TStatisticSelector(config)
    results = selector.run()

    logger.info("\n✅ Phase 2.10 completed successfully")


def run_phase2_15(config: Config, logger, device: str):
    """Run Phase 2.15: Layer-wise Analysis Visualization

    NOTE: This phase was developed but NOT included in the final paper.
    See phase2_15_layerwise_visualization/layerwise_visualizer.py for detailed rationale.
    The code is preserved for supplementary materials and future analysis.
    """
    from phase2_15_layerwise_visualization.layerwise_visualizer import LayerwiseVisualizer

    logger.info("Starting Phase 2.15: Layer-wise Analysis Visualization")
    logger.info("NOTE: This visualization is NOT included in the final paper (see code comments)")
    logger.info("Creating heatmaps for t-statistics and separation scores across layers")

    # Log configuration
    logger.info("\n" + config.dump(phase="2.15"))

    # Create and run visualizer
    visualizer = LayerwiseVisualizer(config)
    results = visualizer.run()

    logger.info("\n✅ Phase 2.15 completed successfully")


def run_phase3(config: Config, logger, device: str):
    """Run Phase 3: Validation"""
    from common.utils import discover_latest_phase_output
    
    logger.info("Starting Phase 3: Validation")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="3"))
    
    # Get SAE results from input or auto-discovery
    if hasattr(config, '_input_file') and config._input_file:
        logger.info(f"Using specified SAE results: {config._input_file}")
        sae_results_path = config._input_file
    else:
        logger.info("Auto-discovering SAE results from Phase 2.5...")
        sae_results_path = discover_latest_phase_output("2.5")
        if not sae_results_path:
            logger.error(f"No SAE results found in {config.phase2_5_output_dir}. Please run Phase 2.5 first or specify --input")
            sys.exit(1)
        logger.info(f"Found SAE results: {sae_results_path}")
    
    # Always auto-discover dataset (Phase 3 needs both)
    logger.info("Auto-discovering dataset from Phase 1...")
    dataset_path = discover_latest_phase_output("1")
    if not dataset_path:
        logger.error(f"No dataset found in {config.phase1_output_dir}. Please run Phase 1 first")
        sys.exit(1)
    logger.info(f"Found dataset: {dataset_path}")
    
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"SAE Results: {sae_results_path}")
    logger.info(f"Temperatures to test: {config.validation_temperatures}")
    logger.info(f"Steering coefficients: {config.validation_steering_coeffs}")
    
    # TODO: Implement validation
    logger.info("Validation not yet implemented")
    logger.info("This phase will run statistical validation and model steering experiments")
    logger.info(f"Results will be saved to {config.phase3_output_dir}/")


def run_phase3_5(config: Config, logger, device: str):
    """Run Phase 3.5: Temperature Robustness Testing"""
    from phase3_5_temperature_robustness.temperature_runner import TemperatureRobustnessRunner
    
    logger.info("Starting Phase 3.5: Temperature Robustness Testing")
    logger.info("Will auto-discover best layers from Phase 2.5 output")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="3.5"))
    
    # Create and run temperature robustness runner
    runner = TemperatureRobustnessRunner(config)
    metadata = runner.run()
    
    logger.info("\n✅ Phase 3.5 completed successfully")


def run_phase3_6(config: Config, logger, device: str):
    """Run Phase 3.6: Hyperparameter Tuning Set Processing"""
    from phase3_6.hyperparameter_runner import HyperparameterDataRunner
    
    logger.info("Starting Phase 3.6: Hyperparameter Tuning Set Processing")
    logger.info("Will auto-discover best layers from Phase 3.5 output")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="3.6"))
    
    # Create and run hyperparameter runner
    runner = HyperparameterDataRunner(config)
    metadata = runner.run()
    
    logger.info("\n✅ Phase 3.6 completed successfully")


def run_phase7_3(config: Config, logger, device: str):
    """Run Phase 7.3: Instruction-Tuned Model Baseline Generation"""
    from phase7_3_instruct_baseline.instruct_baseline_runner import InstructBaselineRunner
    
    logger.info("Starting Phase 7.3: Instruction-Tuned Model Baseline Generation")
    logger.info("Will auto-discover best layers from Phase 2.5/2.10 output")
    logger.info("Using validation split from Phase 0.1 (388 problems)")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="7.3"))
    
    # Create and run instruction-tuned baseline runner
    runner = InstructBaselineRunner(config)
    metadata = runner.run()
    
    logger.info("\n✅ Phase 7.3 completed successfully")


def run_phase7_6(config: Config, logger, device: str):
    """Run Phase 7.6: Instruction-Tuned Model Steering Analysis"""
    from phase7_6_instruct_steering.instruct_steering_analyzer import InstructSteeringAnalyzer
    
    logger.info("Starting Phase 7.6: Instruction-Tuned Model Steering Analysis")
    logger.info("Will test PVA feature transfer on instruction-tuned model")
    logger.info("Using Phase 7.3 baseline and Phase 2.5 PVA features")
    logger.info(f"Steering coefficients - Correct: {config.phase4_8_correct_coefficient}, Incorrect: {config.phase4_8_incorrect_coefficient}")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="7.6"))
    
    # Create and run instruction-tuned steering analyzer
    analyzer = InstructSteeringAnalyzer(config)
    results = analyzer.run()
    
    logger.info("\n✅ Phase 7.6 completed successfully")


def run_phase7_9(config: Config, logger, device: str):
    """Run Phase 7.9: Universality Analysis"""
    from phase7_9_universality_analysis.universality_analysis import UniversalityAnalyzer

    logger.info("Starting Phase 7.9: Universality Analysis")
    logger.info("Comparing PVA features across base and instruction-tuned models")

    # Log configuration
    logger.info("\n" + config.dump(phase="7.9"))

    # Create and run universality analyzer
    analyzer = UniversalityAnalyzer(config)
    analyzer.load_data()
    results = analyzer.run()

    logger.info("\n✅ Phase 7.9 completed successfully")


def run_phase7_12(config: Config, logger, device: str):
    """Run Phase 7.12: AUROC and F1 Evaluation for Instruction-Tuned Model"""
    import sys
    from pathlib import Path

    logger.info("Starting Phase 7.12: AUROC and F1 Evaluation for Instruction-Tuned Model")
    logger.info("This phase evaluates PVA features on instruction-tuned model outputs from Phase 7.3")

    # Log configuration
    logger.info("\n" + config.dump(phase="7.12"))

    # Set up sys.argv for the instruct_auroc_f1_evaluator script
    original_argv = sys.argv.copy()

    try:
        # Build new argv
        sys.argv = ["instruct_auroc_f1_evaluator.py"]

        # Add optional arguments if provided
        if hasattr(config, '_input_file') and config._input_file:
            input_path = Path(config._input_file)
            if "phase0_1" in str(input_path):
                sys.argv.extend(["--phase0-1-dir", str(input_path.parent)])
            elif "phase7_3" in str(input_path):
                sys.argv.extend(["--phase7-3-dir", str(input_path.parent)])
            else:
                logger.warning(f"Input file {input_path} not recognized as Phase 0.1 or 7.3 output")

        # Use standard phase output directory with dataset suffix
        base_output_dir = get_phase_dir('7.12')
        if config.dataset_name != "mbpp":
            output_dir = f"{base_output_dir}_{config.dataset_name}"
        else:
            output_dir = base_output_dir
        sys.argv.extend(["--output-dir", output_dir])

        logger.info(f"Running Phase 7.12 evaluator with args: {sys.argv[1:]}")

        # Import and run the main function directly
        from phase7_12.instruct_auroc_f1_evaluator import main
        main()

        logger.info("\n✅ Phase 7.12 completed successfully")
        logger.info(f"Results saved to: {output_dir}")

    finally:
        # Restore original argv
        sys.argv = original_argv


def run_phase8_1(config: Config, logger, device: str):
    """Run Phase 8.1: Percentile Threshold Calculator"""
    from phase8_1_threshold_calculator.runner import run_phase_8_1

    logger.info("\n" + "="*60)
    logger.info("PHASE 8.1: PERCENTILE THRESHOLD CALCULATOR")
    logger.info("="*60)
    logger.info("Calculates percentile-based thresholds from Phase 3.6 hyperparameter dataset")
    logger.info("Thresholds will be used by Phase 8.3 for selective steering")
    logger.info("")

    # Log configuration
    logger.info("\n" + config.dump(phase="8.1"))

    # Run threshold calculator
    results = run_phase_8_1(config)

    logger.info("\n✅ Phase 8.1 completed successfully")


def run_phase8_2(config: Config, logger, device: str):
    """Run Phase 8.2: Percentile Threshold Optimizer"""
    from phase8_2_threshold_optimizer.runner import run_phase_8_2

    logger.info("\n" + "="*60)
    logger.info("PHASE 8.2: PERCENTILE THRESHOLD OPTIMIZER")
    logger.info("="*60)
    logger.info("Optimizes threshold selection across percentiles using hyperparameter dataset")
    logger.info("Tests percentiles from Phase 8.1 to maximize net benefit (correction - corruption)")
    logger.info("")

    # Log configuration
    logger.info("\n" + config.dump(phase="8.2"))

    # Run threshold optimizer
    results = run_phase_8_2(config)

    logger.info("\n✅ Phase 8.2 completed successfully")


def run_phase8_3(config: Config, logger, device: str):
    """Run Phase 8.3: Selective Steering Based on Threshold Analysis"""
    from phase8_3_selective_steering.selective_steering_analyzer import SelectiveSteeringAnalyzer

    logger.info("Starting Phase 8.3: Selective Steering Based on Threshold Analysis")
    logger.info("Will apply steering only when predictive direction exceeds optimal threshold")
    logger.info("Automatically discovers Phase 3.8 threshold and Phase 3.5 baseline")

    # Log configuration
    logger.info("\n" + config.dump(phase="8.3"))

    # Create and run selective steering analyzer
    analyzer = SelectiveSteeringAnalyzer(config)
    results = analyzer.run()

    logger.info("\n✅ Phase 8.3 completed successfully")


def run_phase3_8(config: Config, logger, device: str):
    """Run Phase 3.8: AUROC and F1 Evaluation"""
    import sys
    from pathlib import Path

    logger.info("Starting Phase 3.8: AUROC and F1 Evaluation for PVA-SAE")
    logger.info("This phase evaluates bidirectional SAE features using AUROC and F1 metrics")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="3.8"))
    
    # Set up sys.argv for the auroc_f1_evaluator script
    original_argv = sys.argv.copy()
    
    try:
        # Build new argv
        sys.argv = ["auroc_f1_evaluator.py"]
        
        # Add optional arguments if provided
        if hasattr(config, '_input_file') and config._input_file:
            # Parse input file to determine which phase it's from
            input_path = Path(config._input_file)
            if "phase0_1" in str(input_path):
                sys.argv.extend(["--phase0-1-dir", str(input_path.parent)])
            elif "phase3_5" in str(input_path):
                sys.argv.extend(["--phase3-5-dir", str(input_path.parent)])
            else:
                logger.warning(f"Input file {input_path} not recognized as Phase 0.1 or 3.5 output")
        
        # Use standard phase output directory
        output_dir = get_phase_dir('3.8')
        sys.argv.extend(["--output-dir", output_dir])
        
        logger.info(f"Running Phase 3.8 evaluator with args: {sys.argv[1:]}")
        
        # Import and run the main function directly
        from phase3_8.auroc_f1_evaluator import main
        main()
        
        logger.info("\n✅ Phase 3.8 completed successfully")
        logger.info(f"Results saved to: {output_dir}")
        
    finally:
        # Restore original argv
        sys.argv = original_argv


def run_phase3_10(config: Config, logger, device: str):
    """Run Phase 3.10: Temperature-Based AUROC Analysis"""
    from phase3_10_temperature_auroc_f1.temperature_evaluator import TemperatureAUROCEvaluator

    logger.info("Starting Phase 3.10: Temperature-Based AUROC Analysis")
    logger.info("Using per-sample analysis (no aggregation)")

    # Log configuration
    logger.info("\n" + config.dump(phase="3.10"))

    # Create and run evaluator
    evaluator = TemperatureAUROCEvaluator(config)
    results = evaluator.run()

    logger.info("\n✅ Phase 3.10 completed successfully")
    logger.info(f"Results saved to: {config.phase3_10_output_dir}")


def run_phase3_11(config: Config, logger, device: str):
    """Run Phase 3.11: Temperature Trends Visualization Update"""
    from phase3_11_temperature_trends_updated.temperature_trends_visualizer import TemperatureTrendsVisualizer

    logger.info("Starting Phase 3.11: Temperature Trends Visualization Update")
    logger.info("Updating legend terminology: 'preferring' → 'predicting'")

    # Log configuration
    logger.info("\n" + config.dump(phase="3.11"))

    # Create and run visualizer
    visualizer = TemperatureTrendsVisualizer(config)
    visualizer.run()

    logger.info("\n✅ Phase 3.11 completed successfully")
    logger.info(f"Results saved to: {config.phase3_11_output_dir}")


def run_phase3_12(config: Config, logger, device: str):
    """Run Phase 3.12: Difficulty-Based AUROC Analysis"""
    import sys
    from pathlib import Path
    
    logger.info("Starting Phase 3.12: Difficulty-Based AUROC Analysis for PVA-SAE")
    logger.info("This phase evaluates PVA features across different problem difficulty levels")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="3.12"))
    
    # Set up sys.argv for the difficulty_evaluator script
    original_argv = sys.argv.copy()
    
    try:
        # Build new argv
        sys.argv = ["difficulty_evaluator.py"]
        
        # Add optional arguments if provided
        if hasattr(config, '_input_file') and config._input_file:
            # Parse input file to determine which phase it's from
            input_path = Path(config._input_file)
            if "phase0_1" in str(input_path):
                sys.argv.extend(["--phase0-1-dir", str(input_path.parent)])
            elif "phase3_5" in str(input_path):
                sys.argv.extend(["--phase3-5-dir", str(input_path.parent)])
            elif "phase3_8" in str(input_path):
                sys.argv.extend(["--phase3-8-dir", str(input_path.parent)])
            else:
                logger.warning(f"Input file {input_path} not recognized as Phase 0.1, 3.5, or 3.8 output")
        
        # Use standard phase output directory
        output_dir = get_phase_dir('3.12')
        sys.argv.extend(["--output-dir", output_dir])
        
        logger.info(f"Running Phase 3.12 evaluator with args: {sys.argv[1:]}")
        
        # Import and run the main function directly
        from phase3_12_difficulty_auroc_f1.difficulty_evaluator import main
        main()
        
        logger.info("\n✅ Phase 3.12 completed successfully")
        logger.info(f"Results saved to: {output_dir}")
        
    finally:
        # Restore original argv
        sys.argv = original_argv


def run_phase4_5(config: Config, logger, device: str):
    """Run Phase 4.5: Steering Coefficient Selection"""
    from phase4_5_model_steering.steering_coefficient_selector import SteeringCoefficientSelector
    
    logger.info("Starting Phase 4.5: Steering Coefficient Selection")
    logger.info(f"Experiment mode: {config.phase4_5_experiment_mode}")
    logger.info("Will auto-discover PVA features from Phase 2.5 and baseline from Phase 3.6")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="4.5"))
    
    # Create and run steering coefficient selector
    selector = SteeringCoefficientSelector(config)
    results = selector.run()
    
    logger.info("\n✅ Phase 4.5 completed successfully")


def run_phase4_6(config: Config, logger, device: str):
    """Run Phase 4.6: Golden Section Search Coefficient Refinement"""
    from phase4_6_binary_refinement.golden_section_refiner import GoldenSectionCoefficientRefiner
    
    logger.info("Starting Phase 4.6: Golden Section Search Coefficient Refinement")
    logger.info(f"Experiment mode: {config.phase4_6_experiment_mode}")
    logger.info("Will refine coefficients found in Phase 4.5 using golden section search")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="4.6"))
    
    # Create and run golden section coefficient refiner
    refiner = GoldenSectionCoefficientRefiner(config)
    results = refiner.run()
    
    logger.info("\n✅ Phase 4.6 completed successfully")


def run_phase4_7(config: Config, logger, device: str):
    """Run Phase 4.7: Coefficient Optimization Visualization"""
    from phase4_7_coefficient_visualization.coefficient_plotter import CoefficientVisualizer
    from pathlib import Path

    logger.info("Starting Phase 4.7: Coefficient Optimization Visualization")
    logger.info("Will visualize coefficient search from Phases 4.5 and 4.6")

    # Setup paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    output_dir = data_dir / "phase4_7"

    # Create visualizer and generate plots
    visualizer = CoefficientVisualizer(data_dir, output_dir)
    visualizer.generate_all_plots()

    logger.info("\n✅ Phase 4.7 completed successfully")


def run_phase4_8(config: Config, logger, device: str):
    """Run Phase 4.8: Steering Effect Analysis"""
    from phase4_8_steering_analysis.steering_effect_analyzer import SteeringEffectAnalyzer

    logger.info("Starting Phase 4.8: Steering Effect Analysis")
    logger.info(f"Experiment mode: {config.phase4_8_experiment_mode}")
    logger.info("Will auto-discover PVA features from Phase 2.5 and baseline from Phase 3.5")

    # Log configuration
    logger.info("\n" + config.dump(phase="4.8"))

    # Create and run steering effect analyzer
    analyzer = SteeringEffectAnalyzer(config)
    results = analyzer.run()

    logger.info("\n✅ Phase 4.8 completed successfully")


def run_phase4_10(config: Config, logger, device: str):
    """Run Phase 4.10: Zero-Discrimination Feature Selection"""
    from phase4_10_zero_discrimination.zero_discrimination_selector import ZeroDiscriminationSelector
    
    logger.info("Starting Phase 4.10: Zero-Discrimination Feature Selection")
    logger.info("Will identify SAE features with zero separation scores between correct/incorrect code")
    logger.info("These features serve as rigorous baseline controls for steering experiments")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="4.10"))
    
    # Create and run zero-discrimination feature selector
    selector = ZeroDiscriminationSelector(config)
    results = selector.run()
    
    logger.info("\n✅ Phase 4.10 completed successfully")


def run_phase4_12(config: Config, logger, device: str):
    """Run Phase 4.12: Zero-Discrimination Steering Generation"""
    from phase4_12_zero_disc_steering.zero_disc_steering_generator import ZeroDiscSteeringGenerator
    
    logger.info("Starting Phase 4.12: Zero-Discrimination Steering Generation")
    logger.info("Will apply zero-discrimination features as statistical control vs targeted steering")
    logger.info("This validates that PVA features have specific causal effects on program correctness")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="4.12"))
    
    # Create and run zero-discrimination steering generator
    generator = ZeroDiscSteeringGenerator(config)
    results = generator.run()
    
    logger.info("\n✅ Phase 4.12 completed successfully")


def run_phase4_14(config: Config, logger, device: str):
    """Run Phase 4.14: Statistical Significance Testing"""
    from phase4_14_statistical_significance.significance_tester import SignificanceTester
    
    logger.info("Starting Phase 4.14: Statistical Significance Testing")
    logger.info("Will compare targeted vs zero-discrimination steering using binomial tests")
    logger.info("This validates that PVA features have specific causal effects")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="4.14"))
    
    # Create and run significance tester
    tester = SignificanceTester(config)
    results = tester.run()
    
    logger.info("\n✅ Phase 4.14 completed successfully")


def run_phase5_3(config: Config, logger, device: str):
    """Run Phase 5.3: Weight Orthogonalization Analysis"""
    from phase5_3_weight_orthogonalization.weight_orthogonalizer import WeightOrthogonalizer
    
    logger.info("Starting Phase 5.3: Weight Orthogonalization Analysis")
    logger.info("Will auto-discover PVA features from Phase 2.5 and baseline from Phase 3.5")
    logger.info("This phase permanently modifies model weights to remove PVA information")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="5.3"))
    
    # Create and run weight orthogonalizer
    orthogonalizer = WeightOrthogonalizer(config)
    results = orthogonalizer.run()
    
    logger.info("\n✅ Phase 5.3 completed successfully")


def run_phase5_6(config: Config, logger, device: str):
    """Run Phase 5.6: Zero-Discrimination Weight Orthogonalization"""
    from phase5_6_zero_disc_orthogonalization.zero_disc_weight_orthogonalizer import ZeroDiscWeightOrthogonalizer
    
    logger.info("Starting Phase 5.6: Zero-Discrimination Weight Orthogonalization")
    logger.info("Control experiment using zero-disc features from Phase 4.10")
    logger.info("Expected: minimal effects compared to Phase 5.3 PVA orthogonalization")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="5.6"))
    
    # Create and run orthogonalizer
    orthogonalizer = ZeroDiscWeightOrthogonalizer(config)
    results = orthogonalizer.run()
    
    logger.info("\n✅ Phase 5.6 completed successfully")


def run_phase5_9(config: Config, logger, device: str):
    """Run Phase 5.9: Weight Orthogonalization Statistical Significance"""
    from phase5_9_orthogonalization_significance.orthogonalization_significance_tester import OrthogonalizationSignificanceTester
    
    logger.info("Starting Phase 5.9: Weight Orthogonalization Statistical Significance Testing")
    logger.info("Triangulation analysis comparing baseline, zero-disc, and PVA orthogonalization")
    logger.info("Will perform 6 binomial tests to validate causal effects")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="5.9"))
    
    # Create and run significance tester
    tester = OrthogonalizationSignificanceTester(config)
    results = tester.run()
    
    logger.info("\n✅ Phase 5.9 completed successfully")


def run_phase6_3(config: Config, logger, device: str):
    """Run Phase 6.3: Attention Pattern Analysis"""
    from phase6_3_attention_analysis.attention_analyzer import AttentionAnalyzer
    
    logger.info("Starting Phase 6.3: Attention Pattern Analysis")
    logger.info("Will analyze attention from Phase 3.5 (baseline) and Phase 4.8 (steered)")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="6.3"))
    
    # Create and run analyzer
    analyzer = AttentionAnalyzer(config)
    results = analyzer.run()
    
    logger.info("\n✅ Phase 6.3 completed successfully")


def cleanup_gpu_command(args, logger):
    """Clean GPU memory and zombie contexts"""
    print(f"\n{'='*60}")
    print(f"GPU CLEANUP TOOL")
    print(f"{'='*60}\n")
    
    if not torch.cuda.is_available():
        print("❌ No CUDA GPUs detected")
        return
    
    import gc
    import subprocess
    
    # Step 1: Force Python garbage collection
    print("1. Running Python garbage collection...")
    gc.collect()
    print("   ✓ Garbage collection completed")
    
    # Step 2: Clean all GPUs
    gpu_count = torch.cuda.device_count()
    print(f"\n2. Cleaning {gpu_count} GPU(s)...")
    
    for i in range(gpu_count):
        print(f"\n   GPU {i}:")
        try:
            # Get initial memory stats
            from common.gpu_utils import get_gpu_memory_info
            before = get_gpu_memory_info(i)
            print(f"   - Before: {before.get('allocated_mb', 0):.1f}MB allocated, {before.get('reserved_mb', 0):.1f}MB reserved")
            
            # Clean GPU
            cleanup_gpu_memory(i)
            
            # Get after stats
            after = get_gpu_memory_info(i)
            print(f"   - After:  {after.get('allocated_mb', 0):.1f}MB allocated, {after.get('reserved_mb', 0):.1f}MB reserved")
            print(f"   ✓ Cleaned")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Step 3: Aggressive cleanup if requested
    if args.aggressive:
        print("\n3. Running aggressive cleanup...")
        
        # Kill user's Python processes
        try:
            result = subprocess.run(['pkill', '-u', environ.get('USER', ''), '-f', 'python'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("   ✓ Killed hanging Python processes")
            else:
                print("   - No Python processes to kill")
        except Exception as e:
            print(f"   ❌ Failed to kill processes: {e}")
        
        # Clear shared memory
        try:
            result = subprocess.run("ipcs -m | grep $USER | awk '{print $2}' | xargs -n1 ipcrm -m 2>/dev/null || true", 
                                  shell=True, capture_output=True, text=True)
            print("   ✓ Cleared shared memory segments")
        except Exception as e:
            print(f"   - Could not clear shared memory: {e}")
    
    # Step 4: Show final GPU status
    print("\n4. Final GPU Status:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu', '--format=csv'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("   Could not get nvidia-smi status")
    except:
        print("   nvidia-smi not available")
    
    print(f"\n{'='*60}")
    print("✓ GPU cleanup completed")
    print(f"{'='*60}\n")


def test_gpu_setup(args, logger):
    """Test GPU detection and memory availability"""
    import torch
    from common.utils import get_memory_usage, detect_device
    
    logger.info("Testing GPU setup...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Running on CPU.")
        logger.warning("CUDA not available")
        return
    
    # Get device and GPU count
    device = detect_device()
    gpu_count = torch.cuda.device_count()
    
    print(f"\n{'='*60}")
    print(f"GPU DETECTION REPORT")
    print(f"{'='*60}")
    print(f"✓ CUDA is available")
    print(f"✓ {gpu_count} GPU(s) detected")
    print(f"✓ Default device: {device}")
    print()
    
    # List all GPUs
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print(f"\n{'='*60}")
    print(f"MEMORY STATUS")
    print(f"{'='*60}")
    
    # Get detailed memory stats
    memory_stats = get_memory_usage()
    
    # CPU Memory
    print(f"\nCPU Memory:")
    print(f"  Used: {memory_stats['cpu']['used_gb']:.1f}GB")
    print(f"  Available: {memory_stats['cpu']['available_gb']:.1f}GB")
    print(f"  Total: {memory_stats['cpu']['total_gb']:.1f}GB")
    print(f"  Usage: {memory_stats['cpu']['percent']:.1f}%")
    
    # GPU Memory
    if memory_stats['gpu']:
        print(f"\nGPU Memory:")
        for gpu_id, gpu_stats in memory_stats['gpu'].items():
            print(f"\n  {gpu_id}:")
            print(f"    Allocated: {gpu_stats['allocated']:.1f}GB")
            print(f"    Reserved: {gpu_stats['reserved']:.1f}GB")
            print(f"    Total: {gpu_stats['total']:.1f}GB")
    
    if args.detailed:
        print(f"\n{'='*60}")
        print(f"DETAILED GPU INFORMATION")
        print(f"{'='*60}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i} Properties:")
            print(f"  Name: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / (1024**3):.1f}GB")
            print(f"  Multiprocessor Count: {props.multi_processor_count}")
            # These attributes might not be available in all PyTorch versions
            if hasattr(props, 'max_threads_per_block'):
                print(f"  Max Threads per Block: {props.max_threads_per_block}")
            if hasattr(props, 'max_threads_per_multi_processor'):
                print(f"  Max Threads per Multiprocessor: {props.max_threads_per_multi_processor}")
    
    # Test simple tensor operation
    print(f"\n{'='*60}")
    print(f"FUNCTIONALITY TEST")
    print(f"{'='*60}")
    
    try:
        # Create a small tensor on each GPU
        for i in range(gpu_count):
            device = torch.device(f'cuda:{i}')
            tensor = torch.randn(1000, 1000, device=device)
            result = tensor @ tensor.T
            print(f"✓ GPU {i}: Matrix multiplication test passed")
            del tensor, result
            torch.cuda.empty_cache()
        
        print("\n✓ All GPUs are functional")
        
    except Exception as e:
        print(f"\n❌ GPU functionality test failed: {str(e)}")
        logger.error(f"GPU test failed: {str(e)}")
    
    print(f"\n{'='*60}\n")


def show_status(args, logger):
    """Show current system status"""
    import pandas as pd
    
    print(f"\n{'='*50}")
    print("SYSTEM STATUS")
    print(f"{'='*50}")
    
    # Check phases implementation
    print("\n📋 Phases:")
    implemented_phases = []
    placeholder_phases = []
    
    # Check run functions in this file for implementation status
    current_file_content = Path(__file__).read_text()
    
    # Phase 0 - check if run_phase0 has real implementation
    if "run_phase0" in current_file_content and "MBPPPreprocessor" in current_file_content:
        implemented_phases.append(0)
    else:
        placeholder_phases.append(0)
    
    # Phase 1 - check if run_phase1 has real implementation  
    if "run_phase1" in current_file_content and "Phase1Orchestrator" in current_file_content:
        implemented_phases.append(1)
    else:
        placeholder_phases.append(1)
    
    # Phase 2.5 - check if run_phase2_5 has TODO or placeholder text
    if "# TODO: Implement SAE analysis" in current_file_content or "SAE Analysis not yet implemented" in current_file_content:
        placeholder_phases.append(2.5)
    else:
        implemented_phases.append(2.5)
    
    # Phase 2.10 - check if run_phase2_10 has real implementation
    if "run_phase2_10" in current_file_content and "TStatisticSelector" in current_file_content:
        implemented_phases.append(2.10)
    else:
        placeholder_phases.append(2.10)
    
    # Phase 3 - check if run_phase3 has TODO or placeholder text
    if "# TODO: Implement validation" in current_file_content or "Validation not yet implemented" in current_file_content:
        placeholder_phases.append(3)
    else:
        implemented_phases.append(3)
    
    impl_str = ",".join(map(str, implemented_phases)) if implemented_phases else "None"
    place_str = ",".join(map(str, placeholder_phases)) if placeholder_phases else "None"
    print(f"   ✅ Implemented: {impl_str}")
    print(f"   ❌ Placeholder: {place_str}")
    
    # Check latest dataset
    print("\n📊 Latest Dataset:")
    from common.utils import discover_latest_phase_output
    latest_dataset = discover_latest_phase_output("1")
    if latest_dataset:
        try:
            df = pd.read_parquet(latest_dataset)
            correct_count = df['test_passed'].sum()
            total_count = len(df)
            correct_rate = (correct_count / total_count * 100) if total_count > 0 else 0
            
            dataset_name = Path(latest_dataset).name
            print(f"   ✅ {dataset_name}")
            print(f"   📈 {total_count} records, {correct_rate:.0f}% correct")
        except Exception as e:
            print(f"   ❌ Error reading dataset: {e}")
    else:
        print(f"   ❌ No datasets found in {get_phase_dir('1')}")
    
    # Check GPU status
    print("\n🖥️  GPU:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   ✅ {gpu_count} GPU(s) available")
    else:
        print("   ❌ CUDA unavailable (CPU mode)")
    
    print(f"\n{'='*50}\n")


def validate_system(args, logger):
    """Validate system dependencies and setup"""
    print(f"\n{'='*50}")
    print("SYSTEM VALIDATION")
    print(f"{'='*50}")
    
    validation_results = []
    
    # Check HF Token
    print("\n🔑 Hugging Face Token:")
    try:
        from huggingface_hub import HfApi
        
        # Try to get user info - this uses saved token automatically
        api = HfApi()
        user_info = api.whoami()
        if user_info:
            print(f"   ✅ HF Token accessible (logged in as: {user_info['name']})")
            validation_results.append(("HF Token", True))
        else:
            print("   ❌ HF Token not found")
            validation_results.append(("HF Token", False))
    except Exception as e:
        print(f"   ❌ HF Token error: {e}")
        validation_results.append(("HF Token", False))
    
    # Check Model accessibility
    print("\n🤖 Models:")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        print("   ✅ Gemma-2-2b accessible")
        validation_results.append(("Model", True))
    except Exception as e:
        print(f"   ❌ Model error: {e}")
        validation_results.append(("Model", False))
    
    # Check Tokenizer
    print("\n🔤 Tokenizer:")
    try:
        if 'tokenizer' in locals():
            test_tokens = tokenizer("Hello world", return_tensors="pt")
            print("   ✅ Tokenizer works")
            validation_results.append(("Tokenizer", True))
        else:
            raise Exception("Tokenizer not loaded")
    except Exception as e:
        print(f"   ❌ Tokenizer error: {e}")
        validation_results.append(("Tokenizer", False))
    
    # Check GemmaScope (SAE)
    print("\n🧠 GemmaScope:")
    try:
        import requests
        # Just check if we can access the model hub page
        response = requests.head("https://huggingface.co/google/gemma-scope-2b-pt-res", timeout=10)
        if response.status_code == 200:
            print("   ✅ GemmaScope accessible")
            validation_results.append(("GemmaScope", True))
        else:
            print("   ❌ GemmaScope not accessible")
            validation_results.append(("GemmaScope", False))
    except Exception as e:
        print(f"   ❌ GemmaScope error: {e}")
        validation_results.append(("GemmaScope", False))
    
    # Check CUDA
    print("\n🖥️  CUDA:")
    if torch.cuda.is_available():
        print("   ✅ CUDA available")
        validation_results.append(("CUDA", True))
    else:
        print("   ❌ CUDA unavailable")
        validation_results.append(("CUDA", False))
    
    # Check dataset readability
    print("\n📊 Dataset:")
    try:
        from common.utils import discover_latest_phase_output
        import pandas as pd
        latest_dataset = discover_latest_phase_output("1")
        if latest_dataset:
            df = pd.read_parquet(latest_dataset)
            print(f"   ✅ Dataset readable ({len(df)} records)")
            validation_results.append(("Dataset", True))
        else:
            print(f"   ❌ No datasets found in {get_phase_dir('1')}")
            validation_results.append(("Dataset", False))
    except Exception as e:
        print(f"   ❌ Dataset error: {e}")
        validation_results.append(("Dataset", False))
    
    # Summary
    print(f"\n{'='*50}")
    print("VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in validation_results if result)
    total = len(validation_results)
    
    for component, result in validation_results:
        status = "✅" if result else "❌"
        print(f"   {status} {component}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    print(f"{'='*50}\n")


def test_phase1(args, logger, device: str):
    """Quick test of Phase 1 with 10 records"""
    import pandas as pd
    from common.utils import discover_latest_phase_output, get_phase_dir
    
    print(f"\n{'='*50}")
    print("PHASE 1 QUICK TEST")
    print(f"{'='*50}")
    
    # Use auto-discovery to find latest dataset from Phase 1
    print("\n🔍 Auto-discovering dataset from Phase 1...")
    latest_dataset = discover_latest_phase_output("1")
    if not latest_dataset:
        print(f"❌ No existing datasets found in {get_phase_dir('1')}. Run Phase 1 first.")
        return
    
    print(f"\n📊 Using dataset: {Path(latest_dataset).name}")
    
    try:
        # Load dataset and take first 10 records
        df = pd.read_parquet(latest_dataset)
        if len(df) < 10:
            print(f"⚠️  Dataset only has {len(df)} records (using all)")
            test_df = df
        else:
            test_df = df.head(10)
            print(f"📝 Testing with first 10 records")
        
        # Calculate correct rate
        correct_count = test_df['test_passed'].sum()
        total_count = len(test_df)
        correct_rate = (correct_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n📈 Results:")
        print(f"   Records tested: {total_count}")
        print(f"   Correct solutions: {correct_count}")
        print(f"   Correct rate: {correct_rate:.1f}%")
        
        # Check if rate is reasonable (>10%)
        if correct_rate >= 10:
            print(f"\n✅ Phase 1 test PASSED (correct rate >= 10%)")
        else:
            print(f"\n❌ Phase 1 test FAILED (correct rate < 10%)")
            print("   This might indicate issues with test execution or model performance")
        
    except Exception as e:
        print(f"\n❌ Phase 1 test FAILED: {e}")
        logger.error(f"Phase 1 test error: {e}")
    
    print(f"{'='*50}\n")


def test_phase2(args, logger, device: str):
    """Quick test of Phase 2.5 SAE analysis with saved activations"""
    # Phase 2.5 is CPU-only and loads saved activations
    device = "cpu"
    
    # Run simplified version of Phase 2.5
    config = Config.from_args(args, phase="2.5")
    config.dataset_end_idx = 10  # Limit to 10 samples for testing
    
    run_phase2_5(config, logger, device)
    
    print(f"\n{'='*50}")
    print("PHASE 2.5 QUICK TEST")
    print(f"{'='*50}")
    
    # Use auto-discovery to find latest dataset from Phase 1
    print("\n🔍 Auto-discovering dataset from Phase 1...")
    latest_dataset = discover_latest_phase_output("1")
    if not latest_dataset:
        print(f"❌ No datasets found in {get_phase_dir('1')}. Run Phase 1 first.")
        return
    
    print(f"\n📊 Using dataset: {Path(latest_dataset).name}")
    
    try:
        # Load dataset and sample records
        df = pd.read_parquet(latest_dataset)
        
        # Get correct and incorrect samples
        correct_df = df[df['test_passed'] == True]
        incorrect_df = df[df['test_passed'] == False]
        
        n_samples = min(args.samples, 3)  # Limit to max 3 for speed
        if len(correct_df) < n_samples or len(incorrect_df) < n_samples:
            print(f"⚠️  Not enough samples. Using available: {len(correct_df)} correct, {len(incorrect_df)} incorrect")
            n_samples = min(len(correct_df), len(incorrect_df), n_samples)
        
        if n_samples == 0:
            print("❌ No valid samples found in dataset")
            return
        
        # Sample data
        correct_sample = correct_df.sample(n_samples, random_state=42)
        incorrect_sample = incorrect_df.sample(n_samples, random_state=42)
        
        print(f"📝 Testing with {n_samples} samples per class")
        print(f"🎯 Target layer: {args.layer}")
        
        # Create prompts using generated code (simplified format for testing)
        correct_prompts = [
            f"Task {row['task_id']}:\n{row['generated_code']}" 
            for _, row in correct_sample.iterrows()
        ]
        incorrect_prompts = [
            f"Task {row['task_id']}:\n{row['generated_code']}" 
            for _, row in incorrect_sample.iterrows()
        ]
        
        print(f"\n🤖 Loading Gemma-2B model...")
        
        # Load model (use 2B for testing speed)
        import time
        start_time = time.time()
        
        try:
            model = HookedTransformer.from_pretrained(
                "google/gemma-2-2b",
                device=device,
                dtype=torch.float32  # Use float32 for compatibility with SAE
            )
            load_time = time.time() - start_time
            print(f"   Model loaded in {load_time:.1f}s")
        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
            return
        
        # Create test config based on unified Config
        from common.config import Config
        test_config = Config()
        test_config.activation_layers = [args.layer]  # Test single layer
        test_config.sae_save_after_each_layer = False  # Skip checkpointing for test
        test_config.sae_cleanup_after_layer = True  # Clean up memory
        test_config.sae_checkpoint_dir = f"{get_phase_dir('2.5')}/test_checkpoints"  # Separate test dir
        
        print(f"🧠 Initializing SAE pipeline...")
        
        # Initialize pipeline
        # TODO: EnhancedSAEAnalysisPipeline not implemented - this test function needs updating
        print("❌ Error: test-sae-gpu function is not currently implemented")
        print("   Use regular 'phase 2.5' command instead")
        return
        
        # pipeline = EnhancedSAEAnalysisPipeline(model, test_config, device=device)
        # 
        # print(f"⚙️  Running SAE analysis on layer {args.layer}...")
        # 
        # # Run analysis
        # results = pipeline.analyze_all_residual_layers(
        #     correct_prompts=correct_prompts,
        #     incorrect_prompts=incorrect_prompts,
        #     layer_indices=[args.layer]
        # )
        
        # Check results
        if len(results.layer_results) > 0:
            layer_result = results.layer_results[args.layer]
            
            print(f"\n📈 Results:")
            print(f"   Layer analyzed: {args.layer}")
            print(f"   Correct samples: {results.n_correct_samples}")
            print(f"   Incorrect samples: {results.n_incorrect_samples}")
            print(f"   Best correct feature: {layer_result.correct_direction.feature_idx}")
            print(f"   Correct separation score: {layer_result.correct_direction.separation_score:.3f}")
            print(f"   Best incorrect feature: {layer_result.incorrect_direction.feature_idx}")
            print(f"   Incorrect separation score: {layer_result.incorrect_direction.separation_score:.3f}")
            
            # Check if we found meaningful separation
            max_separation = max(
                layer_result.correct_direction.separation_score,
                layer_result.incorrect_direction.separation_score
            )
            
            if max_separation > 0.05:  # Threshold for meaningful separation
                print(f"\n✅ Phase 2 test PASSED (max separation: {max_separation:.3f} > 0.05)")
            else:
                print(f"\n⚠️  Phase 2 test WARNING (low separation: {max_separation:.3f} <= 0.05)")
                print("   This might be normal for small samples or specific layers")
        else:
            print(f"\n❌ Phase 2 test FAILED (no results generated)")
        
    except ImportError as e:
        print(f"\n❌ Phase 2 test FAILED (import error): {e}")
        print("   Make sure all dependencies are installed")
        logger.error(f"Phase 2 import error: {e}")
    except Exception as e:
        print(f"\n❌ Phase 2 test FAILED: {e}")
        logger.error(f"Phase 2 test error: {e}")
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    print(f"{'='*50}\n")


def main():
    """Main entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle case where no command is provided
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Set global phase context first (before any logging)
    if args.command == 'phase' and hasattr(args, 'phase'):
        # Preserve .0 suffix for whole numbers: 0.0 -> "0.0", 1.0 -> "1.0", 0.1 -> "0.1"
        # Handle special cases like 3.10 which would become "3.1" with str()
        if args.phase == 3.10:
            phase_str = "3.10"
        elif args.phase == 3.11:
            phase_str = "3.11"
        elif args.phase == 3.12:
            phase_str = "3.12"
        else:
            phase_str = str(args.phase)
        set_logging_phase(phase_str)
    
    # For non-phase commands, create logger without phase context
    # For phase commands, delay logger creation until after phase is set
    if args.command != 'phase':
        logger = get_logger("main")
    else:
        # Logger will be created after phase context is fully established
        logger = None
    
    # Detect device once for the entire application
    device = "cpu"  # Default to CPU
    try:
        from common.utils import detect_device
        device = str(detect_device())
        if logger:
            logger.info(f"Detected device: {device}")
    except Exception as e:
        if logger:
            logger.error(f"Device detection failed: {e}")
            logger.info("Falling back to CPU")
        device = "cpu"
    
    # Handle different commands
    if args.command == 'test-gpu':
        test_gpu_setup(args, logger)
        return
    
    elif args.command == 'cleanup-gpu':
        cleanup_gpu_command(args, logger)
        return
    
    elif args.command == 'status':
        show_status(args, logger)
        return
    
    elif args.command == 'validate':
        validate_system(args, logger)
        return
    
    elif args.command == 'test-phase1':
        test_phase1(args, logger, device)
        return
    
    elif args.command == 'test-phase2':
        test_phase2(args, logger, device)
        return
    
    elif args.command == 'phase':
        # Now create logger after phase context is set
        logger = get_logger("main")
        
        # Log device info now that we have a logger
        logger.info(f"Detected device: {device}")
        
        # Create unified config from args
        # Use the properly formatted phase_str we created earlier
        if args.phase == 3.10:
            phase_str_for_config = "3.10"
        elif args.phase == 3.11:
            phase_str_for_config = "3.11"
        elif args.phase == 3.12:
            phase_str_for_config = "3.12"
        else:
            phase_str_for_config = str(args.phase)
        config = Config.from_args(args, phase=phase_str_for_config)
        
        # Store input file path if provided
        if args.input:
            config._input_file = args.input
        
        # Handle phase-specific special arguments
        if args.phase == 0 and hasattr(args, 'dry_run'):
            config._dry_run = args.dry_run
        
        if args.phase == 0.1 and hasattr(args, 'generate_report'):
            config._generate_report = args.generate_report
        
        if args.phase == 2.2 and hasattr(args, 'run_count'):
            config._run_count = args.run_count
        
        # Handle Phase 4.5 experiment mode arguments
        if args.phase == 4.5:
            if hasattr(args, 'correction_only') and args.correction_only:
                config.phase4_5_experiment_mode = 'correction'
            elif hasattr(args, 'corruption_only') and args.corruption_only:
                config.phase4_5_experiment_mode = 'corruption'
            else:
                config.phase4_5_experiment_mode = 'all'
        
        # Handle Phase 4.6 experiment mode arguments
        if args.phase == 4.6:
            if hasattr(args, 'correction_only') and args.correction_only:
                config.phase4_6_experiment_mode = 'correction'
            elif hasattr(args, 'corruption_only') and args.corruption_only:
                config.phase4_6_experiment_mode = 'corruption'
            else:
                config.phase4_6_experiment_mode = 'all'
        
        # Handle Phase 4.8 experiment mode arguments
        if args.phase == 4.8:
            if hasattr(args, 'preservation_only') and args.preservation_only:
                config.phase4_8_experiment_mode = 'preservation'
            elif hasattr(args, 'correction_only') and args.correction_only:
                config.phase4_8_experiment_mode = 'correction'
            elif hasattr(args, 'corruption_only') and args.corruption_only:
                config.phase4_8_experiment_mode = 'corruption'
            else:
                config.phase4_8_experiment_mode = 'all'
        
        # Validate config for the phase
        try:
            config.validate(str(args.phase))
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)
        
        # Show config and exit if requested
        if args.show_config:
            print("\n" + config.dump(phase=str(args.phase)))
            sys.exit(0)
        
        # Display phase info
        phase_names = {
            0: "Difficulty Analysis",
            0.1: "Problem Splitting",
            0.2: "HumanEval to MBPP Conversion",
            0.3: "HumanEval Import Scanning",
            1: "Dataset Building",
            2.2: "Pile Activation Caching",
            2.5: "SAE Analysis with Pile Filtering",
            2.10: "T-Statistic Latent Selection",
            2.15: "Layer-wise Analysis Visualization",
            3: "Validation",
            3.5: "Temperature Robustness",
            3.6: "Hyperparameter Tuning Set Processing",
            3.8: "AUROC and F1 Evaluation",
            3.10: "Temperature-Based AUROC Analysis",
            3.11: "Temperature Trends Visualization Update",
            3.12: "Difficulty-Based AUROC Analysis",
            4.5: "Steering Coefficient Selection",
            4.6: "Golden Section Search Coefficient Refinement",
            4.7: "Coefficient Optimization Visualization",
            4.8: "Steering Effect Analysis",
            4.10: "Random PVA Feature Selection",
            4.12: "Random Steering Analysis",
            4.14: "Statistical Significance Testing",
            5.3: "Weight Orthogonalization Analysis",
            5.6: "Zero-Discrimination Weight Orthogonalization",
            5.9: "Weight Orthogonalization Statistical Significance",
            6.3: "Attention Pattern Analysis",
            7.3: "Instruction-Tuned Model Baseline",
            7.6: "Instruction-Tuned Model Steering Analysis",
            7.9: "Universality Analysis",
            7.12: "Instruction-Tuned Model AUROC/F1 Evaluation",
            8.1: "Percentile Threshold Calculator",
            8.2: "Percentile Threshold Optimizer",
            8.3: "Selective Steering Based on Threshold Analysis"
        }
        
        print(f"\n{'='*60}")
        # Use properly formatted phase string for display
        # Handle float precision issues (2.10 becomes 2.1, 3.10 becomes 3.1, etc.)
        if abs(args.phase - 2.1) < 0.01:
            display_phase = "2.10"
            phase_key = 2.10
        elif abs(args.phase - 3.1) < 0.01:
            display_phase = "3.10"
            phase_key = 3.10
        elif abs(args.phase - 3.11) < 0.01:
            display_phase = "3.11"
            phase_key = 3.11
        elif abs(args.phase - 3.12) < 0.01:
            display_phase = "3.12"
            phase_key = 3.12
        else:
            display_phase = str(args.phase)
            phase_key = args.phase
        print(f"PHASE {display_phase}: {phase_names[phase_key].upper()}")
        print(f"{'='*60}")
        
        try:
            # Run selected phase with unified config
            if args.phase == 0:
                dry_run = getattr(config, '_dry_run', False)
                run_phase0(config, logger, device, dry_run=dry_run)
            elif args.phase == 1:
                run_phase1(config, logger, device)
            elif args.phase == 0.1:
                run_phase0_1(config, logger, device)
            elif args.phase == 0.2:
                run_phase0_2(config, logger, device)
            elif args.phase == 0.3:
                run_phase0_3(config, logger, device)
            elif args.phase == 2.2:
                run_phase2_2(config, logger, device)
            elif args.phase == 2.5:
                run_phase2_5(config, logger, device)
            elif args.phase == 2.10:
                run_phase2_10(config, logger, device)
            elif args.phase == 2.15:
                run_phase2_15(config, logger, device)
            elif args.phase == 3:
                run_phase3(config, logger, device)
            elif args.phase == 3.5:
                run_phase3_5(config, logger, device)
            elif args.phase == 3.6:
                run_phase3_6(config, logger, device)
            elif args.phase == 3.8:
                run_phase3_8(config, logger, device)
            elif args.phase == 3.10:
                run_phase3_10(config, logger, device)
            elif args.phase == 3.11:
                run_phase3_11(config, logger, device)
            elif args.phase == 3.12:
                run_phase3_12(config, logger, device)
            elif args.phase == 4.5:
                run_phase4_5(config, logger, device)
            elif args.phase == 4.6:
                run_phase4_6(config, logger, device)
            elif args.phase == 4.7:
                run_phase4_7(config, logger, device)
            elif args.phase == 4.8:
                run_phase4_8(config, logger, device)
            elif args.phase == 4.10:
                run_phase4_10(config, logger, device)
            elif args.phase == 4.12:
                run_phase4_12(config, logger, device)
            elif args.phase == 4.14:
                run_phase4_14(config, logger, device)
            elif args.phase == 5.3:
                run_phase5_3(config, logger, device)
            elif args.phase == 5.6:
                run_phase5_6(config, logger, device)
            elif args.phase == 5.9:
                run_phase5_9(config, logger, device)
            elif args.phase == 6.3:
                run_phase6_3(config, logger, device)
            elif args.phase == 7.3:
                run_phase7_3(config, logger, device)
            elif args.phase == 7.6:
                run_phase7_6(config, logger, device)
            elif args.phase == 7.9:
                run_phase7_9(config, logger, device)
            elif args.phase == 7.12:
                run_phase7_12(config, logger, device)
            elif args.phase == 8.1:
                run_phase8_1(config, logger, device)
            elif args.phase == 8.2:
                run_phase8_2(config, logger, device)
            elif args.phase == 8.3:
                run_phase8_3(config, logger, device)

            print(f"✅ Phase {args.phase} completed successfully!")
            
        except Exception as e:
            logger.error(f"Phase {args.phase} failed: {str(e)}")
            if config.verbose:
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
            sys.exit(1)


if __name__ == "__main__":
    main()