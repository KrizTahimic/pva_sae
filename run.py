#!/usr/bin/env python3
"""
Single entry point for all PVA-SAE thesis phases.

Usage:
    python3 run.py phase 0                                      # Difficulty analysis
    python3 run.py phase 1 --model google/gemma-2-2b           # Dataset building (single GPU)
    python3 run.py phase 2                                      # SAE analysis (auto-discovers input)
    python3 run.py phase 3                                      # Validation (auto-discovers input)
    
Manual input override:
    python3 run.py phase 1 --input data/phase0/mapping.parquet  # Use specific difficulty mapping
    python3 run.py phase 2 --input data/phase1_0/dataset.parquet  # Use specific dataset
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

from common.logging import LoggingManager, set_logging_phase
from common.gpu_utils import cleanup_gpu_memory, ensure_gpu_available, setup_cuda_environment
from common import MAX_NEW_TOKENS
from common.config import Config

# Import centralized phase directory function
from common.utils import get_phase_dir


def setup_argument_parser():
    """Setup command line argument parser with phase-specific argument groups"""
    parser = ArgumentParser(
        description="Run PVA-SAE thesis phases",
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
        choices=[0, 0.1, 1, 1.2, 2, 3],
        help='Phase to run: 0=Difficulty Analysis, 0.1=Problem Splitting, 1=Dataset Building, 1.2=Temperature Generation, 2=SAE Analysis, 3=Validation'
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
             'Phase 2: dataset (.parquet), '
             'Phase 3: SAE results (.json)'
    )
    
    # Phase 0: Difficulty Analysis arguments
    phase0_group = phase_parser.add_argument_group('Phase 0: Difficulty Analysis')
    phase0_group.add_argument(
        '--output-dir',
        type=str,
        default=get_phase_dir('0'),
        help='Directory to save difficulty mapping'
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
        default='google/gemma-2-2b',
        help='Model name to use for dataset building'
    )
    phase1_group.add_argument(
        '--start',
        type=int,
        default=0,
        help='Starting index for MBPP dataset'
    )
    phase1_group.add_argument(
        '--end',
        type=int,
        default=973,
        help='Ending index for MBPP dataset (inclusive)'
    )
    phase1_group.add_argument(
        '--dataset-dir',
        type=str,
        default=get_phase_dir('1'),
        help='Directory for dataset files'
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
    
    # Phase 1.2: Temperature Generation arguments
    phase1_2_group = phase_parser.add_argument_group('Phase 1.2: Temperature Generation')
    phase1_2_group.add_argument(
        '--samples',
        type=int,
        help='Number of validation samples to process (for testing). Default: all'
    )
    phase1_2_group.add_argument(
        '--test-temps',
        nargs='+',
        type=float,
        help='Override temperatures for testing (e.g., --test-temps 0.3 0.6)'
    )
    phase1_2_group.add_argument(
        '--test-samples-per-temp',
        type=int,
        help='Override samples per temperature for testing (default: 5)'
    )
    
    # Phase 2: SAE Analysis arguments
    phase2_group = phase_parser.add_argument_group('Phase 2: SAE Analysis')
    phase2_group.add_argument(
        '--split',
        type=str,
        choices=['sae', 'hyperparams', 'validation'],
        default='sae',
        help='Which data split to analyze (default: sae)'
    )
    phase2_group.add_argument(
        '--sae-model',
        type=str,
        help='Path to SAE model'
    )
    phase2_group.add_argument(
        '--latent-threshold',
        type=float,
        default=0.02,
        help='Activation threshold for latent filtering'
    )
    phase2_group.add_argument(
        '--pile-filter',
        action='store_true',
        help='Apply Pile dataset filtering to remove general language features'
    )
    phase2_group.add_argument(
        '--pile-threshold',
        type=float,
        default=0.02,
        help='Maximum Pile activation frequency (default: 2%%)'
    )
    phase2_group.add_argument(
        '--pile-samples',
        type=int,
        default=10000,
        help='Number of Pile samples to use (default: 10000)'
    )
    
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
    
    return parser


def validate_phase_arguments(args):
    """Validate phase-specific argument requirements"""
    if args.phase == 1:
        # Phase 1 requires model
        if not args.model:
            raise ValueError("Phase 1 requires --model argument")
    
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
    from phase1_0_dataset_building import Phase1Orchestrator
    from pathlib import Path
    
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
    if torch.cuda.is_available():
        logger.info("Setting up CUDA environment and cleaning GPU memory...")
        setup_cuda_environment()
        cleanup_gpu_memory()
        
        # Ensure GPU is responsive
        gpu_device = int(environ.get('CUDA_VISIBLE_DEVICES', '0'))
        if not ensure_gpu_available(gpu_device):
            logger.warning(f"GPU {gpu_device} not responsive, attempting cleanup...")
            cleanup_gpu_memory(gpu_device)
    
    # Create orchestrator with SAE split
    orchestrator = Phase1Orchestrator(
        split_name='sae',  # Always use SAE split for Phase 1
        config=config
    )
    
    # Build dataset
    dataset_path = orchestrator.build_dataset(
        start_idx=config.dataset_start_idx,
        end_idx=config.dataset_end_idx
    )
    
    logger.info("✅ Phase 1 completed successfully")
    logger.info(f"Dataset saved to: {dataset_path}")


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


def run_phase1_2(config: Config, logger, device: str):
    """Run Phase 1.2: Temperature Variation Generation for validation split"""
    from phase1_2_temperature_generation import TemperatureVariationGenerator
    from common.models import ModelManager
    from common.utils import discover_latest_phase_output
    from common.gpu_utils import setup_cuda_environment
    
    logger.info("Starting Phase 1.2: Temperature Variation Generation")
    logger.info("This phase generates multiple temperature variations for the validation split")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="1.2"))
    
    # Setup CUDA for generation (only for NVIDIA GPUs)
    if device == "cuda":
        setup_cuda_environment()
    
    # Check for test mode arguments
    test_mode = False
    temperatures = config.temperature_variation_temps
    samples_per_temp = config.temperature_samples_per_temp
    output_dir = config.phase1_2_output_dir
    
    # Override for testing if requested
    if hasattr(config, '_test_temps') and config._test_temps:
        temperatures = config._test_temps
        test_mode = True
        logger.info(f"Test mode: Using temperatures {temperatures}")
    
    if hasattr(config, '_test_samples_per_temp') and config._test_samples_per_temp:
        samples_per_temp = config._test_samples_per_temp
        test_mode = True
        logger.info(f"Test mode: Using {samples_per_temp} samples per temperature")
    
    if test_mode:
        # Use test output directory to avoid mixing with production data
        output_dir = "data/test_phase1_2"
        logger.info(f"Test mode: Output will be saved to {output_dir}")
    
    # Update config with test mode overrides if needed
    if hasattr(config, '_test_temps') and config._test_temps:
        config.temperature_variation_temps = config._test_temps
    if hasattr(config, '_test_samples_per_temp') and config._test_samples_per_temp:
        config.temperature_samples_per_temp = config._test_samples_per_temp
    if test_mode:
        config.phase1_2_output_dir = output_dir
    
    # Check Phase 0.1 outputs exist
    validation_file = Path(config.phase0_1_output_dir) / "validation_mbpp.parquet"
    if not validation_file.exists():
        logger.error(f"Validation data not found at {validation_file}")
        logger.error("Please run Phase 0.1 first to create problem splits")
        sys.exit(1)
    
    # Handle --samples argument for testing with subset
    if hasattr(config, '_samples') and config._samples:
        import pandas as pd
        validation_df = pd.read_parquet(validation_file)
        
        # Take subset of validation data
        subset_size = min(config._samples, len(validation_df))
        subset_df = validation_df.head(subset_size)
        
        logger.info(f"Test mode: Using {subset_size} validation tasks out of {len(validation_df)}")
        
        # Create temporary validation file with subset
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="phase1_2_test_"))
        temp_validation_file = temp_dir / "validation_mbpp.parquet"
        
        subset_df.to_parquet(temp_validation_file, index=False)
        
        # Update config to use temporary directory
        config.phase0_1_output_dir = str(temp_dir)
        test_mode = True
    
    # Initialize model manager
    logger.info(f"Loading model: {config.model_name}")
    model_manager = ModelManager(config)
    
    try:
        # Load model
        model_manager.load_model()
        logger.info(f"Model loaded successfully on {model_manager.device}")
        
        # Create temperature generator
        generator = TemperatureVariationGenerator(
            model_manager=model_manager,
            config=config
        )
        
        # Run generation
        metadata = generator.run()
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("TEMPERATURE GENERATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Temperatures generated: {config.temperature_variation_temps}")
        logger.info(f"Tasks processed: {metadata['n_tasks']}")
        logger.info(f"Total samples generated: {metadata['n_total_samples']}")
        
        for temp_str, stats in metadata['temperature_stats'].items():
            logger.info(f"\nTemperature {temp_str}:")
            logger.info(f"  - Pass rate: {stats['pass_rate']:.1%}")
            logger.info(f"  - Correct: {stats['n_correct']}")
            logger.info(f"  - Incorrect: {stats['n_incorrect']}")
            logger.info(f"  - Avg generation time: {stats['avg_generation_time']:.2f}s")
        
        logger.info(f"\n✅ Phase 1.2 completed successfully")
        logger.info(f"Results saved to: {config.phase1_2_output_dir}")
        
    except Exception as e:
        logger.error(f"Phase 1.2 failed: {e}")
        # Always show traceback for debugging
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        # Cleanup
        if model_manager:
            model_manager.cleanup()
        cleanup_gpu_memory()
        
        # Clean up temporary directory if used
        if test_mode and 'temp_dir' in locals() and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


def run_phase2(config: Config, logger, device: str):
    """Run Phase 2: SAE Analysis (CPU-only, using saved activations)"""
    import pandas as pd
    import json
    from pathlib import Path
    from phase2_sae_analysis.activation_loader import ActivationLoader
    from phase2_sae_analysis.sae_analyzer import load_gemma_scope_sae, compute_separation_scores
    from phase2_sae_analysis.pile_filter import PileFilter
    # SAELayerConfig removed - using unified Config
    from common.utils import discover_latest_phase_output
    
    logger.info("Starting Phase 2: SAE Analysis (CPU-only)")
    logger.info("Phase 2 now loads saved activations from Phase 1 - no GPU required")
    
    # Get split to analyze
    split_name = getattr(config, '_split', 'sae')
    logger.info(f"Analyzing split: {split_name}")
    
    # Log configuration
    logger.info("\n" + config.dump(phase="2"))
    
    # Force CPU usage for Phase 2
    device = "cpu"
    
    # Load split data from Phase 0.1
    split_file = Path(config.phase0_1_output_dir) / f"{split_name}_mbpp.parquet"
    if not split_file.exists():
        logger.error(f"Split data not found: {split_file}")
        logger.error("Please run Phase 0.1 first to create problem splits")
        sys.exit(1)
    
    split_df = pd.read_parquet(split_file)
    split_task_ids = split_df['task_id'].tolist()
    logger.info(f"Loaded {len(split_task_ids)} task IDs for {split_name} split")
    
    # Get dataset path from input or auto-discovery
    if hasattr(config, '_input_file') and config._input_file:
        logger.info(f"Using specified dataset: {config._input_file}")
        dataset_path = config._input_file
    else:
        logger.info("Auto-discovering dataset from Phase 1...")
        dataset_path = discover_latest_phase_output("1")
        if not dataset_path:
            logger.error(f"No dataset found in {config.phase1_output_dir}. Please run Phase 1 first or specify --input")
            sys.exit(1)
        logger.info(f"Found dataset: {dataset_path}")
    
    # Load dataset and filter by split
    logger.info("Loading dataset metadata...")
    df = pd.read_parquet(dataset_path)
    dataset_path = Path(dataset_path)
    
    # Filter dataset by split task IDs
    df_split = df[df['task_id'].isin(split_task_ids)]
    
    # Get task IDs by category for this split
    correct_task_ids = df_split[df_split['test_passed'] == True]['task_id'].tolist()
    incorrect_task_ids = df_split[df_split['test_passed'] == False]['task_id'].tolist()
    
    logger.info(f"Split '{split_name}' stats: {len(correct_task_ids)} correct, {len(incorrect_task_ids)} incorrect")
    
    # Determine activation directory based on split
    if split_name == 'validation':
        # Check if temperature variations exist
        temp_activation_dir = Path(config.phase1_2_output_dir) / "activations"
        if temp_activation_dir.exists():
            logger.info("Using temperature-varied activations for validation split")
            activation_dirs = [temp_activation_dir]
        else:
            logger.info("No temperature variations found, using base activations")
            activation_dirs = [dataset_path.parent / "activations"]
    else:
        # Use base activations for SAE and hyperparams splits
        activation_dirs = [dataset_path.parent / "activations"]
    
    # For now, use the first activation directory (we'll enhance this later for temperature aggregation)
    activation_dir = activation_dirs[0]
    if not activation_dir.exists():
        logger.error(f"Activation directory not found: {activation_dir}")
        logger.error("No activation files found. Please run Phase 1 first to generate activations.")
        sys.exit(1)
    
    logger.info(f"Loading activations from: {activation_dir}")
    activation_loader = ActivationLoader(activation_dir)
    
    # Show activation summary
    summary = activation_loader.get_summary()
    logger.info(f"Found activations: {summary['n_correct_tasks']} correct, "
                f"{summary['n_incorrect_tasks']} incorrect, "
                f"{summary['n_layers']} layers: {summary['layers']}")
    
    # Use available layers from the activation summary
    available_layers = summary['layers']
    
    # Create results storage
    from phase2_sae_analysis.sae_analyzer import (
        SAEAnalysisResults, PVALatentDirection, SeparationScores,
        MultiLayerSAEResults
    )
    
    layer_results = {}
    
    # Process each layer
    logger.info("Processing SAE analysis for each layer...")
    for layer_idx in available_layers:
        logger.info(f"\nAnalyzing layer {layer_idx}...")
        
        try:
            # Load saved activations (use all task IDs from the split)
            correct_acts = activation_loader.load_batch(
                correct_task_ids,
                layer=layer_idx,
                category="correct"
            )
            incorrect_acts = activation_loader.load_batch(
                incorrect_task_ids,
                layer=layer_idx,
                category="incorrect"
            )
            
            logger.info(f"Loaded activations - correct: {correct_acts.shape}, incorrect: {incorrect_acts.shape}")
            
            # Load SAE for this layer
            sae_id = f"layer_{layer_idx}/width_{config.sae_width}/average_l0_{config.sae_sparsity}"
            sae = load_gemma_scope_sae(
                repo_id=config.sae_repo_id,
                sae_id=sae_id,
                device=device
            )
            
            # Apply SAE encoding
            import torch
            with torch.no_grad():
                correct_sae_acts = sae.encode(correct_acts)
                incorrect_sae_acts = sae.encode(incorrect_acts)
            
            # Compute separation scores
            scores = compute_separation_scores(correct_sae_acts, incorrect_sae_acts)
            
            # Create PVA directions
            correct_direction = PVALatentDirection(
                direction_type="correct",
                layer=layer_idx,
                feature_idx=scores.best_correct_idx,
                separation_score=scores.s_correct[scores.best_correct_idx].item(),
                f_correct=scores.f_correct[scores.best_correct_idx].item(),
                f_incorrect=scores.f_incorrect[scores.best_correct_idx].item()
            )
            
            incorrect_direction = PVALatentDirection(
                direction_type="incorrect",
                layer=layer_idx,
                feature_idx=scores.best_incorrect_idx,
                separation_score=scores.s_incorrect[scores.best_incorrect_idx].item(),
                f_correct=scores.f_correct[scores.best_incorrect_idx].item(),
                f_incorrect=scores.f_incorrect[scores.best_incorrect_idx].item()
            )
            
            # Store results
            layer_results[layer_idx] = SAEAnalysisResults(
                layer_idx=layer_idx,
                correct_direction=correct_direction,
                incorrect_direction=incorrect_direction,
                separation_scores=scores,
                correct_sae_activations=correct_sae_acts,
                incorrect_sae_activations=incorrect_sae_acts
            )
            
            logger.info(f"Layer {layer_idx} - Best correct feature: {correct_direction.feature_idx} "
                       f"(score: {correct_direction.separation_score:.3f})")
            logger.info(f"Layer {layer_idx} - Best incorrect feature: {incorrect_direction.feature_idx} "
                       f"(score: {incorrect_direction.separation_score:.3f})")
            
            # Clean up to save memory
            del sae, correct_acts, incorrect_acts, correct_sae_acts, incorrect_sae_acts
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Failed to analyze layer {layer_idx}: {e}")
            continue
    
    # Create multi-layer results
    results = MultiLayerSAEResults(
        layer_results=layer_results,
        layer_indices=list(layer_results.keys()),
        model_name=config.model_name,
        n_correct_samples=len(correct_task_ids[:100]),
        n_incorrect_samples=len(incorrect_task_ids[:100]),
        hook_component=config.sae_hook_component
    )
    
    # Print initial results
    logger.info("\n" + results.summary())
    
    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(config.phase2_output_dir) / f"multi_layer_results_{timestamp}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results.save_to_file(str(results_file))
    logger.info(f"\nResults saved to: {results_file}")
    
    # Apply Pile filtering if requested
    if config.pile_filter_enabled:
        logger.info(f"\nPile filtering requires loading the model temporarily.")
        logger.info("Skipping Pile filtering in this CPU-only implementation.")
        # TODO: Consider pre-computing Pile frequencies in a separate step
    
    logger.info("\n✅ Phase 2 completed successfully")


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
        logger.info("Auto-discovering SAE results from Phase 2...")
        sae_results_path = discover_latest_phase_output("2")
        if not sae_results_path:
            logger.error(f"No SAE results found in {config.phase2_output_dir}. Please run Phase 2 first or specify --input")
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
    
    # Phase 2 - check if run_phase2 has TODO or placeholder text
    if "# TODO: Implement SAE analysis" in current_file_content or "SAE Analysis not yet implemented" in current_file_content:
        placeholder_phases.append(2)
    else:
        implemented_phases.append(2)
    
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
    from phase1_0_dataset_building import Phase1Orchestrator
    from common.utils import discover_latest_phase_output
    
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
    """Quick test of Phase 2 SAE analysis with saved activations"""
    # Phase 2 is CPU-only and loads saved activations
    device = "cpu"
    
    # Run simplified version of Phase 2
    config = Config.from_args(args, phase="2")
    config.dataset_end_idx = 10  # Limit to 10 samples for testing
    
    run_phase2(config, logger, device)
    
    print(f"\n{'='*50}")
    print("PHASE 2 QUICK TEST")
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
        test_config.sae_checkpoint_dir = f"{get_phase_dir('2')}/test_checkpoints"  # Separate test dir
        
        print(f"🧠 Initializing SAE pipeline...")
        
        # Initialize pipeline
        pipeline = EnhancedSAEAnalysisPipeline(model, test_config, device=device)
        
        print(f"⚙️  Running SAE analysis on layer {args.layer}...")
        
        # Run analysis
        results = pipeline.analyze_all_residual_layers(
            correct_prompts=correct_prompts,
            incorrect_prompts=incorrect_prompts,
            layer_indices=[args.layer]
        )
        
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
    
    # Set global phase context first
    phase = args.phase if hasattr(args, 'phase') and args.phase else None
    if phase is not None:
        set_logging_phase(phase)
    
    # Setup logging with phase context
    log_level = "DEBUG" if hasattr(args, 'verbose') and args.verbose else "INFO"
    logging_manager = LoggingManager(
        phase=phase,
        log_level=log_level, 
        log_dir="data/logs"
    )
    logger = logging_manager.setup_logging("main")
    
    # Detect device once for the entire application
    try:
        from common.utils import detect_device
        device = str(detect_device())
        logger.info(f"Detected device: {device}")
    except Exception as e:
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
        # Create unified config from args
        config = Config.from_args(args, phase=str(args.phase))
        
        # Store input file path if provided
        if args.input:
            config._input_file = args.input
        
        # Handle phase-specific special arguments
        if args.phase == 0 and hasattr(args, 'dry_run'):
            config._dry_run = args.dry_run
        
        if args.phase == 0.1 and hasattr(args, 'generate_report'):
            config._generate_report = args.generate_report
        
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
            1: "Dataset Building",
            1.2: "Temperature Generation", 
            2: "SAE Analysis",
            3: "Validation"
        }
        
        print(f"\n{'='*60}")
        print(f"PHASE {args.phase}: {phase_names[args.phase].upper()}")
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
            elif args.phase == 1.2:
                run_phase1_2(config, logger, device)
            elif args.phase == 2:
                run_phase2(config, logger, device)
            elif args.phase == 3:
                run_phase3(config, logger, device)
            
            print(f"✅ Phase {args.phase} completed successfully!")
            
        except Exception as e:
            logger.error(f"Phase {args.phase} failed: {str(e)}")
            if config.verbose:
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
            sys.exit(1)


if __name__ == "__main__":
    main()