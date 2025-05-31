#!/usr/bin/env python3
"""
Single entry point for all PVA-SAE thesis phases.

Usage:
    python3 run.py --phase 0                                    # Difficulty analysis
    python3 run.py --phase 1 --model google/gemma-2-9b         # Dataset building
    python3 run.py --phase 2 --dataset data/datasets/latest.parquet  # SAE analysis
    python3 run.py --phase 3 --dataset data/datasets/latest.parquet  # Validation
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common.logging import LoggingManager


def setup_argument_parser():
    """Setup command line argument parser with phase-specific argument groups"""
    parser = argparse.ArgumentParser(
        description="Run PVA-SAE thesis phases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    
    # Phase command (existing functionality)
    phase_parser = subparsers.add_parser('phase', help='Run a specific phase')
    
    # Required phase selection
    phase_parser.add_argument(
        'phase',
        type=int,
        choices=[0, 1, 2, 3],
        help='Phase to run: 0=Difficulty Analysis, 1=Dataset Building, 2=SAE Analysis, 3=Validation'
    )
    
    # Global arguments (add to phase parser)
    phase_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Phase 0: Difficulty Analysis arguments
    phase0_group = phase_parser.add_argument_group('Phase 0: Difficulty Analysis')
    phase0_group.add_argument(
        '--output-dir',
        type=str,
        default='data/datasets',
        help='Directory to save difficulty mapping'
    )
    phase0_group.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving difficulty mapping to file'
    )
    phase0_group.add_argument(
        '--load-existing',
        type=str,
        help='Load and validate existing difficulty mapping file'
    )
    
    # Phase 1: Dataset Building arguments
    phase1_group = phase_parser.add_argument_group('Phase 1: Dataset Building')
    phase1_group.add_argument(
        '--model',
        type=str,
        default='google/gemma-2-9b',
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
        default='data/datasets',
        help='Directory for dataset files'
    )
    phase1_group.add_argument(
        '--stream',
        action='store_true',
        help='Stream generation output'
    )
    phase1_group.add_argument(
        '--cleanup',
        action='store_true',
        help='Run cleanup before building'
    )
    phase1_group.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for generation (1 for sequential)'
    )
    phase1_group.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs to use for generation'
    )
    
    # Phase 2: SAE Analysis arguments
    phase2_group = phase_parser.add_argument_group('Phase 2: SAE Analysis')
    phase2_group.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset file for analysis'
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
    
    elif args.phase == 2:
        # Phase 2 requires dataset
        if not args.dataset:
            raise ValueError("Phase 2 requires --dataset argument")
        if not Path(args.dataset).exists():
            raise ValueError(f"Dataset file not found: {args.dataset}")
    
    elif args.phase == 3:
        # Phase 3 requires dataset
        if not args.dataset:
            raise ValueError("Phase 3 requires --dataset argument")
        if not Path(args.dataset).exists():
            raise ValueError(f"Dataset file not found: {args.dataset}")


def run_phase0(args, logger):
    """Run Phase 0: Difficulty Analysis"""
    from phase0_difficulty_analysis.mbpp_preprocessor import MBPPPreprocessor
    
    logger.info("Starting Phase 0: MBPP difficulty analysis")
    
    preprocessor = MBPPPreprocessor(output_dir=args.output_dir)
    
    if args.load_existing:
        logger.info(f"Loading existing difficulty mapping: {args.load_existing}")
        difficulty_mapping = preprocessor.load_existing_mapping(args.load_existing)
        
        is_complete = preprocessor.validate_mapping_completeness(difficulty_mapping)
        
        if is_complete:
            logger.info("✅ Existing difficulty mapping is complete and valid")
            distribution = preprocessor.difficulty_analyzer.get_complexity_distribution()
            if distribution:
                logger.info(f"Complexity distribution - Mean: {distribution['mean']}, Median: {distribution['median']}")
        else:
            logger.error("❌ Existing difficulty mapping is incomplete")
            sys.exit(1)
    else:
        difficulty_mapping = preprocessor.preprocess_dataset(save_mapping=not args.no_save)
        
        if difficulty_mapping:
            logger.info("✅ Phase 0 completed successfully")
            logger.info(f"Analyzed {len(difficulty_mapping)} MBPP problems")
            
            if not args.no_save:
                latest_mapping = preprocessor.get_latest_difficulty_mapping_path()
                logger.info(f"Difficulty mapping available at: {latest_mapping}")
        else:
            logger.error("❌ Phase 0 failed")
            sys.exit(1)


def run_phase1(args, logger):
    """Run Phase 1: Dataset Building"""
    from phase1_dataset_building import DatasetBuildingOrchestrator
    
    logger.info("Starting Phase 1: Dataset Building")
    logger.info(f"Model: {args.model}, Range: {args.start}-{args.end}")
    logger.info(f"Batch size: {args.batch_size}, GPUs: {args.num_gpus}")
    
    tester = DatasetBuildingOrchestrator(
        model_name=args.model,
        dataset_dir=args.dataset_dir
    )
    
    if args.cleanup:
        dataset_path = tester.build_dataset_simple_with_cleanup(
            start_idx=args.start,
            end_idx=args.end,
            stream=args.stream,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus
        )
    else:
        dataset_path = tester.build_dataset_simple(
            start_idx=args.start,
            end_idx=args.end,
            stream=args.stream,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus
        )
    
    logger.info("✅ Phase 1 completed successfully")
    logger.info(f"Dataset saved to: {dataset_path}")


def run_phase2(args, logger):
    """Run Phase 2: SAE Analysis"""
    logger.info("Starting Phase 2: SAE Analysis")
    logger.info(f"Dataset: {args.dataset}")
    
    # TODO: Implement SAE analysis
    logger.info("SAE Analysis not yet implemented")
    logger.info("This phase will analyze latent representations using Sparse Autoencoders")


def run_phase3(args, logger):
    """Run Phase 3: Validation"""
    logger.info("Starting Phase 3: Validation")
    logger.info(f"Dataset: {args.dataset}")
    
    # TODO: Implement validation
    logger.info("Validation not yet implemented")
    logger.info("This phase will run statistical validation and model steering experiments")


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
            print(f"  Max Threads per Block: {props.max_threads_per_block}")
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


def main():
    """Main entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle case where no command is provided
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = "DEBUG" if hasattr(args, 'verbose') and args.verbose else "INFO"
    logging_manager = LoggingManager(log_level=log_level, log_dir="data/logs")
    logger = logging_manager.setup_logging(__name__)
    
    # Handle different commands
    if args.command == 'test-gpu':
        test_gpu_setup(args, logger)
        return
    
    elif args.command == 'phase':
        # Validate phase-specific arguments
        try:
            validate_phase_arguments(args)
        except ValueError as e:
            logger.error(f"Argument validation failed: {e}")
            sys.exit(1)
        
        # Display phase info
        phase_names = {
            0: "Difficulty Analysis",
            1: "Dataset Building", 
            2: "SAE Analysis",
            3: "Validation"
        }
        
        print(f"\n{'='*60}")
        print(f"PHASE {args.phase}: {phase_names[args.phase].upper()}")
        print(f"{'='*60}")
        
        try:
            # Run selected phase
            if args.phase == 0:
                run_phase0(args, logger)
            elif args.phase == 1:
                run_phase1(args, logger)
            elif args.phase == 2:
                run_phase2(args, logger)
            elif args.phase == 3:
                run_phase3(args, logger)
            
            print(f"✅ Phase {args.phase} completed successfully!")
            
        except Exception as e:
            logger.error(f"Phase {args.phase} failed: {str(e)}")
            if args.verbose:
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
            sys.exit(1)


if __name__ == "__main__":
    main()