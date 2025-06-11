#!/usr/bin/env python3
"""
Single entry point for all PVA-SAE thesis phases.

Usage:
    python3 run.py phase 0                                      # Difficulty analysis
    python3 run.py phase 1 --model google/gemma-2-2b           # Dataset building (single GPU)
    python3 run.py phase 2                                      # SAE analysis (auto-discovers input)
    python3 run.py phase 3                                      # Validation (auto-discovers input)
    
For multi-GPU dataset building, use:
    python3 multi_gpu_launcher.py --phase 1 --num-gpus 3 --model google/gemma-2-2b
"""

import argparse
import sys
import os
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common.logging import LoggingManager
from common.gpu_utils import cleanup_gpu_memory, ensure_gpu_available, setup_cuda_environment
from common import MAX_NEW_TOKENS


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
        default='data/phase0',
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
        default='data/phase1_0',
        help='Directory for dataset files'
    )
    phase1_group.add_argument(
        '--difficulty-mapping',
        type=str,
        help='Path to difficulty mapping file (default: auto-discover from phase0)'
    )
    phase1_group.add_argument(
        '--no-auto-discover',
        action='store_true',
        help='Disable auto-discovery of difficulty mapping from phase0'
    )
    
    # Phase 2: SAE Analysis arguments
    phase2_group = phase_parser.add_argument_group('Phase 2: SAE Analysis')
    phase2_group.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset file for analysis (default: auto-discover from phase1)'
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
    
    # Phases 2-3 use auto-discovery, so no validation required
    # If explicit dataset is provided, validate it exists
    if args.dataset and not Path(args.dataset).exists():
        raise ValueError(f"Dataset file not found: {args.dataset}")


def run_phase0(args, logger, device: str):
    """Run Phase 0: Difficulty Analysis"""
    from phase0_difficulty_analysis.mbpp_preprocessor import MBPPPreprocessor
    
    logger.info("Starting Phase 0: MBPP difficulty analysis")
    
    # Initialize preprocessor with output directory (default: data/phase0)
    preprocessor = MBPPPreprocessor(output_dir=args.output_dir)
    
    # COMPUTATION PATH: Analyze all 974 MBPP problems from scratch
    # This loops through dataset, calculates cyclomatic complexity, test metrics
    # Creates new timestamped files: mbpp_difficulty_mapping_{timestamp}.parquet + .summary.json
    try:
        difficulty_mapping = preprocessor.preprocess_dataset(save_mapping=not args.dry_run)
        
        logger.info("✅ Phase 0 completed successfully")
        logger.info(f"Analyzed {len(difficulty_mapping)} MBPP problems")
        
        # FILE CREATION: Show path to newly created parquet file (only if files were saved)
        if not args.dry_run:
            # Gets most recent file by modification time from data/datasets/
            latest_mapping = preprocessor.get_latest_difficulty_mapping_path()
            logger.info(f"Difficulty mapping available at: {latest_mapping}")
            
    except Exception as e:
        logger.error(f"❌ Phase 0 failed: {str(e)}")
        sys.exit(1)


def run_phase1(args, logger, device: str):
    """Run Phase 1: Dataset Building"""
    from phase1_0_dataset_building import Phase1Orchestrator
    from phase0_difficulty_analysis.difficulty_analyzer import MBPPDifficultyAnalyzer
    from common.utils import discover_latest_phase0_mapping
    from common import ModelConfiguration, DatasetConfiguration, RobustnessConfig
    
    logger.info("Starting Phase 1: Dataset Building")
    logger.info(f"Model: {args.model}, Range: {args.start}-{args.end}")
    logger.info("Processing mode: Sequential (use multi_gpu_launcher.py for parallel processing)")
    
    # Load difficulty mapping from Phase 0
    difficulty_mapping = None
    if not args.no_auto_discover:
        if args.difficulty_mapping:
            # Use explicitly provided mapping
            logger.info(f"Loading difficulty mapping from: {args.difficulty_mapping}")
            difficulty_mapping = MBPPDifficultyAnalyzer.load_difficulty_mapping(args.difficulty_mapping)
        else:
            # Auto-discover from phase0 (required)
            logger.info("Auto-discovering difficulty mapping from Phase 0...")
            mapping_path = discover_latest_phase0_mapping()
            if not mapping_path:
                logger.error("No difficulty mapping found in data/phase0/")
                logger.error("Phase 1 requires Phase 0 difficulty analysis to be completed first.")
                logger.error("Please run: python3 run.py phase 0")
                sys.exit(1)
            
            logger.info(f"Found difficulty mapping: {mapping_path}")
            difficulty_mapping = MBPPDifficultyAnalyzer.load_difficulty_mapping(mapping_path)
    
    # Setup CUDA environment and cleanup GPUs before starting
    if torch.cuda.is_available():
        logger.info("Setting up CUDA environment and cleaning GPU memory...")
        setup_cuda_environment()
        cleanup_gpu_memory()
        
        # Ensure GPU is responsive
        gpu_device = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
        if not ensure_gpu_available(gpu_device):
            logger.warning(f"GPU {gpu_device} not responsive, attempting cleanup...")
            cleanup_gpu_memory(gpu_device)
    
    # Create configuration objects
    model_config = ModelConfiguration(
        model_name=args.model,
        max_new_tokens=MAX_NEW_TOKENS
    )
    
    dataset_config = DatasetConfiguration(
        dataset_dir=args.dataset_dir,
        start_idx=args.start,
        end_idx=args.end
    )
    
    robustness_config = RobustnessConfig()
    
    # Create orchestrator with configs
    orchestrator = Phase1Orchestrator(
        difficulty_mapping=difficulty_mapping,
        model_config=model_config,
        dataset_config=dataset_config,
        robustness_config=robustness_config
    )
    
    # Build dataset
    dataset_path = orchestrator.build_dataset(
        start_idx=args.start,
        end_idx=args.end
    )
    
    logger.info("✅ Phase 1 completed successfully")
    logger.info(f"Dataset saved to: {dataset_path}")


def run_phase2(args, logger, device: str):
    """Run Phase 2: SAE Analysis (CPU-only, using saved activations)"""
    import pandas as pd
    from pathlib import Path
    from phase2_sae_analysis.activation_loader import ActivationLoader
    from phase2_sae_analysis.sae_analyzer import load_gemma_scope_sae, compute_separation_scores
    from phase2_sae_analysis.pile_filter import PileFilter
    from common.config import SAELayerConfig
    from common.utils import discover_latest_phase1_dataset
    
    logger.info("Starting Phase 2: SAE Analysis (CPU-only)")
    logger.info("Phase 2 now loads saved activations from Phase 1 - no GPU required")
    
    # Force CPU usage for Phase 2
    device = "cpu"
    
    # Auto-discover dataset if not provided
    if not args.dataset:
        logger.info("Auto-discovering dataset from Phase 1...")
        dataset_path = discover_latest_phase1_dataset()
        if dataset_path:
            logger.info(f"Found dataset: {dataset_path}")
            args.dataset = dataset_path
        else:
            logger.error("No dataset found in data/phase1_0. Please run Phase 1 first or specify --dataset")
            sys.exit(1)
    else:
        logger.info(f"Using specified dataset: {args.dataset}")
    
    # Load dataset to get task IDs
    logger.info("Loading dataset metadata...")
    df = pd.read_parquet(args.dataset)
    dataset_path = Path(args.dataset)
    
    # Get task IDs by category
    correct_task_ids = df[df['test_passed'] == True]['task_id'].tolist()
    incorrect_task_ids = df[df['test_passed'] == False]['task_id'].tolist()
    
    logger.info(f"Dataset stats: {len(correct_task_ids)} correct, {len(incorrect_task_ids)} incorrect")
    
    # Initialize activation loader
    activation_dir = dataset_path.parent / "activations"
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
    
    # Configure SAE analysis
    sae_config = SAELayerConfig(
        gemma_2b_layers=summary['layers'],  # Use available layers
        save_after_each_layer=True,
        cleanup_after_layer=True,
        checkpoint_dir="data/phase2/checkpoints"
    )
    
    # Create results storage
    from phase2_sae_analysis.sae_analyzer import (
        SAEAnalysisResults, PVALatentDirection, SeparationScores,
        MultiLayerSAEResults
    )
    
    layer_results = {}
    
    # Process each layer
    logger.info("Processing SAE analysis for each layer...")
    for layer_idx in sae_config.gemma_2b_layers:
        logger.info(f"\nAnalyzing layer {layer_idx}...")
        
        try:
            # Load saved activations
            correct_acts = activation_loader.load_batch(
                correct_task_ids[:100],  # Limit for initial run
                layer=layer_idx,
                category="correct"
            )
            incorrect_acts = activation_loader.load_batch(
                incorrect_task_ids[:100],  # Limit for initial run
                layer=layer_idx,
                category="incorrect"
            )
            
            logger.info(f"Loaded activations - correct: {correct_acts.shape}, incorrect: {incorrect_acts.shape}")
            
            # Load SAE for this layer
            sae_id = f"layer_{layer_idx}/width_{sae_config.sae_width}/average_l0_{sae_config.sae_sparsity}"
            sae = load_gemma_scope_sae(
                repo_id=sae_config.sae_repo_id,
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
        model_name="google/gemma-2-2b",
        n_correct_samples=len(correct_task_ids[:100]),
        n_incorrect_samples=len(incorrect_task_ids[:100]),
        hook_component=sae_config.hook_component
    )
    
    # Print initial results
    logger.info("\n" + results.summary())
    
    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("data/phase2") / f"multi_layer_results_{timestamp}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results.save_to_file(str(results_file))
    logger.info(f"\nResults saved to: {results_file}")
    
    # Apply Pile filtering if requested
    if args.pile_filter:
        logger.info(f"\nPile filtering requires loading the model temporarily.")
        logger.info("Skipping Pile filtering in this CPU-only implementation.")
        # TODO: Consider pre-computing Pile frequencies in a separate step
    
    logger.info("\n✅ Phase 2 completed successfully")


def run_phase3(args, logger, device: str):
    """Run Phase 3: Validation"""
    from common.utils import discover_latest_phase2_results, discover_latest_phase1_dataset
    
    logger.info("Starting Phase 3: Validation")
    
    # Auto-discover Phase 2 results if not provided
    if not hasattr(args, 'sae_results') or not args.sae_results:
        logger.info("Auto-discovering SAE results from Phase 2...")
        sae_results_path = discover_latest_phase2_results()
        if sae_results_path:
            logger.info(f"Found SAE results: {sae_results_path}")
            args.sae_results = sae_results_path
        else:
            logger.error("No SAE results found in data/phase2. Please run Phase 2 first")
            sys.exit(1)
    
    # Auto-discover dataset if not provided
    if not args.dataset:
        logger.info("Auto-discovering dataset from Phase 1...")
        dataset_path = discover_latest_phase1_dataset()
        if dataset_path:
            logger.info(f"Found dataset: {dataset_path}")
            args.dataset = dataset_path
        else:
            logger.error("No dataset found in data/phase1_0. Please run Phase 1 first")
            sys.exit(1)
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"SAE Results: {args.sae_results}")
    
    # TODO: Implement validation
    logger.info("Validation not yet implemented")
    logger.info("This phase will run statistical validation and model steering experiments")
    logger.info("Results will be saved to data/phase3/")


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
            result = subprocess.run(['pkill', '-u', os.environ.get('USER', ''), '-f', 'python'], 
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
    from common.utils import discover_latest_phase1_dataset
    latest_dataset = discover_latest_phase1_dataset()
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
        print("   ❌ No datasets found in data/phase1_0")
    
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
        from common.utils import discover_latest_phase1_dataset
        import pandas as pd
        latest_dataset = discover_latest_phase1_dataset()
        if latest_dataset:
            df = pd.read_parquet(latest_dataset)
            print(f"   ✅ Dataset readable ({len(df)} records)")
            validation_results.append(("Dataset", True))
        else:
            print("   ❌ No datasets found in data/phase1_0")
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
    from common.utils import discover_latest_phase1_dataset
    
    print(f"\n{'='*50}")
    print("PHASE 1 QUICK TEST")
    print(f"{'='*50}")
    
    # Use auto-discovery to find latest dataset from Phase 1
    print("\n🔍 Auto-discovering dataset from Phase 1...")
    latest_dataset = discover_latest_phase1_dataset()
    if not latest_dataset:
        print("❌ No existing datasets found in data/phase1_0. Run Phase 1 first.")
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
    """Quick test of Phase 2 SAE analysis with small sample"""
    import pandas as pd
    from transformer_lens import HookedTransformer
    from phase2_sae_analysis.sae_analyzer import EnhancedSAEAnalysisPipeline
    from common.config import SAELayerConfig
    from common.utils import discover_latest_phase1_dataset
    
    print(f"\n{'='*50}")
    print("PHASE 2 QUICK TEST")
    print(f"{'='*50}")
    
    # Use auto-discovery to find latest dataset from Phase 1
    print("\n🔍 Auto-discovering dataset from Phase 1...")
    latest_dataset = discover_latest_phase1_dataset()
    if not latest_dataset:
        print("❌ No datasets found in data/phase1_0. Run Phase 1 first.")
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
        
        # Configure for single layer test
        sae_config = SAELayerConfig(
            gemma_2b_layers=[args.layer],  # Test single layer
            save_after_each_layer=False,       # Skip checkpointing for test
            cleanup_after_layer=True,          # Clean up memory
            checkpoint_dir="data/phase2/test_checkpoints"  # Separate test dir
        )
        
        print(f"🧠 Initializing SAE pipeline...")
        
        # Initialize pipeline
        pipeline = EnhancedSAEAnalysisPipeline(model, sae_config, device=device)
        
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
        print("   Make sure TransformerLens and dependencies are installed")
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
    
    # Setup logging
    log_level = "DEBUG" if hasattr(args, 'verbose') and args.verbose else "INFO"
    logging_manager = LoggingManager(log_level=log_level, log_dir="data/logs")
    logger = logging_manager.setup_logging(__name__)
    
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
                run_phase0(args, logger, device)
            elif args.phase == 1:
                run_phase1(args, logger, device)
            elif args.phase == 2:
                run_phase2(args, logger, device)
            elif args.phase == 3:
                run_phase3(args, logger, device)
            
            print(f"✅ Phase {args.phase} completed successfully!")
            
        except Exception as e:
            logger.error(f"Phase {args.phase} failed: {str(e)}")
            if args.verbose:
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
            sys.exit(1)


if __name__ == "__main__":
    main()