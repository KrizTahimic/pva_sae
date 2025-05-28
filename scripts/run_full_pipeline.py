#!/usr/bin/env python3
"""
Main orchestration script for running the complete PVA-SAE three-phase pipeline.

This script coordinates:
1. Phase 1: Dataset Building
2. Phase 2: SAE Analysis  
3. Phase 3: Validation (Statistical, Robustness, Steering)
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from common import (
    ExperimentConfig,
    ModelConfiguration,
    DatasetConfiguration,
    RobustnessConfig,
    AnalysisConfig,
    ValidationConfig,
    LoggingManager,
    ExperimentLogger
)
from phase1_dataset_building import ProductionDatasetBuilder
from orchestration.pipeline import ThesisPipeline


def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Run the complete PVA-SAE thesis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='pva_sae_experiment',
        help='Name for this experiment run'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments',
        help='Directory for experiment outputs'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemma-2-9b',
        help='Model name to use'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2000,
        help='Maximum tokens to generate'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Starting index for MBPP dataset'
    )
    parser.add_argument(
        '--end-idx',
        type=int,
        default=973,
        help='Ending index for MBPP dataset (inclusive)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='data/datasets',
        help='Directory for dataset files'
    )
    
    # Phase control
    parser.add_argument(
        '--phases',
        nargs='+',
        choices=['phase1', 'phase2', 'phase3', 'all'],
        default=['all'],
        help='Which phases to run'
    )
    parser.add_argument(
        '--skip-phase1',
        action='store_true',
        help='Skip dataset building (use existing dataset)'
    )
    parser.add_argument(
        '--dataset-file',
        type=str,
        help='Path to existing dataset file (for skipping phase 1)'
    )
    
    # Phase 1 specific
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Stream generation output'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint file'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=50,
        help='Checkpoint frequency'
    )
    
    # Phase 2 specific
    parser.add_argument(
        '--sae-model',
        type=str,
        help='Path to SAE model'
    )
    parser.add_argument(
        '--latent-threshold',
        type=float,
        default=0.02,
        help='Activation threshold for latent filtering'
    )
    
    # Phase 3 specific
    parser.add_argument(
        '--temperatures',
        nargs='+',
        type=float,
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
        help='Temperature values for robustness testing'
    )
    parser.add_argument(
        '--steering-coeffs',
        nargs='+',
        type=float,
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
        help='Steering coefficients'
    )
    
    # Debugging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration without running'
    )
    
    return parser


def load_or_create_config(args):
    """Load configuration from file or create from arguments"""
    if args.config:
        # Load from file
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        
        # Create config objects
        experiment_config = ExperimentConfig(**config_data.get('experiment', {}))
        model_config = ModelConfiguration(**config_data.get('model', {}))
        dataset_config = DatasetConfiguration(**config_data.get('dataset', {}))
        robustness_config = RobustnessConfig(**config_data.get('robustness', {}))
        analysis_config = AnalysisConfig(**config_data.get('analysis', {}))
        validation_config = ValidationConfig(**config_data.get('validation', {}))
    else:
        # Create from arguments
        experiment_config = ExperimentConfig(
            experiment_name=args.experiment_name,
            seed=42
        )
        
        model_config = ModelConfiguration(
            model_name=args.model,
            max_new_tokens=args.max_tokens
        )
        
        dataset_config = DatasetConfiguration(
            dataset_dir=args.dataset_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
        
        robustness_config = RobustnessConfig(
            checkpoint_frequency=args.checkpoint_freq,
            show_progress_bar=True
        )
        
        analysis_config = AnalysisConfig(
            sae_model_path=args.sae_model,
            latent_threshold=args.latent_threshold
        )
        
        validation_config = ValidationConfig(
            temperature_values=args.temperatures,
            steering_coefficients=args.steering_coeffs
        )
    
    return {
        'experiment': experiment_config,
        'model': model_config,
        'dataset': dataset_config,
        'robustness': robustness_config,
        'analysis': analysis_config,
        'validation': validation_config
    }


def run_phase1(configs, args, logger):
    """Run Phase 1: Dataset Building"""
    print("\n" + "="*80)
    print("PHASE 1: DATASET BUILDING")
    print("="*80)
    
    if args.skip_phase1:
        if not args.dataset_file:
            raise ValueError("--dataset-file required when skipping phase 1")
        print(f"Skipping Phase 1, using existing dataset: {args.dataset_file}")
        return args.dataset_file
    
    # Create tester
    tester = ProductionDatasetBuilder(
        model_name=configs['model'].model_name,
        debug=args.debug,
        dataset_dir=configs['dataset'].dataset_dir,
        max_new_tokens=configs['model'].max_new_tokens,
        robustness_config=configs['robustness']
    )
    
    # Build dataset
    dataset_path = tester.build_dataset_production(
        start_idx=configs['dataset'].start_idx,
        end_idx=configs['dataset'].end_idx,
        stream=args.stream,
        resume_from_checkpoint=args.resume
    )
    
    logger.log_event("phase1_complete", "Dataset building completed", {
        'dataset_path': dataset_path,
        'total_processed': tester.hardened_builder.total_processed
    })
    
    return dataset_path


def run_phase2(dataset_path, configs, args, logger):
    """Run Phase 2: SAE Analysis"""
    print("\n" + "="*80)
    print("PHASE 2: SAE ANALYSIS")
    print("="*80)
    
    # TODO: Implement SAE analysis
    print("SAE Analysis not yet implemented")
    logger.log_event("phase2_skipped", "SAE Analysis not implemented")
    
    return None


def run_phase3(dataset_path, sae_results, configs, args, logger):
    """Run Phase 3: Validation"""
    print("\n" + "="*80)
    print("PHASE 3: VALIDATION")
    print("="*80)
    
    # TODO: Implement validation
    print("Validation not yet implemented")
    logger.log_event("phase3_skipped", "Validation not implemented")
    
    return None


def main():
    """Main entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Load configurations
    configs = load_or_create_config(args)
    
    # Setup logging
    logging_manager = LoggingManager(
        log_dir='data/logs',
        log_level='DEBUG' if args.debug else 'INFO'
    )
    logger = logging_manager.setup_logging('thesis_pipeline')
    
    # Setup experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=args.experiment_name,
        output_dir=args.output_dir
    )
    
    # Log experiment start
    exp_logger.log_event("experiment_start", "Starting thesis pipeline", {
        'phases': args.phases,
        'model': configs['model'].model_name
    })
    
    if args.dry_run:
        print("\nDRY RUN - Configuration:")
        print(json.dumps({
            k: v.to_dict() if hasattr(v, 'to_dict') else v
            for k, v in configs.items()
        }, indent=2))
        return
    
    try:
        # Determine which phases to run
        phases_to_run = set()
        if 'all' in args.phases:
            phases_to_run = {'phase1', 'phase2', 'phase3'}
        else:
            phases_to_run = set(args.phases)
        
        # Run phases
        dataset_path = None
        sae_results = None
        validation_results = None
        
        if 'phase1' in phases_to_run:
            dataset_path = run_phase1(configs, args, exp_logger)
        elif args.dataset_file:
            dataset_path = args.dataset_file
        
        if 'phase2' in phases_to_run and dataset_path:
            sae_results = run_phase2(dataset_path, configs, args, exp_logger)
        
        if 'phase3' in phases_to_run and dataset_path:
            validation_results = run_phase3(dataset_path, sae_results, configs, args, exp_logger)
        
        # Log experiment completion
        exp_logger.log_event("experiment_complete", "Pipeline completed successfully")
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        
        # Save experiment log
        exp_logger.save()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        exp_logger.log_event("experiment_failed", f"Pipeline failed: {str(e)}")
        exp_logger.save()
        raise


if __name__ == "__main__":
    main()