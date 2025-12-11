#!/usr/bin/env python3
"""
Usage:
    python3 run.py phase 0                                      # Difficulty analysis
    python3 run.py phase 1                                      # Dataset building (single GPU)
    python3 run.py phase 2.5                                    # SAE analysis (auto-discovers input)
    python3 run.py phase 3.8                                    # AUROC/F1 evaluation

Range selection (for testing or subsetting):
    python3 run.py phase 1 --start 0 --end 10                   # Process first 10 problems
    python3 run.py phase 3.5 --start 0 --end 5                  # Validate first 5 problems

Manual input override:
    python3 run.py phase 2.5 --input data/phase1_0/dataset.parquet

Model/dataset configuration is controlled via config.py (not CLI args).
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common.logging import set_logging_phase, get_logger
from common.config import Config

# Import phase registry for single source of truth
from common.phase_registry import get_all_phase_ids, get_phase_choices_help
from common.phase_runner import run_phase as generic_run_phase, can_use_generic_runner


def setup_argument_parser():
    """Setup command line argument parser with phase-specific argument groups"""
    parser = ArgumentParser(
        description="Run phases",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Phase command (main functionality)
    phase_parser = subparsers.add_parser('phase', help='Run a specific phase')
    
    # Required phase selection - uses string IDs from registry (avoids float precision issues)
    phase_parser.add_argument(
        'phase',
        type=str,
        choices=get_all_phase_ids(),
        help=f'Phase to run: {get_phase_choices_help()}'
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
    
    # Experiment mode arguments for steering phases (4.5, 4.6, 4.8)
    phase_parser.add_argument(
        '--correction-only',
        action='store_true',
        help='Run only correction experiment (phases 4.5, 4.6, 4.8)'
    )
    phase_parser.add_argument(
        '--corruption-only',
        action='store_true',
        help='Run only corruption experiment (phases 4.5, 4.6, 4.8)'
    )
    phase_parser.add_argument(
        '--preservation-only',
        action='store_true',
        help='Run only preservation experiment (phase 4.8)'
    )

    # Visualization mode
    phase_parser.add_argument(
        '--viz-only',
        action='store_true',
        dest='viz_only',
        help='Regenerate visualizations without recomputing (requires previous run)'
    )

    return parser


def validate_phase_arguments(args):
    """Validate phase-specific argument requirements"""
    # Phase 1 model is now optional - defaults to config.py setting

    # Validate input file if provided
    if args.input and not Path(args.input).exists():
        raise ValueError(f"Input file not found: {args.input}")


def main():
    """Main entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle case where no command is provided
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Set global phase context first (before any logging)
    # args.phase is now a string, so no float precision issues
    if args.command == 'phase' and hasattr(args, 'phase'):
        set_logging_phase(args.phase)
    
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
    
    # Handle phase command
    if args.command == 'phase':
        # Now create logger after phase context is set
        logger = get_logger("main")
        
        # Log device info now that we have a logger
        logger.info(f"Detected device: {device}")
        
        # Create unified config from args
        # args.phase is now a string, so no conversion needed
        config = Config.from_args(args, phase=args.phase)
        
        # Store input file path if provided
        if args.input:
            config._input_file = args.input

        # Handle experiment mode arguments using phase registry
        from common.phase_registry import get_phase
        phase_info = get_phase(args.phase)
        if phase_info.experiment_modes:
            config_attr = phase_info.experiment_modes["config_attr"]
            mode = "all"  # default
            for flag, mode_value in phase_info.experiment_modes["flags"]:
                if getattr(args, flag, False):
                    mode = mode_value
                    break
            setattr(config, config_attr, mode)

        # Validate config for the phase
        try:
            config.validate(args.phase)
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)

        # Show config and exit if requested
        if args.show_config:
            print("\n" + config.dump(phase=args.phase))
            sys.exit(0)
        
        # Display phase info using registry (single source of truth)
        phase_info = get_phase(args.phase)
        print(f"\n{'='*60}")
        print(f"PHASE {args.phase}: {phase_info.name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Run selected phase using the generic runner
            # All phases now follow the standard Runner(config).run() pattern
            if can_use_generic_runner(args.phase):
                generic_run_phase(args.phase, config, device)
            else:
                # Phase 3 is a placeholder - not implemented
                logger.warning(f"Phase {args.phase} is not implemented yet")
                sys.exit(0)

            print(f"✅ Phase {args.phase} completed successfully!")
            
        except Exception as e:
            logger.error(f"Phase {args.phase} failed: {str(e)}")
            if config.verbose:
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
            sys.exit(1)


if __name__ == "__main__":
    main()