#!/usr/bin/env python3
"""
Production Dataset Building Script

This script demonstrates how to use the hardened dataset building pipeline
for processing the full MBPP dataset (974 records) with production-grade
reliability features.

Usage:
    python run_production_build.py [options]

Options:
    --model MODEL        Model name (default: google/gemma-2-2b)
    --start START        Starting index (default: 0)
    --end END           Ending index (default: 973)
    --checkpoint FREQ    Checkpoint frequency (default: 50)
    --no-resume         Disable checkpoint resume
    --test-run          Run with only 10 records for testing
"""

import argparse
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interp.data_processing_hardened import (
    ProductionMBPPTester,
    HardeningConfig,
    create_production_config,
    estimate_processing_time
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Build MBPP dataset with production hardening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run with 10 records
  python run_production_build.py --test-run
  
  # Full production run with gemma-2-9b
  python run_production_build.py --model google/gemma-2-9b
  
  # Resume from checkpoint
  python run_production_build.py --model google/gemma-2-9b
  
  # Custom range with frequent checkpoints
  python run_production_build.py --start 100 --end 500 --checkpoint 25
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b",
        help="Model name to use (default: google/gemma-2-2b)"
    )
    
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting record index (default: 0)"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        default=973,
        help="Ending record index (default: 973)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=50,
        help="Checkpoint frequency in records (default: 50)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume functionality"
    )
    
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run with only 10 records for testing"
    )
    
    parser.add_argument(
        "--max-memory",
        type=float,
        default=100.0,
        help="Max memory usage in GB before warning (default: 100)"
    )
    
    parser.add_argument(
        "--max-gpu-memory",
        type=float,
        default=30.0,
        help="Max GPU memory usage in GB before warning (default: 30)"
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="mbpp_datasets",
        help="Directory for saving datasets (default: mbpp_datasets)"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="mbpp_logs",
        help="Directory for saving logs (default: mbpp_logs)"
    )
    
    return parser.parse_args()

def print_banner():
    """Print startup banner"""
    print("\n" + "="*70)
    print("MBPP DATASET PRODUCTION BUILD")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

def confirm_production_run(args):
    """Confirm with user before starting production run"""
    if args.test_run:
        return True
    
    num_records = args.end - args.start + 1
    estimated_time = estimate_processing_time(num_records, avg_time_per_record=30)
    
    print(f"⚠️  PRODUCTION RUN CONFIRMATION")
    print(f"   Model: {args.model}")
    print(f"   Records: {args.start} to {args.end} ({num_records} total)")
    print(f"   Estimated time: {estimated_time}")
    print(f"   Checkpoint frequency: every {args.checkpoint} records")
    print(f"   Resume enabled: {not args.no_resume}")
    print()
    
    response = input("Continue with production run? (yes/no): ").strip().lower()
    return response in ['yes', 'y']

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Adjust for test run
    if args.test_run:
        args.end = min(args.start + 9, 973)  # Process only 10 records
        args.checkpoint = 5  # More frequent checkpoints for testing
        print("ℹ️  TEST RUN MODE: Processing only 10 records")
    
    # Print banner
    print_banner()
    
    # Confirm production run
    if not confirm_production_run(args):
        print("❌ Production run cancelled by user")
        return 1
    
    try:
        # Create configuration
        print("📋 Creating production configuration...")
        config = create_production_config(
            checkpoint_frequency=args.checkpoint,
            max_memory_gb=args.max_memory,
            max_gpu_memory_gb=args.max_gpu_memory
        )
        
        # Add some production-specific settings
        config.show_progress_bar = True
        config.progress_log_frequency = 10
        config.autosave_frequency = 100
        config.memory_cleanup_frequency = 100
        
        # Initialize tester
        print(f"🔧 Initializing production tester with {args.model}...")
        tester = ProductionMBPPTester(
            model_name=args.model,
            config=config,
            debug=False,
            log_dir=args.log_dir,
            dataset_dir=args.dataset_dir
        )
        
        # Run production build
        print("🚀 Starting production build...")
        print("   (Press Ctrl+C to interrupt safely - progress will be saved)")
        print()
        
        summary = tester.build_dataset_production(
            start_idx=args.start,
            end_idx=args.end,
            resume=not args.no_resume,
            save_config=True
        )
        
        # Success!
        print("\n" + "="*70)
        print("✅ PRODUCTION BUILD COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Show key results
        print(f"\nKey Results:")
        print(f"  Total processed: {summary['total_processed']}")
        print(f"  Success rate: {summary['correct_rate']:.1f}%")
        print(f"  Failed records: {len(summary['failed_records'])}")
        
        if summary['failed_records']:
            print(f"\n⚠️  Failed record indices: {summary['failed_records'][:10]}")
            if len(summary['failed_records']) > 10:
                print(f"    ... and {len(summary['failed_records']) - 10} more")
        
        print(f"\n📁 Results saved in: {args.dataset_dir}")
        print(f"📋 Logs saved in: {args.log_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  BUILD INTERRUPTED BY USER")
        print("✓ Progress has been saved to checkpoint")
        print("ℹ️  Run the same command again to resume from checkpoint")
        return 2
        
    except Exception as e:
        print(f"\n\n❌ BUILD FAILED WITH ERROR:")
        print(f"   {type(e).__name__}: {str(e)}")
        print(f"\n📋 Check the logs for detailed error information")
        print(f"ℹ️  You may be able to resume from checkpoint by running again")
        return 1

if __name__ == "__main__":
    sys.exit(main())