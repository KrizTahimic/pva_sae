#!/usr/bin/env python3
"""
Run Phase 1: Dataset Building

This script runs only the dataset building phase of the thesis pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from phase1_dataset_building import DatasetBuildingOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 1: Dataset Building",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemma-2-9b',
        help='Model name to use'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Starting index'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=973,
        help='Ending index (inclusive)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='data/datasets',
        help='Directory for dataset files'
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Stream generation output'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Run cleanup before building'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("PHASE 1: DATASET BUILDING")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Range: {args.start} to {args.end}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"{'='*60}\n")
    
    try:
        # Create tester
        tester = DatasetBuildingOrchestrator(
            model_name=args.model,
            dataset_dir=args.dataset_dir
        )
        
        # Build dataset
        if args.cleanup:
            dataset_path = tester.build_dataset_simple_with_cleanup(
                start_idx=args.start,
                end_idx=args.end,
                stream=args.stream
            )
        else:
            dataset_path = tester.build_dataset_simple(
                start_idx=args.start,
                end_idx=args.end,
                stream=args.stream
            )
        
        print(f"\n✅ Phase 1 complete!")
        print(f"Dataset saved to: {dataset_path}")
        
    except Exception as e:
        print(f"\n❌ Phase 1 failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())