#!/usr/bin/env python3
"""
Test script to verify the enriched dataset refactoring works correctly.

This script tests:
1. Phase 0 creates enriched dataset with all MBPP fields + complexity
2. Auto-discovery finds the enriched dataset
3. DatasetManager loads from enriched dataset
4. All phases can use the enriched data
"""

import sys
import pandas as pd
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from common.utils import discover_latest_phase_output, get_phase_dir
from phase1_0_dataset_building.dataset_manager import DatasetManager


def test_phase0_output():
    """Test that Phase 0 creates the correct enriched dataset."""
    print("\n=== Testing Phase 0 Output ===")
    
    # Check if Phase 0 output exists
    phase0_dir = Path(get_phase_dir("0"))
    if not phase0_dir.exists():
        print(f"❌ Phase 0 directory does not exist: {phase0_dir}")
        print("   Please run: python3 run.py phase 0")
        return False
    
    # Auto-discover enriched dataset
    enriched_path = discover_latest_phase_output("0")
    if not enriched_path:
        print("❌ No enriched dataset found")
        print("   Expected pattern: mbpp_with_complexity_*.parquet")
        return False
    
    print(f"✅ Found enriched dataset: {enriched_path}")
    
    # Load and verify structure
    try:
        df = pd.read_parquet(enriched_path)
        print(f"✅ Loaded dataset with shape: {df.shape}")
        
        # Check required columns
        required_columns = ['task_id', 'text', 'code', 'test_list', 'test_imports', 'cyclomatic_complexity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print(f"✅ All required columns present: {required_columns}")
        
        # Show sample data
        print("\nSample record:")
        sample = df.iloc[0]
        print(f"  task_id: {sample['task_id']}")
        print(f"  text: {sample['text'][:80]}...")
        print(f"  code length: {len(sample['code'])} chars")
        print(f"  test_list items: {len(sample['test_list'])}")
        print(f"  cyclomatic_complexity: {sample['cyclomatic_complexity']}")
        
        # Check complexity distribution
        print(f"\nComplexity distribution:")
        print(f"  Min: {df['cyclomatic_complexity'].min()}")
        print(f"  Max: {df['cyclomatic_complexity'].max()}")
        print(f"  Mean: {df['cyclomatic_complexity'].mean():.2f}")
        print(f"  Median: {df['cyclomatic_complexity'].median()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load enriched dataset: {e}")
        return False


def test_dataset_manager():
    """Test that DatasetManager correctly loads from enriched dataset."""
    print("\n=== Testing DatasetManager ===")
    
    try:
        # Create DatasetManager (now always uses enriched)
        dm = DatasetManager()
        dm.load_dataset()
        
        print(f"✅ DatasetManager loaded successfully")
        print(f"   Dataset size: {dm.get_size()}")
        
        # Test record access
        record = dm.get_record(0)
        print(f"✅ Retrieved record 0:")
        print(f"   task_id: {record['task_id']}")
        print(f"   Has complexity: {'cyclomatic_complexity' in record}")
        
        if 'cyclomatic_complexity' in record:
            print(f"   Complexity value: {record['cyclomatic_complexity']}")
        else:
            print("❌ Record missing cyclomatic_complexity field")
            return False
        
        # Test prompt generation still works
        prompt = dm.get_prompt_template(0)
        print(f"✅ Generated prompt (length: {len(prompt)} chars)")
        
        return True
        
    except Exception as e:
        print(f"❌ DatasetManager test failed: {e}")
        return False


def test_phase0_1_compatibility():
    """Test that Phase 0.1 can read the enriched dataset."""
    print("\n=== Testing Phase 0.1 Compatibility ===")
    
    enriched_path = discover_latest_phase_output("0")
    if not enriched_path:
        print("❌ No enriched dataset to test with")
        return False
    
    try:
        # Load as Phase 0.1 would
        df = pd.read_parquet(enriched_path)
        
        # Check required columns for Phase 0.1
        required = ['task_id', 'cyclomatic_complexity']
        if all(col in df.columns for col in required):
            print(f"✅ Phase 0.1 compatible - has required columns: {required}")
            return True
        else:
            print(f"❌ Missing columns for Phase 0.1")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test Phase 0.1 compatibility: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Enriched Dataset Refactoring")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings/errors during tests
        format='%(levelname)s: %(message)s'
    )
    
    tests = [
        ("Phase 0 Output", test_phase0_output),
        ("DatasetManager", test_dataset_manager),
        ("Phase 0.1 Compatibility", test_phase0_1_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! The enriched dataset refactoring is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())