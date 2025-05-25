#!/usr/bin/env python3
"""
Test runner script for data_processing.py regression tests

Usage:
    python run_tests.py                  # Run all tests
    python run_tests.py --unit           # Run only unit tests
    python run_tests.py --integration    # Run only integration tests
    python run_tests.py --critical       # Run only critical tests
    python run_tests.py --fast           # Skip slow tests
    python run_tests.py --coverage       # Generate coverage report
    python run_tests.py --specific TestClassName  # Run specific test class
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd: list, cwd: str = None) -> int:
    """Run a command and return the exit code"""
    print(f"Running: {' '.join(cmd)}")
    print("-" * 70)
    
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run regression tests for data_processing.py"
    )
    
    # Test selection arguments
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run only integration tests"
    )
    parser.add_argument(
        "--critical", 
        action="store_true", 
        help="Run only critical tests"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Skip slow tests"
    )
    
    # Coverage arguments
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Generate detailed coverage report"
    )
    parser.add_argument(
        "--no-cov", 
        action="store_true", 
        help="Disable coverage reporting"
    )
    
    # Specific test arguments
    parser.add_argument(
        "--specific", 
        type=str, 
        help="Run specific test class or test method"
    )
    parser.add_argument(
        "--keyword", 
        "-k", 
        type=str, 
        help="Run tests matching keyword expression"
    )
    
    # Output arguments
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", 
        "-q", 
        action="store_true", 
        help="Quiet output"
    )
    
    # Debugging arguments
    parser.add_argument(
        "--pdb", 
        action="store_true", 
        help="Drop into debugger on test failure"
    )
    parser.add_argument(
        "--lf", 
        "--last-failed", 
        action="store_true", 
        help="Run only tests that failed last time"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["pytest"]
    
    # Add marker selections
    markers = []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.critical:
        markers.append("critical")
    if args.fast:
        markers.append("not slow")
    
    if markers:
        if len(markers) == 1:
            cmd.extend(["-m", markers[0]])
        else:
            cmd.extend(["-m", " or ".join(markers)])
    
    # Add coverage options
    if args.coverage:
        cmd.extend([
            "--cov=interp.data_processing",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-branch"
        ])
    elif args.no_cov:
        # Override the default coverage settings from pytest.ini
        cmd.append("--no-cov")
    
    # Add specific test selection
    if args.specific:
        cmd.append(f"tests/test_data_processing.py::{args.specific}")
    elif args.keyword:
        cmd.extend(["-k", args.keyword])
    else:
        cmd.append("tests/test_data_processing.py")
    
    # Add output options
    if args.verbose:
        cmd.append("-vv")
    elif args.quiet:
        cmd.append("-q")
    
    # Add debugging options
    if args.pdb:
        cmd.append("--pdb")
    if args.lf:
        cmd.append("--lf")
    
    # Find project root (where interp/ directory is)
    current_dir = Path(__file__).parent
    if current_dir.name == "tests":
        project_root = current_dir.parent
    else:
        project_root = current_dir
    
    # Run the tests
    print(f"Running tests from: {project_root}")
    exit_code = run_command(cmd, cwd=str(project_root))
    
    # Print results
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✅ All tests passed!")
        
        # If coverage was generated, show where to find the report
        if args.coverage:
            coverage_dir = project_root / "htmlcov"
            if coverage_dir.exists():
                index_file = coverage_dir / "index.html"
                print(f"\n📊 Coverage report: {index_file}")
                print(f"   Open in browser: file://{index_file.absolute()}")
    else:
        print("❌ Some tests failed!")
        print(f"   Exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())