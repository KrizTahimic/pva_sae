#!/usr/bin/env python3
"""Test to demonstrate Python caching behavior."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("Testing Python Import Caching")
print("=" * 50)

# First import
from common.prompt_utils import PromptBuilder
print(f"\n1. First import - Template is:")
print(repr(PromptBuilder.TEMPLATE))

# Test the prompt
prompt = PromptBuilder.build_prompt(
    problem_description="Test problem",
    test_cases="assert test() == True"
)
print(f"\n2. Generated prompt:")
print(repr(prompt))

# Check for unwanted text
if "# Your code here:" in prompt:
    print("\n❌ Found '# Your code here:' in prompt")
else:
    print("\n✅ No '# Your code here:' in prompt")

print("\n" + "=" * 50)
print("Each time you run this script with:")
print("  python3 test_caching.py")
print("It will import fresh modules.")
print("=" * 50)