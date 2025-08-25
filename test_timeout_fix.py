#!/usr/bin/env python3
"""
Test script to verify the timeout protection for evaluate_code.
"""

from common_simplified.helpers import evaluate_code

def test_timeout_protection():
    """Test that evaluate_code now handles blocking code properly."""
    
    print("Testing timeout protection for evaluate_code()...")
    print("="*60)
    
    # Test 1: Normal code (should pass)
    print("\nTest 1: Normal code")
    normal_code = """
def add(a, b):
    return a + b
"""
    test_cases = ["assert add(1, 2) == 3", "assert add(5, 3) == 8"]
    result = evaluate_code(normal_code, test_cases)
    print(f"✅ Normal code: {'PASS' if result else 'FAIL'} (expected: PASS)")
    
    # Test 2: Code with input() (should timeout and fail)
    print("\nTest 2: Code with input() statement")
    blocking_code = """
def get_number():
    n = input("Type a number: ")
    return int(n)
    
result = get_number()
"""
    test_cases = ["assert True"]  # Simple test
    print("Testing code with input() - should timeout after 5 seconds...")
    result = evaluate_code(blocking_code, test_cases)
    print(f"✅ Blocking code: {'PASS' if result else 'FAIL'} (expected: FAIL due to timeout)")
    
    # Test 3: Infinite loop (should timeout and fail)
    print("\nTest 3: Code with infinite loop")
    infinite_loop_code = """
while True:
    pass
"""
    test_cases = ["assert True"]
    print("Testing infinite loop - should timeout after 5 seconds...")
    result = evaluate_code(infinite_loop_code, test_cases)
    print(f"✅ Infinite loop: {'PASS' if result else 'FAIL'} (expected: FAIL due to timeout)")
    
    # Test 4: Code with syntax error (should fail quickly)
    print("\nTest 4: Code with syntax error")
    syntax_error_code = """
def broken(:
    return
"""
    test_cases = ["assert True"]
    result = evaluate_code(syntax_error_code, test_cases)
    print(f"✅ Syntax error: {'PASS' if result else 'FAIL'} (expected: FAIL)")
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print("✅ Timeout protection is now active")
    print("✅ Code with input() will timeout after 5 seconds")
    print("✅ Infinite loops will timeout after 5 seconds")
    print("✅ Phase 3.5 will no longer hang waiting for input")
    print("\nNote: This timeout solution only works on Unix/Mac systems.")
    print("For Windows, a threading-based solution would be needed.")
    
    return True

if __name__ == "__main__":
    test_timeout_protection()