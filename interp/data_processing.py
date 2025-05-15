from datasets import load_dataset
import logging
import os
from datetime import datetime

def test_mbpp_range(start_idx=0, end_idx=3, debug=False):
    """Test MBPP examples with proper logging"""
    
    # Setup logging directory
    log_dir = "mbpp_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mbpp_test_{timestamp}.log")
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # logging.StreamHandler()  # Also output to console
        ]
    )
    
    # Load dataset
    logging.info(f"Loading MBPP dataset")
    dataset = load_dataset("mbpp")
    test_data = dataset['test']
    
    logging.info(f"Testing examples from index {start_idx} to {end_idx}")
    
    # Track overall results
    total_tests = 0
    passed_tests = 0
    
    # Process each example
    for idx in range(start_idx, min(end_idx + 1, len(test_data))):
        record = test_data[idx]
        
        logging.info(f"Processing example {idx} (Task ID: {record['task_id']})")
        
        logging.debug(f"PROBLEM:\n{record['text']}")
        logging.debug(f"CODE:\n{record['code']}")
        logging.debug(f"TEST CASES:")
        for i, test in enumerate(record['test_list']):
            logging.debug(f"  Test {i+1}: {test}")
        
        # Execute the solution code
        namespace = {}
        try:
            exec(record['code'], namespace)
            
            # Run tests
            example_tests = 0
            example_passed = 0
            
            for i, test in enumerate(record['test_list']):
                try:
                    exec(test, namespace)
                    logging.info(f"Test {i+1}: PASSED")
                    example_passed += 1
                except AssertionError:
                    logging.info(f"Test {i+1}: FAILED")
                except Exception as e:
                    logging.info(f"Test {i+1}: ERROR - {str(e)}")
                
                example_tests += 1
            
            total_tests += example_tests
            passed_tests += example_passed
            
            logging.info(f"Example summary: {example_passed}/{example_tests} tests passed")
                
        except Exception as e:
            logging.error(f"Failed to execute code: {str(e)}")
        
        logging.info("-" * 40)
    
    # Log overall summary
    logging.info(f"OVERALL SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    print(f"Logs saved to: {log_file}")
    return log_file

# Example usage
if __name__ == "__main__":
    # Test examples with index 0 through 5 with debug output
    test_mbpp_range(0, 5, debug=False)
    
    # Or for less verbose output:
    # test_mbpp_range(0, 5, debug=False)