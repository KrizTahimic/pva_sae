from datasets import load_dataset
import logging
import os
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class TestResult:
    """Data class to store test execution results"""
    passed: int
    total: int
    errors: List[str]
    
    @property
    def success_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0


class LoggingManager:
    """Handles logging configuration and setup"""
    
    @staticmethod
    def setup_logging(debug: bool = False, log_dir: str = "mbpp_logs") -> str:
        """Configure logging and return log file path"""
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"mbpp_test_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                # logging.StreamHandler()  # Also output to console
            ]
        )
        
        return log_file


class DatasetManager:
    """Handles dataset loading and management"""
    
    def __init__(self):
        self.dataset = None
        self.test_data = None
    
    def load_mbpp_dataset(self):
        """Load MBPP dataset"""
        logging.info("Loading MBPP dataset")
        self.dataset = load_dataset("mbpp")
        self.test_data = self.dataset['test']
        logging.info(f"Dataset loaded with {len(self.test_data)} test examples")
    
    def get_record(self, idx: int) -> Dict:
        """Get a specific record by index"""
        if self.test_data is None:
            raise ValueError("Dataset not loaded. Call load_mbpp_dataset() first.")
        return self.test_data[idx]
    
    def get_dataset_size(self) -> int:
        """Get total number of records"""
        return len(self.test_data) if self.test_data else 0


class TestExecutor:
    """Handles test execution for individual records"""
    
    @staticmethod
    def execute_single_test(test_code: str, namespace: Dict) -> Tuple[bool, Optional[str]]:
        """Execute a single test case and return (success, error_message)"""
        try:
            exec(test_code, namespace)
            return True, None
        except AssertionError:
            return False, "Assertion failed"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def execute_record_tests(record: Dict) -> TestResult:
        """Execute all tests for a single MBPP record"""
        task_id = record['task_id']
        
        logging.debug(f"PROBLEM:\n{record['text']}")
        logging.debug(f"CODE:\n{record['code']}")
        logging.debug("TEST CASES:")
        for i, test in enumerate(record['test_list']):
            logging.debug(f"  Test {i+1}: {test}")
        
        # Execute the solution code
        namespace = {}
        errors = []
        
        try:
            exec(record['code'], namespace)
        except Exception as e:
            error_msg = f"Failed to execute solution code: {str(e)}"
            logging.error(error_msg)
            return TestResult(passed=0, total=len(record['test_list']), errors=[error_msg])
        
        # Run individual tests
        passed_tests = 0
        total_tests = len(record['test_list'])
        
        for i, test in enumerate(record['test_list']):
            success, error_msg = TestExecutor.execute_single_test(test, namespace)
            
            if success:
                logging.info(f"Test {i+1}: PASSED")
                passed_tests += 1
            else:
                status = "FAILED" if error_msg == "Assertion failed" else "ERROR"
                logging.info(f"Test {i+1}: {status}" + (f" - {error_msg}" if error_msg != "Assertion failed" else ""))
                if error_msg != "Assertion failed":
                    errors.append(f"Test {i+1}: {error_msg}")
        
        logging.info(f"Example summary: {passed_tests}/{total_tests} tests passed")
        
        return TestResult(passed=passed_tests, total=total_tests, errors=errors)


class MBPPTester:
    """Main class for testing MBPP examples"""
    
    def __init__(self, debug: bool = False, log_dir: str = "mbpp_logs"):
        self.debug = debug
        self.log_dir = log_dir
        self.log_file = None
        
        # Initialize managers
        self.dataset_manager = DatasetManager()
        
        # Results tracking
        self.total_tests = 0
        self.passed_tests = 0
        self.record_results: List[TestResult] = []
    
    def setup_logging(self) -> str:
        """Setup logging configuration"""
        self.log_file = LoggingManager.setup_logging(self.debug, self.log_dir)
        return self.log_file
    
    def load_dataset(self):
        """Load the MBPP dataset"""
        self.dataset_manager.load_mbpp_dataset()
    
    def test_single_record(self, idx: int) -> TestResult:
        """Test a single MBPP record"""
        record = self.dataset_manager.get_record(idx)
        
        logging.info(f"Processing example {idx} (Task ID: {record['task_id']})")
        
        result = TestExecutor.execute_record_tests(record)
        
        # Update overall statistics
        self.total_tests += result.total
        self.passed_tests += result.passed
        self.record_results.append(result)
        
        logging.info("-" * 40)
        
        return result
    
    def test_range(self, start_idx: int = 0, end_idx: int = 3) -> Dict:
        """Test a range of MBPP examples"""
        # Setup
        self.setup_logging()
        self.load_dataset()
        
        # Validate range
        dataset_size = self.dataset_manager.get_dataset_size()
        actual_end_idx = min(end_idx, dataset_size - 1)
        
        logging.info(f"Testing examples from index {start_idx} to {actual_end_idx}")
        
        # Reset results
        self.total_tests = 0
        self.passed_tests = 0
        self.record_results = []
        
        # Process each example with progress bar
        for idx in tqdm(range(start_idx, actual_end_idx + 1), 
                        desc="Testing MBPP records", 
                        unit="record"):
            self.test_single_record(idx)
        
        # Log overall summary
        summary = self.get_summary()
        logging.info(f"OVERALL SUMMARY: {summary['passed']}/{summary['total']} tests passed ({summary['success_rate']:.1f}%)")
        
        print(f"Logs saved to: {self.log_file}")
        
        return summary
    
    def get_summary(self) -> Dict:
        """Get summary of test results"""
        return {
            'passed': self.passed_tests,
            'total': self.total_tests,
            'success_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0.0,
            'records_tested': len(self.record_results),
            'log_file': self.log_file
        }
    
    def get_detailed_results(self) -> List[TestResult]:
        """Get detailed results for each record"""
        return self.record_results


# Convenience function to maintain backward compatibility
def test_mbpp_range(start_idx: int = 0, end_idx: int = 3, debug: bool = False) -> str:
    """Test MBPP examples with proper logging (backward compatible function)"""
    tester = MBPPTester(debug=debug)
    summary = tester.test_range(start_idx, end_idx)
    return summary['log_file']


# Example usage
if __name__ == "__main__":
    # Using the class-based approach
    tester = MBPPTester(debug=False)
    summary = tester.test_range(0, 5)
    print(f"Final Results: {summary}")
    
    # Or using the backward compatible function
    # test_mbpp_range(0, 5, debug=False)