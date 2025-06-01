"""
Test execution functionality for Phase 1 of the PVA-SAE project.

This module handles the execution of generated code against test cases
to determine correctness according to the pass@1 criterion.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any

from phase1_dataset_building.dataset_manager import CodeTestResult


class TestExecutor:
    """Executes code against test cases"""
    
    @staticmethod
    def run_single_test(test_code: str, namespace: dict) -> Tuple[bool, Optional[str]]:
        """
        Execute single test case and return result
        
        Args:
            test_code: The test code to execute
            namespace: The namespace containing function definitions
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            exec(test_code, namespace)
            return True, None
        except AssertionError:
            return False, "Assertion failed"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def run_code_tests(code: str, test_cases: List[str], task_id: Optional[str] = None) -> CodeTestResult:
        """
        Execute code against multiple test cases
        
        Args:
            code: The generated code to test
            test_cases: List of test case strings
            task_id: Optional task identifier for logging
            
        Returns:
            CodeTestResult: Results of test execution
        """
        logger = logging.getLogger(__name__)
        
        if task_id:
            logger.debug(f"Testing code for task {task_id}")
        
        TestExecutor._log_test_setup(code, test_cases)
        
        # Prepare execution environment
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            logger.error(error_msg)
            return CodeTestResult(passed=0, total=len(test_cases), errors=[error_msg])
        
        # Execute test cases
        return TestExecutor._execute_test_cases(test_cases, namespace)
    
    @staticmethod
    def run_record_tests(record: dict) -> CodeTestResult:
        """
        Execute tests using ground truth code from MBPP record
        
        Args:
            record: MBPP record containing code and test cases
            
        Returns:
            CodeTestResult: Results of test execution
        """
        logger = logging.getLogger(__name__)
        task_id = record['task_id']
        logger.debug(f"PROBLEM:\n{record['text']}")
        
        return TestExecutor.run_code_tests(
            code=record['code'],
            test_cases=record['test_list'],
            task_id=task_id
        )
    
    @staticmethod
    def _log_test_setup(code: str, test_cases: List[str]):
        """Log test setup information"""
        logger = logging.getLogger(__name__)
        logger.debug(f"CODE TO TEST:\n{code}")
        logger.debug("TEST CASES:")
        for i, test in enumerate(test_cases):
            logger.debug(f"  Test {i+1}: {test}")
    
    @staticmethod
    def _execute_test_cases(test_cases: List[str], namespace: dict) -> CodeTestResult:
        """Execute all test cases and collect results"""
        logger = logging.getLogger(__name__)
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        
        for i, test_case in enumerate(test_cases):
            success, error_msg = TestExecutor.run_single_test(test_case, namespace)
            
            if success:
                logger.info(f"Test {i+1}: PASSED")
                passed_tests += 1
            else:
                status = "FAILED" if error_msg == "Assertion failed" else "ERROR"
                log_msg = f"Test {i+1}: {status}"
                if error_msg != "Assertion failed":
                    log_msg += f" - {error_msg}"
                    errors.append(f"Test {i+1}: {error_msg}")
                
                logger.info(log_msg)
        
        logger.info(f"Test summary: {passed_tests}/{total_tests} tests passed")
        return CodeTestResult(passed=passed_tests, total=total_tests, errors=errors)


class SafeTestExecutor(TestExecutor):
    """Safe test executor with timeout and error handling"""
    
    @staticmethod
    def run_code_with_timeout(code: str, test_cases: List[str], 
                              timeout: float = 5.0, task_id: Optional[str] = None) -> CodeTestResult:
        """
        Execute code against test cases with timeout protection
        
        Args:
            code: The generated code to test
            test_cases: List of test case strings
            timeout: Maximum execution time per test case
            task_id: Optional task identifier for logging
            
        Returns:
            CodeTestResult: Results of test execution
        """
        import signal
        import functools
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Test execution timed out")
        
        # For non-Unix systems, fall back to regular execution
        if not hasattr(signal, 'SIGALRM'):
            return TestExecutor.run_code_tests(code, test_cases, task_id)
        
        logger = logging.getLogger(__name__)
        
        if task_id:
            logger.debug(f"Testing code for task {task_id} with timeout {timeout}s")
        
        # Prepare execution environment
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            logger.error(error_msg)
            return CodeTestResult(passed=0, total=len(test_cases), errors=[error_msg])
        
        # Execute test cases with timeout
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        
        for i, test_case in enumerate(test_cases):
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                success, error_msg = TestExecutor.run_single_test(test_case, namespace)
                signal.alarm(0)  # Cancel timeout
                
                if success:
                    logger.info(f"Test {i+1}: PASSED")
                    passed_tests += 1
                else:
                    status = "FAILED" if error_msg == "Assertion failed" else "ERROR"
                    log_msg = f"Test {i+1}: {status}"
                    if error_msg != "Assertion failed":
                        log_msg += f" - {error_msg}"
                        errors.append(f"Test {i+1}: {error_msg}")
                    logger.info(log_msg)
                    
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                error_msg = f"Test {i+1}: TIMEOUT after {timeout}s"
                errors.append(error_msg)
                logger.error(error_msg)
            except Exception as e:
                signal.alarm(0)  # Cancel timeout
                error_msg = f"Test {i+1}: Unexpected error - {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        logger.info(f"Test summary: {passed_tests}/{total_tests} tests passed")
        return CodeTestResult(passed=passed_tests, total=total_tests, errors=errors)
    
    @staticmethod
    def validate_code_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python code syntax before execution
        
        Args:
            code: The code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        import ast
        
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Failed to parse code: {str(e)}"
    
    @staticmethod
    def extract_function_names(code: str) -> List[str]:
        """
        Extract function names from code
        
        Args:
            code: The code to analyze
            
        Returns:
            List of function names defined in the code
        """
        import ast
        
        try:
            tree = ast.parse(code)
            function_names = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_names.append(node.name)
            
            return function_names
        except:
            return []