"""
Solution evaluation functionality for Phase 1 of the PVA-SAE project.

This module handles the evaluation of generated code solutions against test cases
to determine correctness according to the pass@1 criterion for dataset building.
"""

from typing import Optional, Tuple, List, Dict, Any

from phase1_0_dataset_building.dataset_manager import CodeTestResult
from common.logging import get_logger

# Module-level logger
logger = get_logger("solution_evaluator", phase="1.0")


class SolutionEvaluator:
    """Evaluates generated code solutions for correctness"""
    
    @staticmethod
    def _evaluate_single_test(test_code: str, namespace: dict) -> Tuple[bool, Optional[str]]:
        """
        Evaluate single test case and return result
        
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
    def evaluate_solution(code: str, test_cases: List[str], task_id: Optional[str] = None) -> CodeTestResult:
        """
        Evaluate generated code against multiple test cases
        
        Args:
            code: The generated code to evaluate
            test_cases: List of test case strings
            task_id: Optional task identifier for logging
            
        Returns:
            CodeTestResult: Results of solution evaluation
        """
        # Use module-level logger
        
        if task_id:
            logger.debug(f"Evaluating solution for task {task_id}")
        
        SolutionEvaluator._log_evaluation_setup(code, test_cases)
        
        # Prepare execution environment
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            logger.error(error_msg)
            return CodeTestResult(passed=0, total=len(test_cases), errors=[error_msg])
        
        # Evaluate test cases
        return SolutionEvaluator._evaluate_test_cases(test_cases, namespace)
    
    @staticmethod
    def evaluate_mbpp_solution(record: dict) -> CodeTestResult:
        """
        Evaluate solution using MBPP record data
        
        Args:
            record: MBPP record containing code and test cases
            
        Returns:
            CodeTestResult: Results of solution evaluation
        """
        # Use module-level logger
        task_id = record['task_id']
        logger.debug(f"PROBLEM:\n{record['text']}")
        
        return SolutionEvaluator.evaluate_solution(
            code=record['code'],
            test_cases=record['test_list'],
            task_id=task_id
        )
    
    @staticmethod
    def _log_evaluation_setup(code: str, test_cases: List[str]):
        """Log evaluation setup information"""
        # Use module-level logger
        logger.debug(f"CODE TO EVALUATE:\n{code}")
        logger.debug("TEST CASES:")
        for i, test in enumerate(test_cases):
            logger.debug(f"  Test {i+1}: {test}")
    
    @staticmethod
    def _evaluate_test_cases(test_cases: List[str], namespace: dict) -> CodeTestResult:
        """Evaluate all test cases and collect results"""
        # Use module-level logger
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        
        for i, test_case in enumerate(test_cases):
            success, error_msg = SolutionEvaluator._evaluate_single_test(test_case, namespace)
            
            if success:
                logger.debug(f"Test {i+1}: PASSED")
                passed_tests += 1
            else:
                status = "FAILED" if error_msg == "Assertion failed" else "ERROR"
                log_msg = f"Test {i+1}: {status}"
                if error_msg != "Assertion failed":
                    log_msg += f" - {error_msg}"
                    errors.append(f"Test {i+1}: {error_msg}")
                
                logger.debug(log_msg)
        
        logger.info(f"Evaluation summary: {passed_tests}/{total_tests} tests passed")
        return CodeTestResult(passed=passed_tests, total=total_tests, errors=errors)


class SafeSolutionEvaluator(SolutionEvaluator):
    """Safe solution evaluator with timeout and error handling"""
    
    @staticmethod
    def evaluate_solution_with_timeout(code: str, test_cases: List[str], 
                                     timeout: float = 5.0, task_id: Optional[str] = None) -> CodeTestResult:
        """
        Evaluate solution with timeout protection
        
        Args:
            code: The generated code to evaluate
            test_cases: List of test case strings
            timeout: Maximum execution time per test case
            task_id: Optional task identifier for logging
            
        Returns:
            CodeTestResult: Results of solution evaluation
        """
        import signal
        import functools
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Solution evaluation timed out")
        
        # For non-Unix systems, fall back to regular evaluation
        if not hasattr(signal, 'SIGALRM'):
            return SolutionEvaluator.evaluate_solution(code, test_cases, task_id)
        
        # Use module-level logger
        
        if task_id:
            logger.debug(f"Evaluating solution for task {task_id} with timeout {timeout}s")
        
        # Prepare execution environment
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            logger.error(error_msg)
            return CodeTestResult(passed=0, total=len(test_cases), errors=[error_msg])
        
        # Evaluate test cases with timeout
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        
        for i, test_case in enumerate(test_cases):
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                success, error_msg = SolutionEvaluator._evaluate_single_test(test_case, namespace)
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
        
        logger.info(f"Evaluation summary: {passed_tests}/{total_tests} tests passed")
        return CodeTestResult(passed=passed_tests, total=total_tests, errors=errors)
    
    @staticmethod
    def validate_code_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python code syntax before evaluation
        
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