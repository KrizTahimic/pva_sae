from datasets import load_dataset
import logging
import os
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import sys



class DeviceUtils:
    """Utility class for device detection and management"""
    
    @staticmethod
    def detect_best_device() -> str:
        """Detect the best available device"""
        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            logging.info("Using MPS (Apple Silicon) device")
            return "mps"
        # Check for CUDA
        elif torch.cuda.is_available():
            logging.info("Using CUDA device")
            return "cuda"
        # Fallback to CPU
        else:
            logging.info("Using CPU device")
            return "cpu"


class ModelManager:
    """Handles Gemma 2 model loading and code generation"""
    
    def __init__(self, model_name: str = "google/gemma-2-2b", device: Optional[str] = None):
        self.model_name = model_name
        self.device_str = device if device else DeviceUtils.detect_best_device()
        self.device = torch.device(self.device_str)  # Convert to proper torch device
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load Gemma 2 model and tokenizer"""
        if self.model is not None:
            logging.info("Model already loaded")
            print("✓ Model already loaded")
            return
            
        logging.info(f"Loading model: {self.model_name} on device: {self.device}")
        
        try:
            # Disable gradient computation to save memory
            torch.set_grad_enabled(False)
            
            # Load tokenizer
            print("📥 Loading tokenizer...")
            logging.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("✓ Tokenizer loaded")
            
            # Load model with appropriate device mapping
            print("📥 Loading model weights (this may take several minutes)...")
            logging.info("Loading model...")
            if self.device_str == "cuda":
                # Use device_map='auto' for multi-GPU or automatic placement
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map='auto',
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                # For MPS or CPU, load normally then move to device
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                print(f"📤 Moving model to {self.device}...")
                self.model = self.model.to(self.device)  # Now using proper torch.device
            
            # Set model to evaluation mode
            self.model.eval()
            print("🔧 Model set to evaluation mode")
            
            logging.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            print(f"✗ Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def generate_code(self, prompt: str, max_new_tokens: int = 200, stream: bool = True) -> str:
        """Generate code from problem prompt with deterministic output"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            logging.debug(f"Generating code for prompt: {prompt[:100]}...")
            
            # Tokenize input with attention mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                add_special_tokens=True,
                padding=True,
                truncation=True
            )
            
            # Move inputs to the same device as model
            if self.device_str in ["mps", "cuda"]:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Using proper torch.device
            
            if stream:
                print("🤖 Generating code (streaming):")
                print("-" * 50)
                # Create a text streamer for real-time output
                streamer = TextStreamer(
                    self.tokenizer, 
                    skip_prompt=True,  # Don't repeat the input prompt
                    skip_special_tokens=True
                )
            else:
                streamer = None
                print("🤖 Generating code...")
            
            # Generate with deterministic settings
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding (deterministic)
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                    num_return_sequences=1,
                    streamer=streamer  # This enables token-by-token streaming
                )
            
            if stream:
                print("\n" + "-" * 50)
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the generated text
            prompt_length = len(prompt)
            generated_code = generated_text[prompt_length:].strip()
            
            logging.debug(f"Generated code length: {len(generated_code)} characters")
            print("✓ Code generation complete")
            
            return generated_code
            
        except Exception as e:
            logging.error(f"Code generation failed: {str(e)}")
            print(f"✗ Code generation failed: {str(e)}")
            raise RuntimeError(f"Code generation failed: {str(e)}")
        
    
    def test_model(self, stream: bool = True) -> bool:
        """Simple test to verify model is loaded and working"""
        if not self.is_loaded():
            logging.error("Model not loaded for testing")
            return False
        
        try:
            test_prompt = "Write a simple function that adds two numbers:\n# Your code here"
            logging.info("Testing model with simple prompt...")
            print("Testing model generation...")
            
            result = self.generate_code(test_prompt, max_new_tokens=100, stream=stream)
            
            logging.info(f"Model test successful. Generated: {result[:50]}...")
            print(f"\n✓ Model test passed!")
            print(f"Full generated code:\n{result}")
            return True
            
        except Exception as e:
            logging.error(f"Model test failed: {str(e)}")
            print(f"✗ Model test failed: {str(e)}")
            return False

    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None
    
    def get_device_info(self) -> str:
        """Get information about the device being used"""
        return f"Device: {self.device}"
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            logging.info("Unloading model...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            # Clear CUDA cache if using CUDA
            if self.device_str == "cuda":
                torch.cuda.empty_cache()
            
            logging.info("Model unloaded successfully")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'model') and self.model is not None:
            self.unload_model()


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
    print("Testing Gemma 2-2B model loading...")
    
    # Test model loading and generation
    model_manager = ModelManager("google/gemma-2-2b")
    
    try:
        print("Loading model... (this may take a few minutes)")
        print("📥 Loading tokenizer...")
        print("📥 Loading model weights...")
        model_manager.load_model()
        print(f"✓ Model loaded successfully on {model_manager.get_device_info()}")
        
        # Test generation
        if model_manager.test_model():
            print("✓ Model is working correctly!")
        else:
            print("✗ Model test failed!")
            
    except Exception as e:
        print(f"✗ Failed to load model: {e}")

    # Uncomment to clean up model after testing
    # finally:
    #     print("🧹 Cleaning up...")
    #     model_manager.unload_model()
    #     print("Model unloaded.")
    
    print("\n" + "="*50)
    print("Now running MBPP tests...")

    # Using the class-based approach
    tester = MBPPTester(debug=False)
    summary = tester.test_range(0, 5)
    print(f"Final Results: {summary}")
    
    # Or using the backward compatible function
    # test_mbpp_range(0, 5, debug=False)