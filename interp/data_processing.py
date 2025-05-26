from datasets import load_dataset
import logging
import os
import time
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import glob

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODEL_NAME = "google/gemma-2-2b"
DEFAULT_LOG_DIR = "mbpp_logs"
DEFAULT_DATASET_DIR = "mbpp_datasets"


# ============================================================================
# Exception Hierarchy
# ============================================================================

class MBPPFrameworkError(Exception):
    """Base exception for MBPP framework"""
    pass

class ModelError(MBPPFrameworkError):
    """Model-related errors"""
    pass

class DatasetError(MBPPFrameworkError):
    """Dataset-related errors"""  
    pass

class TestExecutionError(MBPPFrameworkError):
    """Test execution errors"""
    pass

class LoggingConfigurationError(MBPPFrameworkError):
    """Logging setup errors"""
    pass

# ============================================================================
# Cleanup Utilities
# ============================================================================

def cleanup_old_files(directory: str, pattern: str, max_files: int = 3):
    """Simple cleanup function - keeps only latest 3 files matching pattern"""
    if not os.path.exists(directory):
        return
    
    files = glob.glob(os.path.join(directory, pattern))
    if len(files) <= max_files:
        return
    
    # Sort by modification time (newest first)
    files_with_times = [(f, os.path.getmtime(f)) for f in files]
    files_with_times.sort(key=lambda x: x[1], reverse=True)
    
    # Delete old files
    for file_path, _ in files_with_times[max_files:]:
        try:
            os.remove(file_path)
            print(f"ℹ️  Cleaned up old file: {os.path.basename(file_path)}")
            logging.info(f"Cleaned up old file: {os.path.basename(file_path)}")
        except Exception as e:
            logging.warning(f"Failed to delete {file_path}: {e}")

def auto_cleanup():
    """Automatically cleanup logs and datasets"""
    print("ℹ️  Performing automatic cleanup...")
    
    # Cleanup logs
    cleanup_old_files(DEFAULT_LOG_DIR, "mbpp_test_*.log", 3)
    
    # Cleanup datasets  
    cleanup_old_files(DEFAULT_DATASET_DIR, "mbpp_dataset_*.parquet", 3)
    cleanup_old_files(DEFAULT_DATASET_DIR, "dataset_results_*.json", 3)
    cleanup_old_files(DEFAULT_DATASET_DIR, "*_metadata.json", 3)
    
    print("✓ Cleanup completed!")

# ============================================================================
# Utility Classes  
# ============================================================================

class DeviceManager:
    """Handles device detection and management"""
    
    @staticmethod
    def detect_best_device() -> str:
        """Detect the best available device"""
        if torch.backends.mps.is_available():
            logging.info("Detected MPS (Apple Silicon) device")
            return "mps"
        elif torch.cuda.is_available():
            logging.info("Detected CUDA device")
            return "cuda"
        else:
            logging.info("Using CPU device")
            return "cpu"
    
    @staticmethod
    def get_torch_device(device_str: str) -> torch.device:
        """Convert device string to torch.device"""
        return torch.device(device_str)
    
    @staticmethod
    def get_optimal_dtype(device_str: str) -> torch.dtype:
        """Get optimal dtype for device"""
        return torch.float16 if device_str == "cuda" else torch.float32

class ConsoleOutput:
    """Centralized console output with consistent formatting"""
    
    @staticmethod
    def success(message: str):
        print(f"✓ {message}")
    
    @staticmethod  
    def error(message: str):
        print(f"✗ {message}")
    
    @staticmethod
    def info(message: str):
        print(f"ℹ️  {message}")
    
    @staticmethod
    def loading(message: str):
        print(f"📥 {message}")
    
    @staticmethod
    def working(message: str):
        print(f"🔧 {message}")
    
    @staticmethod
    def generating(message: str):
        print(f"🤖 {message}")

class ErrorContext:
    """Provides consistent error handling patterns"""
    
    @staticmethod
    def handle_and_raise(exception_class: type, message: str, 
                        original_error: Exception = None, log_error: bool = True):
        """Handle error with logging and raise appropriate exception"""
        if log_error:
            logging.error(message)
            ConsoleOutput.error(message)
        
        if original_error:
            raise exception_class(message) from original_error
        else:
            raise exception_class(message)

# ============================================================================
# Core Configuration Classes
# ============================================================================

class LoggingConfiguration:
    """Manages logging setup and configuration"""
    
    def __init__(self, debug: bool = False, log_dir: str = DEFAULT_LOG_DIR):
        self.debug = debug
        self.log_dir = log_dir
        self.log_file = None
        self._is_configured = False
    
    def setup_logging(self) -> str:
        """Configure logging system and return log file path"""
        if self._is_configured:
            return self.log_file
        
        try:
            self._ensure_log_directory()
            self.log_file = self._create_log_filename()
            self._configure_logging_handlers()
            self._is_configured = True
            
            logging.info("Logging system initialized")
            return self.log_file
            
        except Exception as e:
            raise LoggingConfigurationError(f"Failed to setup logging: {str(e)}") from e
    
    def _ensure_log_directory(self):
        """Create log directory if it doesn't exist"""
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _create_log_filename(self) -> str:
        """Generate timestamped log filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"mbpp_test_{timestamp}.log")
    
    def _configure_logging_handlers(self):
        """Configure logging handlers and formatters"""
        # Clear existing handlers to prevent conflicts
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(self.log_file)],
            force=True
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if logging is configured"""
        return self._is_configured
    
    @staticmethod
    def find_existing_log_file() -> Optional[str]:
        """Find existing log file from configured handlers"""
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename
        return None

class DatasetDirectoryManager:
    """Manages dataset directory structure and file organization"""
    
    def __init__(self, dataset_dir: str = DEFAULT_DATASET_DIR):
        self.dataset_dir = dataset_dir
        self._is_configured = False
    
    def setup_directory(self) -> str:
        """Create dataset directory and return path"""
        if self._is_configured:
            return self.dataset_dir
        
        try:
            self._ensure_dataset_directory()
            self._is_configured = True
            
            logging.info(f"Dataset directory ready: {self.dataset_dir}")
            return self.dataset_dir
            
        except Exception as e:
            raise DatasetError(f"Failed to setup dataset directory: {str(e)}") from e
    
    def _ensure_dataset_directory(self):
        """Create dataset directory if it doesn't exist"""
        os.makedirs(self.dataset_dir, exist_ok=True)
    
    def get_timestamped_filepath(self, base_name: str, extension: str) -> str:
        """Generate timestamped filepath in dataset directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_{timestamp}.{extension}"
        return os.path.join(self.dataset_dir, filename)
    
    def get_filepath(self, filename: str) -> str:
        """Get full filepath in dataset directory"""
        return os.path.join(self.dataset_dir, filename)
    
    @property
    def is_configured(self) -> bool:
        """Check if directory is configured"""
        return self._is_configured
    
    def list_datasets(self) -> dict[str, list[str]]:
        """List all dataset files by type"""
        if not os.path.exists(self.dataset_dir):
            return {'json': [], 'parquet': [], 'other': []}
        
        files = os.listdir(self.dataset_dir)
        datasets = {
            'json': [f for f in files if f.endswith('.json')],
            'parquet': [f for f in files if f.endswith('.parquet')],
            'metadata': [f for f in files if f.endswith('_metadata.json')],
            'other': [f for f in files if not any(f.endswith(ext) for ext in ['.json', '.parquet'])]
        }
        
        return datasets

# ============================================================================
# Model Management Classes
# ============================================================================

class ModelLoader:
    """Handles model and tokenizer loading operations"""
    
    def __init__(self, model_name: str, device_str: str):
        self.model_name = model_name
        self.device_str = device_str
        self.device = DeviceManager.get_torch_device(device_str)
        self.dtype = DeviceManager.get_optimal_dtype(device_str)
    
    def load_tokenizer(self) -> AutoTokenizer:
        """Load and return tokenizer"""
        try:
            ConsoleOutput.loading("Loading tokenizer...")
            logging.info(f"Loading tokenizer for {self.model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            ConsoleOutput.success("Tokenizer loaded")
            logging.info("Tokenizer loaded successfully")
            return tokenizer
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                ModelError, f"Tokenizer loading failed: {str(e)}", e
            )
    
    def load_model_weights(self) -> AutoModelForCausalLM:
        """Load and return model weights"""
        try:
            ConsoleOutput.loading("Loading model weights (this may take several minutes)...")
            logging.info(f"Loading model weights for {self.model_name} on {self.device_str}")
            
            # Disable gradients for inference
            torch.set_grad_enabled(False)
            
            if self.device_str == "cuda":
                model = self._load_cuda_model()
            else:
                model = self._load_non_cuda_model()
            
            ConsoleOutput.success("Model weights loaded")
            logging.info("Model weights loaded successfully")
            return model
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                ModelError, f"Model weight loading failed: {str(e)}", e
            )
    
    def _load_cuda_model(self) -> AutoModelForCausalLM:
        """Load model optimized for CUDA"""
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=self.dtype
        )
    
    def _load_non_cuda_model(self) -> AutoModelForCausalLM:
        """Load model for MPS or CPU"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype
        )
        ConsoleOutput.info(f"Moving model to {self.device}...")
        return model.to(self.device)
    
    def configure_for_inference(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """Configure model for inference mode"""
        try:
            model.eval()
            ConsoleOutput.working("Model configured for inference")
            logging.info("Model set to evaluation mode")
            return model
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                ModelError, f"Model configuration failed: {str(e)}", e
            )

class CodeGenerator:
    """Handles deterministic code generation"""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device_str: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device_str = device_str
        self.device = DeviceManager.get_torch_device(device_str)
    
    def generate(self, prompt: str, max_new_tokens: int = 200, stream: bool = True) -> str:
        """Generate code from prompt with deterministic settings"""
        try:
            logging.debug(f"Generating code for prompt: {prompt[:100]}...")
            
            inputs = self._prepare_inputs(prompt)
            outputs = self._generate_tokens(inputs, max_new_tokens, stream)
            generated_code = self._extract_code(outputs, prompt)
            
            return generated_code
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                ModelError, f"Code generation failed: {str(e)}", e
            )
    
    def _prepare_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        """Tokenize and prepare inputs for generation"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
        
        # Move to device if needed
        if self.device_str in ["mps", "cuda"]:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def _generate_tokens(self, inputs: dict[str, torch.Tensor], 
                        max_new_tokens: int, stream: bool) -> torch.Tensor:
        """Generate tokens using deterministic settings"""
        streamer = self._setup_streaming(stream)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic generation
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0,
                num_return_sequences=1,
                streamer=streamer
            )
        
        if stream:
            print("\n" + "-" * 50)
        
        return outputs
    
    def _setup_streaming(self, stream: bool) -> Optional[TextStreamer]:
        """Setup streaming output if requested"""
        if stream:
            ConsoleOutput.generating("Generating code (streaming):")
            print("-" * 50)
            return TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
        else:
            ConsoleOutput.generating("Generating code...")
            return None
    
    def _extract_code(self, outputs: torch.Tensor, prompt: str) -> str:
        """Extract generated code from model outputs"""
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_length = len(prompt)
        generated_code = generated_text[prompt_length:].strip()
        
        logging.debug(f"Generated code length: {len(generated_code)} characters")
        ConsoleOutput.success("Code generation complete")
        
        return generated_code

class ModelManager:
    """Orchestrates model lifecycle and code generation"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device_str = device if device else DeviceManager.detect_best_device()
        self.device = DeviceManager.get_torch_device(self.device_str)
        
        # Component references
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._loader = None
    
    def load_model(self):
        """Load model components"""
        if self.is_loaded():
            ConsoleOutput.info("Model already loaded")
            logging.info("Model already loaded")
            return
        
        try:
            logging.info(f"Starting model load: {self.model_name} on {self.device_str}")
            
            self._loader = ModelLoader(self.model_name, self.device_str)
            
            # Load components in sequence
            self.tokenizer = self._loader.load_tokenizer()
            self.model = self._loader.load_model_weights()
            self.model = self._loader.configure_for_inference(self.model)
            
            # Initialize generator
            self.generator = CodeGenerator(self.model, self.tokenizer, self.device_str)
            
            logging.info(f"Model pipeline ready on device: {self.device}")
            
        except Exception as e:
            self._cleanup_failed_load()
            raise  # Re-raise the specific exception from loader/generator
    
    def generate_code(self, prompt: str, max_new_tokens: int = 200, stream: bool = True) -> str:
        """Generate code using loaded model"""
        if not self.is_loaded():
            raise ModelError("Model not loaded. Call load_model() first.")
        
        return self.generator.generate(prompt, max_new_tokens, stream)
    
    def test_model_functionality(self, stream: bool = True) -> bool:
        """Test model with simple generation task"""
        if not self.is_loaded():
            logging.error("Cannot test: model not loaded")
            return False
        
        try:
            test_prompt = "Write a simple function that adds two numbers:\n# Your code here"
            logging.info("Running model functionality test...")
            ConsoleOutput.info("Testing model generation...")
            
            result = self.generate_code(test_prompt, max_new_tokens=100, stream=stream)
            
            logging.info(f"Test successful. Generated: {result[:50]}...")
            ConsoleOutput.success("Model functionality test passed!")
            print(f"\nGenerated code:\n{result}")
            return True
            
        except Exception as e:
            logging.error(f"Model test failed: {str(e)}")
            ConsoleOutput.error(f"Model test failed: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model pipeline is ready"""
        return (self.model is not None and 
                self.tokenizer is not None and 
                self.generator is not None)
    
    def get_device_info(self) -> str:
        """Get device information"""
        return f"Device: {self.device}"
    
    def unload_model(self):
        """Clean up model resources"""
        if self.model is not None:
            self._safe_log("Unloading model resources...")
            self._cleanup_all_resources()
            self._safe_log("Model unloaded successfully")
    
    def _cleanup_failed_load(self):
        """Clean up after failed loading attempt"""
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._loader = None
    
    def _cleanup_all_resources(self):
        """Clean up all allocated resources"""
        # Clean up model components
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        # Reset references
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._loader = None
        
        # Clear GPU cache if applicable
        if self.device_str == "cuda":
            torch.cuda.empty_cache()
    
    def _safe_log(self, message: str):
        """Safely log message (handles shutdown scenarios)"""
        try:
            logging.info(message)
        except (AttributeError, TypeError):
            # Handle case where logging is cleaned up during shutdown
            pass
    
    def __del__(self):
        """Cleanup on object destruction"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                self.unload_model()
        except (AttributeError, TypeError):
            # Handle cleanup during Python shutdown
            pass

# ============================================================================
# Data Management Classes
# ============================================================================

@dataclass
class TestResult:
    """Encapsulates test execution results"""
    passed: int
    total: int
    errors: list[str]
    
    @property
    def success_rate(self) -> float:
        """Calculate success percentage"""
        return (self.passed / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def failed(self) -> int:
        """Number of failed tests"""
        return self.total - self.passed

@dataclass
class GenerationResult:
    """Encapsulates code generation and testing results"""
    task_id: str
    problem_text: str
    prompt: str
    generated_code: str
    test_result: TestResult
    is_correct: bool
    generation_time: float
    error_type: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Get test success rate"""
        return self.test_result.success_rate
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'problem_text': self.problem_text,
            'prompt': self.prompt,
            'generated_code': self.generated_code,
            'is_correct': self.is_correct,
            'passed_tests': self.test_result.passed,
            'total_tests': self.test_result.total,
            'success_rate': self.success_rate,
            'test_errors': self.test_result.errors,
            'generation_time': self.generation_time,
            'error_type': self.error_type
        }
    
    def to_dataframe_row(self) -> dict:
        """Convert to flattened dictionary optimized for DataFrame storage"""
        return {
            'task_id': self.task_id,
            'problem_text': self.problem_text,
            'prompt': self.prompt,
            'generated_code': self.generated_code,
            'is_correct': self.is_correct,
            'passed_tests': self.test_result.passed,
            'total_tests': self.test_result.total,
            'success_rate': self.success_rate,
            'generation_time': self.generation_time,
            'error_type': self.error_type if self.error_type else "none",
            'test_errors_json': json.dumps(self.test_result.errors),  # Serialize complex field
            'prompt_length': len(self.prompt),
            'code_length': len(self.generated_code),
            'problem_length': len(self.problem_text)
        }

class PromptTemplateBuilder:
    """Constructs standardized prompt templates from MBPP records"""
    
    # Template constants
    CODE_INITIATOR = "# Your code here"
    
    def __init__(self):
        """Initialize the prompt template builder"""
        logging.info("PromptTemplateBuilder initialized")
    
    def build_prompt(self, record: dict) -> str:
        """
        Build standardized prompt from MBPP record
        
        Args:
            record: MBPP dataset record containing 'text' and 'test_list'
            
        Returns:
            str: Formatted prompt template ready for model input
        """
        try:
            self._validate_record(record)
            
            # Extract components
            problem_description = self._extract_problem_description(record['text'])
            test_cases = self._format_test_cases(record['test_list'])
            
            # Construct standardized template
            prompt = self._construct_template(problem_description, test_cases)
            
            task_id = record.get('task_id', 'unknown')
            logging.debug(f"Built prompt template for task_id: {task_id}")
            
            return prompt
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                DatasetError, f"Failed to build prompt template: {str(e)}", e
            )
    
    def build_batch_prompts(self, records: list[dict]) -> list[str]:
        """
        Build prompts for multiple records
        
        Args:
            records: List of MBPP dataset records
            
        Returns:
            list[str]: List of formatted prompt templates
        """
        prompts = []
        failed_records = []
        
        for i, record in enumerate(records):
            try:
                prompt = self.build_prompt(record)
                prompts.append(prompt)
            except Exception as e:
                task_id = record.get('task_id', f'index_{i}')
                failed_records.append(task_id)
                logging.error(f"Failed to build prompt for task_id {task_id}: {str(e)}")
        
        if failed_records:
            logging.warning(f"Failed to build prompts for {len(failed_records)} records: {failed_records}")
        
        logging.info(f"Successfully built {len(prompts)} prompts from {len(records)} records")
        return prompts
    
    def preview_template(self, record: dict, max_length: int = 200) -> str:
        """
        Generate a preview of the template for debugging
        
        Args:
            record: MBPP dataset record
            max_length: Maximum length for preview
            
        Returns:
            str: Truncated template preview
        """
        prompt = self.build_prompt(record)
        
        if len(prompt) <= max_length:
            return prompt
        
        preview = prompt[:max_length] + "..."
        return preview
    
    def get_template_stats(self, record: dict) -> dict[str, Any]:
        """
        Get statistics about the generated template
        
        Args:
            record: MBPP dataset record
            
        Returns:
            dict containing template statistics
        """
        prompt = self.build_prompt(record)
        
        stats = {
            'total_length': len(prompt),
            'total_lines': prompt.count('\n') + 1,
            'problem_description_length': len(record['text'].strip()),
            'num_test_cases': len(record['test_list']),
            'test_cases_length': sum(len(test.strip()) for test in record['test_list']),
            'task_id': record.get('task_id', 'unknown')
        }
        
        return stats
    
    def _validate_record(self, record: dict):
        """
        Validate MBPP record has required fields
        
        Args:
            record: MBPP dataset record to validate
            
        Raises:
            ValueError: If record is missing required fields or has invalid format
        """
        required_fields = ['text', 'test_list']
        
        # Check required fields exist
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate field types and content
        if not isinstance(record['text'], str):
            raise ValueError("'text' field must be a string")
        
        if not isinstance(record['test_list'], list):
            raise ValueError("'test_list' field must be a list")
        
        if len(record['test_list']) == 0:
            raise ValueError("'test_list' cannot be empty")
        
        # Validate test cases are strings
        for i, test in enumerate(record['test_list']):
            if not isinstance(test, str):
                raise ValueError(f"Test case {i} must be a string")
            if not test.strip():
                raise ValueError(f"Test case {i} cannot be empty")
        
        # Log successful validation
        task_id = record.get('task_id', 'unknown')
        logging.debug(f"Record validation passed for task_id: {task_id}")
    
    def _extract_problem_description(self, text: str) -> str:
        """
        Extract and clean problem description
        
        Args:
            text: Raw problem description from MBPP record
            
        Returns:
            str: Cleaned problem description
        """
        # Clean and normalize the problem description
        description = text.strip()
        
        # Remove any extra whitespace while preserving line breaks
        lines = [line.strip() for line in description.split('\n')]
        cleaned_description = '\n'.join(line for line in lines if line)
        
        return cleaned_description
    
    def _format_test_cases(self, test_list: list[str]) -> str:
        """
        Format test cases for template inclusion
        
        Args:
            test_list: List of test case strings from MBPP record
            
        Returns:
            str: Formatted test cases ready for template
        """
        formatted_tests = []
        
        for test in test_list:
            # Clean up each test case
            cleaned_test = test.strip()
            if cleaned_test:
                formatted_tests.append(cleaned_test)
        
        # Join test cases with newlines
        return '\n'.join(formatted_tests)
    
    def _construct_template(self, problem_description: str, test_cases: str) -> str:
        """
        Construct the final standardized template
        
        Args:
            problem_description: Cleaned problem description
            test_cases: Formatted test cases
            
        Returns:
            str: Complete standardized prompt template
        """
        # Construct template with consistent formatting:
        # 1. Problem description
        # 2. Empty line
        # 3. Test cases
        # 4. Empty line  
        # 5. Code initiator
        template = f"{problem_description}\n\n{test_cases}\n\n{self.CODE_INITIATOR}"
        
        return template

class DatasetManager:
    """Manages MBPP dataset operations"""
    
    def __init__(self):
        self.dataset = None
        self.test_data = None
        self._is_loaded = False
    
    def load_dataset(self):
        """Load MBPP dataset from Hugging Face"""
        if self._is_loaded:
            logging.info("MBPP dataset already loaded")
            return
        
        try:
            logging.info("Loading MBPP dataset...")
            self.dataset = load_dataset("mbpp")
            self.test_data = self.dataset['test']
            self._is_loaded = True
            
            logging.info(f"MBPP dataset loaded: {len(self.test_data)} examples")
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                DatasetError, f"Failed to load MBPP dataset: {str(e)}", e
            )
    
    def get_record(self, idx: int) -> dict:
        """Retrieve record by index with validation"""
        self._ensure_loaded()
        
        try:
            return self.test_data[idx]
        except IndexError as e:
            ErrorContext.handle_and_raise(
                DatasetError, 
                f"Index {idx} out of range. Dataset has {len(self.test_data)} records.",
                e
            )
    
    def get_size(self) -> int:
        """Get dataset size"""
        return len(self.test_data) if self._is_loaded else 0
    
    def is_loaded(self) -> bool:
        """Check if dataset is loaded"""
        return self._is_loaded
    
    def _ensure_loaded(self):
        """Ensure dataset is loaded before access"""
        if not self._is_loaded:
            raise DatasetError("Dataset not loaded. Call load_dataset() first.")

class EnhancedDatasetManager(DatasetManager):
    """Extended DatasetManager with prompt template functionality"""
    
    def __init__(self):
        super().__init__()
        self.template_builder = PromptTemplateBuilder()
    
    def get_prompt_template(self, idx: int) -> str:
        """
        Get standardized prompt template for record by index
        
        Args:
            idx: Index of the record in the dataset
            
        Returns:
            str: Standardized prompt template
        """
        record = self.get_record(idx)
        return self.template_builder.build_prompt(record)
    
    def get_batch_prompts(self, start_idx: int, end_idx: int) -> list[str]:
        """
        Get prompt templates for a range of records
        
        Args:
            start_idx: Starting index (inclusive)
            end_idx: Ending index (inclusive)
            
        Returns:
            list[str]: List of prompt templates
        """
        self._ensure_loaded()
        
        # Validate range
        dataset_size = self.get_size()
        if start_idx < 0 or start_idx >= dataset_size:
            raise DatasetError(f"start_idx {start_idx} out of range [0, {dataset_size-1}]")
        if end_idx < start_idx or end_idx >= dataset_size:
            raise DatasetError(f"end_idx {end_idx} out of range [{start_idx}, {dataset_size-1}]")
        
        # Extract records and build prompts
        records = [self.test_data[i] for i in range(start_idx, end_idx + 1)]
        return self.template_builder.build_batch_prompts(records)
    
    def preview_prompt_template(self, idx: int, max_length: int = 200) -> str:
        """
        Preview prompt template for debugging
        
        Args:
            idx: Index of the record
            max_length: Maximum length for preview
            
        Returns:
            str: Truncated prompt preview
        """
        record = self.get_record(idx)
        return self.template_builder.preview_template(record, max_length)
    
    def get_prompt_stats(self, idx: int) -> dict[str, Any]:
        """
        Get statistics for prompt template
        
        Args:
            idx: Index of the record
            
        Returns:
            dict: Template statistics
        """
        record = self.get_record(idx)
        return self.template_builder.get_template_stats(record)

# ============================================================================
# Test Execution Classes  
# ============================================================================

class TestExecutor:
    """Executes code against test cases"""
    
    @staticmethod
    def run_single_test(test_code: str, namespace: dict) -> tuple[bool, Optional[str]]:
        """Execute single test case and return result"""
        try:
            exec(test_code, namespace)
            return True, None
        except AssertionError:
            return False, "Assertion failed"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def run_code_tests(code: str, test_cases: list[str], task_id: Optional[str] = None) -> TestResult:
        """Execute code against multiple test cases"""
        if task_id:
            logging.debug(f"Testing code for task {task_id}")
        
        TestExecutor._log_test_setup(code, test_cases)
        
        # Prepare execution environment
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            logging.error(error_msg)
            return TestResult(passed=0, total=len(test_cases), errors=[error_msg])
        
        # Execute test cases
        return TestExecutor._execute_test_cases(test_cases, namespace)
    
    @staticmethod
    def run_record_tests(record: dict) -> TestResult:
        """Execute tests using ground truth code from MBPP record"""
        task_id = record['task_id']
        logging.debug(f"PROBLEM:\n{record['text']}")
        
        return TestExecutor.run_code_tests(
            code=record['code'],
            test_cases=record['test_list'],
            task_id=task_id
        )
    
    @staticmethod
    def _log_test_setup(code: str, test_cases: list[str]):
        """Log test setup information"""
        logging.debug(f"CODE TO TEST:\n{code}")
        logging.debug("TEST CASES:")
        for i, test in enumerate(test_cases):
            logging.debug(f"  Test {i+1}: {test}")
    
    @staticmethod
    def _execute_test_cases(test_cases: list[str], namespace: dict) -> TestResult:
        """Execute all test cases and collect results"""
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        
        for i, test_case in enumerate(test_cases):
            success, error_msg = TestExecutor.run_single_test(test_case, namespace)
            
            if success:
                logging.info(f"Test {i+1}: PASSED")
                passed_tests += 1
            else:
                status = "FAILED" if error_msg == "Assertion failed" else "ERROR"
                log_msg = f"Test {i+1}: {status}"
                if error_msg != "Assertion failed":
                    log_msg += f" - {error_msg}"
                    errors.append(f"Test {i+1}: {error_msg}")
                
                logging.info(log_msg)
        
        logging.info(f"Test summary: {passed_tests}/{total_tests} tests passed")
        return TestResult(passed=passed_tests, total=total_tests, errors=errors)

# ============================================================================
# Dataset Building Pipeline Classes
# ============================================================================

class DatasetBuilder:
    """Builds dataset by generating and classifying code solutions"""
    
    def __init__(self, model_manager: ModelManager, dataset_manager: EnhancedDatasetManager,
                 max_new_tokens: int = 200, stream_output: bool = False, 
                 dataset_dir: str = DEFAULT_DATASET_DIR):
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.max_new_tokens = max_new_tokens
        self.stream_output = stream_output
        
        # Directory management
        self.directory_manager = DatasetDirectoryManager(dataset_dir)
        
        # Results tracking
        self.generation_results: list[GenerationResult] = []
        self.total_processed = 0
        self.correct_solutions = 0
        self.incorrect_solutions = 0
    
    def build_dataset(self, start_idx: int = 0, end_idx: int = 2) -> list[GenerationResult]:
        """
        Build dataset by processing MBPP records and generating solutions
        
        Args:
            start_idx: Starting index for MBPP records
            end_idx: Ending index for MBPP records (inclusive)
            
        Returns:
            list[GenerationResult]: Results for each processed record
        """
        try:
            self._validate_prerequisites()
            self._validate_range(start_idx, end_idx)
            
            logging.info(f"Starting dataset building for records {start_idx} to {end_idx}")
            ConsoleOutput.info(f"Building dataset for {end_idx - start_idx + 1} records...")
            
            # Reset statistics
            self._reset_statistics()
            
            # Process records with progress tracking
            results = self._process_record_batch(start_idx, end_idx)
            
            # Log final statistics
            self._log_final_statistics()
            
            return results
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                DatasetError, f"Dataset building failed: {str(e)}", e
            )
    
    def process_single_record(self, idx: int) -> GenerationResult:
        """
        Process a single MBPP record: generate code and test it
        
        Args:
            idx: Index of MBPP record to process
            
        Returns:
            GenerationResult: Complete result with generation and testing info
        """
        try:
            # Get record and build prompt
            record = self.dataset_manager.get_record(idx)
            prompt = self.dataset_manager.get_prompt_template(idx)
            
            task_id = record['task_id']
            logging.info(f"Processing record {idx} (Task ID: {task_id})")
            
            if self.stream_output:
                print(f"\n{'='*60}")
                print(f"PROCESSING TASK {task_id} (Record {idx})")
                print(f"{'='*60}")
                print(f"PROBLEM: {record['text']}")
                print(f"{'='*60}")
            
            # Generate code
            generation_start = time.time()
            generated_code = self._generate_code_safely(prompt, task_id)
            generation_time = time.time() - generation_start
            
            # Test generated code
            test_result = self._test_generated_code(generated_code, record, task_id)
            
            # Classify result
            is_correct, error_type = self._classify_solution(test_result, generated_code)
            
            # Create result object
            result = GenerationResult(
                task_id=task_id,
                problem_text=record['text'],
                prompt=prompt,
                generated_code=generated_code,
                test_result=test_result,
                is_correct=is_correct,
                generation_time=generation_time,
                error_type=error_type
            )
            
            # Update statistics
            self._update_statistics(result)
            
            # Log result
            self._log_single_result(result, idx)
            
            return result
            
        except Exception as e:
            # Create failed result for consistency
            error_msg = str(e)
            logging.error(f"Failed to process record {idx}: {error_msg}")
            
            return GenerationResult(
                task_id=f"failed_{idx}",
                problem_text="",
                prompt="",
                generated_code="",
                test_result=TestResult(passed=0, total=0, errors=[error_msg]),
                is_correct=False,
                generation_time=0.0,
                error_type="generation_failed"
            )
    
    def get_statistics(self) -> dict[str, Any]:
        """Get current dataset building statistics"""
        return {
            'total_processed': self.total_processed,
            'correct_solutions': self.correct_solutions,
            'incorrect_solutions': self.incorrect_solutions,
            'correct_rate': (self.correct_solutions / self.total_processed * 100) 
                           if self.total_processed > 0 else 0.0,
            'results_count': len(self.generation_results)
        }
    
    def save_results(self, filepath: str = None) -> str:
        """Save generation results to JSON file"""
        # Setup directory
        self.directory_manager.setup_directory()
        
        if filepath is None:
            filepath = self.directory_manager.get_timestamped_filepath("dataset_results", "json")
        elif not os.path.isabs(filepath):
            # If relative path, put it in dataset directory
            filepath = self.directory_manager.get_filepath(filepath)
        
        try:
            results_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_name': self.model_manager.model_name,
                    'total_processed': self.total_processed,
                    'statistics': self.get_statistics(),
                    'dataset_directory': self.directory_manager.dataset_dir
                },
                'results': [result.to_dict() for result in self.generation_results]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Results saved to {filepath}")
            ConsoleOutput.success(f"JSON results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                DatasetError, f"Failed to save results: {str(e)}", e
            )
    
    def save_dataframe(self, filepath: str = None) -> str:
        """Save generation results as DataFrame in Parquet format"""
        # Setup directory
        self.directory_manager.setup_directory()
        
        if filepath is None:
            filepath = self.directory_manager.get_timestamped_filepath("mbpp_dataset", "parquet")
        elif not os.path.isabs(filepath):
            # If relative path, put it in dataset directory
            filepath = self.directory_manager.get_filepath(filepath)
        
        try:
            # Convert results to DataFrame-optimized format
            df_rows = [result.to_dataframe_row() for result in self.generation_results]
            df = pd.DataFrame(df_rows)
            
            # Add metadata columns
            df['dataset_created'] = datetime.now().isoformat()
            df['model_name'] = self.model_manager.model_name
            df['record_index'] = range(len(df))
            
            # Save as Parquet (efficient binary format)
            df.to_parquet(filepath, index=False)
            
            # Also save metadata separately
            metadata = {
                'creation_timestamp': datetime.now().isoformat(),
                'model_name': self.model_manager.model_name,
                'total_records': len(df),
                'columns': list(df.columns),
                'statistics': self.get_statistics(),
                'dataframe_file': os.path.basename(filepath),
                'dataset_directory': self.directory_manager.dataset_dir
            }
            
            metadata_file = filepath.replace('.parquet', '_metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logging.info(f"DataFrame saved to {filepath}")
            logging.info(f"Metadata saved to {metadata_file}")
            ConsoleOutput.success(f"Parquet dataset saved to: {filepath}")
            ConsoleOutput.info(f"Metadata saved to: {os.path.basename(metadata_file)}")
            
            # Display DataFrame info
            self._display_dataframe_info(df)
            
            return filepath
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                DatasetError, f"Failed to save DataFrame: {str(e)}", e
            )
    
    def save_both_formats(self, base_filepath: str = None) -> tuple[str, str]:
        """Save results in both JSON and Parquet formats"""
        # Setup directory
        self.directory_manager.setup_directory()
        
        if base_filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filepath = f"mbpp_dataset_{timestamp}"
        
        # Save both formats
        json_file = self.save_results(f"{base_filepath}.json")
        parquet_file = self.save_dataframe(f"{base_filepath}.parquet")
        
        ConsoleOutput.success("Dataset saved in both formats:")
        ConsoleOutput.info(f"  📄 JSON: {os.path.basename(json_file)}")
        ConsoleOutput.info(f"  📊 Parquet: {os.path.basename(parquet_file)}")
        ConsoleOutput.info(f"  📁 Directory: {self.directory_manager.dataset_dir}")
        
        return json_file, parquet_file
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get current results as a pandas DataFrame"""
        df_rows = [result.to_dataframe_row() for result in self.generation_results]
        df = pd.DataFrame(df_rows)
        
        # Add metadata columns
        df['dataset_created'] = datetime.now().isoformat()
        df['model_name'] = self.model_manager.model_name
        df['record_index'] = range(len(df))
        
        return df
    
    def analyze_results(self) -> dict[str, Any]:
        """Analyze current results using pandas operations"""
        if not self.generation_results:
            return {'error': 'No results to analyze'}
        
        df = self.get_dataframe()
        
        analysis = {
            'overview': {
                'total_records': len(df),
                'correct_solutions': df['is_correct'].sum(),
                'success_rate': df['is_correct'].mean() * 100,
                'avg_generation_time': df['generation_time'].mean(),
                'avg_code_length': df['code_length'].mean()
            },
            'error_analysis': df['error_type'].value_counts().to_dict(),
            'test_performance': {
                'avg_tests_passed': df['passed_tests'].mean(),
                'perfect_scores': (df['passed_tests'] == df['total_tests']).sum(),
                'zero_scores': (df['passed_tests'] == 0).sum()
            },
            'timing_stats': {
                'min_time': df['generation_time'].min(),
                'max_time': df['generation_time'].max(),
                'median_time': df['generation_time'].median()
            }
        }
        
        return analysis
    
    def _display_dataframe_info(self, df: pd.DataFrame):
        """Display helpful DataFrame information"""
        print(f"\n{'='*50}")
        print("DATAFRAME SUMMARY")
        print(f"{'='*50}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nData types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        print(f"\nSuccess Rate Analysis:")
        print(f"  Correct: {df['is_correct'].sum()}")
        print(f"  Incorrect: {(~df['is_correct']).sum()}")
        print(f"  Success Rate: {df['is_correct'].mean()*100:.1f}%")
        
        if 'error_type' in df.columns:
            print(f"\nError Type Distribution:")
            error_counts = df['error_type'].value_counts()
            for error_type, count in error_counts.items():
                print(f"  {error_type}: {count}")
        
        print(f"{'='*50}")
    
    def _validate_prerequisites(self):
        """Ensure all components are ready"""
        if not self.model_manager.is_loaded():
            raise DatasetError("Model not loaded. Call model_manager.load_model() first.")
        
        if not self.dataset_manager.is_loaded():
            raise DatasetError("Dataset not loaded. Call dataset_manager.load_dataset() first.")
    
    def _validate_range(self, start_idx: int, end_idx: int):
        """Validate processing range"""
        dataset_size = self.dataset_manager.get_size()
        
        if start_idx < 0 or start_idx >= dataset_size:
            raise ValueError(f"start_idx {start_idx} out of range [0, {dataset_size-1}]")
        if end_idx < start_idx or end_idx >= dataset_size:
            raise ValueError(f"end_idx {end_idx} out of range [{start_idx}, {dataset_size-1}]")
    
    def _reset_statistics(self):
        """Reset processing statistics"""
        self.generation_results = []
        self.total_processed = 0
        self.correct_solutions = 0
        self.incorrect_solutions = 0
    
    def _process_record_batch(self, start_idx: int, end_idx: int) -> list[GenerationResult]:
        """Process batch of records with progress tracking"""
        results = []
        
        for idx in tqdm(range(start_idx, end_idx + 1),
                       desc="Generating solutions",
                       unit="problem"):
            try:
                result = self.process_single_record(idx)
                results.append(result)
                self.generation_results.append(result)
                
            except Exception as e:
                logging.error(f"Failed to process record {idx}: {str(e)}")
                ConsoleOutput.error(f"Failed to process record {idx}")
                # Continue with next record
                continue
        
        return results
    
    def _generate_code_safely(self, prompt: str, task_id: str) -> str:
        """Generate code with error handling"""
        try:
            generated_code = self.model_manager.generate_code(
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                stream=self.stream_output
            )
            
            if not generated_code.strip():
                raise ModelError("Generated empty code")
            
            return generated_code
            
        except Exception as e:
            error_msg = f"Code generation failed for task {task_id}: {str(e)}"
            logging.error(error_msg)
            raise ModelError(error_msg) from e
    
    def _test_generated_code(self, generated_code: str, record: dict, task_id: str) -> TestResult:
        """Test generated code against MBPP test cases"""
        try:
            test_cases = record['test_list']
            test_result = TestExecutor.run_code_tests(
                code=generated_code,
                test_cases=test_cases,
                task_id=task_id
            )
            
            return test_result
            
        except Exception as e:
            error_msg = f"Testing failed for task {task_id}: {str(e)}"
            logging.error(error_msg)
            # Return failed test result
            return TestResult(passed=0, total=len(record['test_list']), 
                            errors=[str(e)])
    
    def _classify_solution(self, test_result: TestResult, generated_code: str) -> tuple[bool, Optional[str]]:
        """
        Classify solution according to methodology:
        - Correct: passes all 3 test cases on first attempt (pass@1)
        - Incorrect: fails any test case, compilation errors, or runtime exceptions
        """
        # Check if passes all tests (pass@1 criterion)
        is_correct = (test_result.passed == test_result.total and test_result.total > 0)
        
        # Determine error type if incorrect
        error_type = None
        if not is_correct:
            if test_result.total == 0:
                error_type = "compilation_error"
            elif test_result.passed == 0:
                error_type = "all_tests_failed"
            elif test_result.passed < test_result.total:
                error_type = "partial_test_failure"
            
            # Check for runtime errors in error messages
            if any("Error" in error or "Exception" in error for error in test_result.errors):
                error_type = "runtime_error"
        
        return is_correct, error_type
    
    def _update_statistics(self, result: GenerationResult):
        """Update processing statistics"""
        self.total_processed += 1
        if result.is_correct:
            self.correct_solutions += 1
        else:
            self.incorrect_solutions += 1
    
    def _log_single_result(self, result: GenerationResult, idx: int):
        """Log result for single record"""
        status = "CORRECT" if result.is_correct else "INCORRECT"
        test_summary = f"{result.test_result.passed}/{result.test_result.total}"
        
        log_msg = (f"Record {idx} ({result.task_id}): {status} "
                  f"[Tests: {test_summary}, Time: {result.generation_time:.2f}s]")
        
        if result.error_type:
            log_msg += f" [Error: {result.error_type}]"
        
        logging.info(log_msg)
        
        if self.stream_output:
            color = "✓" if result.is_correct else "✗"
            print(f"\n{color} {status}: {test_summary} tests passed")
            if result.error_type:
                print(f"  Error type: {result.error_type}")
    
    def _log_final_statistics(self):
        """Log final dataset building statistics"""
        stats = self.get_statistics()
        
        summary_msg = (f"Dataset building complete: {stats['total_processed']} records processed, "
                      f"{stats['correct_solutions']} correct ({stats['correct_rate']:.1f}%), "
                      f"{stats['incorrect_solutions']} incorrect")
        
        logging.info(summary_msg)
        ConsoleOutput.success(summary_msg)

# ============================================================================
# Main Orchestration Classes
# ============================================================================

class MBPPTester:
    """Main orchestrator for MBPP testing workflow"""
    
    def __init__(self, debug: bool = False, log_dir: str = DEFAULT_LOG_DIR):
        self.debug = debug
        self.log_dir = log_dir
        
        # Initialize core components
        self.logging_config = LoggingConfiguration(debug, log_dir)
        self.dataset_manager = EnhancedDatasetManager()  # Use enhanced version
        
        # Results tracking
        self.total_tests = 0
        self.passed_tests = 0
        self.record_results: list[TestResult] = []
    
    @property
    def log_file(self) -> Optional[str]:
        """Get current log file path"""
        return self.logging_config.log_file
    
    def setup_logging(self) -> str:
        """Configure logging system"""
        return self.logging_config.setup_logging()
    
    def ensure_dataset_ready(self):
        """Ensure dataset is loaded and ready"""
        if not self.dataset_manager.is_loaded():
            self.dataset_manager.load_dataset()
    
    def test_single_record(self, idx: int) -> TestResult:
        """Test single MBPP record by index"""
        try:
            record = self.dataset_manager.get_record(idx)
            
            logging.info(f"Processing record {idx} (Task ID: {record['task_id']})")
            
            result = TestExecutor.run_record_tests(record)
            self._update_overall_stats(result)
            
            logging.info("-" * 40)
            return result
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                TestExecutionError, f"Failed to test record {idx}: {str(e)}", e
            )
    
    def test_range(self, start_idx: int = 0, end_idx: int = 3) -> dict[str, Any]:
        """Test range of MBPP records"""
        try:
            # Ensure prerequisites
            self._ensure_prerequisites()
            
            # Validate range
            validated_end = self._validate_test_range(start_idx, end_idx)
            
            # Execute tests
            self._run_test_sequence(start_idx, validated_end)
            
            # Generate final summary
            return self._create_summary()
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                TestExecutionError, f"Test range execution failed: {str(e)}", e
            )
    
    def get_summary(self) -> dict[str, Any]:
        """Get current test results summary"""
        return {
            'passed': self.passed_tests,
            'total': self.total_tests,
            'success_rate': self._calculate_success_rate(),
            'records_tested': len(self.record_results),
            'log_file': self.log_file
        }
    
    def get_detailed_results(self) -> list[TestResult]:
        """Get detailed results for each tested record"""
        return self.record_results.copy()
    
    def _ensure_prerequisites(self):
        """Ensure all prerequisites are met"""
        if not self.logging_config.is_configured:
            self.setup_logging()
        self.ensure_dataset_ready()
    
    def _validate_test_range(self, start_idx: int, end_idx: int) -> int:
        """Validate and adjust test range"""
        dataset_size = self.dataset_manager.get_size()
        
        if start_idx < 0:
            raise ValueError(f"start_idx must be >= 0, got {start_idx}")
        if start_idx >= dataset_size:
            raise ValueError(f"start_idx {start_idx} >= dataset size {dataset_size}")
        
        validated_end = min(end_idx, dataset_size - 1)
        logging.info(f"Testing records {start_idx} to {validated_end}")
        return validated_end
    
    def _run_test_sequence(self, start_idx: int, end_idx: int):
        """Execute the test sequence with progress tracking"""
        self._reset_statistics()
        
        for idx in tqdm(range(start_idx, end_idx + 1),
                        desc="Testing MBPP records",
                        unit="record"):
            self.test_single_record(idx)
    
    def _reset_statistics(self):
        """Reset test statistics for new run"""
        self.total_tests = 0
        self.passed_tests = 0
        self.record_results = []
    
    def _update_overall_stats(self, result: TestResult):
        """Update overall statistics with new result"""
        self.total_tests += result.total
        self.passed_tests += result.passed
        self.record_results.append(result)
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        return (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0.0
    
    def _create_summary(self) -> dict[str, Any]:
        """Create and log final summary"""
        summary = self.get_summary()
        
        logging.info(
            f"FINAL SUMMARY: {summary['passed']}/{summary['total']} tests passed "
            f"({summary['success_rate']:.1f}%)"
        )
        
        if self.log_file:
            ConsoleOutput.info(f"Results logged to: {self.log_file}")
        
        return summary

class EnhancedMBPPTester(MBPPTester):
    """Extended MBPPTester with dataset building capabilities"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, debug: bool = False, 
                 log_dir: str = DEFAULT_LOG_DIR, dataset_dir: str = DEFAULT_DATASET_DIR):
        super().__init__(debug, log_dir)
        self.model_manager = ModelManager(model_name)
        self.dataset_dir = dataset_dir
        self.dataset_builder = None
    
    def setup_components(self):
        """Setup all required components"""
        # Setup logging
        if not self.logging_config.is_configured:
            self.setup_logging()
        
        # Load dataset
        self.ensure_dataset_ready()
        
        # Load model
        if not self.model_manager.is_loaded():
            ConsoleOutput.info("Loading model for dataset building...")
            self.model_manager.load_model()
            ConsoleOutput.success("Model loaded successfully!")
        
        # Initialize dataset builder with custom directory
        self.dataset_builder = DatasetBuilder(
            model_manager=self.model_manager,
            dataset_manager=self.dataset_manager,
            max_new_tokens=200,
            stream_output=False,
            dataset_dir=self.dataset_dir
        )
    
    def build_dataset_mvp_with_cleanup(self, start_idx: int = 0, end_idx: int = 2, 
                                     save_format: str = "both") -> dict[str, Any]:
        """Build MVP dataset with automatic cleanup"""
        
        # Perform cleanup first
        auto_cleanup()
        
        # Call the existing method
        return self.build_dataset_mvp(start_idx, end_idx, save_format)
    
    def build_dataset_mvp(self, start_idx: int = 0, end_idx: int = 2, 
                         save_format: str = "both") -> dict[str, Any]:
        """
        Build MVP dataset with 3 records (or specified range)
        
        Args:
            start_idx: Starting record index
            end_idx: Ending record index (inclusive)
            save_format: "json", "parquet", or "both"
            
        Returns:
            dict: Summary of dataset building results
        """
        try:
            # Setup components
            self.setup_components()
            
            ConsoleOutput.info(f"Building MVP dataset for records {start_idx} to {end_idx}")
            
            # Build dataset
            results = self.dataset_builder.build_dataset(start_idx, end_idx)
            
            # Save results in requested format
            saved_files = {}
            if save_format in ["json", "both"]:
                saved_files['json'] = self.dataset_builder.save_results()
            
            if save_format in ["parquet", "both"]:
                saved_files['parquet'] = self.dataset_builder.save_dataframe()
            
            if save_format == "both":
                ConsoleOutput.success("Dataset saved in both JSON and Parquet formats!")
            
            # Create summary
            stats = self.dataset_builder.get_statistics()
            analysis = self.dataset_builder.analyze_results()
            
            summary = {
                **stats,
                'saved_files': saved_files,
                'log_file': self.log_file,
                'results': results,
                'analysis': analysis
            }
            
            # Display summary
            self._display_mvp_summary(summary)
            
            return summary
            
        except Exception as e:
            ErrorContext.handle_and_raise(
                DatasetError, f"MVP dataset building failed: {str(e)}", e
            )
    
    def _display_mvp_summary(self, summary: dict):
        """Display formatted summary of MVP results"""
        print(f"\n{'='*60}")
        print("MVP DATASET BUILDING SUMMARY")
        print(f"{'='*60}")
        print(f"Records processed: {summary['total_processed']}")
        print(f"Correct solutions: {summary['correct_solutions']}")
        print(f"Incorrect solutions: {summary['incorrect_solutions']}")
        print(f"Success rate: {summary['correct_rate']:.1f}%")
        
        if 'saved_files' in summary:
            print(f"\nSaved files:")
            for format_type, filepath in summary['saved_files'].items():
                filename = os.path.basename(filepath)
                print(f"  📄 {format_type.upper()}: {filename}")
            
            # Show directory path
            if summary['saved_files']:
                first_file = list(summary['saved_files'].values())[0]
                directory = os.path.dirname(first_file)
                print(f"  📁 Directory: {directory}")
        
        print(f"\nLogs saved to: {summary['log_file']}")
        
        # Display analysis if available
        if 'analysis' in summary and 'overview' in summary['analysis']:
            analysis = summary['analysis']
            print(f"\nDetailed Analysis:")
            print(f"  Avg generation time: {analysis['overview']['avg_generation_time']:.2f}s")
            print(f"  Avg code length: {analysis['overview']['avg_code_length']:.0f} chars")
            
            if 'error_analysis' in analysis:
                print(f"  Error breakdown: {analysis['error_analysis']}")
        
        print(f"{'='*60}")

# ============================================================================
# Data Analysis Utilities
# ============================================================================

def load_dataset_dataframe(filepath: str) -> pd.DataFrame:
    """Load MBPP dataset from Parquet file"""
    try:
        # Handle both absolute and relative paths
        if not os.path.isabs(filepath) and not os.path.exists(filepath):
            # Try looking in default dataset directory
            full_path = os.path.join(DEFAULT_DATASET_DIR, filepath)
            if os.path.exists(full_path):
                filepath = full_path
        
        df = pd.read_parquet(filepath)
        ConsoleOutput.success(f"Loaded dataset: {df.shape[0]} records from {os.path.basename(filepath)}")
        return df
    except Exception as e:
        ErrorContext.handle_and_raise(
            DatasetError, f"Failed to load dataset from {filepath}: {str(e)}", e
        )

def analyze_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Perform comprehensive analysis of MBPP dataset"""
    analysis = {
        'overview': {
            'total_records': len(df),
            'correct_solutions': df['is_correct'].sum() if 'is_correct' in df else 0,
            'success_rate': df['is_correct'].mean() * 100 if 'is_correct' in df else 0,
        }
    }
    
    # Add detailed analysis if columns exist
    if 'generation_time' in df.columns:
        analysis['timing'] = {
            'avg_time': df['generation_time'].mean(),
            'min_time': df['generation_time'].min(),
            'max_time': df['generation_time'].max(),
            'median_time': df['generation_time'].median()
        }
    
    if 'error_type' in df.columns:
        analysis['errors'] = df['error_type'].value_counts().to_dict()
    
    if 'code_length' in df.columns:
        analysis['code_stats'] = {
            'avg_length': df['code_length'].mean(),
            'min_length': df['code_length'].min(),
            'max_length': df['code_length'].max()
        }
    
    return analysis

def list_available_datasets(dataset_dir: str = DEFAULT_DATASET_DIR) -> dict[str, list[str]]:
    """List all available datasets in the dataset directory"""
    manager = DatasetDirectoryManager(dataset_dir)
    datasets = manager.list_datasets()
    
    if any(datasets.values()):
        print(f"\n📁 Available datasets in '{dataset_dir}':")
        print("="*50)
        
        if datasets['parquet']:
            print("📊 Parquet datasets:")
            for filename in sorted(datasets['parquet']):
                print(f"  • {filename}")
        
        if datasets['json']:
            print("\n📄 JSON datasets:")
            for filename in sorted(datasets['json']):
                if not filename.endswith('_metadata.json'):  # Skip metadata files
                    print(f"  • {filename}")
        
        if datasets['metadata']:
            print(f"\n📋 Metadata files: {len(datasets['metadata'])}")
        
        print("="*50)
    else:
        print(f"📁 No datasets found in '{dataset_dir}'")
    
    return datasets

def get_dataset_info(filepath: str) -> dict[str, Any]:
    """Get comprehensive information about a dataset file"""
    try:
        # Handle both absolute and relative paths
        if not os.path.isabs(filepath):
            full_path = os.path.join(DEFAULT_DATASET_DIR, filepath)
            if os.path.exists(full_path):
                filepath = full_path
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        info = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'size_mb': os.path.getsize(filepath) / (1024 * 1024),
            'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
        }
        
        # Try to load metadata if it exists
        metadata_file = filepath.replace('.parquet', '_metadata.json').replace('.json', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                info['metadata'] = metadata
        
        # If it's a parquet file, get basic DataFrame info
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
            info['records'] = len(df)
            info['columns'] = list(df.columns)
            if 'is_correct' in df.columns:
                info['success_rate'] = df['is_correct'].mean() * 100
        
        return info
        
    except Exception as e:
        ErrorContext.handle_and_raise(
            DatasetError, f"Failed to get dataset info: {str(e)}", e
        )

# ============================================================================
# Convenience Functions
# ============================================================================

def test_mbpp_range(start_idx: int = 0, end_idx: int = 3, debug: bool = False) -> str:
    """
    Convenience function to test MBPP range with automatic setup
    
    Returns:
        str: Path to log file
    """
    tester = MBPPTester(debug=debug)
    summary = tester.test_range(start_idx, end_idx)
    return summary['log_file']

def create_prompt_from_record(record: dict) -> str:
    """
    Convenience function to create prompt from MBPP record
    
    Args:
        record: MBPP dataset record
        
    Returns:
        str: Standardized prompt template
    """
    builder = PromptTemplateBuilder()
    return builder.build_prompt(record)

def preview_mbpp_prompt(idx: int = 0) -> str:
    """
    Convenience function to preview prompt for MBPP record
    
    Args:
        idx: Index of record to preview (default: 0)
        
    Returns:
        str: Preview of prompt template
    """
    dataset_manager = EnhancedDatasetManager()
    dataset_manager.load_dataset()
    return dataset_manager.preview_prompt_template(idx)

def build_mvp_dataset(start_idx: int = 0, end_idx: int = 2, 
                     model_name: str = DEFAULT_MODEL_NAME,
                     save_format: str = "both") -> Union[str, tuple[str, str]]:
    """
    Convenience function to build MVP dataset
    
    Args:
        start_idx: Starting record index
        end_idx: Ending record index  
        model_name: Model to use for generation
        save_format: "json", "parquet", or "both"
    
    Returns:
        str or tuple: Path(s) to saved results file(s)
    """
    tester = EnhancedMBPPTester(model_name=model_name, debug=False)
    summary = tester.build_dataset_mvp_with_cleanup(start_idx, end_idx, save_format=save_format)
    
    if save_format == "both":
        return summary['saved_files']['json'], summary['saved_files']['parquet']
    elif save_format == "parquet":
        return summary['saved_files']['parquet']
    else:
        return summary['saved_files']['json']

def quick_analyze(filepath: str) -> dict[str, Any]:
    """Quick analysis of saved MBPP dataset"""
    df = load_dataset_dataframe(filepath)
    return analyze_dataset(df)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    try:
        # Perform cleanup before starting
        auto_cleanup()
        
        # Initialize logging
        logging_config = LoggingConfiguration(debug=False, log_dir="mbpp_logs")
        log_file = logging_config.setup_logging()
        ConsoleOutput.info(f"Logging initialized: {log_file}")
        
        # Test prompt template functionality
        ConsoleOutput.info("Testing prompt template functionality...")
        
        dataset_manager = EnhancedDatasetManager()
        dataset_manager.load_dataset()
        
        # Show sample prompt
        prompt = dataset_manager.get_prompt_template(0)
        ConsoleOutput.success("Sample prompt generated!")
        print(f"\nSample prompt:\n{'-'*50}")
        print(prompt)
        print(f"{'-'*50}")
        
        # Test model loading
        ConsoleOutput.info(f"Testing {DEFAULT_MODEL_NAME} model loading...")
        model_manager = ModelManager(DEFAULT_MODEL_NAME)
        
        try:
            ConsoleOutput.info("Loading model (this may take several minutes)...")
            model_manager.load_model()
            ConsoleOutput.success(f"Model ready: {model_manager.get_device_info()}")
            
            # Test model functionality
            if model_manager.test_model_functionality():
                ConsoleOutput.success("Model is working correctly!")
            else:
                ConsoleOutput.error("Model functionality test failed!")
                
        except ModelError as e:
            ConsoleOutput.error(f"Model error: {e}")
        except Exception as e:
            ConsoleOutput.error(f"Unexpected error: {e}")
            logging.error(f"Unexpected error: {e}", exc_info=True)
        
        # Test MBPP examples with ground truth
        print("\n" + "="*50)
        ConsoleOutput.info("Running MBPP ground truth tests...")
        
        tester = MBPPTester(debug=False)
        summary = tester.test_range(0, 2)
        
        ConsoleOutput.success(f"Ground truth testing complete: {summary}")
        
        # NEW: Test MVP Dataset Building with DataFrame storage and cleanup
        print("\n" + "="*50)
        ConsoleOutput.info("Building MVP Dataset with DataFrame Storage and Cleanup...")
        
        enhanced_tester = EnhancedMBPPTester(model_name=DEFAULT_MODEL_NAME, debug=False)
        mvp_summary = enhanced_tester.build_dataset_mvp_with_cleanup(
            start_idx=0, 
            end_idx=2, 
            save_format="both"  # Save in both JSON and Parquet formats
        )
        
        ConsoleOutput.success("MVP dataset building with cleanup completed!")
        
        # Show some results
        if mvp_summary['results']:
            print(f"\nSample generated code from first record:")
            print(f"{'='*50}")
            first_result = mvp_summary['results'][0]
            print(f"Task: {first_result.task_id}")
            print(f"Correct: {first_result.is_correct}")
            print(f"Generated code preview:\n{first_result.generated_code[:200]}...")
            print(f"{'='*50}")
        
        # Demonstrate DataFrame analysis
        if 'parquet' in mvp_summary.get('saved_files', {}):
            print(f"\nDemonstrating DataFrame analysis...")
            parquet_file = mvp_summary['saved_files']['parquet']
            analysis = quick_analyze(parquet_file)
            print(f"Analysis results: {analysis}")
        
        ConsoleOutput.info(f"Full logs: {log_file}")
        
    except Exception as e:
        ConsoleOutput.error(f"Application failed: {e}")
        if 'logging' in globals():
            logging.error(f"Application failed: {e}", exc_info=True)