"""
Regression tests for interp/data_processing.py

These tests ensure the MBPP framework continues working correctly
as the codebase evolves. Heavy operations are mocked for speed.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import torch
import logging
import os
from datetime import datetime
import tempfile
import shutil

# Import all classes and functions from data_processing
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interp.data_processing import (
    # Exceptions
    MBPPFrameworkError, ModelError, DatasetError, TestExecutionError, 
    LoggingConfigurationError,
    
    # Utility Classes
    DeviceManager, ConsoleOutput, ErrorContext,
    
    # Core Configuration
    LoggingConfiguration,
    
    # Model Management
    ModelLoader, CodeGenerator, ModelManager,
    
    # Data Management
    TestResult, DatasetManager,
    
    # Test Execution
    TestExecutor,
    
    # Main Orchestration
    MBPPTester,
    
    # Convenience Functions
    test_mbpp_range
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_torch_devices():
    """Mock torch device availability"""
    with patch('torch.backends.mps.is_available') as mock_mps, \
         patch('torch.cuda.is_available') as mock_cuda:
        yield mock_mps, mock_cuda


@pytest.fixture
def mock_model_loading():
    """Mock model and tokenizer loading"""
    with patch('interp.data_processing.AutoModelForCausalLM') as mock_model_class, \
         patch('interp.data_processing.AutoTokenizer') as mock_tokenizer_class:
        
        # Create mock instances
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Configure the mock classes to return the mock instances
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Set some attributes on the mocks
        mock_tokenizer.eos_token_id = 1
        
        yield mock_model, mock_tokenizer, mock_model_class, mock_tokenizer_class


@pytest.fixture
def mock_dataset():
    """Mock MBPP dataset"""
    with patch('interp.data_processing.load_dataset') as mock_load:
        mock_dataset = {
            'test': [
                {
                    'task_id': 1,
                    'text': 'Write a function to add two numbers',
                    'code': 'def add(a, b):\n    return a + b',
                    'test_list': [
                        'assert add(1, 2) == 3',
                        'assert add(-1, 1) == 0',
                        'assert add(0, 0) == 0'
                    ]
                },
                {
                    'task_id': 2,
                    'text': 'Write a function to multiply two numbers',
                    'code': 'def multiply(a, b):\n    return a * b',
                    'test_list': [
                        'assert multiply(2, 3) == 6',
                        'assert multiply(-2, 3) == -6'
                    ]
                }
            ]
        }
        mock_load.return_value = mock_dataset
        yield mock_dataset


# ============================================================================
# Test Exception Hierarchy
# ============================================================================

class TestExceptionHierarchy:
    """Test custom exception classes"""
    
    def test_base_exception(self):
        """Test MBPPFrameworkError is the base exception"""
        assert issubclass(ModelError, MBPPFrameworkError)
        assert issubclass(DatasetError, MBPPFrameworkError)
        assert issubclass(TestExecutionError, MBPPFrameworkError)
        assert issubclass(LoggingConfigurationError, MBPPFrameworkError)
    
    def test_exception_messages(self):
        """Test exceptions can be raised with messages"""
        with pytest.raises(ModelError, match="test model error"):
            raise ModelError("test model error")
        
        with pytest.raises(DatasetError, match="test dataset error"):
            raise DatasetError("test dataset error")


# ============================================================================
# Test Utility Classes
# ============================================================================

class TestDeviceManager:
    """Test DeviceManager functionality"""
    
    def test_detect_mps_device(self, mock_torch_devices):
        """Test MPS device detection on Apple Silicon"""
        mock_mps, mock_cuda = mock_torch_devices
        mock_mps.return_value = True
        mock_cuda.return_value = False
        
        device = DeviceManager.detect_best_device()
        assert device == "mps"
    
    def test_detect_cuda_device(self, mock_torch_devices):
        """Test CUDA device detection"""
        mock_mps, mock_cuda = mock_torch_devices
        mock_mps.return_value = False
        mock_cuda.return_value = True
        
        device = DeviceManager.detect_best_device()
        assert device == "cuda"
    
    def test_detect_cpu_device(self, mock_torch_devices):
        """Test CPU fallback when no GPU available"""
        mock_mps, mock_cuda = mock_torch_devices
        mock_mps.return_value = False
        mock_cuda.return_value = False
        
        device = DeviceManager.detect_best_device()
        assert device == "cpu"
    
    def test_get_torch_device(self):
        """Test torch.device creation"""
        device = DeviceManager.get_torch_device("cpu")
        assert isinstance(device, torch.device)
        assert device.type == "cpu"
    
    def test_get_optimal_dtype(self):
        """Test dtype selection based on device"""
        assert DeviceManager.get_optimal_dtype("cuda") == torch.float16
        assert DeviceManager.get_optimal_dtype("cpu") == torch.float32
        assert DeviceManager.get_optimal_dtype("mps") == torch.float32


class TestConsoleOutput:
    """Test ConsoleOutput formatting"""
    
    def test_output_methods(self, capsys):
        """Test all console output methods"""
        ConsoleOutput.success("test success")
        ConsoleOutput.error("test error")
        ConsoleOutput.info("test info")
        ConsoleOutput.loading("test loading")
        ConsoleOutput.working("test working")
        ConsoleOutput.generating("test generating")
        
        captured = capsys.readouterr()
        assert "✓ test success" in captured.out
        assert "✗ test error" in captured.out
        assert "ℹ️  test info" in captured.out
        assert "📥 test loading" in captured.out
        assert "🔧 test working" in captured.out
        assert "🤖 test generating" in captured.out


class TestErrorContext:
    """Test ErrorContext error handling"""
    
    def test_handle_and_raise_without_original(self):
        """Test raising exception without original error"""
        with pytest.raises(ModelError, match="test error"):
            ErrorContext.handle_and_raise(ModelError, "test error")
    
    def test_handle_and_raise_with_original(self):
        """Test raising exception with original error"""
        original = ValueError("original error")
        with pytest.raises(ModelError, match="test error"):
            try:
                ErrorContext.handle_and_raise(ModelError, "test error", original)
            except ModelError as e:
                assert e.__cause__ == original
                raise
    
    @patch('interp.data_processing.logging.error')
    def test_logging_behavior(self, mock_log):
        """Test that errors are logged"""
        with pytest.raises(ModelError):
            ErrorContext.handle_and_raise(ModelError, "test error", log_error=True)
        
        mock_log.assert_called_once_with("test error")


# ============================================================================
# Test Core Configuration Classes
# ============================================================================

class TestLoggingConfiguration:
    """Test LoggingConfiguration functionality"""
    
    def test_initialization(self):
        """Test LoggingConfiguration initialization"""
        config = LoggingConfiguration(debug=True, log_dir="test_logs")
        assert config.debug is True
        assert config.log_dir == "test_logs"
        assert config.log_file is None
        assert not config._is_configured
    
    def test_setup_logging(self, temp_log_dir):
        """Test logging setup process"""
        config = LoggingConfiguration(debug=False, log_dir=temp_log_dir)
        log_file = config.setup_logging()
        
        assert config.is_configured
        assert log_file is not None
        assert os.path.exists(log_file)
        assert temp_log_dir in log_file
        assert "mbpp_test_" in log_file
        assert ".log" in log_file
    
    def test_setup_logging_idempotent(self, temp_log_dir):
        """Test that setup_logging is idempotent"""
        config = LoggingConfiguration(debug=False, log_dir=temp_log_dir)
        log_file1 = config.setup_logging()
        log_file2 = config.setup_logging()
        
        assert log_file1 == log_file2
        assert config.is_configured
    
    def test_log_directory_creation(self, temp_log_dir):
        """Test that log directory is created if it doesn't exist"""
        non_existent_dir = os.path.join(temp_log_dir, "new_logs")
        config = LoggingConfiguration(log_dir=non_existent_dir)
        
        config.setup_logging()
        assert os.path.exists(non_existent_dir)
    
    def test_debug_logging_level(self, temp_log_dir):
        """Test debug logging level setting"""
        config = LoggingConfiguration(debug=True, log_dir=temp_log_dir)
        config.setup_logging()
        
        # The root logger should be at DEBUG level
        assert logging.getLogger().level == logging.DEBUG


# ============================================================================
# Test Model Management Classes
# ============================================================================

class TestModelLoader:
    """Test ModelLoader functionality"""
    
    def test_initialization(self):
        """Test ModelLoader initialization"""
        loader = ModelLoader("test-model", "cpu")
        assert loader.model_name == "test-model"
        assert loader.device_str == "cpu"
        assert isinstance(loader.device, torch.device)
        assert loader.dtype == torch.float32
    
    def test_load_tokenizer_success(self, mock_model_loading):
        """Test successful tokenizer loading"""
        mock_model, mock_tokenizer, _, mock_tokenizer_class = mock_model_loading
        loader = ModelLoader("test-model", "cpu")
        
        tokenizer = loader.load_tokenizer()
        
        assert tokenizer == mock_tokenizer
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
    
    def test_load_tokenizer_failure(self, mock_model_loading):
        """Test tokenizer loading failure"""
        _, _, _, mock_tokenizer_class = mock_model_loading
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Load failed")
        
        loader = ModelLoader("test-model", "cpu")
        
        with pytest.raises(ModelError, match="Tokenizer loading failed"):
            loader.load_tokenizer()
    
    @patch('torch.set_grad_enabled')
    def test_load_model_weights_cpu(self, mock_grad, mock_model_loading):
        """Test model loading on CPU"""
        mock_model, _, mock_model_class, _ = mock_model_loading
        loader = ModelLoader("test-model", "cpu")
        
        model = loader.load_model_weights()
        
        assert model == mock_model
        mock_grad.assert_called_once_with(False)
        mock_model_class.from_pretrained.assert_called_once()
        mock_model.to.assert_called_once()
    
    @patch('torch.set_grad_enabled')
    def test_load_model_weights_cuda(self, mock_grad, mock_model_loading):
        """Test model loading on CUDA"""
        mock_model, _, mock_model_class, _ = mock_model_loading
        loader = ModelLoader("test-model", "cuda")
        
        model = loader.load_model_weights()
        
        assert model == mock_model
        mock_grad.assert_called_once_with(False)
        # For CUDA, should use device_map='auto'
        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == 'auto'
    
    def test_configure_for_inference(self, mock_model_loading):
        """Test model configuration for inference"""
        mock_model, _, _, _ = mock_model_loading
        loader = ModelLoader("test-model", "cpu")
        
        configured_model = loader.configure_for_inference(mock_model)
        
        assert configured_model == mock_model
        mock_model.eval.assert_called_once()


class TestCodeGenerator:
    """Test CodeGenerator functionality"""
    
    def test_initialization(self, mock_model_loading):
        """Test CodeGenerator initialization"""
        mock_model, mock_tokenizer, _, _ = mock_model_loading
        generator = CodeGenerator(mock_model, mock_tokenizer, "cpu")
        
        assert generator.model == mock_model
        assert generator.tokenizer == mock_tokenizer
        assert generator.device_str == "cpu"
    
    @patch('torch.no_grad')
    def test_generate_without_streaming(self, mock_no_grad, mock_model_loading):
        """Test code generation without streaming"""
        mock_model, mock_tokenizer, _, _ = mock_model_loading
        
        # Set up tokenizer mock
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.decode.return_value = "prompt\ngenerated code"
        
        # Set up model mock
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        generator = CodeGenerator(mock_model, mock_tokenizer, "cpu")
        result = generator.generate("prompt", max_new_tokens=10, stream=False)
        
        assert result == "generated code"
        mock_model.generate.assert_called_once()
    
    def test_generate_with_cuda_device(self, mock_model_loading):
        """Test that inputs are moved to CUDA device"""
        mock_model, mock_tokenizer, _, _ = mock_model_loading
        
        # Create mock tensors with to() method
        mock_input_ids = MagicMock()
        mock_attention_mask = MagicMock()
        
        mock_tokenizer.return_value = {
            'input_ids': mock_input_ids,
            'attention_mask': mock_attention_mask
        }
        
        generator = CodeGenerator(mock_model, mock_tokenizer, "cuda")
        
        try:
            generator.generate("test prompt", stream=False)
        except:
            pass  # We expect this to fail, just testing device movement
        
        # Check that tensors were moved to device
        assert mock_input_ids.to.called or mock_attention_mask.to.called


class TestModelManager:
    """Test ModelManager orchestration"""
    
    def test_initialization(self):
        """Test ModelManager initialization"""
        manager = ModelManager("test-model", "cpu")
        assert manager.model_name == "test-model"
        assert manager.device_str == "cpu"
        assert manager.model is None
        assert manager.tokenizer is None
        assert not manager.is_loaded()
    
    @patch('interp.data_processing.ModelLoader')
    @patch('interp.data_processing.CodeGenerator')
    def test_load_model_success(self, mock_generator_class, mock_loader_class):
        """Test successful model loading"""
        # Set up mocks
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        mock_loader.load_tokenizer.return_value = mock_tokenizer
        mock_loader.load_model_weights.return_value = mock_model
        mock_loader.configure_for_inference.return_value = mock_model
        
        mock_loader_class.return_value = mock_loader
        
        # Test loading
        manager = ModelManager("test-model", "cpu")
        manager.load_model()
        
        assert manager.is_loaded()
        assert manager.model == mock_model
        assert manager.tokenizer == mock_tokenizer
        assert manager.generator is not None
    
    def test_load_model_idempotent(self, mock_model_loading):
        """Test that load_model is idempotent"""
        manager = ModelManager("test-model", "cpu")
        
        # Manually set as loaded
        manager.model = "dummy"
        manager.tokenizer = "dummy"
        manager.generator = "dummy"
        
        # Try to load again - should not do anything
        manager.load_model()
        
        assert manager.model == "dummy"  # Unchanged
    
    def test_generate_code_without_loading(self):
        """Test that generate_code fails if model not loaded"""
        manager = ModelManager("test-model", "cpu")
        
        with pytest.raises(ModelError, match="Model not loaded"):
            manager.generate_code("test prompt")
    
    @patch('torch.cuda.empty_cache')
    def test_unload_model_cuda(self, mock_empty_cache):
        """Test model unloading with CUDA cleanup"""
        manager = ModelManager("test-model", "cuda")
        manager.model = MagicMock()
        manager.tokenizer = MagicMock()
        manager.generator = MagicMock()
        
        manager.unload_model()
        
        assert manager.model is None
        assert manager.tokenizer is None
        assert manager.generator is None
        mock_empty_cache.assert_called_once()


# ============================================================================
# Test Data Management Classes
# ============================================================================

class TestTestResult:
    """Test TestResult dataclass"""
    
    def test_initialization(self):
        """Test TestResult initialization"""
        result = TestResult(passed=8, total=10, errors=["error1", "error2"])
        assert result.passed == 8
        assert result.total == 10
        assert len(result.errors) == 2
    
    def test_success_rate(self):
        """Test success rate calculation"""
        result = TestResult(passed=8, total=10, errors=[])
        assert result.success_rate == 80.0
        
        result = TestResult(passed=0, total=0, errors=[])
        assert result.success_rate == 0.0
    
    def test_failed_count(self):
        """Test failed count calculation"""
        result = TestResult(passed=8, total=10, errors=[])
        assert result.failed == 2


class TestDatasetManager:
    """Test DatasetManager functionality"""
    
    def test_initialization(self):
        """Test DatasetManager initialization"""
        manager = DatasetManager()
        assert manager.dataset is None
        assert manager.test_data is None
        assert not manager.is_loaded()
    
    def test_load_dataset_success(self, mock_dataset):
        """Test successful dataset loading"""
        manager = DatasetManager()
        manager.load_dataset()
        
        assert manager.is_loaded()
        assert manager.dataset is not None
        assert len(manager.test_data) == 2
    
    def test_load_dataset_idempotent(self, mock_dataset):
        """Test that load_dataset is idempotent"""
        manager = DatasetManager()
        manager.load_dataset()
        
        # Mark the initial dataset
        initial_dataset = manager.dataset
        
        # Load again
        manager.load_dataset()
        
        # Should be the same object (not reloaded)
        assert manager.dataset is initial_dataset
    
    def test_get_record_success(self, mock_dataset):
        """Test successful record retrieval"""
        manager = DatasetManager()
        manager.load_dataset()
        
        record = manager.get_record(0)
        assert record['task_id'] == 1
        assert 'add' in record['code']
    
    def test_get_record_without_loading(self):
        """Test record retrieval without loading dataset"""
        manager = DatasetManager()
        
        with pytest.raises(DatasetError, match="Dataset not loaded"):
            manager.get_record(0)
    
    def test_get_record_out_of_range(self, mock_dataset):
        """Test record retrieval with invalid index"""
        manager = DatasetManager()
        manager.load_dataset()
        
        with pytest.raises(DatasetError, match="out of range"):
            manager.get_record(100)
    
    def test_get_size(self, mock_dataset):
        """Test dataset size retrieval"""
        manager = DatasetManager()
        assert manager.get_size() == 0
        
        manager.load_dataset()
        assert manager.get_size() == 2


# ============================================================================
# Test Execution Classes
# ============================================================================

class TestTestExecutor:
    """Test TestExecutor functionality"""
    
    def test_run_single_test_success(self):
        """Test successful single test execution"""
        namespace = {'add': lambda a, b: a + b}
        success, error = TestExecutor.run_single_test("assert add(1, 2) == 3", namespace)
        
        assert success is True
        assert error is None
    
    def test_run_single_test_assertion_failure(self):
        """Test single test with assertion failure"""
        namespace = {'add': lambda a, b: a + b}
        success, error = TestExecutor.run_single_test("assert add(1, 2) == 4", namespace)
        
        assert success is False
        assert error == "Assertion failed"
    
    def test_run_single_test_exception(self):
        """Test single test with exception"""
        namespace = {}
        success, error = TestExecutor.run_single_test("undefined_function()", namespace)
        
        assert success is False
        assert "undefined_function" in error
    
    def test_run_code_tests_all_pass(self):
        """Test code execution with all tests passing"""
        code = "def add(a, b):\n    return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
        
        result = TestExecutor.run_code_tests(code, tests)
        
        assert result.passed == 2
        assert result.total == 2
        assert len(result.errors) == 0
    
    def test_run_code_tests_some_fail(self):
        """Test code execution with some tests failing"""
        code = "def add(a, b):\n    return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]
        
        result = TestExecutor.run_code_tests(code, tests)
        
        assert result.passed == 1
        assert result.total == 2
        assert len(result.errors) == 0  # Assertion failures don't go in errors
    
    def test_run_code_tests_code_error(self):
        """Test code execution with code compilation error"""
        code = "def add(a, b)\n    return a + b"  # Missing colon
        tests = ["assert add(1, 2) == 3"]
        
        result = TestExecutor.run_code_tests(code, tests)
        
        assert result.passed == 0
        assert result.total == 1
        assert len(result.errors) == 1
        assert "Code execution failed" in result.errors[0]
    
    def test_run_record_tests(self):
        """Test execution using MBPP record"""
        record = {
            'task_id': 1,
            'text': 'Write a function to add two numbers',
            'code': 'def add(a, b):\n    return a + b',
            'test_list': ['assert add(1, 2) == 3', 'assert add(-1, 1) == 0']
        }
        
        result = TestExecutor.run_record_tests(record)
        
        assert result.passed == 2
        assert result.total == 2


# ============================================================================
# Test Main Orchestration Class
# ============================================================================

class TestMBPPTester:
    """Test MBPPTester main orchestration"""
    
    def test_initialization(self):
        """Test MBPPTester initialization"""
        tester = MBPPTester(debug=True, log_dir="test_logs")
        assert tester.debug is True
        assert tester.log_dir == "test_logs"
        assert tester.total_tests == 0
        assert tester.passed_tests == 0
    
    def test_setup_logging(self, temp_log_dir):
        """Test logging setup through MBPPTester"""
        tester = MBPPTester(log_dir=temp_log_dir)
        log_file = tester.setup_logging()
        
        assert log_file is not None
        assert os.path.exists(log_file)
        assert tester.log_file == log_file
    
    @patch('interp.data_processing.TestExecutor.run_record_tests')
    def test_test_single_record(self, mock_run_tests, mock_dataset):
        """Test single record testing"""
        mock_result = TestResult(passed=2, total=3, errors=[])
        mock_run_tests.return_value = mock_result
        
        tester = MBPPTester()
        tester.ensure_dataset_ready()
        
        result = tester.test_single_record(0)
        
        assert result == mock_result
        assert tester.total_tests == 3
        assert tester.passed_tests == 2
    
    @patch('interp.data_processing.tqdm')
    @patch('interp.data_processing.TestExecutor.run_record_tests')
    def test_test_range(self, mock_run_tests, mock_tqdm, mock_dataset, temp_log_dir):
        """Test range testing functionality"""
        # Set up mocks
        mock_result = TestResult(passed=2, total=3, errors=[])
        mock_run_tests.return_value = mock_result
        mock_tqdm.side_effect = lambda x, **kwargs: x  # Pass through
        
        tester = MBPPTester(log_dir=temp_log_dir)
        summary = tester.test_range(0, 1)
        
        assert summary['passed'] == 4  # 2 records * 2 passed each
        assert summary['total'] == 6   # 2 records * 3 tests each
        assert summary['success_rate'] == pytest.approx(66.67, rel=0.01)
        assert summary['records_tested'] == 2
    
    def test_test_range_validation(self, mock_dataset):
        """Test range validation in test_range"""
        tester = MBPPTester()
        tester.ensure_dataset_ready()
        
        # Test negative start index
        with pytest.raises(ValueError, match="start_idx must be >= 0"):
            tester.test_range(-1, 5)
        
        # Test start index >= dataset size
        with pytest.raises(ValueError, match="start_idx .* >= dataset size"):
            tester.test_range(100, 105)
    
    def test_get_summary(self):
        """Test summary generation"""
        tester = MBPPTester()
        tester.total_tests = 10
        tester.passed_tests = 8
        
        summary = tester.get_summary()
        
        assert summary['passed'] == 8
        assert summary['total'] == 10
        assert summary['success_rate'] == 80.0
        assert summary['records_tested'] == 0


# ============================================================================
# Test Convenience Functions
# ============================================================================

@patch('interp.data_processing.MBPPTester')
def test_test_mbpp_range_convenience(mock_tester_class, temp_log_dir):
    """Test the convenience function test_mbpp_range"""
    # Set up mock
    mock_tester = MagicMock()
    mock_tester.test_range.return_value = {'log_file': '/path/to/log.log'}
    mock_tester_class.return_value = mock_tester
    
    log_file = test_mbpp_range(0, 5, debug=True)
    
    assert log_file == '/path/to/log.log'
    mock_tester_class.assert_called_once_with(debug=True)
    mock_tester.test_range.assert_called_once_with(0, 5)


# ============================================================================
# Integration Tests (with mocking)
# ============================================================================

class TestIntegration:
    """Integration tests with mocked external dependencies"""
    
    @patch('interp.data_processing.load_dataset')
    @patch('interp.data_processing.TestExecutor.run_record_tests')
    def test_full_workflow(self, mock_run_tests, mock_load_dataset, temp_log_dir):
        """Test complete workflow from initialization to results"""
        # Set up dataset mock
        mock_load_dataset.return_value = {
            'test': [
                {'task_id': 1, 'text': 'test1', 'code': 'code1', 'test_list': []},
                {'task_id': 2, 'text': 'test2', 'code': 'code2', 'test_list': []}
            ]
        }
        
        # Set up test execution mock
        mock_run_tests.return_value = TestResult(passed=3, total=3, errors=[])
        
        # Run full workflow
        tester = MBPPTester(log_dir=temp_log_dir)
        summary = tester.test_range(0, 1)
        
        # Verify results
        assert summary['passed'] == 6
        assert summary['total'] == 6
        assert summary['success_rate'] == 100.0
        assert os.path.exists(summary['log_file'])
    
    def test_error_propagation(self, mock_dataset):
        """Test that errors propagate correctly through the system"""
        tester = MBPPTester()
        tester.ensure_dataset_ready()
        
        # Test with an index that will cause an error
        with pytest.raises(DatasetError):
            tester.test_single_record(999)


# ============================================================================
# Performance Tests (ensure mocking works correctly)
# ============================================================================

class TestPerformance:
    """Test that heavy operations are properly mocked"""
    
    @patch('interp.data_processing.AutoModelForCausalLM.from_pretrained')
    def test_model_loading_is_mocked(self, mock_from_pretrained):
        """Ensure model loading is fast due to mocking"""
        import time
        
        mock_from_pretrained.return_value = MagicMock()
        
        start = time.time()
        loader = ModelLoader("test-model", "cpu")
        loader.load_model_weights()
        duration = time.time() - start
        
        # Should be very fast (< 0.1 seconds) if properly mocked
        assert duration < 0.1
    
    @patch('interp.data_processing.load_dataset')
    def test_dataset_loading_is_mocked(self, mock_load):
        """Ensure dataset loading is fast due to mocking"""
        import time
        
        mock_load.return_value = {'test': []}
        
        start = time.time()
        manager = DatasetManager()
        manager.load_dataset()
        duration = time.time() - start
        
        # Should be very fast (< 0.1 seconds) if properly mocked
        assert duration < 0.1