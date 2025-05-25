# Regression Tests for data_processing.py

This directory contains comprehensive regression tests for the MBPP framework in `interp/data_processing.py`. These tests ensure that the codebase continues to work correctly as you develop new features.

## 🚀 Quick Start

### Installation

```bash
# Install test dependencies
pip install -r test-requirements.txt

# Or install with main dependencies
pip install -r requirements.txt -r test-requirements.txt
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/test_data_processing.py

# Or use the test runner script
python run_tests.py

# Run only fast unit tests
python run_tests.py --unit --fast

# Run with verbose output
python run_tests.py -v

# Run specific test class
python run_tests.py --specific TestModelManager
```

## 📋 Test Coverage

The test suite covers all major components:

### 1. **Exception Hierarchy** (`TestExceptionHierarchy`)
- Tests custom exception classes
- Verifies exception inheritance
- Tests exception messages

### 2. **Utility Classes**
- `TestDeviceManager` - Device detection (MPS/CUDA/CPU)
- `TestConsoleOutput` - Console formatting
- `TestErrorContext` - Error handling patterns

### 3. **Core Configuration** (`TestLoggingConfiguration`)
- Logging setup and configuration
- Log file creation
- Debug mode handling

### 4. **Model Management**
- `TestModelLoader` - Model/tokenizer loading (mocked)
- `TestCodeGenerator` - Code generation logic
- `TestModelManager` - Model lifecycle orchestration

### 5. **Data Management**
- `TestTestResult` - Test result dataclass
- `TestDatasetManager` - MBPP dataset handling (mocked)

### 6. **Test Execution** (`TestTestExecutor`)
- Code execution against test cases
- Test result collection
- Error handling during execution

### 7. **Main Orchestration** (`TestMBPPTester`)
- Full workflow testing
- Range testing
- Result summarization

## 🎯 Testing Philosophy

### Mocking Strategy

All heavy operations are mocked for speed:
- **Model Loading**: `AutoModelForCausalLM.from_pretrained()` is mocked
- **Tokenizer Loading**: `AutoTokenizer.from_pretrained()` is mocked
- **Dataset Loading**: `load_dataset()` returns mock data
- **File I/O**: Temporary directories are used for logs

### Test Types

Tests are organized with markers:
- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Integration tests with multiple components
- `@pytest.mark.slow` - Slower tests (can be skipped)
- `@pytest.mark.critical` - Must-pass tests for core functionality

## 🛠️ Test Utilities

### Fixtures

- `temp_log_dir` - Provides temporary directory for log files
- `mock_torch_devices` - Mocks PyTorch device availability
- `mock_model_loading` - Mocks model and tokenizer loading
- `mock_dataset` - Provides mock MBPP dataset

### Running Specific Tests

```bash
# Run a specific test class
pytest tests/test_data_processing.py::TestModelManager

# Run a specific test method
pytest tests/test_data_processing.py::TestModelManager::test_load_model_success

# Run tests matching a pattern
pytest -k "test_load"

# Run only tests that failed last time
pytest --lf
```

## 📊 Coverage Reports

After running tests with coverage:

```bash
# View coverage in terminal
python run_tests.py --coverage

# Open HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## 🐛 Debugging Tests

```bash
# Drop into debugger on failure
python run_tests.py --pdb

# Run with maximum verbosity
pytest -vvv tests/test_data_processing.py

# Show print statements during tests
pytest -s tests/test_data_processing.py
```

## ✅ Best Practices

1. **Run tests before committing** - Ensures nothing is broken
2. **Add tests for new features** - Maintain coverage
3. **Use mocks for external dependencies** - Keep tests fast
4. **Update tests when changing behavior** - Tests document expected behavior
5. **Check coverage reports** - Identify untested code paths

## 🔧 Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r test-requirements.txt
    pytest tests/test_data_processing.py --cov=interp.data_processing
```

## 📝 Adding New Tests

When adding new functionality to `data_processing.py`:

1. Add corresponding test class in `test_data_processing.py`
2. Mock any external dependencies
3. Test both success and failure cases
4. Add appropriate markers (`@pytest.mark.unit`, etc.)
5. Run tests to ensure they pass

Example test structure:
```python
class TestNewFeature:
    """Test new feature functionality"""
    
    @pytest.mark.unit
    def test_success_case(self):
        """Test normal operation"""
        # Arrange
        # Act  
        # Assert
        
    @pytest.mark.unit
    def test_error_case(self):
        """Test error handling"""
        with pytest.raises(ExpectedException):
            # Test error condition
```

## 🚨 Common Issues

### Import Errors
- Ensure you're running tests from the project root
- Check that `interp/` is in the Python path

### Slow Tests
- Check that mocks are working (tests should complete in < 5 seconds)
- Use `--fast` flag to skip slow tests

### Coverage Gaps
- Check the HTML coverage report for uncovered lines
- Add tests for any critical uncovered code

## 📞 Support

If you encounter issues with the tests:
1. Check this README for solutions
2. Ensure all dependencies are installed
3. Run with verbose output for more details
4. Check that mocks are properly configured