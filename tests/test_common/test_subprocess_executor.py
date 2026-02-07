"""Tests for common.subprocess_executor module."""

import pytest

from common.subprocess_executor import _classify_exception, execute_code_with_hard_timeout


# ---------------------------------------------------------------------------
# Tests for _classify_exception()
# ---------------------------------------------------------------------------

class TestClassifyException:
    """Tests for the _classify_exception() pure function."""

    def test_syntax_error(self):
        error_type, exc_class = _classify_exception(SyntaxError("invalid syntax"))
        assert error_type == "syntax"
        assert exc_class == "SyntaxError"

    def test_indentation_error(self):
        """IndentationError is a subclass of SyntaxError and should map to 'syntax'."""
        error_type, exc_class = _classify_exception(IndentationError("unexpected indent"))
        assert error_type == "syntax"
        assert exc_class == "IndentationError"

    def test_name_error(self):
        error_type, exc_class = _classify_exception(NameError("name 'x' is not defined"))
        assert error_type == "name"
        assert exc_class == "NameError"

    def test_attribute_error(self):
        error_type, exc_class = _classify_exception(AttributeError("has no attribute 'foo'"))
        assert error_type == "name"
        assert exc_class == "AttributeError"

    def test_unbound_local_error(self):
        error_type, exc_class = _classify_exception(UnboundLocalError("referenced before assignment"))
        assert error_type == "name"
        assert exc_class == "UnboundLocalError"

    def test_type_error(self):
        error_type, exc_class = _classify_exception(TypeError("unsupported operand"))
        assert error_type == "type"
        assert exc_class == "TypeError"

    def test_assertion_error(self):
        error_type, exc_class = _classify_exception(AssertionError())
        assert error_type == "logic"
        assert exc_class == "AssertionError"

    def test_generic_exception_falls_through_to_runtime(self):
        error_type, exc_class = _classify_exception(Exception("something went wrong"))
        assert error_type == "runtime"
        assert exc_class == "Exception"

    def test_value_error_falls_through_to_runtime(self):
        error_type, exc_class = _classify_exception(ValueError("bad value"))
        assert error_type == "runtime"
        assert exc_class == "ValueError"

    def test_zero_division_error_falls_through_to_runtime(self):
        error_type, exc_class = _classify_exception(ZeroDivisionError("division by zero"))
        assert error_type == "runtime"
        assert exc_class == "ZeroDivisionError"


# ---------------------------------------------------------------------------
# Tests for execute_code_with_hard_timeout()
# ---------------------------------------------------------------------------

class TestExecuteCodeWithHardTimeout:
    """Tests for the execute_code_with_hard_timeout() function."""

    def test_passing_code(self):
        result = execute_code_with_hard_timeout(
            code="x = 1 + 1",
            test_list=["assert x == 2"],
            timeout_seconds=5,
        )
        assert result.passed is True
        assert result.error_type == "passed"
        assert result.error_message is None
        assert result.exception_class is None

    def test_syntax_error_code(self):
        result = execute_code_with_hard_timeout(
            code="def foo(:",
            test_list=[],
            timeout_seconds=5,
        )
        assert result.passed is False
        assert result.error_type == "syntax"
        assert result.exception_class == "SyntaxError"
        assert result.error_message is not None

    def test_timeout(self):
        result = execute_code_with_hard_timeout(
            code="import time; time.sleep(100)",
            test_list=[],
            timeout_seconds=1,
        )
        assert result.passed is False
        assert result.error_type == "timeout"
        assert result.exception_class == "TimeoutError"
        assert "1 seconds" in result.error_message

    def test_runtime_error_division_by_zero(self):
        result = execute_code_with_hard_timeout(
            code="1/0",
            test_list=[],
            timeout_seconds=5,
        )
        assert result.passed is False
        assert result.error_type == "runtime"
        assert result.exception_class == "ZeroDivisionError"

    def test_test_assertion_failure(self):
        result = execute_code_with_hard_timeout(
            code="x = 42",
            test_list=["assert x == 99"],
            timeout_seconds=5,
        )
        assert result.passed is False
        assert result.error_type == "logic"
        assert result.exception_class == "AssertionError"

    def test_passing_code_no_tests(self):
        """Code that executes without error and has no tests should pass."""
        result = execute_code_with_hard_timeout(
            code="x = 1 + 1",
            test_list=[],
            timeout_seconds=5,
        )
        assert result.passed is True
        assert result.error_type == "passed"
