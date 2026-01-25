"""
Subprocess-based code execution with hard timeout.

This module provides reliable timeout protection for code execution in
parallel workers. The main process can use signal-based timeouts (SIGALRM),
but these don't work in subprocess workers. This module spawns a fresh
process for each code execution that can be forcibly terminated.

Key insight: CUDA context issues only affect model inference, NOT code
execution. exec() is pure Python/CPU - we can spawn a subprocess for
just code evaluation without GPU concerns.
"""

import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Optional

from common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SubprocessResult:
    """Result from subprocess code execution."""
    passed: bool
    error_type: str  # "passed", "syntax", "name", "type", "logic", "runtime", "timeout"
    error_message: Optional[str]
    exception_class: Optional[str]


def _classify_exception(exc: Exception) -> tuple[str, str]:
    """Classify an exception into an error type category."""
    exc_class = type(exc).__name__

    if isinstance(exc, (SyntaxError, IndentationError)):
        return "syntax", exc_class
    if isinstance(exc, (NameError, AttributeError, UnboundLocalError)):
        return "name", exc_class
    if isinstance(exc, TypeError):
        return "type", exc_class
    if isinstance(exc, AssertionError):
        return "logic", exc_class
    return "runtime", exc_class


def _execute_in_subprocess(code: str, test_list: list[str], import_code: Optional[str], result_queue: mp.Queue):
    """
    Worker function that runs in isolated subprocess.

    This function executes in a fresh process that can be terminated
    if the code hangs (e.g., infinite loop).
    """
    try:
        # Step 1: Compile check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            error_type, exc_class = _classify_exception(e)
            result_queue.put(SubprocessResult(False, error_type, str(e), exc_class))
            return

        # Step 2: Prepare namespace with imports
        namespace = {}
        if import_code:
            try:
                exec(import_code, namespace)
            except Exception:
                # Import failures are non-fatal - continue without those imports
                pass

        # Step 3: Execute code definition
        try:
            exec(code, namespace)
        except Exception as e:
            error_type, exc_class = _classify_exception(e)
            result_queue.put(SubprocessResult(False, error_type, str(e), exc_class))
            return

        # Step 4: Run tests
        for test in test_list:
            try:
                exec(test, namespace)
            except Exception as e:
                error_type, exc_class = _classify_exception(e)
                result_queue.put(SubprocessResult(False, error_type, str(e), exc_class))
                return

        # All passed
        result_queue.put(SubprocessResult(True, "passed", None, None))

    except Exception as e:
        # Catch-all for unexpected errors
        result_queue.put(SubprocessResult(False, "runtime", str(e), type(e).__name__))


def execute_code_with_hard_timeout(
    code: str,
    test_list: list[str],
    timeout_seconds: int = 5,
    import_code: Optional[str] = None
) -> SubprocessResult:
    """
    Execute code with hard timeout using subprocess.

    Spawns a separate process that can be forcibly terminated.
    This is the only reliable way to kill infinite loops in worker processes.

    Args:
        code: Generated code to execute
        test_list: List of test assertion strings
        timeout_seconds: Maximum execution time before killing the process
        import_code: Optional import statements to execute before code

    Returns:
        SubprocessResult with passed status, error_type, error_message, and exception_class
    """
    # Use 'spawn' context to ensure clean process without inherited state
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()

    process = ctx.Process(
        target=_execute_in_subprocess,
        args=(code, test_list, import_code, result_queue)
    )
    process.start()

    try:
        # Wait for result with timeout
        result = result_queue.get(timeout=timeout_seconds)
        process.join(timeout=1)
        return result

    except queue.Empty:
        # Timeout - kill the process
        logger.debug(f"Code execution timeout after {timeout_seconds}s, terminating subprocess")
        process.terminate()
        process.join(timeout=2)

        if process.is_alive():
            # Force kill if terminate didn't work
            logger.debug("Subprocess still alive after terminate, using kill")
            process.kill()
            process.join(timeout=1)

        return SubprocessResult(
            passed=False,
            error_type="timeout",
            error_message=f"Code execution exceeded {timeout_seconds} seconds",
            exception_class="TimeoutError"
        )

    finally:
        # Cleanup: ensure process is dead
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
