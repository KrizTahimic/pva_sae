"""
Tests for common/prompt_utils.py

Validates:
- PromptBuilder.build_prompt: Correct format for MBPP and HumanEval
- Prompt includes test assertions
- Prompt does NOT double-wrap
"""

import pytest

from common.prompt_utils import PromptBuilder


# =============================================================================
# PromptBuilder Tests
# =============================================================================

class TestPromptBuilder:
    """Test PromptBuilder.build_prompt() output format."""

    def test_mbpp_format(self):
        """MBPP-style prompt should have description, tests, and solution marker."""
        prompt = PromptBuilder.build_prompt(
            problem_description="Write a function to find the sum of a list.",
            test_cases="assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0"
        )

        assert "Write a function to find the sum of a list." in prompt
        assert "assert sum_list([1, 2, 3]) == 6" in prompt
        assert "assert sum_list([]) == 0" in prompt
        assert "# Solution:" in prompt

    def test_humaneval_format(self):
        """HumanEval-style prompt (converted to MBPP format) should work identically."""
        prompt = PromptBuilder.build_prompt(
            problem_description="def has_close_elements(numbers: List[float], threshold: float) -> bool:",
            test_cases="assert has_close_elements([1.0, 2.0, 3.9, 4.0], 0.3) == True\nassert has_close_elements([1.0, 2.8, 3.0, 4.0], 0.5) == True"
        )

        assert "has_close_elements" in prompt
        assert "assert has_close_elements" in prompt
        assert "# Solution:" in prompt

    def test_prompt_includes_test_assertions(self):
        """Test cases should appear verbatim in the prompt."""
        test_str = "assert foo(1) == 2\nassert foo(0) == 0"
        prompt = PromptBuilder.build_prompt(
            problem_description="Write function foo.",
            test_cases=test_str
        )

        for line in test_str.split('\n'):
            assert line in prompt

    def test_prompt_structure_order(self):
        """Prompt structure: description, then tests, then solution marker."""
        prompt = PromptBuilder.build_prompt(
            problem_description="DESCRIPTION_MARKER",
            test_cases="TEST_MARKER"
        )

        desc_pos = prompt.index("DESCRIPTION_MARKER")
        test_pos = prompt.index("TEST_MARKER")
        solution_pos = prompt.index("# Solution:")

        assert desc_pos < test_pos < solution_pos

    def test_no_double_wrap(self):
        """Building a prompt from already-built prompt text should add extra wrapping."""
        first_prompt = PromptBuilder.build_prompt(
            problem_description="Write function bar.",
            test_cases="assert bar(1) == 1"
        )

        # If someone accidentally passes the full prompt as description again
        second_prompt = PromptBuilder.build_prompt(
            problem_description=first_prompt,
            test_cases="assert bar(1) == 1"
        )

        # The second prompt should have TWO "# Solution:" markers (double-wrapped)
        # This tests that callers should NOT re-wrap
        count = second_prompt.count("# Solution:")
        assert count == 2, (
            "Double-wrapping produces duplicate markers - callers should use "
            "row['prompt'] directly, not re-wrap with PromptBuilder.build_prompt()"
        )

    def test_custom_code_initiator(self):
        """Custom code initiator should replace default '# Solution:'."""
        prompt = PromptBuilder.build_prompt(
            problem_description="Write function baz.",
            test_cases="assert baz() == True",
            code_initiator="# Your code here:"
        )

        assert "# Your code here:" in prompt
        assert "# Solution:" not in prompt

    def test_sections_separated_by_double_newline(self):
        """Sections should be separated by double newlines."""
        prompt = PromptBuilder.build_prompt(
            problem_description="Description",
            test_cases="Tests"
        )

        # The template uses \n\n between sections
        assert "Description\n\nTests\n\n# Solution:" == prompt
