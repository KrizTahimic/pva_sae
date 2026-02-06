"""
Tests for Phase 4.12 - Zero-Discrimination Steering (Double Prompt Wrapping Fix)

Validates:
- prompt_uses_row_directly: row['prompt'] is used as-is without re-wrapping
- no_prompt_builder_import: PromptBuilder is not imported in the module
"""

import pytest
import inspect
import importlib


# =============================================================================
# Prompt Handling Tests
# =============================================================================

class TestPromptHandling:
    """Test that Phase 4.12 uses pre-built prompts directly (no double wrapping)."""

    def test_uses_prebuilt_prompt_directly(self):
        """Verify that row['prompt'] is used directly, not re-wrapped by PromptBuilder.

        The fix ensures that the prompt from Phase 1 (already stored in row['prompt'])
        is passed directly to the tokenizer. Previously, PromptBuilder.build_prompt()
        was called on top of the already-wrapped prompt, causing double wrapping.

        We verify this by inspecting the source code of _apply_zero_disc_steering
        to confirm it assigns row['prompt'] directly to the prompt variable.
        """
        from phase4_12_zero_disc_steering.zero_disc_steering_generator import ZeroDiscSteeringGenerator

        source = inspect.getsource(ZeroDiscSteeringGenerator._apply_zero_disc_steering)

        # Should use row['prompt'] directly
        assert "row['prompt']" in source, (
            "Phase 4.12 should use row['prompt'] directly, not re-wrap via PromptBuilder"
        )

        # Should NOT call PromptBuilder.build_prompt on the prompt
        assert "PromptBuilder" not in source, (
            "Phase 4.12 _apply_zero_disc_steering should not reference PromptBuilder "
            "(prompt is already built in Phase 1)"
        )

        # Should NOT call build_prompt
        assert "build_prompt" not in source, (
            "Phase 4.12 should not call build_prompt() - the prompt is pre-built"
        )

    def test_no_prompt_builder_import(self):
        """Verify that PromptBuilder is not imported in the module.

        After the fix, the import of PromptBuilder was removed since
        the module no longer wraps prompts - it uses row['prompt'] directly.
        """
        import phase4_12_zero_disc_steering.zero_disc_steering_generator as module

        source = inspect.getsource(module)

        # Check that PromptBuilder is not imported at module level
        assert "from common.prompt_utils import PromptBuilder" not in source, (
            "Phase 4.12 should not import PromptBuilder - prompts are pre-built in Phase 1"
        )
        assert "import PromptBuilder" not in source, (
            "Phase 4.12 should not import PromptBuilder in any form"
        )

    def test_prompt_passed_to_tokenizer(self):
        """Verify the prompt variable is passed to self.tokenizer() call."""
        from phase4_12_zero_disc_steering.zero_disc_steering_generator import ZeroDiscSteeringGenerator

        source = inspect.getsource(ZeroDiscSteeringGenerator._apply_zero_disc_steering)

        # The prompt should be passed to tokenizer
        assert "self.tokenizer(" in source, (
            "Phase 4.12 should pass prompt to self.tokenizer()"
        )

        # Verify the pattern: prompt = row['prompt'] followed by tokenizer(prompt, ...)
        lines = source.split('\n')
        found_prompt_assignment = False
        found_tokenizer_call = False

        for line in lines:
            stripped = line.strip()
            if "prompt = row['prompt']" in stripped:
                found_prompt_assignment = True
            if found_prompt_assignment and "self.tokenizer(" in stripped:
                found_tokenizer_call = True
                break

        assert found_prompt_assignment, "Should have prompt = row['prompt'] assignment"
        assert found_tokenizer_call, (
            "Tokenizer call should appear after prompt = row['prompt'] assignment"
        )
