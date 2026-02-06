"""
Tests for prompt construction regression - Prevent double-wrapping.

Validates:
- steering_phases_use_row_prompt_directly: All steering phases use row['prompt'] as-is
  instead of re-wrapping with PromptBuilder.build_prompt()

This is a source-code-level regression test. If a developer accidentally
re-introduces PromptBuilder.build_prompt(problem_description=row['prompt'], ...),
it would cause double-wrapping: the prompt (already wrapped in Phase 1) gets
wrapped again, leading to malformed prompts that waste GPU hours.
"""

import pytest
import inspect


# =============================================================================
# No Double Wrapping Regression Tests
# =============================================================================

class TestNoDoubleWrapping:
    """Source-code-level regression test for prompt double-wrapping bug.

    All steering phases should use row['prompt'] directly because:
    1. Phase 1 already builds the full prompt via PromptBuilder
    2. The prompt is stored in the 'prompt' column of the parquet file
    3. Steering phases should use this pre-built prompt as-is

    Calling PromptBuilder.build_prompt(problem_description=row['prompt'], ...)
    would wrap an already-wrapped prompt, corrupting it.
    """

    # Steering phase files to check
    STEERING_PHASE_FILES = {
        '4.8': (
            'phase4_8_steering_analysis.steering_effect_analyzer',
            'SteeringEffectAnalyzer',
        ),
        '4.12': (
            'phase4_12_zero_disc_steering.zero_disc_steering_generator',
            'ZeroDiscSteeringGenerator',
        ),
        '5.3': (
            'phase5_3_weight_orthogonalization.weight_orthogonalizer',
            'WeightOrthogonalizer',
        ),
        '5.6': (
            'phase5_6_zero_disc_orthogonalization.zero_disc_weight_orthogonalizer',
            'ZeroDiscWeightOrthogonalizer',
        ),
        '7.6': (
            'phase7_6_instruct_steering.instruct_steering_analyzer',
            'InstructSteeringAnalyzer',
        ),
    }

    def test_steering_phases_use_row_prompt_directly(self):
        """All steering phases should use row['prompt'] directly (not PromptBuilder.build_prompt).

        Iterates over all steering phase source files and verifies:
        1. row['prompt'] appears in the code (prompt is taken from the data row)
        2. PromptBuilder.build_prompt is NOT called with row['prompt'] as argument
           (which would cause double-wrapping)
        """
        import importlib

        for phase, (module_path, class_name) in self.STEERING_PHASE_FILES.items():
            module = importlib.import_module(module_path)
            source = inspect.getsource(module)

            # Verify row['prompt'] is used
            assert "row['prompt']" in source, (
                f"Phase {phase} ({module_path}) should use row['prompt'] directly"
            )

            # Verify no double-wrapping pattern: build_prompt(...row['prompt']...)
            # This pattern indicates someone is wrapping an already-wrapped prompt
            assert "build_prompt(problem_description=row['prompt']" not in source, (
                f"Phase {phase} ({module_path}) has double-wrapping bug: "
                f"build_prompt(problem_description=row['prompt']...) wraps an already-wrapped prompt"
            )

            # Verify PromptBuilder is not called on the prompt variable
            assert "PromptBuilder().build_prompt" not in source, (
                f"Phase {phase} ({module_path}) should not call PromptBuilder().build_prompt() "
                f"- prompts are already built in Phase 1"
            )

    def test_phase_4_12_has_no_prompt_builder_import(self):
        """Phase 4.12 specifically should not import PromptBuilder at all.

        Phase 4.12 was the phase where the double-wrapping bug was found and fixed.
        The fix included removing the PromptBuilder import entirely.
        """
        import importlib
        module = importlib.import_module(
            'phase4_12_zero_disc_steering.zero_disc_steering_generator'
        )
        source = inspect.getsource(module)

        assert "from common.prompt_utils import PromptBuilder" not in source, (
            "Phase 4.12 should not import PromptBuilder - it was removed as part of the fix"
        )

    def test_no_build_prompt_call_in_generation_methods(self):
        """Verify that generation/steering methods do not call build_prompt().

        This checks the actual method source code (not the whole module) to ensure
        the prompt construction happens only via row['prompt'].
        """
        import importlib

        # Check specific generation methods in each phase
        generation_methods = {
            '4.12': (
                'phase4_12_zero_disc_steering.zero_disc_steering_generator',
                'ZeroDiscSteeringGenerator',
                '_apply_zero_disc_steering',
            ),
        }

        for phase, (module_path, class_name, method_name) in generation_methods.items():
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            method = getattr(cls, method_name)
            method_source = inspect.getsource(method)

            assert "build_prompt" not in method_source, (
                f"Phase {phase} {class_name}.{method_name}() should not call build_prompt() "
                f"- the prompt is pre-built in Phase 1 and stored in row['prompt']"
            )

    def test_phase_8_2_has_no_prompt_builder_import(self):
        """Phase 8.2 should not import PromptBuilder (H2 fix).

        Phase 8.2 merges Phase 0.1 with Phase 3.6 which provides the 'prompt' column.
        It should use row['prompt'] directly, not re-build prompts.
        """
        import importlib
        module = importlib.import_module(
            'phase8_2_threshold_optimizer.threshold_optimizer'
        )
        source = inspect.getsource(module)

        assert "from common.prompt_utils import PromptBuilder" not in source, (
            "Phase 8.2 should not import PromptBuilder - prompts come from Phase 3.6 data"
        )
        assert "PromptBuilder.build_prompt" not in source, (
            "Phase 8.2 should not call PromptBuilder.build_prompt()"
        )
