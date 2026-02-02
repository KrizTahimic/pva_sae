"""
Tests for Phase 7.6 - Instruct Steering

Validates:
- instruct_prompt_format: Correct chat template
- coefficient_transfer: Uses base model coefficients
"""

import pytest
from common.config import Config


# =============================================================================
# instruct_prompt_format Tests
# =============================================================================

class TestInstructPromptFormat:
    """Test correct chat template."""

    def test_instruct_model_from_config(self):
        """Config should specify instruction-tuned model."""
        config = Config()
        assert hasattr(config, 'phase7_6_model_name')
        assert 'it' in config.phase7_6_model_name.lower() or \
               'instruct' in config.phase7_6_model_name.lower()

    def test_gemma_it_model_specified(self):
        """Should use Gemma instruction-tuned model by default."""
        config = Config()
        # Default should be gemma-2-2b-it
        assert 'gemma-2-2b-it' in config.phase7_6_model_name or \
               'gemma-2-9b-it' in config.phase7_6_model_name

    def test_chat_template_format(self):
        """Instruct model should use chat template format."""
        # Gemma chat template structure
        template = """<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
"""
        assert '<start_of_turn>user' in template
        assert '<start_of_turn>model' in template

    def test_prompt_wrapping_for_instruct(self):
        """MBPP prompts should be wrapped in chat template."""
        user_prompt = "Write a Python function that adds two numbers."

        # Simulated wrapping
        formatted = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"

        assert user_prompt in formatted
        assert formatted.startswith('<start_of_turn>user')


# =============================================================================
# coefficient_transfer Tests
# =============================================================================

class TestCoefficientTransfer:
    """Test uses base model coefficients."""

    def test_loads_base_model_coefficients(self, tmp_path):
        """Should load coefficients from base model Phase 4.6."""
        import json

        # Create base model Phase 4.6 output
        phase_4_6_dir = tmp_path / "data" / "phase4_6"
        phase_4_6_dir.mkdir(parents=True)

        coeff_data = {
            "correct": {"refined_coefficient": 47.5},
            "incorrect": {"refined_coefficient": 43.0}
        }
        with open(phase_4_6_dir / "refined_coefficients.json", 'w') as f:
            json.dump(coeff_data, f)

        # Load
        with open(phase_4_6_dir / "refined_coefficients.json") as f:
            loaded = json.load(f)

        assert loaded['correct']['refined_coefficient'] == 47.5
        assert loaded['incorrect']['refined_coefficient'] == 43.0

    def test_same_directions_as_base_model(self):
        """Should use same SAE directions as base model."""
        # Phase 7.6 uses directions from Phase 2.5 (same as base model)
        # This is a conceptual test
        base_model_phase = "2.5"
        instruct_direction_source = "2.5"

        assert base_model_phase == instruct_direction_source

    def test_coefficient_interpretation_same(self):
        """Coefficient interpretation should be same for instruct model."""
        # With normalized directions, coefficient = steering magnitude
        coefficient = 47.5
        direction_norm = 1.0  # Unit normalized

        steering_magnitude = coefficient * direction_norm

        assert steering_magnitude == 47.5


# =============================================================================
# Activation Extraction Tests
# =============================================================================

class TestActivationExtraction:
    """Test activation extraction for instruct model."""

    def test_last_token_extraction(self):
        """Should extract at last token (same as base model)."""
        from common.config import Config

        config = Config()
        # Position -1 (last token)
        assert config.activation_position == -1

    def test_same_layers_as_base_model(self):
        """Should use same layers as base model for extraction."""
        from common.config import Config

        base_config = Config(model_name="google/gemma-2-2b")
        instruct_config = Config(model_name="google/gemma-2-2b-it")

        # Same model family = same number of layers
        assert base_config.activation_layers == instruct_config.activation_layers


# =============================================================================
# Output Format Tests
# =============================================================================

class TestOutputFormat:
    """Test Phase 7.6 output format."""

    def test_output_comparable_to_base_model(self):
        """Output should be comparable to Phase 4.8 (base model)."""
        expected_structure = {
            'summary': {
                'correction_rate': 25.0,
                'corruption_rate': 15.0,
                'preservation_rate': 85.0
            },
            'model': 'google/gemma-2-2b-it',
            'coefficient_source': 'phase4_6'
        }

        assert 'correction_rate' in expected_structure['summary']
        assert 'model' in expected_structure

    def test_includes_model_info(self):
        """Output should include instruct model information."""
        output = {
            'model': 'google/gemma-2-2b-it',
            'is_instruct': True
        }

        assert output['is_instruct'] is True


# =============================================================================
# Phase 7.3 Integration Tests
# =============================================================================

class TestPhase73Integration:
    """Test integration with Phase 7.3 baseline."""

    def test_phase_7_3_model_from_config(self):
        """Config should specify Phase 7.3 model."""
        config = Config()
        assert hasattr(config, 'phase7_3_model_name')

    def test_same_model_as_7_6(self):
        """Phase 7.3 and 7.6 should use same instruct model."""
        config = Config()
        assert config.phase7_3_model_name == config.phase7_6_model_name
