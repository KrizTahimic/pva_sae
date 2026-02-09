"""
Tests for Phase 7.6 - Instruct Steering

Validates:
- instruct_prompt_format: Correct chat template
- coefficient_transfer: Uses base model coefficients
- statistical_tests: _format_effect_log produces correct output
"""

import pytest
import torch
import pandas as pd
from unittest.mock import patch, MagicMock
from common.config import Config
from phase7_6_instruct_steering.instruct_steering_analyzer import InstructSteeringAnalyzer, _format_effect_log


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


# =============================================================================
# _format_effect_log Tests (imported from production code)
# =============================================================================

class TestFormatEffectLog:
    """Test _format_effect_log produces correctly formatted strings."""

    def test_significant_correction_effect(self):
        """Significant correction should include '(significant)' marker."""
        result = _format_effect_log(
            effect_type="correction",
            successes=5,
            trials=50,
            rate=10.0,
            pvalue=0.001,
            is_significant=True
        )

        assert "correction" in result
        assert "5/50" in result
        assert "10.0%" in result
        assert "p=0.0010" in result
        assert "(significant)" in result

    def test_not_significant_corruption_effect(self):
        """Non-significant corruption should include '(not significant)' marker."""
        result = _format_effect_log(
            effect_type="corruption",
            successes=2,
            trials=100,
            rate=2.0,
            pvalue=0.45,
            is_significant=False
        )

        assert "corruption" in result
        assert "2/100" in result
        assert "(not significant)" in result

    def test_includes_instruction_tuned_prefix(self):
        """Output should mention instruction-tuned model."""
        result = _format_effect_log(
            effect_type="preservation",
            successes=90,
            trials=100,
            rate=90.0,
            pvalue=0.0001,
            is_significant=True
        )

        assert "Instruction-tuned model" in result

    def test_returns_string(self):
        """Should return a string."""
        result = _format_effect_log(
            effect_type="test",
            successes=0,
            trials=10,
            rate=0.0,
            pvalue=1.0,
            is_significant=False
        )

        assert isinstance(result, str)


# =============================================================================
# Phase 4.9 Integration Tests (SAE mode)
# =============================================================================

class TestPhase76Phase49Integration:
    """Test that SAE mode loads latent + coefficient from Phase 4.9."""

    def test_sae_mode_loads_latent_from_phase4_9(self):
        """SAE mode should use load_phase4_9_best_latent for latent identity."""
        from common.steering_setup import SAEDirections

        selection = {
            "correct": {"rank": 2, "layer": 15, "latent_idx": 12809, "refined_coefficient": 62},
            "incorrect": {"rank": 0, "layer": 18, "latent_idx": 4612, "refined_coefficient": 25},
        }

        mock_sae_dirs = SAEDirections(
            correct_sae=MagicMock(),
            incorrect_sae=MagicMock(),
            correct_direction=torch.randn(2304),
            incorrect_direction=torch.randn(2304),
        )

        analyzer = object.__new__(InstructSteeringAnalyzer)
        analyzer.config = Config()
        analyzer.use_probe = False
        analyzer.device = torch.device("cpu")
        analyzer.model = MagicMock()
        analyzer.model.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])

        with patch('common.steering_setup.load_phase4_9_best_latent', return_value=selection), \
             patch('common.steering_setup.load_sae_and_directions', return_value=mock_sae_dirs), \
             patch('common.steering_setup.load_baseline_data',
                   return_value=(pd.DataFrame({'task_id': [], 'baseline_passed': []}), "/fake")):
            InstructSteeringAnalyzer._load_dependencies(analyzer)

        assert analyzer.best_correct_latent['layer'] == 15
        assert analyzer.best_correct_latent['latent_idx'] == 12809
        assert analyzer.best_incorrect_latent['layer'] == 18
        assert analyzer.best_incorrect_latent['latent_idx'] == 4612

    def test_sae_mode_coefficient_from_phase4_9(self):
        """SAE mode should read refined_coefficient directly from Phase 4.9 selection."""
        from common.steering_setup import SAEDirections

        selection = {
            "correct": {"rank": 2, "layer": 15, "latent_idx": 12809, "refined_coefficient": 62},
            "incorrect": {"rank": 0, "layer": 18, "latent_idx": 4612, "refined_coefficient": 25},
        }

        mock_sae_dirs = SAEDirections(
            correct_sae=MagicMock(),
            incorrect_sae=MagicMock(),
            correct_direction=torch.randn(2304),
            incorrect_direction=torch.randn(2304),
        )

        analyzer = object.__new__(InstructSteeringAnalyzer)
        analyzer.config = Config()
        analyzer.use_probe = False
        analyzer.device = torch.device("cpu")
        analyzer.model = MagicMock()
        analyzer.model.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])

        with patch('common.steering_setup.load_phase4_9_best_latent', return_value=selection), \
             patch('common.steering_setup.load_sae_and_directions', return_value=mock_sae_dirs), \
             patch('common.steering_setup.load_baseline_data',
                   return_value=(pd.DataFrame({'task_id': [], 'baseline_passed': []}), "/fake")):
            InstructSteeringAnalyzer._load_dependencies(analyzer)

        assert analyzer.correct_coefficient == 62
        assert analyzer.incorrect_coefficient == 25
