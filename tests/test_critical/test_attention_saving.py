"""
Regression tests: attention patterns saved in both sequential and parallel paths.

Phase 3.5 parallel path (TemperatureEvaluator) previously silently discarded
attention data captured by hooks. Phase 4.8 was not affected.

These tests verify both phases call save_raw_attention_with_boundaries
so Phase 6.3 can find the attention data it needs.
"""

import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from common.activation_hooks import (
    AttentionExtractor,
    save_raw_attention_with_boundaries,
)


class TestPhase35ParallelAttentionSaving:
    """TemperatureEvaluator._generate_temp0_with_activations must save attention."""

    def test_save_task_attention_calls_save_raw(self, tmp_path):
        """_save_task_attention should call save_raw_attention_with_boundaries per layer."""
        from phase3_5_temperature_robustness.temperature_runner import TemperatureEvaluator

        evaluator = TemperatureEvaluator.__new__(TemperatureEvaluator)
        evaluator.output_dir = tmp_path
        evaluator.tokenizer = MagicMock()

        attention_patterns = {
            20: torch.randn(8, 64),
            21: torch.randn(8, 64),
        }
        tokenized_prompt = torch.randint(0, 1000, (1, 64))

        with patch(
            'phase3_5_temperature_robustness.temperature_runner.save_raw_attention_with_boundaries'
        ) as mock_save:
            evaluator._save_task_attention('task_001', attention_patterns, tokenized_prompt)

        assert mock_save.call_count == 2

        call_kwargs = [c.kwargs for c in mock_save.call_args_list]
        saved_layers = {kw['layer_idx'] for kw in call_kwargs}
        assert saved_layers == {20, 21}

        for kw in call_kwargs:
            assert kw['task_id'] == 'task_001'
            assert kw['output_dir'] == tmp_path / "activations" / "attention_patterns"
            assert kw['tokenizer'] is evaluator.tokenizer

    def test_save_task_attention_creates_directory(self, tmp_path):
        """_save_task_attention should create the attention directory."""
        from phase3_5_temperature_robustness.temperature_runner import TemperatureEvaluator

        evaluator = TemperatureEvaluator.__new__(TemperatureEvaluator)
        evaluator.output_dir = tmp_path
        evaluator.tokenizer = MagicMock()

        attention_dir = tmp_path / "activations" / "attention_patterns"
        assert not attention_dir.exists()

        with patch(
            'phase3_5_temperature_robustness.temperature_runner.save_raw_attention_with_boundaries'
        ):
            evaluator._save_task_attention(
                'task_001', {20: torch.randn(8, 64)}, torch.randint(0, 1000, (1, 64))
            )

        assert attention_dir.is_dir()

    def _make_evaluator_for_generate(self, tmp_path, attention_patterns):
        """Helper: build a mocked TemperatureEvaluator for _generate_temp0_with_activations."""
        from phase3_5_temperature_robustness.temperature_runner import TemperatureEvaluator

        evaluator = TemperatureEvaluator.__new__(TemperatureEvaluator)
        evaluator.config = MagicMock()
        evaluator.config.activation_max_length = 512
        evaluator.config.model_max_new_tokens = 256
        evaluator.output_dir = tmp_path
        evaluator.activation_dir = tmp_path / "activations" / "task_activations"
        evaluator.activation_dir.mkdir(parents=True, exist_ok=True)
        evaluator.device = torch.device('cpu')

        # Mock tokenizer — return object must support .to(device)
        input_ids = torch.randint(0, 1000, (1, 32))
        tokenizer_output = MagicMock()
        tokenizer_output.__getitem__ = lambda self, key: {'input_ids': input_ids}[key]
        tokenizer_output.to.return_value = tokenizer_output
        evaluator.tokenizer = MagicMock(return_value=tokenizer_output)
        evaluator.tokenizer.pad_token_id = 0
        evaluator.tokenizer.eos_token_id = 1
        evaluator.tokenizer.decode.return_value = "def solution():\n    return 42"

        # Mock model.generate
        mock_output = MagicMock()
        mock_output.sequences = torch.cat([input_ids, torch.randint(0, 1000, (1, 10))], dim=1)
        evaluator.model = MagicMock()
        evaluator.model.generate.return_value = mock_output

        # Mock extractors
        evaluator.activation_extractor = MagicMock()
        evaluator.activation_extractor.activations = {20: [torch.randn(2304)]}

        evaluator.attention_extractor = MagicMock()
        evaluator.attention_extractor.get_attention_patterns.return_value = attention_patterns

        return evaluator

    def test_generate_temp0_calls_get_attention_patterns(self, tmp_path):
        """_generate_temp0_with_activations must call get_attention_patterns and save."""
        fake_patterns = {20: torch.randn(8, 32)}
        evaluator = self._make_evaluator_for_generate(tmp_path, fake_patterns)

        row = {
            'task_id': 'task_001',
            'test_list': '["assert solution() == 42"]',
        }

        with patch(
            'phase3_5_temperature_robustness.temperature_runner.evaluate_code_with_error_type'
        ) as mock_eval, patch(
            'phase3_5_temperature_robustness.temperature_runner.extract_code',
            return_value="def solution():\n    return 42"
        ), patch(
            'phase3_5_temperature_robustness.temperature_runner.save_raw_attention_with_boundaries'
        ) as mock_save_attn, patch(
            'phase3_5_temperature_robustness.temperature_runner.save_activation'
        ):
            mock_eval.return_value = MagicMock(passed=True, error_type=None)
            evaluator._generate_temp0_with_activations(row, "prompt text")

        # Key assertion: get_attention_patterns was called
        evaluator.attention_extractor.get_attention_patterns.assert_called_once()

        # Key assertion: save was called with the patterns
        assert mock_save_attn.call_count == 1
        kw = mock_save_attn.call_args.kwargs
        assert kw['task_id'] == 'task_001'
        assert kw['layer_idx'] == 20

    def test_empty_attention_patterns_not_saved(self, tmp_path):
        """If get_attention_patterns returns empty dict, no save should happen."""
        evaluator = self._make_evaluator_for_generate(tmp_path, attention_patterns={})
        evaluator.activation_extractor.activations = {}

        row = {
            'task_id': 'task_001',
            'test_list': '["assert solution() == 42"]',
        }

        with patch(
            'phase3_5_temperature_robustness.temperature_runner.evaluate_code_with_error_type'
        ) as mock_eval, patch(
            'phase3_5_temperature_robustness.temperature_runner.extract_code',
            return_value="def solution():\n    return 42"
        ), patch(
            'phase3_5_temperature_robustness.temperature_runner.save_raw_attention_with_boundaries'
        ) as mock_save_attn, patch(
            'phase3_5_temperature_robustness.temperature_runner.save_activation'
        ):
            mock_eval.return_value = MagicMock(passed=True, error_type=None)
            evaluator._generate_temp0_with_activations(row, "prompt text")

        # Should NOT save when patterns are empty
        mock_save_attn.assert_not_called()


class TestPhase35SequentialAttentionSaving:
    """TemperatureRobustnessRunner._save_task_attention must save attention."""

    def test_save_task_attention_calls_save_raw(self, tmp_path):
        """Sequential _save_task_attention should call save_raw_attention_with_boundaries."""
        from phase3_5_temperature_robustness.temperature_runner import TemperatureRobustnessRunner

        runner = TemperatureRobustnessRunner.__new__(TemperatureRobustnessRunner)
        runner.output_dir = tmp_path
        runner.tokenizer = MagicMock()
        runner.last_tokenized_prompt = torch.randint(0, 1000, (1, 64))

        attention_patterns = {20: torch.randn(8, 64)}

        with patch(
            'phase3_5_temperature_robustness.temperature_runner.save_raw_attention_with_boundaries'
        ) as mock_save:
            runner._save_task_attention('task_001', attention_patterns)

        mock_save.assert_called_once()
        kw = mock_save.call_args.kwargs
        assert kw['task_id'] == 'task_001'
        assert kw['layer_idx'] == 20


class TestPhase48AttentionSaving:
    """SteeringEffectAnalyzer._save_steered_attention must save attention."""

    def test_save_steered_attention_calls_save_raw(self, tmp_path):
        """_save_steered_attention should call save_raw_attention_with_boundaries per layer."""
        from phase4_8_steering_analysis.steering_effect_analyzer import SteeringEffectAnalyzer

        analyzer = SteeringEffectAnalyzer.__new__(SteeringEffectAnalyzer)
        analyzer.output_dir = tmp_path
        analyzer.tokenizer = MagicMock()

        attention_patterns = {
            20: torch.randn(8, 64),
            21: torch.randn(8, 64),
        }
        tokenized_prompt = torch.randint(0, 1000, (1, 64))

        with patch(
            'phase4_8_steering_analysis.steering_effect_analyzer.save_raw_attention_with_boundaries'
        ) as mock_save:
            analyzer._save_steered_attention(
                'task_001', 'correct', attention_patterns, tokenized_prompt
            )

        assert mock_save.call_count == 2

        call_kwargs = [c.kwargs for c in mock_save.call_args_list]
        saved_layers = {kw['layer_idx'] for kw in call_kwargs}
        assert saved_layers == {20, 21}

        for kw in call_kwargs:
            assert kw['task_id'] == 'task_001'
            assert kw['output_dir'] == tmp_path / "attention_patterns" / "correct_steering" / "rank_0"

    def test_steering_type_in_directory_path(self, tmp_path):
        """Different steering types should use different subdirectories."""
        from phase4_8_steering_analysis.steering_effect_analyzer import SteeringEffectAnalyzer

        analyzer = SteeringEffectAnalyzer.__new__(SteeringEffectAnalyzer)
        analyzer.output_dir = tmp_path
        analyzer.tokenizer = MagicMock()

        patterns = {20: torch.randn(8, 64)}
        prompt = torch.randint(0, 1000, (1, 64))

        with patch(
            'phase4_8_steering_analysis.steering_effect_analyzer.save_raw_attention_with_boundaries'
        ) as mock_save:
            analyzer._save_steered_attention('t1', 'correct', patterns, prompt)
            analyzer._save_steered_attention('t2', 'incorrect', patterns, prompt)

        dirs_used = {c.kwargs['output_dir'] for c in mock_save.call_args_list}
        assert dirs_used == {
            tmp_path / "attention_patterns" / "correct_steering" / "rank_0",
            tmp_path / "attention_patterns" / "incorrect_steering" / "rank_0",
        }


class TestPhase35And48AttentionPathConsistency:
    """Phase 3.5 baseline attention paths must match what Phase 6.3 expects."""

    def test_phase35_parallel_attention_path(self, tmp_path):
        """Parallel path should save to activations/attention_patterns (same as sequential)."""
        from phase3_5_temperature_robustness.temperature_runner import TemperatureEvaluator

        evaluator = TemperatureEvaluator.__new__(TemperatureEvaluator)
        evaluator.output_dir = tmp_path
        evaluator.tokenizer = MagicMock()

        with patch(
            'phase3_5_temperature_robustness.temperature_runner.save_raw_attention_with_boundaries'
        ) as mock_save:
            evaluator._save_task_attention(
                'task_001', {20: torch.randn(8, 64)}, torch.randint(0, 1000, (1, 64))
            )

        expected_dir = tmp_path / "activations" / "attention_patterns"
        assert mock_save.call_args.kwargs['output_dir'] == expected_dir

    def test_phase35_sequential_attention_path(self, tmp_path):
        """Sequential path should save to activations/attention_patterns."""
        from phase3_5_temperature_robustness.temperature_runner import TemperatureRobustnessRunner

        runner = TemperatureRobustnessRunner.__new__(TemperatureRobustnessRunner)
        runner.output_dir = tmp_path
        runner.tokenizer = MagicMock()
        runner.last_tokenized_prompt = torch.randint(0, 1000, (1, 64))

        with patch(
            'phase3_5_temperature_robustness.temperature_runner.save_raw_attention_with_boundaries'
        ) as mock_save:
            runner._save_task_attention('task_001', {20: torch.randn(8, 64)})

        expected_dir = tmp_path / "activations" / "attention_patterns"
        assert mock_save.call_args.kwargs['output_dir'] == expected_dir

    def test_both_paths_use_same_directory(self, tmp_path):
        """Sequential and parallel paths must write to identical directories."""
        from phase3_5_temperature_robustness.temperature_runner import (
            TemperatureEvaluator, TemperatureRobustnessRunner
        )

        # Parallel path
        evaluator = TemperatureEvaluator.__new__(TemperatureEvaluator)
        evaluator.output_dir = tmp_path
        evaluator.tokenizer = MagicMock()

        # Sequential path
        runner = TemperatureRobustnessRunner.__new__(TemperatureRobustnessRunner)
        runner.output_dir = tmp_path
        runner.tokenizer = MagicMock()
        runner.last_tokenized_prompt = torch.randint(0, 1000, (1, 64))

        dirs = []
        with patch(
            'phase3_5_temperature_robustness.temperature_runner.save_raw_attention_with_boundaries'
        ) as mock_save:
            evaluator._save_task_attention(
                'task_001', {20: torch.randn(8, 64)}, torch.randint(0, 1000, (1, 64))
            )
            dirs.append(mock_save.call_args.kwargs['output_dir'])

            runner._save_task_attention('task_002', {20: torch.randn(8, 64)})
            dirs.append(mock_save.call_args.kwargs['output_dir'])

        assert dirs[0] == dirs[1], "Parallel and sequential paths must save to same directory"
