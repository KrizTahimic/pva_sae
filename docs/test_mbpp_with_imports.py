#!/usr/bin/env python3
"""
Standalone test script to evaluate impact of pre-loading imports for MBPP dataset.

This script copies Phase 3.5 logic but adds hardcoded import pre-loading
to test if it improves pass@1 rates without modifying the main codebase.

Usage:
    # Test on first 10 problems
    python3 test_mbpp_with_imports.py --start 0 --end 9

    # Run on full dataset (default)
    python3 test_mbpp_with_imports.py
"""

import argparse
import gc
import json
import time
import signal
import contextlib
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Import common utilities (these don't need to be copied)
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.activation_hooks import ActivationExtractor, AttentionExtractor, save_raw_attention_with_boundaries
from common.prompt_utils import PromptBuilder
from common.config import Config
from common.logging import get_logger
from common.utils import detect_device

logger = get_logger("test_mbpp_with_imports", phase="TEST")

# ============================================================================
# EMBEDDED HELPER FUNCTIONS (copied and modified from common_simplified/helpers.py)
# ============================================================================

# Combined import list from two sources:
# 1. 55 imports identified from Phase 3.5 library usage analysis
# 2. 30+ imports extracted from original MBPP solutions (Phase 0.1)
# NOTE: __future__ imports MUST be first in Python
MBPP_REQUIRED_IMPORTS = [
    # __future__ imports (must be first)
    "from __future__ import annotations",
    "from __future__ import print_function",

    # Base library imports
    "import array",
    "import bisect",
    "import cmath",
    "import collections",
    "import collections as ct",
    "import copy",
    "import csv",
    "import datetime",
    "import functools",
    "import heapq",
    "import heapq as hq",
    "import itertools",
    "import math",
    "import math as mt",
    "import operator",
    "import os",
    "import random",
    "import re",
    "import string",
    "import sys",
    "import time",

    # Specific imports from standard libraries
    "from array import array",
    "from bisect import bisect",
    "from bisect import bisect_left",
    "from bisect import bisect_right",
    "from collections import Counter",
    "from collections import defaultdict",
    "from collections import deque",
    "from collections import namedtuple",
    "from collections import OrderedDict",
    "from copy import deepcopy",
    "from datetime import date",
    "from functools import lru_cache",
    "from functools import reduce",
    "from heapq import heappop, heappush",
    "from heapq import merge",
    "from itertools import chain",
    "from itertools import combinations",
    "from itertools import combinations_with_replacement",
    "from itertools import compress",
    "from itertools import filterfalse",
    "from itertools import groupby",
    "from itertools import permutations",
    "from itertools import zip_longest, chain, tee",
    "from math import acos",
    "from math import ceil",
    "from math import cos",
    "from math import factorial",
    "from math import gcd",
    "from math import hypot",
    "from math import pi",
    "from math import radians",
    "from math import sin",
    "from math import sqrt",
    "from math import tan, pi",
    "from operator import eq",
    "from operator import itemgetter",
    "from operator import mul",
    "from random import randint",
    "from random import randrange",
    "from re import compile",
    "from re import findall",
    "from re import match",
    "from string import ascii_lowercase",
    "from string import whitespace",
    "from sys import maxsize",
    "from typing import Callable",
    "from typing import List",
    "from typing import Optional",
    "from typing import Tuple",
    "from typing import Union",
]

@contextlib.contextmanager
def timeout(seconds):
    """Context manager for timeout protection."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def evaluate_code_with_imports(code: str, test_list: list) -> bool:
    """
    Evaluate generated code with MBPP imports pre-loaded.

    This version pre-loads all standard library imports discovered from:
    1. Library usage analysis in Phase 3.5 generated code
    2. Actual import statements in original MBPP solutions
    """
    # Create namespace for execution
    namespace = {}

    # PRE-LOAD MBPP IMPORTS (This is the key change!)
    logger.debug(f"Pre-loading {len(MBPP_REQUIRED_IMPORTS)} standard library imports for MBPP")
    import_code = '\n'.join(MBPP_REQUIRED_IMPORTS)
    try:
        exec(import_code, namespace)
    except Exception as e:
        logger.warning(f"Failed to pre-load some imports: {e}")

    # Execute the generated code with timeout
    try:
        with timeout(5):
            exec(code, namespace)
    except TimeoutError:
        logger.debug("Code execution timeout")
        return False
    except Exception as e:
        logger.debug(f"Code execution error: {e}")
        return False

    # Run each test with timeout
    for test in test_list:
        try:
            with timeout(5):
                exec(test, namespace)
        except (TimeoutError, Exception) as e:
            logger.debug(f"Test execution failed: {e}")
            return False

    return True


def extract_code(generated_text: str, prompt: str) -> str:
    """Extract generated code from model output."""
    if generated_text.startswith(prompt):
        code = generated_text[len(prompt):].strip()
    else:
        code = generated_text.strip()

    def_index = code.find('def ')
    search_start = def_index + 4

    for i in range(search_start, len(code) - 1):
        if code[i] == '\n' and i + 1 < len(code):
            next_char = code[i + 1]
            if next_char not in ' \t\n':
                return code[:i].rstrip()

    return code.strip()


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

class ImportTestRunner:
    """Test runner for evaluating import pre-loading impact."""

    def __init__(self, config: Config, start_idx: int = 0, end_idx: int = None):
        self.config = config
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.device = detect_device()

        logger.info(f"Loading model {config.model_name} on device: {self.device}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            device=self.device
        )

        # Initialize seeds for deterministic generation at temperature=0.0
        # CRITICAL: Even with temperature=0.0 and do_sample=False, PyTorch requires
        # explicit seed initialization for deterministic generation across runs.
        # Without these seeds, GPU kernel scheduling and floating-point operations
        # can vary between runs, producing different code.
        import random
        torch.manual_seed(config.evaluation_random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.evaluation_random_seed)
        random.seed(config.evaluation_random_seed)
        np.random.seed(config.evaluation_random_seed)

        # Force deterministic algorithms (warn_only=True to avoid errors on unsupported ops)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Initialized random seeds (seed={config.evaluation_random_seed}) for deterministic generation")

        # Setup activation extraction (reuse Phase 3.5 logic)
        self.best_layers = self._discover_best_features()
        unique_layers = list(set([self.best_layers['correct'], self.best_layers['incorrect']]))
        self.extraction_layers = unique_layers

        logger.info(f"Using layers: {self.extraction_layers}")

        self.activation_extractor = ActivationExtractor(
            self.model,
            layers=self.extraction_layers
        )

        self.attention_extractor = AttentionExtractor(
            self.model,
            layers=self.extraction_layers,
            position=-1
        )

    def _discover_best_features(self) -> Dict[str, int]:
        """Discover best features from Phase 2.10."""
        phase_2_10_dir = Path(getattr(self.config, 'phase2_10_output_dir', 'data/phase2_10'))
        top_features_file = phase_2_10_dir / "top_20_features.json"

        if not top_features_file.exists():
            raise FileNotFoundError(
                f"top_20_features.json not found in Phase 2.10. "
                "Please run Phase 2.10 first."
            )

        with open(top_features_file, 'r') as f:
            top_features = json.load(f)

        best_correct = top_features['correct'][0]
        best_incorrect = top_features['incorrect'][0]

        return {
            'correct': best_correct['layer'],
            'incorrect': best_incorrect['layer'],
            'correct_feature_idx': best_correct['feature_idx'],
            'incorrect_feature_idx': best_incorrect['feature_idx']
        }

    def generate_temp0_with_activations(self, prompt: str):
        """Generate at temperature 0 with activation extraction."""
        self.activation_extractor.setup_hooks()
        self.attention_extractor.setup_hooks()

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.activation_max_length
            ).to(self.device)

            self.last_tokenized_prompt = inputs['input_ids']
            self.activation_extractor.activations.clear()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    temperature=0.0,
                    max_new_tokens=self.config.model_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_attentions=True,
                    return_dict_in_generate=True
                )

            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            activations = self.activation_extractor.get_activations()
            attention_patterns = self.attention_extractor.get_attention_patterns()

            return generated_text, activations, attention_patterns

        finally:
            self.activation_extractor.remove_hooks()
            self.attention_extractor.remove_hooks()

    def run(self):
        """Run the test."""
        logger.info("="*70)
        logger.info("MBPP IMPORT PRE-LOADING TEST")
        logger.info("="*70)
        logger.info(f"Testing with {len(MBPP_REQUIRED_IMPORTS)} standard library imports pre-loaded")
        logger.info(f"Dataset: {self.config.dataset_name}")
        logger.info(f"Model: {self.config.model_name}")

        # Load validation data
        validation_file = Path(self.config.phase0_1_output_dir) / "validation_mbpp.parquet"
        if not validation_file.exists():
            raise FileNotFoundError(f"Validation data not found: {validation_file}")

        validation_data = pd.read_parquet(validation_file)
        logger.info(f"Loaded {len(validation_data)} validation problems")

        # Apply range filtering
        end_idx = self.end_idx if self.end_idx is not None else len(validation_data) - 1
        logger.info(f"Processing rows {self.start_idx} to {end_idx} (inclusive)")
        validation_data = validation_data.iloc[self.start_idx:end_idx+1].copy()
        logger.info(f"Testing on {len(validation_data)} problems")

        # Setup output directory (fixed path, overwrites each run)
        output_dir = Path("data/test_imports")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "activations" / "task_activations").mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {output_dir}")

        # Process all tasks
        results = []

        for idx, row in tqdm(validation_data.iterrows(), total=len(validation_data), desc="Testing"):
            # Build prompt
            test_cases_str = "\n".join([
                test.strip() if test.strip().startswith('assert ') else f"assert {test.strip()}"
                for test in row['test_list']
            ])
            prompt = PromptBuilder.build_prompt(
                problem_description=row['text'],
                test_cases=test_cases_str
            )

            try:
                start_time = time.time()

                # Generate at temperature 0
                generated_text, task_activations, attention_patterns = \
                    self.generate_temp0_with_activations(prompt)

                generation_time = time.time() - start_time

                # Extract code
                generated_code = extract_code(generated_text, prompt)

                # Evaluate with imports pre-loaded
                test_passed = evaluate_code_with_imports(generated_code, row['test_list'])

                # Save activations
                for layer_num, layer_activations in task_activations.items():
                    save_path = output_dir / "activations" / "task_activations" / \
                                f"{row['task_id']}_layer_{layer_num}.npz"
                    np.savez_compressed(save_path, layer_activations.clone().cpu().float().numpy())

                # Record result
                results.append({
                    'task_id': row['task_id'],
                    'temperature': 0.0,
                    'prompt': prompt,
                    'generated_code': generated_code,
                    'test_passed': test_passed,
                    'error_message': None,
                    'generation_time': generation_time,
                    'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
                    'generation_idx': 0,
                    'test_list': json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
                })

                if test_passed:
                    logger.debug(f"✓ Task {row['task_id']} passed")
                else:
                    logger.debug(f"✗ Task {row['task_id']} failed")

            except Exception as e:
                logger.error(f"Error processing task {row['task_id']}: {e}")
                results.append({
                    'task_id': row['task_id'],
                    'temperature': 0.0,
                    'prompt': prompt,
                    'generated_code': "",
                    'test_passed': False,
                    'error_message': str(e),
                    'generation_time': 0.0,
                    'cyclomatic_complexity': row.get('cyclomatic_complexity', 0.0),
                    'generation_idx': 0,
                    'test_list': json.dumps(row['test_list'].tolist() if hasattr(row['test_list'], 'tolist') else row['test_list'])
                })

        # Save results
        df = pd.DataFrame(results)
        output_file = output_dir / "results.parquet"
        df.to_parquet(output_file, index=False)

        # Calculate statistics
        n_passed = sum(1 for r in results if r['test_passed'])
        n_total = len(results)
        pass_rate = (n_passed / n_total * 100) if n_total > 0 else 0.0

        # Save metadata
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "test_description": "MBPP with combined imports from usage analysis and original solutions",
            "imports_preloaded": len(MBPP_REQUIRED_IMPORTS),
            "dataset_range": f"{self.start_idx}-{end_idx}",
            "n_problems_tested": n_total,
            "n_passed": n_passed,
            "n_failed": n_total - n_passed,
            "pass_rate_percent": round(pass_rate, 2),
            "avg_generation_time": np.mean([r['generation_time'] for r in results]),
            "best_layers": self.best_layers,
            "imports": MBPP_REQUIRED_IMPORTS
        }

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create human-readable summary file
        summary_file = output_dir / "SUMMARY.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MBPP IMPORT PRE-LOADING TEST RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Test: {metadata['test_description']}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.config.model_name}\n")
            f.write(f"Dataset: {self.config.dataset_name}\n\n")
            f.write(f"Imports pre-loaded: {metadata['imports_preloaded']}\n")
            f.write(f"Problems tested: {metadata['dataset_range']} ({n_total} problems)\n\n")
            f.write("RESULTS:\n")
            f.write(f"  Passed:  {n_passed}/{n_total} ({pass_rate:.1f}%)\n")
            f.write(f"  Failed:  {n_total - n_passed}/{n_total} ({100-pass_rate:.1f}%)\n\n")
            f.write(f"Average generation time: {metadata['avg_generation_time']:.2f}s\n\n")
            f.write("FILES:\n")
            f.write(f"  - results.parquet     (test results data)\n")
            f.write(f"  - metadata.json       (detailed metadata)\n")
            f.write(f"  - activations/        (activation data)\n")

        # Print summary
        logger.info("="*70)
        logger.info("TEST COMPLETE")
        logger.info("="*70)
        logger.info(f"Problems tested: {n_total}")
        logger.info(f"Passed: {n_passed}")
        logger.info(f"Failed: {n_total - n_passed}")
        logger.info(f"Pass rate: {pass_rate:.2f}%")
        logger.info(f"Avg generation time: {np.mean([r['generation_time'] for r in results]):.2f}s")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  - {summary_file}")
        logger.info(f"  - {output_file}")
        logger.info(f"  - {metadata_file}")
        logger.info("="*70)

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Test MBPP pass@1 rate with import pre-loading"
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start index (default: 0)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End index inclusive (default: last problem)'
    )

    args = parser.parse_args()

    # Load config
    config = Config()

    # Ensure we're using MBPP
    if config.dataset_name != "mbpp":
        logger.warning(f"Config dataset is '{config.dataset_name}', forcing to 'mbpp'")
        config.dataset_name = "mbpp"

    # Run test
    runner = ImportTestRunner(config, start_idx=args.start, end_idx=args.end)
    runner.run()


if __name__ == "__main__":
    main()
