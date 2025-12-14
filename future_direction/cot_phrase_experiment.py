"""
CoT Phrase Experiment: Test if reasoning models produce different language
for initially correct vs initially incorrect code.

Uses DeepSeek-R1-Distill-Llama-8B which naturally produces <think>...</think>
reasoning blocks. We analyze the content within these blocks for
uncertainty/confidence phrases.

Hypothesis:
- Initially incorrect → <think> should contain more uncertain phrases
- Initially correct → <think> should contain more confident phrases

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate pva_sae

    # Test on subset first
    python cot_phrase_experiment.py --end 30

    # Full run
    python cot_phrase_experiment.py [--start N] [--end M]

Output:
    future_direction/cot_phrase_results/cot_outputs.json

Output fields:
    - full_output: Complete model response including <think> and code
    - think_content: Extracted reasoning from <think>...</think> block
"""

import sys
sys.path.insert(0, '..')

import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from tqdm import tqdm

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.prompt_utils import PromptBuilder

# ============================================================================
# EXPANDED PHRASE LISTS
# ============================================================================

# Phrases indicating uncertainty/doubt (expect more in initially INCORRECT)
UNCERTAIN_PHRASES = [
    # Explicit uncertainty
    "not sure", "might be wrong", "maybe", "i think", "i believe",
    "let me check", "could be", "possibly", "perhaps", "probably",
    "uncertain", "unsure", "don't know", "not certain", "might not",

    # Hedging
    "seems like", "looks like", "appears to", "i guess", "i suppose",
    "should work", "might work", "could work", "may work",

    # Self-correction / backtracking
    "hmm", "wait", "actually", "let me reconsider", "on second thought",
    "hold on", "let me think again", "that's not right", "no,",
    "let me try again", "that doesn't work", "i made a mistake",
    "let me redo", "scratch that", "never mind", "oops",

    # Checking / verifying
    "let me verify", "need to check", "double check", "let me test",
    "is this correct", "does this work", "will this work",

    # Complexity acknowledgment
    "tricky", "complicated", "difficult", "challenging", "not straightforward",
    "edge case", "corner case", "careful here", "be careful",
]

# Phrases indicating confidence (expect more in initially CORRECT)
CONFIDENT_PHRASES = [
    # Explicit confidence
    "clearly", "obviously", "definitely", "certainly", "of course",
    "sure", "confident", "no doubt", "without doubt",

    # Correctness claims
    "this works", "this is correct", "this is right", "correct solution",
    "that should work", "this solves", "this handles", "this will work",
    "perfect", "exactly", "precisely",

    # Simplicity
    "simple", "straightforward", "easy", "just need to", "all we need",
    "basic", "trivial", "obvious solution",

    # Completion
    "done", "complete", "finished", "that's it", "all set",
    "and we're done", "and that's all",

    # Understanding
    "i understand", "i see", "got it", "makes sense", "clear",

    # Direct assertions
    "the answer is", "the solution is", "we return", "we output",
    "the result", "final answer",
]

# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def build_prompt_standard(row: dict) -> str:
    """Standard prompt (no CoT) - baseline."""
    return row['prompt']

def build_prompt_cot(row: dict) -> str:
    """Build prompt for DeepSeek-R1-Distill from raw Llama Phase 1 data.

    DeepSeek-R1 models naturally produce <think>...</think> reasoning blocks,
    so we just build the standard prompt without adding CoT instructions.

    Llama Phase 1 data has:
    - 'text': problem description
    - 'test_list': array of test case strings
    """
    problem_description = row['text']
    test_cases = '\n'.join(row['test_list'])

    return PromptBuilder.build_prompt(
        problem_description=problem_description,
        test_cases=test_cases,
        code_initiator="# Solution:"
    )

def extract_think_block(output: str) -> str:
    """Extract content between <think> and </think> tags.

    DeepSeek-R1 models output reasoning in <think>...</think> blocks.
    This is the content we want to analyze for uncertainty/confidence phrases.
    """
    import re
    match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    return match.group(1).strip() if match else output

# ============================================================================
# PHRASE ANALYSIS
# ============================================================================

def count_phrases(text: str, phrase_list: list[str]) -> dict[str, int]:
    """Count occurrences of each phrase in text using word boundaries."""
    import re
    text_lower = text.lower()
    counts = {}
    for phrase in phrase_list:
        # Use word boundaries to avoid matching substrings (e.g., "oops" in "loops")
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            counts[phrase] = len(matches)
    return counts

def analyze_cot_text(cot_text: str) -> dict:
    """Analyze CoT text for phrase patterns."""
    uncertain_counts = count_phrases(cot_text, UNCERTAIN_PHRASES)
    confident_counts = count_phrases(cot_text, CONFIDENT_PHRASES)

    return {
        'uncertain_phrases': uncertain_counts,
        'confident_phrases': confident_counts,
        'total_uncertain': sum(uncertain_counts.values()),
        'total_confident': sum(confident_counts.values()),
        'cot_length_chars': len(cot_text),
        'cot_length_words': len(cot_text.split()),
    }

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

class CoTPhraseExperiment:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
        self.model_name = model_name
        self.device = self._detect_device()
        self.output_dir = Path(__file__).parent / "cot_phrase_results"
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.results_file = self.output_dir / "cot_outputs.json"

        print(f"Using device: {self.device}")
        print(f"Output directory: {self.output_dir}")

    def _detect_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model loaded!")

    def load_data(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> pd.DataFrame:
        """Load Llama Phase 1 data (more appropriate for DeepSeek-R1-Distill which is Llama-based)."""
        data_file = Path(__file__).parent.parent / "data/phase1_0_llama/dataset_sae_20251126_145021.parquet"

        if not data_file.exists():
            raise FileNotFoundError(f"Llama Phase 1 data not found: {data_file}")

        df = pd.read_parquet(data_file)
        print(f"Loaded {len(df)} samples from Llama Phase 1")

        # Add initially_correct label
        df['initially_correct'] = df['baseline_passed']

        n_correct = df['initially_correct'].sum()
        n_incorrect = len(df) - n_correct
        print(f"  Initially correct: {n_correct}")
        print(f"  Initially incorrect: {n_incorrect}")

        # Apply range filtering
        if start_idx is not None or end_idx is not None:
            start = start_idx or 0
            end = end_idx or len(df)
            df = df.iloc[start:end].copy()
            print(f"Filtered to indices {start}-{end}: {len(df)} samples")

        return df

    def generate_cot(self, prompt: str, max_new_tokens: int = 1500) -> str:
        """Generate CoT response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return generated

    def load_checkpoint(self) -> tuple[list[dict], set]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            processed_ids = set(r['task_id'] for r in checkpoint['results'])
            print(f"Loaded checkpoint: {len(checkpoint['results'])} results")
            return checkpoint['results'], processed_ids
        return [], set()

    def save_checkpoint(self, results: list[dict]):
        """Save checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'n_results': len(results),
                'results': results
            }, f, indent=2)

    def run(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None):
        """Run the full experiment."""
        # Load model
        self.load_model()

        # Load data
        df = self.load_data(start_idx, end_idx)

        # Load checkpoint
        results, processed_ids = self.load_checkpoint()

        # Filter already processed
        df_remaining = df[~df['task_id'].isin(processed_ids)]
        print(f"Remaining to process: {len(df_remaining)}")

        if len(df_remaining) == 0:
            print("All samples already processed!")
            return results

        # Process samples
        checkpoint_freq = 10

        for idx, (_, row) in enumerate(tqdm(df_remaining.iterrows(), total=len(df_remaining))):
            row_dict = row.to_dict()

            # Build CoT prompt
            cot_prompt = build_prompt_cot(row_dict)

            # Generate
            start_time = time.time()
            try:
                full_output = self.generate_cot(cot_prompt)
                generation_time = time.time() - start_time

                # Extract <think> block for analysis
                think_content = extract_think_block(full_output)

                # Analyze only the think block (reasoning content)
                analysis = analyze_cot_text(think_content)

                result = {
                    'task_id': row_dict['task_id'],
                    'initially_correct': bool(row_dict['initially_correct']),
                    'baseline_passed': bool(row_dict['baseline_passed']),
                    'prompt': cot_prompt,
                    'full_output': full_output,
                    'think_content': think_content,
                    'generation_time': generation_time,
                    **analysis
                }
                results.append(result)

            except Exception as e:
                print(f"Error on task {row_dict['task_id']}: {e}")
                continue

            # Checkpoint
            if (idx + 1) % checkpoint_freq == 0:
                self.save_checkpoint(results)
                print(f"  Checkpoint saved: {len(results)} results")

        # Final save
        self.save_results(results)

        # Print summary
        self.print_summary(results)

        return results

    def save_results(self, results: list[dict]):
        """Save final results."""
        with open(self.results_file, 'w') as f:
            json.dump({
                'metadata': {
                    'model': self.model_name,
                    'timestamp': datetime.now().isoformat(),
                    'n_total': len(results),
                    'n_uncertain_phrases': len(UNCERTAIN_PHRASES),
                    'n_confident_phrases': len(CONFIDENT_PHRASES),
                },
                'phrase_lists': {
                    'uncertain': UNCERTAIN_PHRASES,
                    'confident': CONFIDENT_PHRASES,
                },
                'results': results
            }, f, indent=2)

        print(f"\nResults saved to: {self.results_file}")

        # Clean up checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("Checkpoint cleaned up")

    def print_summary(self, results: list[dict]):
        """Print summary statistics."""
        correct_results = [r for r in results if r['initially_correct']]
        incorrect_results = [r for r in results if not r['initially_correct']]

        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)

        print(f"\nTotal samples: {len(results)}")
        print(f"  Initially correct: {len(correct_results)}")
        print(f"  Initially incorrect: {len(incorrect_results)}")

        # Phrase statistics
        if correct_results:
            avg_uncertain_correct = sum(r['total_uncertain'] for r in correct_results) / len(correct_results)
            avg_confident_correct = sum(r['total_confident'] for r in correct_results) / len(correct_results)
            print(f"\nInitially CORRECT samples:")
            print(f"  Avg uncertain phrases: {avg_uncertain_correct:.2f}")
            print(f"  Avg confident phrases: {avg_confident_correct:.2f}")

        if incorrect_results:
            avg_uncertain_incorrect = sum(r['total_uncertain'] for r in incorrect_results) / len(incorrect_results)
            avg_confident_incorrect = sum(r['total_confident'] for r in incorrect_results) / len(incorrect_results)
            print(f"\nInitially INCORRECT samples:")
            print(f"  Avg uncertain phrases: {avg_uncertain_incorrect:.2f}")
            print(f"  Avg confident phrases: {avg_confident_incorrect:.2f}")

        # Hypothesis check
        if correct_results and incorrect_results:
            print("\n" + "-"*40)
            print("HYPOTHESIS CHECK:")
            print("-"*40)

            # Hypothesis: incorrect should have MORE uncertain phrases
            if avg_uncertain_incorrect > avg_uncertain_correct:
                print(f"✓ Incorrect has MORE uncertain phrases ({avg_uncertain_incorrect:.2f} > {avg_uncertain_correct:.2f})")
            else:
                print(f"✗ Incorrect has LESS uncertain phrases ({avg_uncertain_incorrect:.2f} <= {avg_uncertain_correct:.2f})")

            # Hypothesis: correct should have MORE confident phrases
            if avg_confident_correct > avg_confident_incorrect:
                print(f"✓ Correct has MORE confident phrases ({avg_confident_correct:.2f} > {avg_confident_incorrect:.2f})")
            else:
                print(f"✗ Correct has LESS confident phrases ({avg_confident_correct:.2f} <= {avg_confident_incorrect:.2f})")

        print("="*60)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CoT Phrase Experiment")
    parser.add_argument('--start', type=int, default=None, help="Start index")
    parser.add_argument('--end', type=int, default=None, help="End index")
    args = parser.parse_args()

    experiment = CoTPhraseExperiment()
    experiment.run(start_idx=args.start, end_idx=args.end)

if __name__ == "__main__":
    main()
