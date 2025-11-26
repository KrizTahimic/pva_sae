"""
Quick HumanEval validation test for LLAMA 3.1 8B.

Goal: Verify LLAMA achieves ~72.6% pass rate (system card claim) using
HumanEval's standardized format.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate pva_sae
    python3 tests/test_llama_humaneval.py --n 20
"""

import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

from common_simplified.model_loader import load_model_and_tokenizer


def extract_function(generated_text: str, prompt: str, entry_point: str) -> str:
    """Extract the completed function from generated text."""
    # Remove prompt if present
    if generated_text.startswith(prompt):
        code = generated_text[len(prompt):]
    else:
        code = generated_text

    # Find where the function ends (next function def or class or end)
    lines = code.split('\n')
    result_lines = []
    in_function = True

    for line in lines:
        # Stop at next top-level definition
        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            if line.startswith('def ') or line.startswith('class ') or line.startswith('@'):
                break
        result_lines.append(line)

    return prompt + '\n'.join(result_lines)


def evaluate_solution(code: str, test_code: str, entry_point: str, timeout: int = 5) -> bool:
    """Run HumanEval test against generated code."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError()

    # Combine code and test
    full_code = code + '\n' + test_code + f'\ncheck({entry_point})'

    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        namespace = {}
        exec(full_code, namespace)
        return True
    except AssertionError:
        return False
    except TimeoutError:
        return False
    except Exception as e:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20, help='Number of samples to test')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    print("=" * 60)
    print("LLAMA 3.1 8B HumanEval Validation Test")
    print("Expected pass rate: ~72.6% (system card)")
    print("=" * 60)

    # Load model
    print("\nLoading LLAMA 3.1 8B...")
    model, tokenizer = load_model_and_tokenizer('meta-llama/Llama-3.1-8B', args.device)
    model.eval()

    # Load HumanEval
    print("Loading HumanEval dataset...")
    ds = load_dataset('openai_humaneval', split='test')

    # Limit to n samples
    n = min(args.n, len(ds))
    print(f"\nTesting on {n} samples (temp=0, greedy decoding)")
    print("-" * 60)

    passed = 0
    failed = 0

    for i in tqdm(range(n), desc="Evaluating"):
        sample = ds[i]
        prompt = sample['prompt']
        test_code = sample['test']
        entry_point = sample['entry_point']

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors='pt').to(args.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract function
        code = extract_function(generated_text, prompt, entry_point)

        # Evaluate
        success = evaluate_solution(code, test_code, entry_point)

        if success:
            passed += 1
        else:
            failed += 1

        # Print progress
        if (i + 1) % 5 == 0:
            rate = 100 * passed / (i + 1)
            tqdm.write(f"  [{i+1}/{n}] Pass rate: {rate:.1f}% ({passed}/{i+1})")

    # Final results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    pass_rate = 100 * passed / n
    print(f"Passed: {passed}/{n} ({pass_rate:.1f}%)")
    print(f"Failed: {failed}/{n} ({100-pass_rate:.1f}%)")
    print()
    print(f"Expected: ~72.6%")
    print(f"Actual:   {pass_rate:.1f}%")

    if pass_rate >= 65:
        print("\n[OK] Pass rate is in expected range!")
    else:
        print(f"\n[WARNING] Pass rate {pass_rate:.1f}% is lower than expected 72.6%")

    # Show a few examples
    print("\n" + "=" * 60)
    print("SAMPLE GENERATIONS")
    print("=" * 60)

    for i in range(min(3, n)):
        sample = ds[i]
        prompt = sample['prompt']

        inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = extract_function(generated_text, prompt, sample['entry_point'])
        success = evaluate_solution(code, sample['test'], sample['entry_point'])

        print(f"\n--- Sample {i+1}: {sample['task_id']} ({'PASS' if success else 'FAIL'}) ---")
        print(f"Entry point: {sample['entry_point']}")
        print(f"Generated code (first 400 chars):")
        print(code[:400])
        if len(code) > 400:
            print("...")


if __name__ == '__main__':
    main()
