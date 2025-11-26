"""
Test HumanEval with MBPP-style prompt format.

Goal: Validate if converting HumanEval to MBPP-style format eliminates
the "comment-only" generation failures where the model outputs:
    # def function_name(...) -> type:
instead of actual code.

Known failing tasks (comment-only outputs): 3, 4, 5, 8, 16, etc.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate pva_sae
    python3 tests/test_humaneval_mbpp_style.py --n 20
    python3 tests/test_humaneval_mbpp_style.py --tasks 3,4,5,8,16  # Test specific failing tasks
"""

import argparse
import torch
import re
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import sys
sys.path.insert(0, '.')

from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import evaluate_code
from common.prompt_utils import PromptBuilder


def extract_docstring_description(prompt: str) -> str:
    """
    Extract the problem description from HumanEval's docstring.

    HumanEval format:
        def function_name(args) -> type:
            '''Problem description here
            >>> example1
            >>> example2
            '''

    Returns:
        The problem description text (without examples)
    """
    # Find docstring between triple quotes
    docstring_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if not docstring_match:
        docstring_match = re.search(r"'''(.*?)'''", prompt, re.DOTALL)

    if not docstring_match:
        return "Write a function that solves the problem."

    docstring = docstring_match.group(1).strip()

    # Remove examples (lines starting with >>>)
    lines = docstring.split('\n')
    description_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('>>>') or stripped.startswith('...'):
            break  # Stop at first example
        description_lines.append(line)

    description = '\n'.join(description_lines).strip()
    return description if description else "Write a function that solves the problem."


def extract_function_signature(prompt: str) -> tuple[str, str, str]:
    """
    Extract function name, args, and return type from HumanEval prompt.

    Returns:
        (function_name, args_str, return_type)
    """
    # Match: def function_name(args) -> type:
    match = re.search(r'def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^:]+))?:', prompt, re.DOTALL)
    if match:
        func_name = match.group(1)
        args = match.group(2).strip()
        return_type = match.group(3).strip() if match.group(3) else ""
        return func_name, args, return_type
    return "solution", "", ""


def convert_humaneval_test_to_asserts(test_code: str, entry_point: str) -> list[str]:
    """
    Convert HumanEval's check() function tests to simple assert statements.

    HumanEval test format:
        def check(candidate):
            assert candidate(1, 2) == 3
            assert candidate([1,2,3]) == 6

    Returns:
        List of assert strings like: ["assert func(1, 2) == 3", ...]
    """
    asserts = []

    # Find all assert statements
    for line in test_code.split('\n'):
        line = line.strip()
        if line.startswith('assert '):
            # Replace 'candidate' with actual function name
            assert_stmt = line.replace('candidate', entry_point)
            asserts.append(assert_stmt)

    return asserts


def convert_to_mbpp_style(humaneval_sample: dict) -> dict:
    """
    Convert HumanEval problem to MBPP-style format.

    MBPP format:
        Problem description text

        assert func(input1) == output1
        assert func(input2) == output2

        # Solution:

    Returns:
        Dict with 'prompt', 'test_list', 'entry_point'
    """
    prompt = humaneval_sample['prompt']
    test_code = humaneval_sample['test']
    entry_point = humaneval_sample['entry_point']

    # Extract components
    description = extract_docstring_description(prompt)
    func_name, args, return_type = extract_function_signature(prompt)
    asserts = convert_humaneval_test_to_asserts(test_code, entry_point)

    # Build problem description
    problem_desc = f"Write a function called `{entry_point}` that {description.lower() if description[0].isupper() else description}"
    if args:
        problem_desc += f"\n\nFunction signature: def {entry_point}({args})"
        if return_type:
            problem_desc += f" -> {return_type}"

    # Limit to 3-5 asserts to keep prompt manageable
    test_asserts = asserts[:5] if len(asserts) > 5 else asserts
    test_cases_str = '\n'.join(test_asserts)

    # Build MBPP-style prompt
    mbpp_prompt = PromptBuilder.build_prompt(
        problem_description=problem_desc,
        test_cases=test_cases_str,
        code_initiator="# Solution:"
    )

    return {
        'prompt': mbpp_prompt,
        'test_list': asserts,  # All asserts for evaluation
        'entry_point': entry_point,
        'original_prompt': prompt,
        'original_test': test_code
    }


def extract_code_mbpp_style(generated_text: str, prompt: str) -> str:
    """
    Extract generated code using MBPP-style extraction.
    Same logic as common_simplified/helpers.py extract_code()
    """
    code = None

    # Method 1: Try exact prompt match
    if generated_text.startswith(prompt):
        code = generated_text[len(prompt):].strip()

    # Method 2: Look for "# Solution:" marker
    if code is None:
        solution_marker = "# Solution:"
        marker_idx = generated_text.find(solution_marker)
        if marker_idx != -1:
            code = generated_text[marker_idx + len(solution_marker):].strip()

    # Method 3: Find last assert and extract after
    if code is None:
        last_assert_idx = generated_text.rfind("assert ")
        if last_assert_idx != -1:
            newline_after_assert = generated_text.find('\n', last_assert_idx)
            if newline_after_assert != -1:
                code = generated_text[newline_after_assert:].strip()

    # Method 4: Fallback
    if code is None:
        code = generated_text.strip()

    # Find and extract function definition
    def_index = code.find('def ')
    if def_index == -1:
        return code.strip()

    code = code[def_index:]

    # Find function end
    search_start = 4
    for i in range(search_start, len(code) - 1):
        if code[i] == '\n' and i + 1 < len(code):
            next_char = code[i + 1]
            if next_char not in ' \t\n':
                return code[:i].rstrip()

    return code.strip()


def is_comment_only(code: str) -> bool:
    """Check if extracted code is just comments (no actual code)."""
    lines = code.strip().split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            return False
    return True


# Using evaluate_code from common_simplified/helpers.py which has proper pre-imports


def main():
    parser = argparse.ArgumentParser(description="Test HumanEval with MBPP-style prompts")
    parser.add_argument('--n', type=int, default=20, help='Number of samples to test')
    parser.add_argument('--tasks', type=str, default=None,
                        help='Comma-separated task indices to test (e.g., 3,4,5,8,16)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--model', type=str, default='google/gemma-2-2b',
                        help='Model to use (default: gemma-2-2b)')
    parser.add_argument('--show-prompts', action='store_true',
                        help='Show original vs MBPP-style prompts')
    parser.add_argument('--output-dir', type=str, default='data/test_humaneval_mbpp_style',
                        help='Output directory for results parquet')
    args = parser.parse_args()

    print("=" * 70)
    print("HumanEval MBPP-Style Format Test")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Goal: Test if MBPP-style format eliminates comment-only generation failures")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    model.eval()

    # Load HumanEval
    print("Loading HumanEval dataset...")
    ds = load_dataset('openai_humaneval', split='test')

    # Determine which tasks to test
    if args.tasks:
        task_indices = [int(t.strip()) for t in args.tasks.split(',')]
        print(f"Testing specific tasks: {task_indices}")
    else:
        task_indices = list(range(min(args.n, len(ds))))
        print(f"Testing first {len(task_indices)} tasks")

    print("-" * 70)

    results = {
        'passed': 0,
        'failed': 0,
        'comment_only': 0,
        'details': []
    }

    for idx in tqdm(task_indices, desc="Evaluating"):
        sample = ds[idx]
        task_id = sample['task_id']

        # Convert to MBPP-style
        mbpp_style = convert_to_mbpp_style(sample)

        if args.show_prompts:
            print(f"\n--- Task {idx}: {task_id} ---")
            print("ORIGINAL PROMPT:")
            print(sample['prompt'][:500])
            print("\nMBPP-STYLE PROMPT:")
            print(mbpp_style['prompt'][:500])
            print()

        # Generate
        inputs = tokenizer(mbpp_style['prompt'], return_tensors='pt').to(args.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract code
        extracted_code = extract_code_mbpp_style(generated_text, mbpp_style['prompt'])

        # Check if comment-only
        is_comment = is_comment_only(extracted_code)

        # Evaluate (using evaluate_code which has proper pre-imports like List, Dict, etc.)
        success = False
        if not is_comment:
            success = evaluate_code(extracted_code, mbpp_style['test_list'])

        # Record result with full data for parquet inspection
        result_detail = {
            'task_id': task_id,
            'task_idx': idx,
            'passed': success,
            'is_comment_only': is_comment,
            'original_prompt': sample['prompt'],
            'mbpp_style_prompt': mbpp_style['prompt'],
            'generated_text': generated_text,
            'extracted_code': extracted_code,
            'test_list': mbpp_style['test_list'],
            'entry_point': mbpp_style['entry_point'],
        }
        results['details'].append(result_detail)

        if success:
            results['passed'] += 1
            status = "PASS"
        elif is_comment:
            results['comment_only'] += 1
            results['failed'] += 1
            status = "COMMENT-ONLY"
        else:
            results['failed'] += 1
            status = "FAIL"

        tqdm.write(f"  Task {idx} ({task_id}): {status}")
        if is_comment:
            tqdm.write(f"    Code: {extracted_code[:100]}...")

    # Print summary
    total = len(task_indices)
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total tested: {total}")
    print(f"Passed:       {results['passed']} ({100*results['passed']/total:.1f}%)")
    print(f"Failed:       {results['failed']} ({100*results['failed']/total:.1f}%)")
    print(f"Comment-only: {results['comment_only']} ({100*results['comment_only']/total:.1f}%)")
    print()

    if results['comment_only'] == 0:
        print("[SUCCESS] No comment-only generations! MBPP-style format works.")
    else:
        print(f"[WARNING] Still have {results['comment_only']} comment-only generations.")
        print("Tasks with comment-only outputs:")
        for d in results['details']:
            if d['is_comment_only']:
                print(f"  - Task {d['task_idx']} ({d['task_id']})")

    # Show examples
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS")
    print("=" * 70)

    for i, d in enumerate(results['details'][:5]):
        status = "PASS" if d['passed'] else ("COMMENT" if d['is_comment_only'] else "FAIL")
        print(f"\n--- Task {d['task_idx']} ({d['task_id']}) [{status}] ---")
        print(f"Extracted code preview:")
        print(d['extracted_code'][:200])
        if len(d['extracted_code']) >= 200:
            print("...")

    # Save results to parquet for inspection
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame from details
    df = pd.DataFrame(results['details'])

    # Generate output filename with model name
    model_suffix = args.model.split('/')[-1].replace('-', '_').lower()
    output_file = output_dir / f"results_{model_suffix}.parquet"

    df.to_parquet(output_file, index=False)
    print("\n" + "=" * 70)
    print("OUTPUT SAVED")
    print("=" * 70)
    print(f"Results saved to: {output_file}")
    print(f"Total records: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nInspect with notebook or pandas:")
    print(f"  df = pd.read_parquet('{output_file}')")
    print(f"  df[df['is_comment_only'] == True]  # View comment-only failures")
    print(f"  df[df['passed'] == False]  # View all failures")


if __name__ == '__main__':
    main()
