#!/usr/bin/env python3
"""
Debug code extraction and evaluation in Phase 4.5 context.
"""

import torch
import pandas as pd
import json
from pathlib import Path
from common.config import Config
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import extract_code, evaluate_code
from common.utils import detect_device, discover_latest_phase_output
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae
from common.steering_metrics import create_steering_hook

def debug_extraction():
    # Setup
    config = Config()
    device = detect_device()
    
    print("="*80)
    print("CODE EXTRACTION AND EVALUATION DEBUG")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(config.model_name, device=device)
    model.eval()
    
    # Load Phase 3.6 data
    phase3_6_output = discover_latest_phase_output("3.6")
    if not phase3_6_output:
        print("ERROR: Phase 3.6 output not found!")
        return
    
    baseline_file = Path(phase3_6_output).parent / "dataset_hyperparams_temp_0_0.parquet"
    data = pd.read_parquet(baseline_file)
    
    # Get incorrect problems
    incorrect_data = data[data['test_passed'] == False]
    print(f"\nFound {len(incorrect_data)} initially incorrect problems")
    
    # Test first problem
    if len(incorrect_data) == 0:
        print("No incorrect problems found!")
        return
    
    row = incorrect_data.iloc[0]
    task_id = row['task_id']
    prompt = row['prompt']
    baseline_code = row['generated_code']
    
    # Parse test list
    if isinstance(row['test_list'], str):
        test_list = json.loads(row['test_list'])
    else:
        test_list = row['test_list']
    
    print(f"\n{'='*40}")
    print(f"TASK {task_id}")
    print(f"{'='*40}")
    
    print("\nPROMPT (first 300 chars):")
    print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    
    print("\nBASELINE CODE FROM DATASET:")
    print(baseline_code[:500] if len(baseline_code) > 500 else baseline_code)
    
    print(f"\nBASELINE EVALUATION: {'PASS' if row['test_passed'] else 'FAIL'}")
    
    # Re-evaluate baseline code to verify
    print("\nRE-EVALUATING BASELINE CODE:")
    baseline_eval = evaluate_code(baseline_code, test_list)
    print(f"  Result: {'PASS' if baseline_eval else 'FAIL'}")
    
    # Generate new code without steering
    print("\n" + "="*40)
    print("REGENERATING WITHOUT STEERING")
    print("="*40)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.model_max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only new tokens
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    print("RAW GENERATED TEXT:")
    print(f"  Length: {len(generated_text)}")
    print(f"  First 200 chars: {repr(generated_text[:200])}")
    
    # Extract code
    extracted_code = extract_code(generated_text, prompt)
    
    print("\nEXTRACTED CODE:")
    print(f"  Length: {len(extracted_code)}")
    print(f"  Code:\n{extracted_code[:500] if len(extracted_code) > 500 else extracted_code}")
    
    # Evaluate
    test_passed = evaluate_code(extracted_code, test_list)
    print(f"\nEVALUATION: {'PASS' if test_passed else 'FAIL'}")
    
    # Now test with steering
    print("\n" + "="*40)
    print("WITH STEERING (coefficient=30)")
    print("="*40)
    
    # Load features
    phase2_5_output = discover_latest_phase_output("2.5")
    if not phase2_5_output:
        print("ERROR: Phase 2.5 output not found!")
        return
    
    features_file = Path(phase2_5_output).parent / "top_20_features.json"
    with open(features_file, 'r') as f:
        top_features = json.load(f)
    
    best_correct = top_features['correct'][0]
    print(f"Using feature: Layer {best_correct['layer']}, Index {best_correct['feature_idx']}, Score {best_correct['separation_score']}")
    
    # Load SAE
    sae = load_gemma_scope_sae(best_correct['layer'], device)
    decoder_direction = sae.W_dec[best_correct['feature_idx']].detach()
    
    # Ensure dtype compatibility
    model_dtype = next(model.parameters()).dtype
    decoder_direction = decoder_direction.to(dtype=model_dtype)
    
    # Apply steering
    hook_fn = create_steering_hook(decoder_direction, 30.0)
    target_module = model.model.layers[best_correct['layer']]
    hook_handle = target_module.register_forward_pre_hook(hook_fn)
    
    try:
        with torch.no_grad():
            outputs_steered = model.generate(
                **inputs,
                max_new_tokens=config.model_max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        generated_text_steered = tokenizer.decode(
            outputs_steered[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        print("RAW GENERATED TEXT (STEERED):")
        print(f"  Length: {len(generated_text_steered)}")
        print(f"  First 200 chars: {repr(generated_text_steered[:200])}")
        
        # Extract code
        extracted_code_steered = extract_code(generated_text_steered, prompt)
        
        print("\nEXTRACTED CODE (STEERED):")
        print(f"  Length: {len(extracted_code_steered)}")
        print(f"  Code:\n{extracted_code_steered[:500] if len(extracted_code_steered) > 500 else extracted_code_steered}")
        
        # Evaluate
        test_passed_steered = evaluate_code(extracted_code_steered, test_list)
        print(f"\nEVALUATION: {'PASS' if test_passed_steered else 'FAIL'}")
        
        # Compare
        print("\n" + "="*40)
        print("COMPARISON")
        print("="*40)
        print(f"Baseline from dataset: {'PASS' if row['test_passed'] else 'FAIL'}")
        print(f"Regenerated (no steering): {'PASS' if test_passed else 'FAIL'}")
        print(f"With steering: {'PASS' if test_passed_steered else 'FAIL'}")
        
        if extracted_code != extracted_code_steered:
            print("\n✓ Code CHANGED with steering")
        else:
            print("\n✗ Code UNCHANGED with steering")
            
    finally:
        hook_handle.remove()

if __name__ == "__main__":
    debug_extraction()