"""
Debug script to investigate Phase 4.5 generation issues.
"""

import torch
import pandas as pd
from pathlib import Path
from common.config import Config
from common.prompt_utils import PromptBuilder
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import extract_code, evaluate_code
from common.utils import detect_device, discover_latest_phase_output
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae
from common.steering_metrics import create_steering_hook
import json

def debug_generation():
    # Load config
    config = Config()
    device = detect_device()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        config.model_name,
        device=device
    )
    model.eval()
    
    # Load Phase 3.6 baseline data
    print("\nLoading Phase 3.6 baseline data...")
    phase3_6_output = discover_latest_phase_output("3.6")
    if not phase3_6_output:
        print("Phase 3.6 output not found!")
        return
    
    baseline_file = Path(phase3_6_output).parent / "dataset_hyperparams_temp_0_0.parquet"
    baseline_data = pd.read_parquet(baseline_file)
    print(f"Loaded {len(baseline_data)} problems")
    
    # Get one initially incorrect problem
    incorrect_data = baseline_data[baseline_data['test_passed'] == False]
    if len(incorrect_data) == 0:
        print("No incorrect problems found!")
        return
    
    test_row = incorrect_data.iloc[0]
    print(f"\nUsing task {test_row['task_id']} (initially incorrect)")
    
    # Extract prompt and test cases
    prompt = test_row['prompt']
    test_cases = json.loads(test_row['test_list']) if isinstance(test_row['test_list'], str) else test_row['test_list']
    
    print("\n" + "="*60)
    print("PROMPT:")
    print("="*60)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    print("\n" + "="*60)
    print("BASELINE (NO STEERING) GENERATION:")
    print("="*60)
    
    # Generate without steering
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
    print(repr(generated_text[:500]))
    
    print("\n" + "-"*40)
    print("EXTRACTED CODE:")
    extracted_code = extract_code(generated_text, prompt)
    print(extracted_code[:500] if len(extracted_code) > 500 else extracted_code)
    
    print("\n" + "-"*40)
    print("EVALUATION:")
    test_passed = evaluate_code(extracted_code, test_cases)
    print(f"Tests passed: {test_passed}")
    
    # Now test with steering
    print("\n" + "="*60)
    print("WITH STEERING (coefficient=30.0):")
    print("="*60)
    
    # Load Phase 2.5 features
    phase2_5_output = discover_latest_phase_output("2.5")
    if not phase2_5_output:
        print("Phase 2.5 output not found!")
        return
    
    features_file = Path(phase2_5_output).parent / "top_20_features.json"
    with open(features_file, 'r') as f:
        top_features = json.load(f)
    
    best_correct_feature = top_features['correct'][0]
    print(f"Using feature: Layer {best_correct_feature['layer']}, Index {best_correct_feature['feature_idx']}")
    
    # Load SAE and get decoder direction
    correct_sae = load_gemma_scope_sae(best_correct_feature['layer'], device)
    decoder_direction = correct_sae.W_dec[best_correct_feature['feature_idx']].detach()
    
    # Ensure correct dtype
    model_dtype = next(model.parameters()).dtype
    decoder_direction = decoder_direction.to(dtype=model_dtype)
    
    # Create and register steering hook
    hook_fn = create_steering_hook(decoder_direction, 30.0)
    target_module = model.model.layers[best_correct_feature['layer']]
    hook_handle = target_module.register_forward_pre_hook(hook_fn)
    
    try:
        # Generate with steering
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
        print(repr(generated_text_steered[:500]))
        
        print("\n" + "-"*40)
        print("EXTRACTED CODE (STEERED):")
        extracted_code_steered = extract_code(generated_text_steered, prompt)
        print(extracted_code_steered[:500] if len(extracted_code_steered) > 500 else extracted_code_steered)
        
        print("\n" + "-"*40)
        print("EVALUATION (STEERED):")
        test_passed_steered = evaluate_code(extracted_code_steered, test_cases)
        print(f"Tests passed: {test_passed_steered}")
        
        print("\n" + "-"*40)
        print("COMPARISON:")
        print(f"Baseline: {'PASS' if test_row['test_passed'] else 'FAIL'}")
        print(f"Regenerated (no steering): {'PASS' if test_passed else 'FAIL'}")
        print(f"With steering: {'PASS' if test_passed_steered else 'FAIL'}")
        
        if extracted_code != extracted_code_steered:
            print("\nCode changed with steering!")
        else:
            print("\nCode identical with/without steering")
            
    finally:
        # Remove hook
        hook_handle.remove()

if __name__ == "__main__":
    debug_generation()