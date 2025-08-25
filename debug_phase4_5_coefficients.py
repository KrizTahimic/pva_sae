"""
Debug script to test steering across multiple coefficients.
"""

import torch
import pandas as pd
from pathlib import Path
from common.config import Config
from common_simplified.model_loader import load_model_and_tokenizer
from common_simplified.helpers import extract_code, evaluate_code
from common.utils import detect_device, discover_latest_phase_output
from phase2_5_simplified.sae_analyzer import load_gemma_scope_sae
from common.steering_metrics import create_steering_hook
import json

def test_steering_coefficients():
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
    
    # Get initially incorrect problems
    incorrect_data = baseline_data[baseline_data['test_passed'] == False]
    print(f"Found {len(incorrect_data)} initially incorrect problems")
    
    # Load Phase 2.5 features
    phase2_5_output = discover_latest_phase_output("2.5")
    if not phase2_5_output:
        print("Phase 2.5 output not found!")
        return
    
    features_file = Path(phase2_5_output).parent / "top_20_features.json"
    with open(features_file, 'r') as f:
        top_features = json.load(f)
    
    best_correct_feature = top_features['correct'][0]
    print(f"\nUsing 'correct' feature: Layer {best_correct_feature['layer']}, Index {best_correct_feature['feature_idx']}")
    
    # Load SAE and get decoder direction
    correct_sae = load_gemma_scope_sae(best_correct_feature['layer'], device)
    decoder_direction = correct_sae.W_dec[best_correct_feature['feature_idx']].detach()
    
    # Ensure correct dtype
    model_dtype = next(model.parameters()).dtype
    decoder_direction = decoder_direction.to(dtype=model_dtype)
    
    # Test different coefficients (fewer for speed)
    coefficients = [0, 20, 30, 50]
    
    print("\n" + "="*80)
    print("TESTING STEERING COEFFICIENTS")
    print("="*80)
    
    # Test first 3 incorrect problems
    for idx, (_, row) in enumerate(incorrect_data.head(3).iterrows()):
        task_id = row['task_id']
        prompt = row['prompt']
        test_cases = json.loads(row['test_list']) if isinstance(row['test_list'], str) else row['test_list']
        
        print(f"\n[Task {task_id}]")
        
        results = []
        for coeff in coefficients:
            if coeff == 0:
                # No steering (baseline)
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
                
                generated_text = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                extracted_code = extract_code(generated_text, prompt)
                test_passed = evaluate_code(extracted_code, test_cases)
                results.append((coeff, test_passed))
                
            else:
                # With steering
                hook_fn = create_steering_hook(decoder_direction, float(coeff))
                target_module = model.model.layers[best_correct_feature['layer']]
                hook_handle = target_module.register_forward_pre_hook(hook_fn)
                
                try:
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
                    
                    generated_text = tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    extracted_code = extract_code(generated_text, prompt)
                    test_passed = evaluate_code(extracted_code, test_cases)
                    results.append((coeff, test_passed))
                finally:
                    hook_handle.remove()
        
        # Print results for this task
        result_str = " | ".join([f"{c}: {'✓' if p else '✗'}" for c, p in results])
        print(f"  {result_str}")
        
        # Check if any coefficient corrected the problem
        if any(p for c, p in results if c > 0):
            print(f"  → Corrected at coefficient(s): {[c for c, p in results if c > 0 and p]}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✓ = test passed, ✗ = test failed")
    print("Coefficient 0 = no steering (baseline)")

if __name__ == "__main__":
    test_steering_coefficients()