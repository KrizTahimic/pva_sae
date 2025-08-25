#!/usr/bin/env python3
"""
Test script to verify Phase 1 generation fix.
Tests that problematic tasks don't hang the generation.
"""

import time
import torch
from pathlib import Path
from common.config import Config
from common.utils import detect_device
from common_simplified.model_loader import load_model_and_tokenizer
from common.prompt_utils import PromptBuilder
from common_simplified.helpers import extract_code, evaluate_code

def test_generation_timeout():
    """Test that generation doesn't hang on difficult problems."""
    
    print("Testing Phase 1 generation fix...")
    print(f"MAX_NEW_TOKENS set to: {Config().model_max_new_tokens}")
    
    # Load model
    device = detect_device()
    print(f"Using device: {device}")
    
    config = Config()
    model, tokenizer = load_model_and_tokenizer(
        config.model_name,
        device=device
    )
    model.eval()
    
    # Create a deliberately complex prompt that might cause long generation
    complex_prompt = PromptBuilder.build_prompt(
        problem_description="""Write a function that implements a complete chess game engine with:
        - Full move validation for all piece types
        - Check and checkmate detection
        - En passant and castling rules
        - Board state management
        - Move history tracking
        - Undo/redo functionality
        - FEN notation support
        - Performance optimization using bitboards
        The function should handle all edge cases and be production-ready.""",
        test_cases="assert chess_engine() is not None"
    )
    
    print("\nTesting generation with complex prompt...")
    print("This should complete within reasonable time (< 30 seconds)")
    
    start_time = time.time()
    
    try:
        inputs = tokenizer(
            complex_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.model_max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"\n✅ Generation completed in {generation_time:.2f} seconds")
        print(f"Generated {len(generated_text)} characters")
        
        if generation_time > 60:
            print("⚠️  WARNING: Generation still took over 60 seconds")
            print("Consider reducing MAX_NEW_TOKENS further")
        elif generation_time > 30:
            print("⚠️  Generation took 30-60 seconds - acceptable but could be improved")
        else:
            print("✅ Generation time is reasonable")
            
        # Show snippet of generated code
        extracted_code = extract_code(generated_text, complex_prompt)
        print(f"\nFirst 500 chars of generated code:")
        print(extracted_code[:500])
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return False
    
    print("\n" + "="*60)
    print("Test complete. The fix should prevent hanging on difficult tasks.")
    print("Recommendations:")
    print("1. MAX_NEW_TOKENS has been reduced to 800 (from 2000)")
    print("2. Warning logs added for slow generations (>60s)")
    print("3. Warning logs added for excessively long code (>3000 chars)")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_generation_timeout()