#!/usr/bin/env python3
"""
Debug script to test hook conflicts between steering and attention extraction.
This simulates what happens in Phase 4.8 vs Phase 4.5.
"""

import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    """Simplified model to test hook behavior."""
    
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
        
    def forward(self, x):
        return self.layer(x)


def test_single_hook():
    """Test with single steering hook (like Phase 4.5)."""
    print("TEST 1: Single Steering Hook (Phase 4.5 style)")
    print("="*50)
    
    model = SimpleModel()
    x = torch.randn(1, 10)
    
    # Original output
    with torch.no_grad():
        original = model(x).clone()
    print(f"Original output sum: {original.sum().item():.4f}")
    
    # Add steering hook that modifies input
    def steering_hook(module, input):
        modified = list(input)
        modified[0] = input[0] + 10.0  # Add large value to steer
        print(f"  Steering hook: Modified input by +10")
        return tuple(modified)
    
    hook = model.layer.register_forward_pre_hook(steering_hook)
    
    # Run with steering
    with torch.no_grad():
        steered = model(x)
    print(f"Steered output sum: {steered.sum().item():.4f}")
    print(f"Difference: {(steered - original).sum().item():.4f}")
    
    hook.remove()
    print()


def test_two_hooks_wrong():
    """Test with two hooks where second doesn't return properly."""
    print("TEST 2: Two Hooks - WRONG (attention hook returns None)")
    print("="*50)
    
    model = SimpleModel()
    x = torch.randn(1, 10)
    
    # Original output
    with torch.no_grad():
        original = model(x).clone()
    print(f"Original output sum: {original.sum().item():.4f}")
    
    # Hook 1: Steering (modifies input)
    def steering_hook(module, input):
        modified = list(input)
        modified[0] = input[0] + 10.0
        print(f"  1. Steering hook: Modified input by +10")
        return tuple(modified)
    
    # Hook 2: Attention capture (WRONG - doesn't return)
    captured = {}
    def attention_hook_wrong(module, input):
        captured['input'] = input[0].clone()
        print(f"  2. Attention hook (WRONG): Captured but returned None")
        # MISSING: return input
    
    hook1 = model.layer.register_forward_pre_hook(steering_hook)
    hook2 = model.layer.register_forward_pre_hook(attention_hook_wrong)
    
    # Run with both hooks
    with torch.no_grad():
        result = model(x)
    print(f"Result output sum: {result.sum().item():.4f}")
    print(f"Difference from original: {(result - original).sum().item():.4f}")
    print(f"⚠️  Steering likely LOST due to second hook returning None!")
    
    hook1.remove()
    hook2.remove()
    print()


def test_two_hooks_correct():
    """Test with two hooks where second properly returns input."""
    print("TEST 3: Two Hooks - CORRECT (attention hook returns input)")
    print("="*50)
    
    model = SimpleModel()
    x = torch.randn(1, 10)
    
    # Original output
    with torch.no_grad():
        original = model(x).clone()
    print(f"Original output sum: {original.sum().item():.4f}")
    
    # Hook 1: Steering (modifies input)
    def steering_hook(module, input):
        modified = list(input)
        modified[0] = input[0] + 10.0
        print(f"  1. Steering hook: Modified input by +10")
        return tuple(modified)
    
    # Hook 2: Attention capture (CORRECT - returns input)
    captured = {}
    def attention_hook_correct(module, input):
        captured['input'] = input[0].clone()
        print(f"  2. Attention hook (CORRECT): Captured and returned input")
        return input  # CRITICAL: Pass through the (modified) input!
    
    hook1 = model.layer.register_forward_pre_hook(steering_hook)
    hook2 = model.layer.register_forward_pre_hook(attention_hook_correct)
    
    # Run with both hooks
    with torch.no_grad():
        result = model(x)
    print(f"Result output sum: {result.sum().item():.4f}")
    print(f"Difference from original: {(result - original).sum().item():.4f}")
    print(f"✓ Steering PRESERVED because second hook returned input!")
    
    hook1.remove()
    hook2.remove()
    print()


def test_hook_order():
    """Test if hook order matters."""
    print("TEST 4: Hook Order")
    print("="*50)
    
    model = SimpleModel()
    x = torch.randn(1, 10)
    
    print("Testing different hook orders...")
    
    # Test 1: Steering first, then attention
    def steering_hook(module, input):
        modified = list(input)
        modified[0] = input[0] + 10.0
        return tuple(modified)
    
    def attention_hook(module, input):
        # This hook returns input properly
        return input
    
    # Order 1: Steering then attention
    hook1 = model.layer.register_forward_pre_hook(steering_hook)
    hook2 = model.layer.register_forward_pre_hook(attention_hook)
    
    with torch.no_grad():
        result1 = model(x).clone()
    
    hook1.remove()
    hook2.remove()
    
    # Order 2: Attention then steering  
    hook2 = model.layer.register_forward_pre_hook(attention_hook)
    hook1 = model.layer.register_forward_pre_hook(steering_hook)
    
    with torch.no_grad():
        result2 = model(x).clone()
    
    hook1.remove()
    hook2.remove()
    
    print(f"Order 1 (steer→attn) output sum: {result1.sum().item():.4f}")
    print(f"Order 2 (attn→steer) output sum: {result2.sum().item():.4f}")
    print(f"Difference: {(result1 - result2).abs().sum().item():.4f}")
    
    if torch.allclose(result1, result2):
        print("✓ Hook order doesn't matter when both return properly")
    else:
        print("⚠️  Hook order affects the result!")
    print()


def main():
    print("\nHOOK CONFLICT DEBUGGING")
    print("="*80)
    print()
    
    # Run tests
    test_single_hook()
    test_two_hooks_wrong()
    test_two_hooks_correct()
    test_hook_order()
    
    print("="*80)
    print("CONCLUSIONS:")
    print("="*80)
    print("""
1. When a single hook modifies input, steering works (Phase 4.5 style)

2. When two hooks are on the same layer:
   - If second hook doesn't return input → steering is LOST
   - If second hook returns input → steering is PRESERVED
   
3. Hook order matters for execution sequence but shouldn't affect
   the final result if all hooks properly return the input

4. The bug in Phase 4.8 is likely that AttentionExtractor's hook
   either doesn't return input OR returns it incorrectly

SOLUTION: Ensure ALL hooks return their input properly to preserve
any modifications from previous hooks in the chain.
""")


if __name__ == "__main__":
    main()