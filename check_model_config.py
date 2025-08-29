#!/usr/bin/env python3
"""
Quick script to check Gemma-2-2b model configuration, especially attention heads.
"""

from transformers import AutoConfig, AutoModelForCausalLM
import torch

def check_gemma_config():
    """Load and inspect Gemma-2-2b configuration."""
    
    model_name = "google/gemma-2-2b"
    
    print(f"Checking configuration for: {model_name}")
    print("=" * 60)
    
    # Load just the config (lightweight)
    config = AutoConfig.from_pretrained(model_name)
    
    # Print key attention-related parameters
    print("\nAttention Configuration:")
    print(f"  - Number of attention heads: {config.num_attention_heads}")
    
    # Check for key-value heads (for grouped query attention)
    if hasattr(config, 'num_key_value_heads'):
        print(f"  - Number of key-value heads: {config.num_key_value_heads}")
        print(f"  - Grouped Query Attention: {'Yes' if config.num_key_value_heads < config.num_attention_heads else 'No'}")
    
    print(f"\nModel Architecture:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Number of hidden layers: {config.num_hidden_layers}")
    print(f"  - Head dimension: {config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads}")
    print(f"  - Vocab size: {config.vocab_size}")
    
    # Additional architecture details
    if hasattr(config, 'intermediate_size'):
        print(f"  - Intermediate size (FFN): {config.intermediate_size}")
    
    print(f"\nTotal Parameters (approximate): {config.num_hidden_layers * config.hidden_size * config.hidden_size * 12 / 1e9:.2f}B")
    
    return config

if __name__ == "__main__":
    config = check_gemma_config()
    
    # Save to file for reference
    print("\n" + "=" * 60)
    print("Saving configuration details to gemma_config.txt...")
    
    with open("gemma_config.txt", "w") as f:
        f.write(f"Gemma-2-2b Configuration\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Number of attention heads: {config.num_attention_heads}\n")
        f.write(f"Number of hidden layers: {config.num_hidden_layers}\n")
        f.write(f"Hidden size: {config.hidden_size}\n")
        if hasattr(config, 'num_key_value_heads'):
            f.write(f"Number of key-value heads: {config.num_key_value_heads}\n")
    
    print("Done! Check gemma_config.txt for details.")